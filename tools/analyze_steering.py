#!/usr/bin/env python3
"""Inspect capture stability and held-out contrast separation without inference."""
import argparse
import array
import json
import math
from pathlib import Path
import statistics
from build_steering import read_floats


def corpus(directory):
    meta=json.loads((directory/'capture.json').read_text())
    # Legacy v1 captures used exactly these defaults but omitted the fields.
    meta.setdefault('capture_kind','paired_instructions_v1')
    meta.setdefault('position','last_rendered_prompt_token_including_assistant_opening')
    meta.setdefault('backend','production_prefill_cpu_completion')
    meta.setdefault('site','post_layer_residual_v1')
    count=meta['layers']*meta['width']
    return meta, {pole:[read_floats(directory/f'{i:04d}-{pole}.f32',count)
                       for i in range(meta['pairs'])] for pole in ('positive','negative')}


def norm(values):
    return math.sqrt(math.fsum(x*x for x in values))


def compare(left,right):
    lm,la=corpus(left);rm,ra=corpus(right)
    for key in ('dataset_sha256','model_sha256','layers','width','pairs','position','site'):
        assert lm.get(key)==rm.get(key),key
    result=[];w=lm['width']
    for layer in range(lm['layers']):
        errors=[];contrasts=[];cosines=[]
        for i in range(lm['pairs']):
            rows={k:[a[i][layer*w:(layer+1)*w] for a in (la[k],ra[k])]
                  for k in ('positive','negative')}
            for l,r in rows.values(): errors.append(norm(a-b for a,b in zip(l,r))/max(norm(l),1e-30))
            dl=[a-b for a,b in zip(rows['positive'][0],rows['negative'][0])]
            dr=[a-b for a,b in zip(rows['positive'][1],rows['negative'][1])]
            contrasts.append(norm(a-b for a,b in zip(dl,dr))/max(norm(dl),1e-30))
            cosines.append(math.fsum(a*b for a,b in zip(dl,dr))/max(norm(dl)*norm(dr),1e-30))
        result.append(dict(layer=layer,max_activation_relative_l2=max(errors),
                           max_contrast_relative_l2=max(contrasts),min_contrast_cosine=min(cosines)))
    return dict(left=str(left),right=str(right),layers=result)


def separation(train,validation):
    tm,ta=corpus(train);vm,va=corpus(validation);w=tm['width']
    for key in ('layers','width','model_sha256','position','capture_kind','backend','site'):
        assert tm.get(key)==vm.get(key),key
    result=[]
    for layer in range(tm['layers']):
        means={pole:[math.fsum(a[layer*w+k] for a in ta[pole])/tm['pairs'] for k in range(w)]
               for pole in ('positive','negative')}
        v=[p-n for p,n in zip(means['positive'],means['negative'])]
        center=[(p+n)/2 for p,n in zip(means['positive'],means['negative'])]
        vn=norm(v)
        scores={pole:[math.fsum((a[layer*w+k]-center[k])*v[k] for k in range(w))/max(vn,1e-30)
                       for a in va[pole]] for pole in ('positive','negative')}
        accuracy=(sum(s>0 for s in scores['positive'])+sum(s<0 for s in scores['negative']))/(2*vm['pairs'])
        activation=statistics.median(norm(a[layer*w:(layer+1)*w]) for rows in ta.values() for a in rows)
        result.append(dict(layer=layer,direction_norm=vn,activation_norm=activation,
            direction_activation_ratio=vn/max(activation,1e-30),validation_accuracy=accuracy,
            validation_margin=min(scores['positive'])-max(scores['negative']),scores=scores))
    return dict(train=str(train),validation=str(validation),layers=result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--compare-backends',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=compare(a.train,a.validation) if a.compare_backends else separation(a.train,a.validation)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for row in result['layers']:print({k:v for k,v in row.items() if k!='scores'})


if __name__=='__main__':main()
