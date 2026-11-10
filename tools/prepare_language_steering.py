#!/usr/bin/env python3
"""Build a bounded single-layer language sweep from capture separation reports."""
import argparse
import json
import math
from pathlib import Path
import shutil
from build_steering import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-prefix',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--relative-levels',default='0.01,0.03,0.1,-0.1')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    levels=[float(x) for x in a.relative_levels.split(',')]
    if not levels or any(not math.isfinite(x) or x==0 for x in levels):p.error('Expected finite nonzero levels')
    configs=[];selected={}
    for recipe in ('completions','instructions'):
        report=json.loads(Path(f'{a.capture_prefix}-{recipe}-separation.json').read_text())
        ranked=sorted((r for r in report['layers'] if 4<=r['layer']<=27 and
                       r['validation_accuracy']>=0.8 and r['validation_margin']>0),
                      key=lambda r:(r['validation_accuracy'],r['validation_margin']/max(r['direction_norm'],1e-30)),reverse=True)
        chosen=[]
        for row in ranked:
            if all(abs(row['layer']-c['layer'])>=8 for c in chosen):chosen.append(row)
            if len(chosen)==2:break
        selected[recipe]=chosen
        for row in chosen:
            layer=row['layer'];directory=a.out/f'{recipe}-l{layer}'
            identity=build(Path(f'{a.capture_prefix}-train-{recipe}'),
                Path(f'steering/examples/language/train-{recipe}.jsonl'),directory,'additive',[layer],
                'https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF',
                '7b92b5b28818151e8669af2e45e88d6086f490dd',('italian','english'))
            registered=a.registry/identity
            if registered.exists():
                for name in ('manifest.json','vectors.f32'):
                    assert (registered/name).read_bytes()==(directory/name).read_bytes()
            else:shutil.copytree(directory,registered)
            for relative in levels:
                configs.append(dict(name=f'{recipe}_l{layer}_r{relative}',artifact=identity,
                    strength=relative/row['direction_activation_ratio'],relative_displacement_estimate=relative,
                    layer=layer,capture_recipe=recipe))
    (a.out/'configs.json').write_text(json.dumps(configs,indent=2)+'\n')
    (a.out/'selected-layers.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(f'{len(configs)} fixed candidate settings prepared; all imports are validated again by the runtime')


if __name__=='__main__':main()
