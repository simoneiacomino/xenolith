#!/usr/bin/env python3
"""Prepare a norm-calibrated additive verbosity sweep at two individual layers."""
import argparse
import json
import math
from pathlib import Path
import shutil
from analyze_steering import separation
from build_steering import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--validation-capture',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--relative-levels',default='0.1,0.3,0.6,-0.3')
    p.add_argument('--layers',help='Explicit comma-separated diagnostic shortlist instead of automatic selection')
    a=p.parse_args();levels=[float(s) for s in a.relative_levels.split(',')]
    if not levels or any(not math.isfinite(s) or s==0 for s in levels):p.error('Expected finite nonzero levels')
    report=separation(a.capture,a.validation_capture)
    ranked=sorted((r for r in report['layers'] if 4<=r['layer']<=27 and
                   r['validation_accuracy']>=0.8 and r['validation_margin']>0),
                  key=lambda r:(r['validation_accuracy'],r['validation_margin']/max(r['direction_norm'],1e-30)),reverse=True)
    selected=[]
    if a.layers:
        wanted=[int(x) for x in a.layers.split(',')]
        if len(set(wanted))!=len(wanted):p.error('Duplicate layer')
        by_layer={r['layer']:r for r in ranked}
        if any(l not in by_layer for l in wanted):p.error('Explicit layers must pass the capture-separation screen')
        selected=[by_layer[l] for l in wanted]
    else:
        for r in ranked:
            if all(abs(r['layer']-s['layer'])>=8 for s in selected):selected.append(r)
            if len(selected)==2:break
    if not selected:raise SystemExit('No layer passes the predeclared capture-separation screen')
    a.out.mkdir(parents=True,exist_ok=False);configs=[]
    for row in selected:
        layer=row['layer'];out=a.out/f'layer-{layer}'
        identity=build(a.capture,a.dataset,out,'additive',[layer],
            'https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF',
            '7b92b5b28818151e8669af2e45e88d6086f490dd',('succinct','detailed'))
        target=a.registry/identity
        if target.exists():
            for name in ('manifest.json','vectors.f32'):assert (target/name).read_bytes()==(out/name).read_bytes()
        else:shutil.copytree(out,target)
        for relative in levels:
            configs.append(dict(name=f'verbosity_l{layer}_r{relative}',artifact=identity,
                strength=relative/row['direction_activation_ratio'],layer=layer,relative_displacement_estimate=relative))
    (a.out/'configs.json').write_text(json.dumps(configs,indent=2)+'\n')
    (a.out/'separation.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'selected-layers.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(f'{len(configs)} settings prepared at layers {[r["layer"] for r in selected]}')


if __name__=='__main__':main()
