#!/usr/bin/env python3
"""Compare F32 encoder dumps; limits are fixed before evaluating candidates."""
import argparse
from array import array
import json
import hashlib
try:
    import numpy as np
except ImportError:
    np = None
import math
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("reference",type=Path)
p.add_argument("candidate",type=Path)
p.add_argument("--output",type=Path)
p.add_argument("--stage",action="append",choices=['pos_embd']+[f'layer_out-{i}' for i in range(27)]+['std_scaled','embedding'],help="Compare selected stages; default: all 30 stages")
a=p.parse_args()
if a.output and a.output.exists():
    p.error('--output must be a new report path to avoid stale comparison results')
results=[];ok=True
for name in a.stage or ['pos_embd']+[f'layer_out-{i}' for i in range(27)]+['std_scaled','embedding']:
    rp=a.reference/(name+'.f32')
    if not rp.exists():rp=a.reference/(name+'.bin')
    cp=a.candidate/(name+'.f32')
    if not rp.exists() or not cp.exists():raise SystemExit('missing dump: '+name)
    rb=rp.read_bytes();cb=cp.read_bytes()
    if len(rb)!=len(cb) or not rb or len(rb)%4:raise SystemExit('invalid dump size: '+name)
    if np is not None:
        r=np.frombuffer(rb,dtype='<f4').astype(np.float64)
        c=np.frombuffer(cb,dtype='<f4').astype(np.float64)
        if not np.isfinite(r).all() or not np.isfinite(c).all():raise SystemExit('nonfinite dump: '+name)
        ref=float(np.sum(r*r));cand=float(np.sum(c*c));delta=r-c
        rms=math.sqrt(float(np.sum(delta*delta))/max(ref,1e-30))
        cosine=float(np.sum(r*c))/math.sqrt(max(ref*cand,1e-30))
        maximum=float(np.max(np.abs(delta)))
    else:
        r=array('f');r.frombytes(rb);c=array('f');c.frombytes(cb)
        if not all(math.isfinite(x) for x in r) or not all(math.isfinite(x) for x in c):raise SystemExit('nonfinite dump: '+name)
        ref=math.fsum(x*x for x in r);cand=math.fsum(x*x for x in c)
        rms=math.sqrt(math.fsum((x-y)**2 for x,y in zip(r,c))/max(ref,1e-30))
        cosine=math.fsum(x*y for x,y in zip(r,c))/math.sqrt(max(ref*cand,1e-30))
        maximum=max(abs(x-y) for x,y in zip(r,c))
    passed=rms<=0.005 and cosine>=0.99998
    results.append({'stage':name,'relative_rms':rms,'cosine':cosine,'max_abs':maximum,'pass':passed,'elements':len(rb)//4,'reference_sha256':hashlib.sha256(rb).hexdigest(),'candidate_sha256':hashlib.sha256(cb).hexdigest()})
    ok &= passed
    print(name,f'rms {rms:.6g} cosine {cosine:.9f}', 'PASS' if passed else 'FAIL')
report={'limits':{'relative_rms':0.005,'cosine':0.99998},'pass':ok,'stages':results}
if a.output:
    with a.output.open('x') as output:
        output.write(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if ok else 1)
