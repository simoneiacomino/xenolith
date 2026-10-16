#!/usr/bin/env python3
"""Independent F32 Gemma4 vision oracle over the locked F16 GGUF weights.

Equations follow pinned Transformers; dense inputs round to F16 to match
ggml's F16 dot-product path. This is test tooling, not a runtime backend.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np
from vision_artifacts import inventory,digest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("mmproj",type=Path)
p.add_argument("prepared",type=Path)
p.add_argument("output",type=Path,help="New directory; existing paths are rejected")
p.add_argument("--layers",type=int,default=27)
a=p.parse_args()
lock=json.loads((Path(__file__).resolve().parent.parent/"tests/vision/manifest.json").read_text())
if np.__version__ != lock['references']['numpy_version']:
    p.error(f"requires NumPy {lock['references']['numpy_version']}; found {np.__version__}")
if not 1 <= a.layers <= 27:
    p.error('--layers must be between 1 and 27; use 27 for a complete encoder reference')
artifact=next(v for v in lock['artifacts'] if v['filename']=='mmproj-F16.gguf')
if not a.mmproj.is_file() or a.mmproj.stat().st_size != artifact['size_bytes'] or digest(a.mmproj) != artifact['sha256']:
    p.error('mmproj must match the pinned mmproj-F16.gguf SHA-256')
raw=a.prepared.read_bytes()
if len(raw) < 8:
    p.error('prepared RGB input must contain a width/height header')
w,h=struct.unpack('<II',raw[:8])
if not w or not h or w%48 or h%48 or w*h > 1120*48*48 or len(raw) != 8+w*h*3:
    p.error('prepared RGB input must have positive dimensions divisible by 48, at most 1120 pooled rows, and exactly width*height*3 pixel bytes')
try:
    a.output.mkdir(parents=True,exist_ok=False)
except FileExistsError:
    p.error('output must be a new directory to avoid stale reference stages')
info=inventory(a.mmproj);tensors={t['name']:t for t in info['tensors']}
def tensor(name):
    t=tensors[name]
    return np.memmap(a.mmproj,dtype='<f2' if t['type']==1 else '<f4',mode='r',
                     offset=info['data_offset']+t['offset'],shape=tuple(reversed(t['shape'])))
def mm(x,name):
    return x.astype(np.float16).astype(np.float32)@tensor(name).astype(np.float32).T
def norm(x,name=None):
    y=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True,dtype=np.float32)+np.float32(1e-6))
    return y if name is None else y*tensor(name)
def save(name,x):
    if not np.isfinite(x).all():raise ValueError("nonfinite reference stage: "+name)
    x.astype('<f4').tofile(a.output/(name+'.f32'))
    (a.output/(name+'.json')).write_text(json.dumps({'shape':list(x.shape),'sha256':hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()}))

pixels=np.frombuffer(raw[8:],np.uint8).reshape(h,w,3).astype(np.float32)/255
px,py=w//16,h//16;n=px*py
patches=(pixels*2-1).reshape(py,16,px,16,3).transpose(0,2,4,1,3).reshape(n,768)
x=patches.astype(np.float16).astype(np.float32)@tensor('v.patch_embd.weight').reshape(1152,768).astype(np.float32).T
pos=tensor('v.position_embd.weight');positions=np.arange(n)
x=x+pos[0,positions%px]+pos[1,positions//px];save('pos_embd',x)
freq=1/(100**(np.arange(18,dtype=np.float32)/18))
def rope(y):
    y=y.reshape(n,16,72).copy()
    for offset,coords in [(0,positions%px),(36,positions//px)]:
        angle=coords[:,None].astype(np.float32)*freq[None,:]
        c=np.cos(angle)[:,None,:];s=np.sin(angle)[:,None,:]
        u=y[:,:,offset:offset+18].copy();v=y[:,:,offset+18:offset+36].copy()
        y[:,:,offset:offset+18]=u*c-v*s;y[:,:,offset+18:offset+36]=u*s+v*c
    return y
for layer in range(a.layers):
    pre=f'v.blk.{layer}.'
    inp=norm(x,pre+'ln1.weight')
    if layer==0:save('layer_inp_normed-0',inp)
    q=norm(mm(inp,pre+'attn_q.weight').reshape(n,16,72),pre+'attn_q_norm.weight')
    k=norm(mm(inp,pre+'attn_k.weight').reshape(n,16,72),pre+'attn_k_norm.weight')
    if layer==0:save('Qcur-0',q)
    q=rope(q);k=rope(k)
    v=norm(mm(inp,pre+'attn_v.weight').reshape(n,16,72))
    if layer==0:
        save('Qcur_pos-0',q);save('Kcur_pos-0',k);save('Vcur_normed-0',v)
    attention=np.empty_like(v)
    for head in range(16):
        score=q[:,head]@k[:,head].T
        score-=score.max(axis=-1,keepdims=True)
        probability=np.exp(score);probability/=probability.sum(axis=-1,keepdims=True)
        attention[:,head]=probability@v[:,head]
    attn=norm(mm(attention.reshape(n,1152),pre+'attn_out.weight'),pre+'attn_post_norm.weight')
    if layer==0:save('attn_post_normed-0',attn)
    x=x+attn
    inp=norm(x,pre+'ln2.weight');gate=mm(inp,pre+'ffn_gate.weight');up=mm(inp,pre+'ffn_up.weight')
    if layer==0:save('ffn_gate-0',gate);save('ffn_up-0',up)
    gelu=0.5*gate*(1+np.tanh(np.float32(math.sqrt(2/math.pi))*(gate+np.float32(0.044715)*gate*gate*gate)))
    if layer==0:save('ffn_geglu-0',gelu*up)
    ffn=norm(mm(gelu*up,pre+'ffn_down.weight'),pre+'ffn_post_norm.weight')
    if layer==0:save('ffn_post_normed-0',ffn)
    x=x+ffn
    save(f'layer_out-{layer}',x)
    print('layer',layer,flush=True)
pooled=x.reshape(py//3,3,px//3,3,1152).mean(axis=(1,3),dtype=np.float32).reshape(-1,1152)*np.float32(math.sqrt(1152))
save('pooled',pooled)
scaled=(pooled-tensor('v.std_bias'))*tensor('v.std_scale');save('std_scaled',scaled)
embedding=mm(norm(scaled),'mm.input_projection.weight');save('embedding',embedding)
(a.output/'manifest.json').write_text(json.dumps({'numpy_version':np.__version__,'mmproj_sha256':artifact['sha256'],
    'prepared_sha256':hashlib.sha256(raw).hexdigest(),'layers':a.layers,'complete_encoder':a.layers==27,
    'rows':len(embedding),'width':2816,'revision':'numpy-gemma4v-f16-dots/v1'},indent=2)+'\n')
