#!/usr/bin/env python3
"""Analytical capture-separation and conservative language-label checks."""
import array
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_steering import compare,separation
from evaluate_language_steering import language,constraint

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary)
    for split in ('train','validation'):
        d=root/split;d.mkdir()
        (d/'capture.json').write_text(json.dumps(dict(layers=1,width=2,pairs=2,
            model_sha256='0'*64,dataset_sha256='1'*64)))
        if split=='validation':
            meta=json.loads((d/'capture.json').read_text())
            meta.update(capture_kind='paired_instructions_v1',position='last_rendered_prompt_token_including_assistant_opening',
                        backend='production_prefill_cpu_completion')
            (d/'capture.json').write_text(json.dumps(meta))
        for i in range(2):
            for pole,sign in (('positive',1),('negative',-1)):
                a=array.array('f',[sign*(i+1),2])
                if sys.byteorder!='little':a.byteswap()
                (d/f'{i:04d}-{pole}.f32').write_bytes(a.tobytes())
    report=separation(root/'train',root/'validation')['layers'][0]
    assert report['validation_accuracy']==1 and report['validation_margin']>0
    same=compare(root/'train',root/'validation')['layers'][0]
    assert same['max_activation_relative_l2']==0 and same['max_contrast_relative_l2']==0
    assert same['min_contrast_cosine']==1
assert language('La temperatura è una misura del calore, che si legge con un termometro.')['label']=='it'
assert language('The temperature is measured with a thermometer and it can change with time.')['label']=='en'
assert language('42')['label']=='mixed_or_unknown'
assert language('El metal es un buen conductor porque los electrones pueden transferir energía y calor.')['label']!='it'
assert constraint('max(low, min(n, high))',dict(kind='python',function='clamp')) is False
print('capture separation, backend comparison and language/definition checks: PASS')
