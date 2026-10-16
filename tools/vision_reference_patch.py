#!/usr/bin/env python3
"""Align the isolated CPU vision oracle's GELU precision with F32 intermediates.

Apply only to a separate copy of the pinned llama.cpp sources. Language-model
comparisons keep the original CPU library, including its FP16 GELU tables.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
a = p.parse_args()
lock = json.loads((Path(__file__).resolve().parent.parent / 'tests/vision/manifest.json').read_text())['references']
revision = subprocess.run(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True, capture_output=True)
if revision.returncode or revision.stdout.strip() != lock['llama_cpp_commit']:
    p.error('source must be a separate checkout of pinned llama.cpp commit ' + lock['llama_cpp_commit'])
file = a.source / 'ggml/src/ggml-cpu/vec.h'
original = file.read_bytes()
old = lock['encoder_adapter']['vec_h_original_sha256']
new = lock['encoder_adapter']['vec_h_adapted_sha256']
digest = hashlib.sha256(original).hexdigest()
if digest == new:
    print('F32 GELU adapter already applied')
elif digest != old:
    raise SystemExit('pinned vec.h checksum mismatch')
else:
    changed = original.replace(b'#define GGML_GELU_FP16\n', b'/* Vision oracle: evaluate GELU in F32, without the FP16 LUT shortcut. */\n')
    if hashlib.sha256(changed).hexdigest() != new:
        raise SystemExit('adapted vec.h checksum mismatch')
    file.write_bytes(changed)
    print('F32 GELU adapter applied', new)
