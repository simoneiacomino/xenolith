#!/usr/bin/env python3
"""Exercise the real projector loader with sparse GGUFs from the audited inventory."""
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def pack(fmt, *values):
    return struct.pack('<' + fmt, *values)


def string(value):
    data = value.encode()
    return pack('Q', len(data)) + data


def value(v):
    if isinstance(v, bool):
        return 7, pack('?', v)
    if isinstance(v, str):
        return 8, string(v)
    if isinstance(v, int):
        return 4, pack('I', v)
    if isinstance(v, float):
        return 6, pack('f', v)
    if isinstance(v, list):
        items = [value(x) for x in v]
        assert items and all(t == items[0][0] for t, _ in items)
        return 9, pack('IQ', items[0][0], len(items)) + b''.join(b for _, b in items)
    raise ValueError(v)


def write_gguf(path, spec, duplicate=None):
    meta = list(spec['metadata'].items())
    if duplicate:
        meta.append((duplicate, spec['metadata'][duplicate]))
    tensors = spec['tensors']
    data = pack('IIQQ', 0x46554747, 3, len(tensors), len(meta))
    for key, v in meta:
        t, b = value(v)
        data += string(key) + pack('I', t) + b
    for t in tensors:
        shape = t['shape']
        data += string(t['name']) + pack('I', len(shape))
        data += pack('Q' * len(shape), *shape) + pack('IQ', t['type'], t['offset'])
    align = spec['metadata'].get('general.alignment', 32)
    # Invalid alignment fixtures still need a finite, well-formed test file.
    align = align if 0 < align <= 4096 else 32
    offset = (len(data) + align - 1) // align * align
    end = max(t['offset'] + math.prod(t['shape']) * (2 if t['type'] == 1 else 4)
              for t in tensors)
    with path.open('wb') as f:
        f.write(data)
        f.truncate(offset + end)
    return offset


def main():
    spec = json.loads((ROOT / 'tests/vision/mmproj-inventory.json').read_text())
    with tempfile.TemporaryDirectory(prefix='xenolith-vision-weights-') as directory:
        tmp = Path(directory)
        wrapper = tmp / 'loader.c'
        wrapper.write_text('''#include "vision_internal.h"
#include <string.h>
int load(const char *path, unsigned char *hash) {
    xe_vision_model m; char error[256];
    if (!xe_vision_model_open(&m,path,error,sizeof error)) return 0;
    memcpy(hash,m.sha256,32); xe_vision_model_close(&m); return 1;
}
''')
        library = tmp / 'loader.so'
        subprocess.run(shlex.split(os.environ.get('CC', 'cc')) +
                       ['-O2', '-std=c11', '-pthread', '-shared', '-fPIC', '-I', str(ROOT),
                        str(wrapper), str(ROOT / 'vision.c'), str(ROOT / 'format.c'),
                        '-lm', '-o', str(library)], check=True)
        loader = ctypes.CDLL(str(library)).load
        loader.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_ubyte)]
        loader.restype = ctypes.c_int
        path = tmp / 'projector.gguf'

        def check(valid):
            digest = (ctypes.c_ubyte * 32)()
            assert bool(loader(os.fsencode(path), digest)) == valid
            if valid:
                with path.open('rb') as f:
                    assert bytes(digest) == hashlib.file_digest(f, 'sha256').digest()
            return bytes(digest)

        offset = write_gguf(path, spec)
        original = check(True)  # Non-reference tensor contents are accepted.
        with path.open('r+b') as f:
            f.seek(offset)
            f.write(b'\x01')
        assert check(True) != original  # Same schema, different full identity.
        renamed = copy.deepcopy(spec)
        renamed['metadata']['general.name'] = 'Compatible fine-tune with a different header size'
        renamed['tensors'].reverse()
        write_gguf(path, renamed)
        assert check(True) != original  # Header/name/order are not an allowlist.

        cases = 0
        def reject(modify, duplicate=None, truncate=False):
            nonlocal cases
            changed = copy.deepcopy(spec)
            modify(changed)
            write_gguf(path, changed, duplicate)
            if truncate:
                with path.open('r+b') as f:
                    f.truncate(path.stat().st_size - 1)
            check(False)
            cases += 1

        for key in ('clip.vision.patch_size', 'clip.vision.block_count',
                    'clip.vision.attention.head_count', 'clip.vision.projection_dim'):
            reject(lambda s: s['metadata'].__setitem__(key, 1))
            reject(lambda s: s['metadata'].pop(key))
        for key, v in [('general.architecture', 'other'), ('clip.vision.projector_type', 'other'),
                       ('clip.vision.image_mean', [1.0, 0.0, 0.0]),
                       ('clip.vision.image_std', [1.0, 2.0, 1.0]),
                       ('clip.vision.attention.layer_norm_epsilon', 0.1),
                       ('clip.has_vision_encoder', False), ('clip.use_gelu', False),
                       ('general.alignment', 3), ('general.alignment', 1)]:
            reject(lambda s: s['metadata'].__setitem__(key, v))
        reject(lambda s: None, duplicate='clip.vision.patch_size')
        reject(lambda s: s['tensors'][0]['shape'].__setitem__(0, 1151))
        reject(lambda s: s['tensors'][0].__setitem__('type', 0))
        reject(lambda s: s['tensors'][0].__setitem__('name', 'missing.weight'))
        reject(lambda s: s['tensors'][1].__setitem__('offset', 0))
        reject(lambda s: s['tensors'][1].__setitem__('offset', 1))
        reject(lambda s: s['tensors'][1].__setitem__('name', s['tensors'][0]['name']))
        reject(lambda s: s['tensors'].pop())
        reject(lambda s: None, truncate=True)
        print(f'vision weights: 3 compatible variants and {cases} invalid schemas PASS')


if __name__ == '__main__':
    main()
