#!/usr/bin/env python3
"""Exercise paired extraction arithmetic and provenance without a model."""
import array
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

spec = importlib.util.spec_from_file_location('build_steering', Path(__file__).resolve().parents[1] / 'tools/build_steering.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
with tempfile.TemporaryDirectory(prefix='xe-builder-test-') as tmp:
    root = Path(tmp)
    dataset = root / 'pairs.jsonl'
    dataset.write_text('{"positive":"short","negative":"long"}\n')
    capture = root / 'capture'
    capture.mkdir()
    metadata = {'version': 1, 'layers': 2, 'width': 3, 'pairs': 1,
                'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                'model_sha256': '0'*64, 'template_sha256': '1'*64,
                'engine_commit': 'test', 'reasoning': 'off', 'numeric_settings': 'test'}
    (capture / 'capture.json').write_text(json.dumps(metadata))
    for label, values in [('positive', [3, 4, 0, 1, 0, 0]), ('negative', [0]*6)]:
        floats = array.array('f', values)
        if sys.byteorder != 'little':
            floats.byteswap()
        (capture / f'0000-{label}.f32').write_bytes(floats.tobytes())
    identity = builder.build(capture, dataset, root/'projection', 'projection', [0], 'test', 'test', ('a', 'b'))
    payload = (root/'projection/vectors.f32').read_bytes()
    floats = array.array('f'); floats.frombytes(payload)
    if sys.byteorder != 'little':
        floats.byteswap()
    assert abs(floats[0] - 0.6) < 1e-6 and abs(floats[1] - 0.8) < 1e-6
    assert list(floats[3:]) == [0, 0, 0]
    expected = hashlib.sha256(b'xenolith-steering-v1\0' + (root/'projection/manifest.json').read_bytes() + payload).hexdigest()
    assert expected == identity
    builder.build(capture, dataset, root/'additive', 'additive', [0], 'test', 'test', ('a', 'b'))
    raw = builder.read_floats(root/'additive/vectors.f32', 6)
    assert list(raw) == [3, 4, 0, 0, 0, 0]
    for label, values in [('positive', [2, 1, 0, 1, 0, 0]), ('negative', [1, 0, 0, 0, 0, 0])]:
        values = array.array('f', values)
        if sys.byteorder != 'little': values.byteswap()
        (capture / f'0000-{label}.f32').write_bytes(values.tobytes())
    builder.build(capture, dataset, root/'orthogonal', 'projection', [0], 'test', 'test', ('a', 'b'), True)
    values = builder.read_floats(root/'orthogonal/vectors.f32', 6)
    assert list(values) == [0, 1, 0, 0, 0, 0]
    dataset.write_text('{"prompt":"question","positive":"answer it","negative":"answer en"}\n')
    metadata.update(capture_kind='paired_completions_v1',position='mean_assistant_completion_tokens',
                    dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
    (capture/'capture.json').write_text(json.dumps(metadata))
    builder.build(capture,dataset,root/'completion','additive',[0],'test','test',('it','en'))
    provenance=json.loads((root/'completion/manifest.json').read_text())['provenance']
    assert provenance['method']=='paired_completion_mean_difference_v1'
    assert provenance['position']=='mean_assistant_completion_tokens'
    metadata['site']='post_ffn_output_v1';(capture/'capture.json').write_text(json.dumps(metadata))
    builder.build(capture,dataset,root/'ffn','projection',[0],'test','test',('a','b'))
    assert json.loads((root/'ffn/manifest.json').read_text())['site']=='post_ffn_output_v1'
    dataset.write_text('{"positive":"changed","negative":"long"}\n')
    try:
        builder.build(capture, dataset, root/'bad', 'projection', [0], 'test', 'test', ('a', 'b'))
        raise AssertionError('Mismatched dataset accepted')
    except ValueError:
        pass
    dataset.write_text('{"positive":"missing negative"}\n')
    try:
        builder.read_pairs(dataset)
        raise AssertionError('Incomplete pair accepted')
    except ValueError:
        pass
print('steering builder: mean differences, normalization, masks and provenance PASS')
