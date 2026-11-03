#!/usr/bin/env python3
"""Build native Xenolith steering artifacts from paired local activation captures."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def read_pairs(path):
    raw = Path(path).read_bytes()
    pairs = []
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                            for k in ('positive', 'negative')):
            raise ValueError(f'Invalid positive/negative strings on line {number}')
        if 'system' in row and not isinstance(row['system'], str):
            raise ValueError(f'Invalid system string on line {number}')
        if 'prompt' in row and (not isinstance(row['prompt'], str) or not row['prompt'].strip()):
            raise ValueError(f'Invalid shared prompt on line {number}')
        pairs.append(row)
    if not pairs:
        raise ValueError('The dataset is empty')
    return raw, pairs


def read_floats(path, count):
    raw = Path(path).read_bytes()
    if len(raw) != count * 4:
        raise ValueError(f'Wrong activation shape: {path}')
    values = array.array('f')
    values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f'Non-finite activations: {path}')
    return values


def build(capture, dataset, out, mode, selected, source, revision, labels, orthogonalize=False):
    raw, pairs = read_pairs(dataset)
    metadata = json.loads((capture / 'capture.json').read_text())
    layers, width = metadata['layers'], metadata['width']
    if metadata.get('version') != 1 or metadata['pairs'] != len(pairs):
        raise ValueError('Capture version or pair count does not match dataset')
    if metadata.get('dataset_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('Capture dataset identity does not match input')
    kind=metadata.get('capture_kind','paired_instructions_v1')
    if kind not in ('paired_instructions_v1','paired_completions_v1'):
        raise ValueError('Unsupported capture kind')
    completion = kind == 'paired_completions_v1'
    if completion and (metadata.get('position')!='mean_assistant_completion_tokens' or metadata['reasoning']!='off'):
        raise ValueError('Unsupported completion position or reasoning setting')
    if any(('prompt' in row) != completion for row in pairs):
        raise ValueError('Dataset prompt/completion structure does not match capture kind')
    site=metadata.get('site','post_layer_residual_v1')
    if site not in ('post_layer_residual_v1','post_ffn_output_v1'):raise ValueError('Unsupported capture site')
    active = sorted(set(selected if selected is not None else range(layers)))
    if not active or any(i < 0 or i >= layers for i in active):
        raise ValueError('Invalid layer selection')
    # Kahan accumulation avoids retaining the complete activation corpus in RAM.
    means = [0.0] * (layers * width)
    corrections = [0.0] * len(means)
    control = [0.0] * len(means) if orthogonalize else None
    control_corrections = [0.0] * len(means) if orthogonalize else None
    for i in range(len(pairs)):
        positive = read_floats(capture / f'{i:04d}-positive.f32', len(means))
        negative = read_floats(capture / f'{i:04d}-negative.f32', len(means))
        for j, (a, b) in enumerate(zip(positive, negative)):
            delta = (a - b) / len(pairs) - corrections[j]
            new = means[j] + delta
            corrections[j] = (new - means[j]) - delta
            means[j] = new
            if orthogonalize:
                delta = b / len(pairs) - control_corrections[j]
                updated = control[j] + delta
                control_corrections[j] = (updated - control[j]) - delta
                control[j] = updated
    rows = array.array('f', [0.0] * len(means))
    for layer in active:
        values = means[layer * width:(layer + 1) * width]
        if orthogonalize:
            base = control[layer * width:(layer + 1) * width]
            denominator = math.fsum(x*x for x in base)
            if denominator > 1e-24:
                projection = math.fsum(a*b for a, b in zip(values, base)) / denominator
                values = [a - projection*b for a, b in zip(values, base)]
        norm = math.sqrt(math.fsum(x * x for x in values))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f'Layer {layer} has no usable contrastive direction; exclude it explicitly')
        scale = 1 / norm if mode == 'projection' else 1
        rows[layer * width:(layer + 1) * width] = array.array('f', (x * scale for x in values))
    if sys.byteorder != 'little':
        rows.byteswap()
    payload = rows.tobytes()
    manifest = {
        'version': 1, 'layers': layers, 'width': width,
        'site': site, 'dtype': 'f32le', 'payload': 'vectors.f32',
        'payload_sha256': hashlib.sha256(payload).hexdigest(),
        'model_sha256': metadata['model_sha256'], 'active_layers': active,
        'mode': mode, 'normalization': 'unit' if mode == 'projection' else 'none',
        'token_policy': 'all_text',
        'provenance': {
            'model_source': source, 'model_revision': revision,
            'dataset_sha256': hashlib.sha256(raw).hexdigest(),
            'template_sha256': metadata['template_sha256'],
            'method': ('paired_completion_mean_difference' if completion else 'paired_difference_of_means') + ('_orthogonalized_control_v1' if orthogonalize else '_v1'),
            'position': metadata.get('position', 'last_rendered_prompt_token_including_assistant_opening'),
            'reasoning': metadata['reasoning'], 'engine_commit': metadata['engine_commit'],
            'numeric_settings': metadata['numeric_settings'],
            'positive_label': labels[0], 'negative_label': labels[1],
            'evaluation': 'unevaluated; no recommended strength',
        },
    }
    # Manifest bytes participate in artifact identity. Preserve these exact bytes.
    encoded = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False) + '\n').encode()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'vectors.f32').write_bytes(payload)
    (out / 'manifest.json').write_bytes(encoded)
    identity = hashlib.sha256(b'xenolith-steering-v1\0' + encoded + payload).hexdigest()
    print(json.dumps({'artifact': identity, 'manifest': str(out / 'manifest.json'),
                      'mode': mode, 'layers': active}))
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xenolith', default='./xenolith')
    p.add_argument('--model', help='Exact GGUF used for extraction and inference')
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', type=Path, required=True, help='New artifact directory')
    p.add_argument('--capture-dir', type=Path, help='Reuse an existing matching capture')
    p.add_argument('--mode', choices=('projection', 'additive'), default='projection')
    p.add_argument('--layers', help='Comma-separated zero-based layer indices; default all')
    p.add_argument('--ctx', type=int, default=512)
    p.add_argument('--thinking', action='store_true')
    p.add_argument('--completions', action='store_true', help='Capture paired assistant answers to a shared prompt')
    p.add_argument('--cpu', action='store_true', help='Diagnostic tokenwise CPU capture, including the prompt')
    p.add_argument('--capture-site',choices=('post_layer_residual_v1','post_ffn_output_v1'),default='post_layer_residual_v1')
    p.add_argument('--orthogonalize-control', action='store_true',
                   help='Remove the component parallel to the negative activation mean before normalization')
    p.add_argument('--model-source', required=True)
    p.add_argument('--model-revision', required=True)
    p.add_argument('--positive-label', default='succinct')
    p.add_argument('--negative-label', default='detailed')
    args = p.parse_args()
    read_pairs(args.dataset)
    selected = [int(i) for i in args.layers.split(',')] if args.layers else None
    def run(capture):
        return build(capture, args.dataset, args.out, args.mode, selected,
                     args.model_source, args.model_revision, (args.positive_label, args.negative_label),
                     args.orthogonalize_control)
    if args.capture_dir:
        run(args.capture_dir)
    else:
        if not args.model:
            p.error('--model is required without --capture-dir')
        with tempfile.TemporaryDirectory(prefix='xe-steering-') as temporary:
            capture = Path(temporary) / 'capture'
            cmd = [args.xenolith, 'steering-capture', args.model, '--dataset', args.dataset,
                   '--out', str(capture), '--ctx', str(args.ctx)]
            if args.thinking:
                cmd.append('--thinking')
            if args.completions: cmd.append('--completions')
            if args.cpu: cmd.append('--cpu')
            cmd+=['--capture-site',args.capture_site]
            subprocess.run(cmd, check=True)
            run(capture)


if __name__ == '__main__':
    main()
