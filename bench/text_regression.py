#!/usr/bin/env python3
"""Measure text throughput with the matrix or paired ABBA regression workload.

Run on the target Linux machine with no other inference workloads active.
Executables must already be built. Output must be a new directory. Completion
means valid measurements were collected; it is not a performance verdict.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def measurements(path, prompt, generated, repetitions, context, batch):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = [(prompt, 0)] if prompt else []
    if generated:
        expected.append((0, generated))
    if len(rows) != len(expected):
        raise ValueError(f'{path.name}: expected {len(expected)} metric records')
    for row, (pp, tg) in zip(rows, expected):
        if not isinstance(row, dict) or (row.get('n_prompt'), row.get('n_gen'), row.get('n_depth')) != (pp, tg, 0):
            raise ValueError(f'{path.name}: unexpected benchmark workload')
        # Older baselines omit these fields; all commands are in provenance.
        if row.get('context_window', context) != context or row.get('text_prefill_rows', batch) != batch:
            raise ValueError(f'{path.name}: unexpected context or prefill size')
        for key in ('avg_ns', 'avg_ts', 'samples_ns', 'samples_ts'):
            values = row.get(key)
            values = values if key.startswith('samples_') else [values]
            if not isinstance(values, list) or len(values) != (repetitions if key.startswith('samples_') else 1):
                raise ValueError(f'{path.name}: invalid {key} count')
            if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f'{path.name}: invalid {key} metrics')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--candidate', default=Path('./xenolith'), type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--output', required=True, type=Path, help='New directory; existing paths are rejected')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--paired', action='store_true', help='ABBA baseline/candidate pp2048 and tg128, at text prefill 512')
    a = p.parse_args()
    if not 1 <= a.repetitions <= 1000:
        p.error('repetitions must be between 1 and 1000')
    if a.paired and not a.baseline:
        p.error('--paired requires --baseline')
    a.model = a.model.resolve()
    if not a.model.is_file():
        p.error('--model must be an existing file')
    executables = {'candidate': a.candidate.resolve()}
    if a.baseline:
        executables['baseline'] = a.baseline.resolve()
    identities = {}
    for variant, executable in executables.items():
        if not executable.is_file() or not os.access(executable, os.X_OK):
            p.error(f'{variant} must be an executable file: {executable}')
        identities[variant] = {'path': str(executable), 'sha256': digest(executable)}
    if a.baseline and (executables['baseline'].samefile(executables['candidate']) or
                       identities['baseline']['sha256'] == identities['candidate']['sha256']):
        p.error('baseline and candidate must be distinct executable builds')
    try:
        a.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        p.error('--output must be a new directory')

    record = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'model': {'path': str(a.model), 'size_bytes': a.model.stat().st_size, 'sha256': digest(a.model)},
              'executables': identities, 'arguments': sys.argv[1:], 'cwd': str(Path.cwd()),
              'repetitions': a.repetitions, 'contexts': [65536, 131072],
              'warmup': True, 'delay_seconds': 1,
              'mode': 'paired ABBA pp2048/tg128 text512' if a.paired else 'full matrix', 'runs': []}

    def save():
        (a.output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')

    save()
    runs = []
    if a.paired:
        for context in record['contexts']:
            for index, variant in enumerate(('baseline', 'candidate', 'candidate', 'baseline'), 1):
                runs.append((f'paired-ctx{context}-{index}-{variant}', variant, context, 512, 2048, 128))
    else:
        for variant in ('baseline', 'candidate') if a.baseline else ('candidate',):
            for context in record['contexts']:
                for batch in ([512, 1024] if variant == 'candidate' else [512]):
                    for prompt in (512, 2048, 8192):
                        runs.append((f'{variant}-ctx{context}-b{batch}-pp{prompt}', variant, context, batch, prompt, 0))
                runs.append((f'{variant}-ctx{context}-tg128', variant, context, 512, 0, 128))
    try:
        for name, variant, context, batch, prompt, generated in runs:
            cmd = [str(executables[variant]), 'bench', str(a.model), '--ctx', str(context),
                   '-p', str(prompt), '-n', str(generated), '-r', str(a.repetitions), '--delay', '1', '-o', 'jsonl']
            if variant == 'candidate':
                cmd += ['--prefill', str(batch)]
            run = {'name': name, 'variant': variant, 'command': cmd, 'status': 'running'}
            record['runs'].append(run)
            save()
            path = a.output / (name + '.jsonl')
            with path.open('x') as out, (a.output / (name + '.log')).open('x') as err:
                result = subprocess.run(cmd, stdout=out, stderr=err)
            run['returncode'] = result.returncode
            if result.returncode:
                raise ValueError(f'{name}: executable exited {result.returncode}; see {name}.log')
            rows = measurements(path, prompt, generated, a.repetitions, context, batch)
            run.update(status='complete', sha256=digest(path), measurements=rows)
            save()
            print(name, 'complete', flush=True)
        for variant, executable in executables.items():
            if digest(executable) != identities[variant]['sha256']:
                raise ValueError(f'{variant} executable changed during measurements')
        summary = []
        for context in record['contexts']:
            grouped = {}
            for run, (_, variant, ctx, batch, _, _) in zip(record['runs'], runs):
                if ctx != context:
                    continue
                for row in run['measurements']:
                    phase = f"pp{row['n_prompt']}" if row['n_prompt'] else f"tg{row['n_gen']}"
                    grouped.setdefault((phase, batch, variant), []).extend(row['samples_ts'])
            for phase, batch in sorted({(phase, batch) for phase, batch, _ in grouped}):
                item = {'context': context, 'phase': phase, 'text_prefill_rows': batch}
                for variant in ('baseline', 'candidate'):
                    samples = grouped.get((phase, batch, variant))
                    if samples:
                        item[variant + '_median_tokens_per_second'] = statistics.median(samples)
                if all(variant + '_median_tokens_per_second' in item for variant in ('baseline', 'candidate')):
                    item['change_percent'] = 100 * (item['candidate_median_tokens_per_second'] / item['baseline_median_tokens_per_second'] - 1)
                summary.append(item)
        record.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
        save()
        (a.output / 'report.json').write_text(json.dumps({'status': 'complete', 'summary': summary}, indent=2) + '\n')
    except (OSError, ValueError, KeyboardInterrupt) as error:
        record.update(status='failed', error=str(error) or type(error).__name__)
        if record['runs'] and record['runs'][-1]['status'] == 'running':
            record['runs'][-1]['status'] = 'failed'
        save()
        raise SystemExit(f'benchmark failed: {record["error"]}')


if __name__ == '__main__':
    main()
