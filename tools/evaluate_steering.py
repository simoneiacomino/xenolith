#!/usr/bin/env python3
"""Record baseline, prompting, and steering responses with comparable local settings."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import time
from steering_wire import Wire


def repetition(text):
    words = text.lower().split()
    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
    return 1 - len(set(grams))/len(grams) if grams else 0


def decode_rate(sample):
    measurement = sample['done'].get('inference', {}).get('decode', {})
    elapsed = measurement.get('elapsed_ms', 0)
    return 1000 * measurement.get('tokens', 0) / elapsed if elapsed > 0 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xenolith', default='./xenolith')
    p.add_argument('--model', required=True)
    p.add_argument('--registry', required=True)
    p.add_argument('--artifact')
    p.add_argument('--configs', type=Path, help='JSON list of named artifact/strength settings')
    p.add_argument('--skip-controls', action='store_true', help='Omit controls when identical controls are already archived')
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--strengths', default='-0.5,-0.25,0.25,0.5')
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ctx', type=int, default=4096)
    p.add_argument('--limit', type=int, help='Use only the first N prompts for an explicitly recorded quick probe')
    args = p.parse_args()
    if bool(args.artifact)==bool(args.configs):
        p.error('Choose either --artifact or --configs')
    args.out.mkdir(parents=True, exist_ok=False)
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    if args.limit is not None:
        if args.limit < 1: p.error('--limit must be positive')
        prompts = prompts[:args.limit]
    if not prompts: p.error('The prompt dataset is empty')
    configs = [] if args.skip_controls else [dict(name='baseline'),
        dict(name='system_prompt',system='You are a helpful assistant. Answer in at most three sentences. Include essential facts. Use no headings, lists, or preamble.')]
    if args.configs:
        configs += json.loads(args.configs.read_text())
    else:
        configs += [dict(name=f'steering_{s}',artifact=args.artifact,strength=float(s)) for s in args.strengths.split(',')]
    if not configs or len({c['name'] for c in configs})!=len(configs):p.error('Expected nonempty uniquely named configurations')
    report = {'platform': platform.platform(), 'created_unix': time.time(), 'arguments': vars(args).copy(),
              'prompts_sha256': hashlib.sha256(args.prompts.read_bytes()).hexdigest(), 'summaries': {},
              'interpretation': 'Response length is not a correctness score. Review archived answers; no automatic factuality claim.'}
    report['arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['arguments'].items()}
    report['configurations']=configs
    if args.artifact:
        report['artifact_manifest'] = json.loads((Path(args.registry)/args.artifact/'manifest.json').read_text())
    else:
        report['artifact_manifests']={c['artifact']:json.loads((Path(args.registry)/c['artifact']/'manifest.json').read_text())
                                     for c in configs if c.get('artifact')}
    report['binary_sha256'] = hashlib.sha256(Path(args.xenolith).read_bytes()).hexdigest()
    report['evaluator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['acceptance_targets'] = {
        'median_words_reduction_fraction': 0.15, 'additional_length_stops': 0,
        'additional_constraint_failures': 0, 'maximum_repeated_4gram_fraction': 0.20,
        'maximum_identical_character_run': 32, 'additional_generation_errors': 0,
        'maximum_replacement_characters': 0,
        'note': 'Experimental screening targets; factual/code correctness also requires answer review.'}

    started = time.monotonic()
    with Wire(args.xenolith, args.model, args.out/'state', args.registry, context=args.ctx) as wire:
        report['describe'] = wire.request(op='describe')
        report['startup_seconds'] = time.monotonic() - started
        # Warm-up excluded from summaries; fresh ephemeral state is used for every sample.
        wire.generate(op='ephemeral', messages=[{'role': 'user', 'text': 'Say hello.'}],
                      temperature=0, seed=1, reasoning=False, max_tokens=16)
        with (args.out/'responses.jsonl').open('w') as stream:
            for config in configs:
                name=config['name'];system=config.get('system','You are a helpful assistant.')
                samples = []
                for index, prompt in enumerate(prompts):
                    request = {'op': 'ephemeral', 'system': system,
                               'messages': [{'role': 'user', 'text': prompt['prompt']}],
                               'temperature': 0, 'seed': 1, 'reasoning': False, 'max_tokens': args.tokens}
                    if config.get('artifact'):
                        request['steering'] = {'artifact': config['artifact'], 'strength': config['strength']}
                    sample = wire.generate(allow_error=True, **request)
                    sample.pop('events')
                    sample.update(configuration=name, index=index, prompt=prompt['prompt'])
                    try:
                        status = Path(f'/proc/{wire.process.pid}/status').read_text()
                        sample['process_memory_kib'] = {line.split(':')[0]: int(line.split()[1])
                            for line in status.splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))}
                    except (OSError, ValueError):
                        sample['process_memory_kib'] = None
                    sample['words'] = len(sample['text'].split())
                    sample['repeated_4gram_fraction'] = repetition(sample['text'])
                    sample['replacement_characters'] = sample['text'].count('\ufffd')
                    sample['longest_character_run'] = max((len(m.group()) for m in re.finditer(r'(.)\1*', sample['text'], re.DOTALL)), default=0)
                    if prompt.get('kind') == 'json':
                        try:
                            value = json.loads(sample['text'])
                            sample['constraint_pass'] = value == prompt['expected']
                        except (ValueError, TypeError):
                            sample['constraint_pass'] = False
                    elif prompt.get('kind') == 'python':
                        text = re.sub(r'^```(?:python)?\s*|\s*```$', '', sample['text'].strip())
                        try:
                            tree = ast.parse(text)
                            sample['constraint_pass'] = any(isinstance(n, ast.FunctionDef) and n.name == prompt['function'] for n in tree.body)
                        except SyntaxError:
                            sample['constraint_pass'] = False
                    elif 'required_facts' in prompt:
                        sample['constraint_pass'] = all(any(word.lower() in sample['text'].lower() for word in group)
                                                        for group in prompt['required_facts'])
                    stream.write(json.dumps(sample, ensure_ascii=False) + '\n'); stream.flush()
                    samples.append(sample)
                    print(f'{name}: {index+1}/{len(prompts)}', flush=True)
                checked = [s for s in samples if 'constraint_pass' in s]
                report['summaries'][name] = {
                    'samples': len(samples), 'median_words': statistics.median(s['words'] for s in samples),
                    'length_stops': sum(s['done'].get('stop') == 'length' for s in samples),
                    'generation_errors': sum(s.get('error') is not None for s in samples),
                    'constraint_passes': sum(s['constraint_pass'] for s in checked), 'constraint_cases': len(checked),
                    'max_repeated_4gram_fraction': max(s['repeated_4gram_fraction'] for s in samples),
                    'max_character_run': max(s['longest_character_run'] for s in samples),
                    'max_replacement_characters': max(s['replacement_characters'] for s in samples),
                    'median_first_text_seconds': statistics.median(s['first_text_seconds'] for s in samples if s['first_text_seconds'] is not None) if any(s['first_text_seconds'] is not None for s in samples) else None,
                    'median_decode_tokens_per_second': statistics.median(
                        decode_rate(s) for s in samples if decode_rate(s) is not None) if any(decode_rate(s) is not None for s in samples) else None,
                }
                (args.out/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
