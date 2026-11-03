#!/usr/bin/env python3
"""Check native capture completion and failure at an unterminated final line."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xenolith', default='./xenolith')
    p.add_argument('--model', required=True)
    args = p.parse_args()
    first = json.dumps({'positive': 'Explain DNS briefly.',
                        'negative': 'Explain DNS in detail.'})
    cases = {
        'valid-eof': (first, True),
        'invalid-pair-eof': (first + '\n' + '{"positive":"missing negative"}', False),
        'overflow-eof': (first + '\n' + json.dumps(
            {'positive': 'word ' * 6000, 'negative': 'Brief answer.'}), False),
    }
    with tempfile.TemporaryDirectory(prefix='xe-capture-check-') as temporary:
        root = Path(temporary)
        for name, (data, expected) in cases.items():
            dataset, out = root / (name + '.jsonl'), root / name
            dataset.write_text(data, encoding='utf-8')  # Intentionally no final newline.
            result = subprocess.run([args.xenolith, 'steering-capture', args.model,
                '--dataset', str(dataset), '--out', str(out), '--ctx', '512'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            assert (result.returncode == 0) == expected, (name, result.stderr.decode(errors='replace'))
            assert (out / '0000-positive.f32').is_file(), name
            assert (out / '0000-negative.f32').is_file(), name
            assert (out / 'capture.json').exists() == expected, name
            if expected:
                assert json.loads((out / 'capture.json').read_text())['pairs'] == 1
            print(f'native capture {name}: PASS', flush=True)
        dataset=root/'completion.jsonl';out=root/'completion'
        dataset.write_text(json.dumps({'prompt':'Name a color.','positive':'Rosso.','negative':'Red.'}))
        result=subprocess.run([args.xenolith,'steering-capture',args.model,'--dataset',str(dataset),
            '--out',str(out),'--ctx','512','--completions','--cpu'],capture_output=True,timeout=180)
        assert result.returncode==0,result.stderr.decode(errors='replace')
        meta=json.loads((out/'capture.json').read_text())
        assert meta['capture_kind']=='paired_completions_v1' and meta['backend']=='cpu_tokenwise'
        for pole in ('positive','negative'):
            trace=json.loads((out/f'0000-{pole}.tokens.json').read_text())
            assert trace['capture_first']==trace['prefix_tokens']
            assert trace['prefix_last_piece']=='<channel|>'
            assert trace['capture_count']==len(trace['tokens'])-trace['prefix_tokens']>0
        print('native completion capture: teacher-forced positions and CPU provenance PASS')
        out=root/'ffn-capture'
        result=subprocess.run([args.xenolith,'steering-capture',args.model,'--dataset',str(dataset),
            '--out',str(out),'--ctx','512','--completions','--capture-site','post_ffn_output_v1'],
            capture_output=True,timeout=180)
        assert result.returncode==0,result.stderr.decode(errors='replace')
        assert json.loads((out/'capture.json').read_text())['site']=='post_ffn_output_v1'
        print('native FFN-site completion capture and provenance: PASS')


if __name__ == '__main__':
    main()
