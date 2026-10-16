#!/usr/bin/env python3
"""Check two image turns using the included quadrants fixture, and two text turns."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import time

from check_support import check, fresh_output, write_report


p = argparse.ArgumentParser(description=__doc__)
for name in ('engine', 'model', 'mmproj', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
a.output = fresh_output(a.output)
image = Path(__file__).resolve().parent / 'fixtures/quadrants.png'
report = {'status': 'RUNNING', 'context': 65536, 'cases': []}
write_report(a.output, report)
try:
    report['image_sha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
    report['executable_sha256'] = hashlib.sha256(a.engine.read_bytes()).hexdigest()
    for visual in (True, False):
        name = 'image' if visual else 'text'
        cmd = [str(a.engine.resolve()), 'chat', str(a.model.resolve()), '--ctx', '65536',
               '--top-k', '1', '-n', '64']
        if visual:
            cmd += ['--mmproj', str(a.mmproj.resolve()), '--image', str(image)]
            prompts = ['What color is the upper-left quadrant? Reply with just the color name.',
                       'What color is the lower-right quadrant in the same image? Reply with just the color name.']
            expected = ['red', 'yellow']
        else:
            prompts = ['Reply with only the word HELLO.', 'Reply with only the word WORLD.']
            expected = ['hello', 'world']
        started = time.monotonic()
        with (a.output / (name + '.stderr')).open('w') as err:
            result = subprocess.run(cmd, input='\n'.join(prompts + ['/exit']) + '\n',
                                    text=True, stdout=subprocess.PIPE, stderr=err,
                                    timeout=600)
        (a.output / (name + '.stdout')).write_text(result.stdout)
        result.check_returncode()
        answers = [re.split(r'\n\[\d+ tokens', s)[0].strip()
                   for s in re.findall(r'assistant> (.*?)(?=you> |\Z)', result.stdout, re.S)]
        check(len(answers) == 2 and all(answers), answers)
        for answer, word in zip(answers, expected):
            check(re.search(r'\b' + word + r'\b', answer, re.I), (word, answers))
        if visual:
            check(result.stdout.count('Image ready.') == 1, result.stdout)
            check((a.output / (name + '.stderr')).read_text().count('encoder_ms=') == 1,
                  'the encoder must run once for both image turns')
        report['cases'].append({'kind': name, 'command': cmd, 'prompts': prompts,
                                'expected': expected, 'answers': answers,
                                'seconds': time.monotonic() - started, 'status': 'PASS'})
        write_report(a.output, report)
        print(name + ' chat: two turns PASS', flush=True)
    report['status'] = 'PASS'
except BaseException as exc:
    report['status'] = 'FAIL'
    report['error'] = repr(exc)
    raise
finally:
    write_report(a.output, report)
