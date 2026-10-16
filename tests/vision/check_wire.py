#!/usr/bin/env python3
"""Model-backed protocol v2 checks, in an isolated state/cache directory."""
import argparse
import base64
import hashlib
from pathlib import Path
from check_support import Wire, check, fresh_output, write_report
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--engine', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--mmproj', type=Path, required=True)
p.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parent / 'fixtures')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--ctx', type=int, choices=(65536, 131072), default=65536)
p.add_argument('--large', action='store_true',
               help='Run the 131072-context, 1120-row checkpoint/restart check instead of the full suite')
a = p.parse_args()
if a.large:
    a.ctx = 131072
a.output = fresh_output(a.output)
state = a.output / 'state'
cache = a.output / 'cache'
report = {'status': 'RUNNING', 'context': a.ctx, 'checks': [], 'generations': []}
client = Wire(a.output, report)
send = client.send
read = client.read
call = client.call
stop = client.stop
write_report(a.output, report)

def record(name):
    report['checks'].append(name)
    write_report(a.output, report)
    print(name, 'PASS', flush=True)

def start(vision=True):
    cmd = [str(a.engine.resolve()), 'wire', str(a.model.resolve()), '--ctx', str(a.ctx),
           '--state', str(state.resolve()), '--cache', str(cache.resolve())]
    if vision:
        cmd += ['--mmproj', str(a.mmproj.resolve())]
    client.start(cmd)

def generate(req, expected=None, exact=False):
    send({'temperature': 1, 'top_k': 1, 'max_tokens': 24, 'reasoning': False, **req})
    text = ''
    events = []
    while True:
        e = read()
        events.append(e)
        check(e.get('ok') is not False, e)
        check(e.get('event') != 'error', e)
        if e.get('event') == 'text_delta':
            text += e['text']
        if e.get('event') == 'done':
            break
    result = {'request': req['op'], 'text': text, 'done': e}
    if a.large:
        result['events'] = events
    report['generations'].append(result)
    if expected:
        check(text.strip().lower() == expected.lower() if exact else expected.lower() in text.lower(), (expected, text))
    return (e, text)
image = lambda data, budget=70: {'data': base64.b64encode(data).decode(), 'budget': budget}
prompt = 'What color is the upper-left quadrant? Reply with just the color name.'

def large_session():
    raw = (a.fixtures / 'quadrants.png').read_bytes()
    report['budget'] = 1120
    start()
    sid = call({'op': 'create', 'system': 'Answer concisely.'})['session']
    call({'op': 'append', 'role': 'user', 'text': prompt, 'image': image(raw, 1120)})
    generate({'op': 'generate', 'max_tokens': 12}, 'red', exact=True)
    ckpt = call({'op': 'checkpoint'})
    check(ckpt['saved'] or ckpt.get('reason') == 'nothing_new', ckpt)
    report['checkpoint'] = ckpt
    history = call({'op': 'history'})
    check(history['entries'][1]['image']['rows'] == 1120, history)
    report['image'] = history['entries'][1]['image']
    stop()
    check(all(key in report['server_memory'][0] for key in ('VmHWM', 'VmRSS', 'VmSwap')),
          'server memory information missing')
    record('1120-row response, memory and checkpoint')
    start()
    opened = call({'op': 'open', 'session': sid})
    check(opened['resume'] == 'snapshot', opened)
    call({'op': 'append', 'role': 'user', 'text': 'Repeat the color from that image. One word.'})
    done, _ = generate({'op': 'generate', 'max_tokens': 12}, 'red', exact=True)
    check(done.get('resume', {}).get('loaded') is True, done)
    stop()
    record('1120-row visual snapshot restored after restart')

def full_suite():
    raw = (a.fixtures / 'quadrants.png').read_bytes()
    other = (a.fixtures / 'quadrants-rotated.png').read_bytes()
    start()
    d = call({'op': 'describe'})
    check(d['protocol'] == 2 and d['vision']['max_bytes'] == 48 * 1024 * 1024)
    record('vision capabilities')
    sid = call({'op': 'create', 'system': 'Answer concisely.'})['session']
    for bad in ({'data': 'YQ=='}, {'data': '!!!!!!!!!!!!'},
                {'data': 'AAAAAAAAAAA=', 'budget': 71},
                {'data': 'AAAAAAAAAAA=', 'id': '0' * 64},
                {'id': '0' * 64}, {'data': 'AAAAAAAAAA=='}):
        call({'op': 'append', 'role': 'user', 'text': prompt, 'image': bad}, False)
    check(len(call({'op': 'history'})['entries']) == 1)
    record('invalid images leave history unchanged')
    marker = call({'op': 'append', 'role': 'user', 'text': prompt, 'image': image(raw)})['marker']
    entries = call({'op': 'history'})['entries']
    meta = entries[-1]['image']
    check(meta['rows'] == 70 and meta['id'] == hashlib.sha256(raw).hexdigest())
    media = call({'op': 'media', 'image_id': meta['id']})
    check(base64.b64decode(media['data']) == raw)
    call({'op': 'media', 'image_id': meta['id'], 'offset': len(raw) + 1}, False)
    call({'op': 'media', 'image_id': meta['id'], 'limit': 0}, False)
    record('image history and source retrieval')
    done, _ = generate({'op': 'generate'}, 'red')
    ckpt = call({'op': 'checkpoint'})
    check(ckpt['saved'] is True, ckpt)
    record('durable response and checkpoint')
    call({'op': 'append', 'role': 'user', 'text': prompt, 'image': image(other)})
    generate({'op': 'generate'}, 'yellow')
    record('second image changes answer with existing KV')
    call({'op': 'rewind', 'marker': marker})
    generate({'op': 'generate'}, 'red')
    ckpt = call({'op': 'checkpoint'})
    check(ckpt['saved'] is True, ckpt)
    record('rewind across second image')
    stop()
    start()
    opened = call({'op': 'open', 'session': sid})
    check(opened['resume'] == 'snapshot', opened)
    check(call({'op': 'history'})['entries'][1]['image'] == meta)
    call({'op': 'append', 'role': 'user', 'text': 'Repeat the color from that image, one word.'})
    done, _ = generate({'op': 'generate'}, 'red')
    check(done.get('resume', {}).get('loaded') is True, done)
    record('restart resumes visual snapshot')
    call({'op': 'rewind', 'marker': marker})
    stop()
    for f in (cache / 'vision-v1').glob('*.xve'):
        f.unlink()
    start()
    call({'op': 'open', 'session': sid})
    generate({'op': 'generate'}, 'red')
    check(list((cache / 'vision-v1').glob('*.xve')))
    record('evicted embeddings rebuilt from persisted media')
    before = call({'op': 'history'})
    generate({'op': 'ephemeral', 'messages': [
        {'role': 'user', 'text': prompt, 'image': image(other)}]}, 'yellow')
    check(call({'op': 'history'}) == before)
    record('ephemeral image preserves durable history')
    call({'op': 'rebuild', 'system': 'Answer concisely.', 'messages': [
        {'role': 'user', 'text': prompt, 'image': {'id': meta['id'], 'budget': 140}}]})
    check(call({'op': 'history'})['entries'][-1]['image']['rows'] == 126)
    generate({'op': 'generate'}, 'red')
    record('rebuild by media ID with variable visual length')
    tools = [{'name': 'camera', 'description': 'Read a camera image.',
              'parameters': {'type': 'object', 'properties': {}}}]
    call({'op': 'rebuild', 'system': 'Answer concisely.', 'tools': tools, 'messages': [
        {'role': 'user', 'text': 'Read the camera and identify the upper-left color.'},
        {'role': 'assistant', 'text': '', 'calls': [{'name': 'camera', 'arguments': {}}]},
        {'role': 'tool', 'tool': 'camera', 'text': prompt, 'image': image(other)}]})
    generate({'op': 'generate'}, 'yellow')
    record('tool-result image')
    before = call({'op': 'history'})
    call({'op': 'rebuild', 'messages': [
        {'role': 'user', 'text': prompt, 'image': image(raw)},
        {'role': 'user', 'image': image(b'broken-image')}]}, False)
    check(call({'op': 'history'}) == before)
    record('failed visual rebuild rolls back the transcript')
    send({'op': 'generate', 'temperature': 1, 'top_k': 1, 'max_tokens': 128,
          'reasoning': {'effort': 'high', 'budget_tokens': 128, 'history': 'discard'}})
    check(read()['event'] == 'start')
    send({'op': 'cancel'})
    cancelled = False
    terminal = None
    while terminal is None:
        e = read()
        if e.get('ok') is True:
            cancelled = True
        check(e.get('event') != 'error', e)
        if e.get('event') == 'done':
            terminal = e
    check(cancelled and terminal['stop'] == 'aborted', terminal)
    check(any((e.get('image') for e in call({'op': 'history'})['entries'])))
    generate({'op': 'generate'})
    record('reasoning cancellation and projection replay retain the image')
    code = (a.fixtures / 'code.png').read_bytes()
    generate({'op': 'ephemeral', 'messages': [{'role': 'user',
        'text': 'Read the four-digit code in the image. Reply with just the digits.',
        'image': image(code, 140)}]}, '7318')
    record('synthetic OCR response')
    stop()
    start(False)
    check(call({'op': 'describe'})['protocol'] == 1)
    call({'op': 'open', 'session': sid}, False)
    call({'op': 'ephemeral', 'messages': [{'role': 'user', 'image': image(raw)}]}, False)
    record('text service rejects visual sessions and requests')
    stop()
try:
    if a.large:
        large_session()
    else:
        full_suite()
    report['status'] = 'PASS'
    write_report(a.output, report)
except BaseException as e:
    report['status'] = 'FAIL'
    report['error'] = repr(e)
    write_report(a.output, report)
    client.abort()
    raise
