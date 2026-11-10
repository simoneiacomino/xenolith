#!/usr/bin/env python3
"""Real-model response prefixes: exact text, replay, recovery and isolation."""
import argparse
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from steering_wire import Wire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xenolith', default='./xenolith')
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    prefix = '  Ecco la risposta: è '
    options = dict(temperature=0, seed=1, max_tokens=4, reasoning=False,
                   response=dict(prefix=prefix))
    with tempfile.TemporaryDirectory(prefix='xe-response-prefix-') as tmp:
        root = Path(tmp)
        common = dict(binary=args.xenolith, model=args.model, state=root/'state',
                      cache=root/'cache', budget='128MiB', context=1024)
        with Wire(**common) as w:
            assert w.request(op='describe')['response']['prefix']['max_bytes'] == 4096
            session = w.request(op='create', system='Be helpful.')['session']
            user = w.request(op='append', role='user', text='Quanto fa 17 per 24?')['marker']
            before = w.request(op='history')
            for bad in (None, 1, 'x'*4097, 'bad\x00text', '<channel|>'):
                w.send(dict(op='generate', reasoning=False, response=dict(prefix=bad)))
                assert w.read().get('ok') is False
                assert w.request(op='history') == before
            for request in (
                dict(reasoning=False, response=dict(prefix='x '*2048)),
                dict(reasoning=dict(effort='low'), response=dict(prefix=prefix)),
                dict(reasoning=dict(effort='low', prefix='thought'), response=dict(prefix=prefix)),
            ):
                result = w.generate(op='generate', allow_error=True, **request)
                assert result['error'], result
                assert w.request(op='history') == before
            result = w.generate(op='generate', **options)
            start = result['events'][0]
            assert start['response_prefix'] == prefix and start['response_prefix_tokens'] > 0
            assert 'reasoning_prefix' not in start
            assert result['reasoning'] == ''
            assert result['done']['usage']['reasoning'] == 0
            assert result['done']['usage']['reasoning_prefix'] == 0
            assert result['done']['usage']['response_prefix'] == start['response_prefix_tokens']
            assert result['done']['usage']['output'] <= 4
            entry = w.request(op='history')['entries'][-1]
            first_marker = entry['marker']
            assert entry['text'] == prefix + result['text'], entry
            assert entry['response_prefix'] == prefix and not entry.get('reasoning')
            # Refuse reinjection into an interrupted assistant turn.
            assert result['done']['stop'] == 'length', result
            saved = w.request(op='history')
            assert w.generate(op='generate', allow_error=True, **options)['error']
            assert w.request(op='history') == saved
            continuation = w.generate(op='generate', temperature=0, seed=1, max_tokens=4)
            assert 'response_prefix' not in continuation['events'][0]
            assert continuation['done']['usage']['response_prefix'] == 0
            w.request(op='checkpoint')
        with Wire(**common) as w:
            w.request(op='open', session=session)
            assert sum(e.get('response_prefix') == prefix for e in w.request(op='history')['entries']) == 1
            w.request(op='rewind', marker=first_marker)
            cold = w.generate(op='generate', temperature=0, seed=1, max_tokens=4)
            assert (cold['text'], cold['reasoning']) == (continuation['text'], continuation['reasoning'])
            w.request(op='rewind', marker=user)
            assert not any(e.get('response_prefix') for e in w.request(op='history')['entries'])
            # Cancellation before prefill still commits the supplied text once.
            w.send(dict(op='generate', **options))
            assert w.read()['response_prefix'] == prefix
            w.send(dict(op='cancel'))
            done = ack = False
            while not (done and ack):
                event = w.read()
                done |= event.get('event') == 'done'
                ack |= event.get('ok') is True
            entry = w.request(op='history')['entries'][-1]
            assert entry['text'] == prefix and entry['response_prefix'] == prefix
            w.generate(op='generate', temperature=0, max_tokens=4)
            ephemeral = w.generate(op='ephemeral', messages=[dict(role='user', text='Quanto fa 17 per 24?')], **options)
            assert ephemeral['events'][0]['response_prefix'] == prefix
            assert ephemeral['reasoning'] == ''
            plain = w.generate(op='ephemeral', messages=[dict(role='user', text='Say hello.')],
                               temperature=0, max_tokens=4, reasoning=False)
            assert 'response_prefix' not in plain['events'][0]
            # Import/rebuild must retain the origin and whitespace of response text.
            text = prefix + '408.\n '
            imported = [dict(role='user', text='Question'), dict(role='assistant', text=text, response_prefix=prefix)]
            w.request(op='rebuild', messages=imported)
            entry = w.request(op='history')['entries'][-1]
            assert entry['text'] == text and entry['response_prefix'] == prefix
            w.request(op='append', role='user', text='Check that answer.')
            thinking = w.generate(op='generate', max_tokens=4, reasoning=dict(effort='low', prefix='Let me verify.'))
            assert thinking['events'][0]['reasoning_prefix'] == 'Let me verify.'
            assert 'response_prefix' not in thinking['events'][0]
            assert any(e.get('response_prefix') == prefix for e in w.request(op='history')['entries'])
            crash_session = w.request(op='create')['session']
            w.request(op='append', role='user', text='Quanto fa 17 per 24?')
            w.send(dict(op='generate', **options))
            assert w.read()['response_prefix'] == prefix
            w.process.kill()
        with Wire(**common) as w:
            w.request(op='open', session=crash_session)
            before = w.request(op='history')
            assert w.generate(op='generate', allow_error=True, reasoning=False, response=dict(prefix='replacement'))['error']
            assert w.generate(op='generate', allow_error=True, reasoning=dict(effort='low'))['error']
            assert w.request(op='history') == before
            recovered = w.generate(op='generate', temperature=0, max_tokens=4)
            assert recovered['events'][0]['response_prefix'] == prefix
            assert w.request(op='history')['entries'][-1]['text'] == prefix + recovered['text']
    print('response prefix: validation, exact text, accounting, continuation, reopen, rewind, cancellation, rebuild, mode switch and crash recovery PASS')


if __name__ == '__main__':
    main()
