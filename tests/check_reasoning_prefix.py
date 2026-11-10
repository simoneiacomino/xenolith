#!/usr/bin/env python3
"""Model integration for reasoning prefixes, replay, cancellation and isolation."""
import argparse
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from steering_wire import Wire


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xenolith', default='./xenolith')
    p.add_argument('--model', required=True)
    p.add_argument('--registry')
    p.add_argument('--projection')
    args = p.parse_args()
    prefix = '  Verifico prima i dati disponibili.\nÈ importante distinguere le ipotesi.\n'
    with tempfile.TemporaryDirectory(prefix='xe-reasoning-prefix-') as tmp:
        root = Path(tmp)
        common = dict(binary=args.xenolith, model=args.model, state=root/'state',
                      cache=root/'cache', budget='128MiB', context=1024)
        options = dict(temperature=0, seed=1, max_tokens=12,
                       reasoning=dict(effort='low', budget_tokens=64, prefix=prefix))
        with Wire(**common) as w:
            assert w.request(op='describe')['reasoning']['prefix']['max_bytes'] == 4096
            session = w.request(op='create', system='Be helpful.')['session']
            user = w.request(op='append', role='user', text='Quanto fa 17 per 24?')['marker']
            before = w.request(op='history')
            w.send(dict(op='generate',reasoning=dict(effort='low',prefix='x '*2048)))
            assert w.read().get('code') == 'context_length_exceeded'
            assert w.request(op='history') == before
            for bad in (None, 1, 'x'*4097, 'bad\x00text', '<channel|>'):
                w.send(dict(op='generate', reasoning=dict(effort='low', prefix=bad)))
                assert w.read().get('ok') is False
                assert w.request(op='history') == before
            result = w.generate(op='generate', **options)
            start = result['events'][0]
            assert start['event'] == 'start' and start['reasoning_prefix'] == prefix
            assert start['reasoning_prefix_tokens'] > 0
            assert result['done']['usage']['output'] == 12
            assert result['done']['usage']['reasoning_prefix'] == start['reasoning_prefix_tokens']
            entry = w.request(op='history')['entries'][-1]
            first_marker = entry['marker']
            assert entry['reasoning_prefix'] == prefix
            assert entry['reasoning'] == prefix + result['reasoning']
            # Refuse reinjection while a turn is open, without modifying history.
            saved = w.request(op='history')
            w.send(dict(op='generate', **options)); assert w.read().get('ok') is False
            assert w.request(op='history') == saved
            # Continue without a prefix, then compare cold replay after reopening.
            continuation = w.generate(op='generate', temperature=0, seed=1, max_tokens=4)
            assert 'reasoning_prefix' not in continuation['events'][0]
            assert continuation['done']['usage']['reasoning_prefix'] == 0
            w.request(op='checkpoint')
        with Wire(**common) as w:
            w.request(op='open', session=session)
            entries = w.request(op='history')['entries']
            assert sum(e.get('reasoning_prefix') == prefix for e in entries) == 1
            w.request(op='rewind',marker=first_marker)
            cold = w.generate(op='generate',temperature=0,seed=1,max_tokens=4)
            assert (cold['text'],cold['reasoning']) == (continuation['text'],continuation['reasoning'])
            w.request(op='rewind', marker=user)
            assert not any(e.get('reasoning_prefix') for e in w.request(op='history')['entries'])
            # Cancellation before prefill must leave a resumable logical prefix.
            w.send(dict(op='generate', **options))
            assert w.read()['event'] == 'start'
            w.send(dict(op='cancel'))
            done = ack = False
            while not (done and ack):
                e = w.read()
                ack |= e.get('ok') is True
                done |= e.get('event') == 'done'
            entry = w.request(op='history')['entries'][-1]
            assert entry['reasoning'] == prefix and entry['reasoning_prefix'] == prefix
            w.generate(op='generate', temperature=0, max_tokens=4)
            # Ephemeral requests neither inherit nor change durable prefix state.
            ephemeral = w.generate(op='ephemeral', messages=[dict(role='user', text='Say hello.')], **options)
            assert ephemeral['events'][0]['reasoning_prefix'] == prefix
            plain = w.generate(op='ephemeral', messages=[dict(role='user', text='Say hello.')],
                               temperature=0, max_tokens=4, reasoning=False)
            assert 'reasoning_prefix' not in plain['events'][0]
            rebuilt = w.request(op='rebuild', messages=[dict(role='user',text='Question'),
                dict(role='assistant',text='Answer',reasoning=prefix+'Finished.',reasoning_prefix=prefix)])
            imported = w.request(op='history')['entries'][-1]
            assert imported['reasoning_prefix'] == prefix and imported['reasoning'] == prefix+'Finished.'
            crash_session = w.request(op='create')['session']
            w.request(op='append',role='user',text='Check this carefully.')
            w.send(dict(op='generate',**options))
            assert w.read()['event'] == 'start'
            w.process.kill()
        with Wire(**common) as w:
            w.request(op='open',session=crash_session)
            recovered = w.generate(op='generate',temperature=0,max_tokens=4)
            assert recovered['events'][0]['reasoning_prefix'] == prefix

        if args.registry and args.projection:
            mixed = dict(common, registry=args.registry)
            with Wire(**mixed) as w:
                w.request(op='create')
                w.request(op='append',role='user',text='Say hello.')
                w.generate(op='generate',temperature=0,max_tokens=96,
                           reasoning=dict(effort='low',budget_tokens=8))
                w.request(op='steer',artifact=args.projection,strength=-0.25)
                w.request(op='append',role='user',text='Quanto fa 17 per 24?')
                result = w.generate(op='generate',**options)
                assert result['events'][0]['reasoning_prefix'] == prefix
                assert w.request(op='history')['entries'][-1]['reasoning_prefix'] == prefix
                assert w.request(op='steer')['steering']['transitions'] == 1
            print('reasoning prefix with activation steering and frozen history: PASS')
    print('reasoning prefix: validation, exact text, accounting, continuation, reopen, rewind, cancellation and isolation PASS')


if __name__ == '__main__':
    main()
