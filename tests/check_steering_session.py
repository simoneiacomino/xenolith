#!/usr/bin/env python3
"""Model integration: transitions, snapshots, restart, eviction and ephemeral isolation."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from steering_wire import Wire


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xenolith',default='./xenolith');p.add_argument('--model',required=True)
    p.add_argument('--registry',required=True);p.add_argument('--projection',required=True);p.add_argument('--additive',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    registry=args.out/'registry';registry.mkdir()
    for identity in (args.projection,args.additive):
        shutil.copytree(Path(args.registry)/identity,registry/identity)
    common=dict(binary=args.xenolith,model=args.model,state=args.out/'state',registry=registry,
                cache=args.out/'cache',budget='1GiB',context=1024)
    generations=dict(temperature=0,seed=1,reasoning=False,max_tokens=12)
    events=[]
    with Wire(**common) as w:
        assert w.request(op='describe')['steering']['enabled']
        created=w.request(op='create',system='You are a helpful assistant.');session=created['session']
        user=w.request(op='append',role='user',text='Explain why a database uses indexes.')
        first=w.request(op='steer',artifact=args.projection,strength=-0.25)['steering'];assert first['position']==0
        before=w.request(op='steer');w.send({'op':'steer','artifact':'f'*64,'strength':1});assert w.read()['ok'] is False
        assert w.request(op='steer')==before
        events.append(w.generate(op='generate',**generations))
        b=w.request(op='steer',artifact=args.additive,strength=0.05)['steering'];assert b['position']>0
        w.request(op='append',role='user',text='Give one practical example.')
        events.append(w.generate(op='generate',**generations))
        off=w.request(op='steer',artifact=None)['steering'];assert off['transitions']==3 and off['artifact'] is None
        checkpoint=w.request(op='checkpoint');assert checkpoint['saved'],checkpoint
        expected=w.request(op='steer')['steering']
        # An explicit ephemeral setting never changes the durable selection.
        events.append(w.generate(op='ephemeral',messages=[{'role':'user','text':'Say hello.'}],
                                 steering={'artifact':args.projection,'strength':-0.25},**generations))
        assert w.request(op='steer')['steering']==expected
        events.append(w.generate(op='ephemeral',messages=[{'role':'user','text':'Say hello.'}],**generations))
        tool={'name':'add','description':'Add two integers.',
              'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}
        for steering in (None,{'artifact':args.projection,'strength':-0.25}):
            sample=w.generate(op='ephemeral',system='Use the supplied tool for arithmetic.',tools=[tool],
                messages=[{'role':'user','text':'Use add to compute 2 + 3.'}],temperature=0,seed=1,
                reasoning=False,max_tokens=128,steering=steering)
            assert sample['calls'] and sample['calls'][0]['name']=='add',sample
            assert sample['calls'][0]['arguments']=={'a':2,'b':3},sample
            events.append(sample)
        # Establish the template's global thinking marker before freezing it.
        thinking_session=w.request(op='create',system='You are a helpful assistant.')['session']
        thinking=dict(temperature=0,seed=1,
                      reasoning={'effort':'low','budget_tokens':64,'history':'discard'},max_tokens=320)
        w.request(op='append',role='user',text='Calculate 137 times 249, checking the arithmetic carefully.')
        events.append(w.generate(op='generate',**thinking))
        w.request(op='steer',artifact=args.additive,strength=0.01)
        w.request(op='append',role='user',text='Verify the result by decomposing 249 as 250 minus 1.')
        thought=w.generate(op='generate',**thinking)
        assert thought['done']['usage']['reasoning']>0 and thought['reasoning'],thought
        assert any(entry.get('reasoning')==thought['reasoning']
                   for entry in w.request(op='history')['entries'])
        events.append(thought)
        # A separate durable conversation has no inherited configuration.
        w.request(op='create');assert w.request(op='steer')['steering']['transitions']==0
        w.request(op='steer',artifact=None);assert w.request(op='steer')['steering']['transitions']==0
        w.request(op='open',session=session);assert w.request(op='steer')['steering']==expected
    with Wire(**common) as w:
        w.request(op='open',session=thinking_session)
        assert any(entry.get('reasoning')==thought['reasoning']
                   for entry in w.request(op='history')['entries'])
        w.request(op='open',session=session);assert w.request(op='steer')['steering']==expected
        w.request(op='append',role='user',text='Summarize the main idea.')
        cached=w.generate(op='generate',**generations);events.append(cached)
        assert cached['done'].get('resume',{}).get('loaded'),cached['done']
        checkpoint=w.request(op='checkpoint');assert checkpoint['saved'],checkpoint
    # Eviction retains transcripts and artifact history. Only files in this test's cache are removed.
    shutil.rmtree(args.out/'cache')
    with Wire(**common) as w:
        w.request(op='open',session=session);assert w.request(op='steer')['steering']==expected
        w.request(op='append',role='user',text='What is the tradeoff?')
        events.append(w.generate(op='generate',**generations))
        w.request(op='rewind',marker=user['marker']);assert w.request(op='steer')['steering']['transitions']==0
        w.request(op='steer',artifact=args.projection,strength=0.25)
        w.request(op='rebuild',system='Be helpful.',messages=[{'role':'user','text':'Say hello.'}])
        rebuilt=w.request(op='steer')['steering'];assert rebuilt['position']==0 and rebuilt['transitions']==1
        events.append(w.generate(op='generate',**generations))
        w.request(op='append',role='user',text='Write a detailed explanation of database transactions with several examples and at least three hundred words.')
        selected=w.request(op='steer')['steering']
        w.send(dict(op='generate',temperature=0,seed=1,reasoning=False,max_tokens=128))
        w.send({'op':'steer','artifact':None})
        busy=False;cancelled=False
        while not busy:
            event=w.read()
            if event.get('ok') is False:
                assert event['code']=='busy',event;busy=True
            assert event.get('event')!='done',event
        w.send({'op':'cancel'})
        ack=False
        while not (cancelled and ack):
            event=w.read()
            if event.get('ok') is True:ack=True
            if event.get('event')=='done':
                assert event['stop']=='aborted',event;cancelled=True
        assert w.request(op='steer')['steering']==selected
        w.request(op='checkpoint')
        present=registry/args.projection;missing=registry/(args.projection+'.temporarily-missing')
        present.rename(missing)
        w.send(dict(op='generate',**generations))
        failed=w.read();assert failed.get('ok') is False and failed['code']=='invalid_request',failed
        missing.rename(present)
        assert w.request(op='steer')['steering']==selected
    # Missing historical data is an explicit error even with saved KV available.
    (registry/args.projection).rename(registry/(args.projection+'.missing'))
    with Wire(**common) as w:
        w.send({'op':'open','session':session})
        reply=w.read();assert reply.get('ok') is False and reply['code']=='invalid_request',reply
    (args.out/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    print('steering session transitions, invalid update atomicity, snapshot resume, eviction replay, rewind, rebuild, busy/cancel, missing artifacts and isolation: PASS')


if __name__=='__main__':main()
