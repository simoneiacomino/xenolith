#!/usr/bin/env python3
"""Verify multi-artifact routing and explicit control omission without a model."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import evaluate_steering as evaluator


class FakeWire:
    calls=[]
    process=SimpleNamespace(pid=-1)
    def __init__(self,*args,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def request(self,**request):return {'steering':{'enabled':True}}
    def generate(self,**request):
        self.calls.append(request)
        return dict(events=[],text='A complete answer.',reasoning='',calls=[],error=None,
                    first_text_seconds=0.01,wall_seconds=0.02,
                    done={'stop':'stop','inference':{'decode':{'tokens':4,'elapsed_ms':10}}})


with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);binary=root/'binary';binary.write_bytes(b'fixture')
    prompts=root/'prompts.jsonl';prompts.write_text('{"prompt":"question"}\n')
    configs=root/'configs.json';settings=[dict(name='first',artifact='a'*64,strength=0.3),
        dict(name='second',artifact='b'*64,strength=-0.2)]
    configs.write_text(json.dumps(settings))
    registry=root/'registry'
    for c in settings:
        d=registry/c['artifact'];d.mkdir(parents=True);(d/'manifest.json').write_text('{"mode":"additive"}')
    for omit in (False,True):
        FakeWire.calls=[];out=root/str(omit)
        argv=['evaluate','--model','fixture','--xenolith',str(binary),'--registry',str(registry),
              '--configs',str(configs),'--prompts',str(prompts),'--out',str(out)]
        if omit:argv.append('--skip-controls')
        with patch.object(sys,'argv',argv),patch.object(evaluator,'Wire',FakeWire),contextlib.redirect_stdout(io.StringIO()):
            evaluator.main()
        calls=[c for c in FakeWire.calls if c.get('steering')]
        assert [c['steering'] for c in calls]==[dict(artifact=c['artifact'],strength=c['strength']) for c in settings]
        report=json.loads((out/'summary.json').read_text())
        assert set(report['summaries'])==({'first','second'} if omit else {'baseline','system_prompt','first','second'})
        assert len(FakeWire.calls)==(3 if omit else 5) # Includes one warm-up.
print('multi-artifact evaluator routing and archived-control omission: PASS')
