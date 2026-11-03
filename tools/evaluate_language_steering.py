#!/usr/bin/env python3
"""Run a fixed language-steering comparison and archive every answer."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from steering_wire import Wire
from evaluate_steering import repetition

SYSTEM='You are a helpful assistant. Answer the question briefly in two or three sentences.'
IT=set('il lo la gli le un uno una di del della dello dei degli delle al alla allo ai alle agli da dal dalla nel nella nei nelle sul sulla con per che è sono come più questo questa questi queste si non può quando anche mentre tra ha hanno essere viene vengono grazie perché quindi attraverso permette permette'.split())
EN=set('the an of is are and to for that it this these those with by from as can which when while has have be been its their because therefore through allows'.split())
ES=set('el los las que es son por para como más esto estos estas puede cuando también mientras hacia sin está están porque cual desde y'.split())


def language(text):
    words=re.findall(r"[^\W\d_]+",text.lower(),re.UNICODE)
    it=sum(w in IT for w in words);en=sum(w in EN for w in words);es=sum(w in ES for w in words)
    label='it' if it>=3 and it>=2*(en+es) else 'en' if en>=3 and en>=2*(it+es) else 'mixed_or_unknown'
    return dict(label=label,italian_markers=it,english_markers=en,spanish_markers=es)


def constraint(text,prompt):
    if prompt.get('kind')=='json':
        try:return json.loads(text)==prompt['expected']
        except ValueError:return False
    if prompt.get('kind')=='python':
        text=re.sub(r'^```(?:python)?\s*|\s*```$','',text.strip())
        try:return any(isinstance(n,ast.FunctionDef) and n.name==prompt['function'] for n in ast.parse(text).body)
        except SyntaxError:return False
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--xenolith',default='./xenolith')
    p.add_argument('--registry',required=True);p.add_argument('--configs',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int);p.add_argument('--tokens',type=int,default=384)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    prompts=[json.loads(x) for x in a.prompts.read_text().splitlines()]
    if a.limit is not None:
        if a.limit<1:p.error('--limit must be positive')
        prompts=prompts[:a.limit]
    configs=[dict(name='baseline'),dict(name='system_italian',system=SYSTEM+' Answer in Italian.')]
    configs+=json.loads(a.configs.read_text())
    if len({c['name'] for c in configs})!=len(configs):p.error('Duplicate configuration name')
    report=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        configs=configs,system=SYSTEM,prompts_sha256=hashlib.sha256(a.prompts.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(Path(a.xenolith).read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        created_unix=time.time(),summaries={},language_measure='stopword heuristic; requires manual review')
    with Wire(a.xenolith,a.model,a.out/'state',a.registry,context=4096) as wire:
        report['describe']=wire.request(op='describe')
        with (a.out/'responses.jsonl').open('x') as stream:
            for config in configs:
                samples=[]
                for index,prompt in enumerate(prompts):
                    request=dict(op='ephemeral',system=config.get('system',SYSTEM),
                        messages=[dict(role='user',text=prompt['prompt'])],temperature=0,seed=1,
                        reasoning=False,max_tokens=a.tokens)
                    if config.get('artifact'):
                        request['steering']=dict(artifact=config['artifact'],strength=config['strength'])
                    r=wire.generate(allow_error=True,**request);r.pop('events')
                    r.update(configuration=config['name'],index=index,prompt=prompt['prompt'],language=language(r['text']),
                        repeated_4gram_fraction=repetition(r['text']),replacement_characters=r['text'].count('\ufffd'),
                        longest_character_run=max((len(m.group()) for m in re.finditer(r'(.)\1*',r['text'],re.DOTALL)),default=0))
                    passed=constraint(r['text'],prompt)
                    if passed is not None:r['constraint_pass']=passed
                    stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();samples.append(r)
                report['summaries'][config['name']]=dict(samples=len(samples),
                    italian=sum(r['language']['label']=='it' for r in samples),
                    english=sum(r['language']['label']=='en' for r in samples),
                    errors=sum(r['error'] is not None for r in samples),
                    length_stops=sum(r['done'].get('stop')=='length' for r in samples),
                    max_repetition=max(r['repeated_4gram_fraction'] for r in samples),
                    max_character_run=max(r['longest_character_run'] for r in samples),
                    replacements=sum(r['replacement_characters'] for r in samples),
                    constraint_passes=sum(r.get('constraint_pass',False) for r in samples),
                    constraint_cases=sum('constraint_pass' in r for r in samples))
                (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
                print(config['name'],report['summaries'][config['name']],flush=True)


if __name__=='__main__':main()
