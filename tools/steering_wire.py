"""Small synchronous wire client shared by steering evaluation and integration tests."""
import json
import os
from pathlib import Path
import select
import subprocess
import time


class Wire:
    def __init__(self, binary, model, state, registry=None, cache=None, budget='0', context=4096):
        self.state = Path(state)
        self.state.mkdir(parents=True, exist_ok=True)
        self.log = (self.state / 'engine.log').open('ab')
        cmd = [str(binary), 'wire', str(model), '--ctx', str(context), '--state', str(state),
               '--cache-budget', str(budget)]
        if registry:
            cmd += ['--steering-dir', str(registry)]
        if cache:
            cmd += ['--cache', str(cache)]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, bufsize=0)
        self.buffer = b''

    def send(self, request):
        self.process.stdin.write((json.dumps(request, ensure_ascii=False, allow_nan=False) + '\n').encode())
        self.process.stdin.flush()

    def read(self, timeout=600):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError(f'Wire timeout; inspect {self.state / "engine.log"}')
            part = os.read(self.process.stdout.fileno(), 65536)
            if not part:
                raise RuntimeError(f'Engine exited ({self.process.poll()}); inspect {self.state / "engine.log"}')
            self.buffer += part
        line, self.buffer = self.buffer.split(b'\n', 1)
        return json.loads(line)

    def request(self, **request):
        self.send(request)
        reply = self.read()
        if not reply.get('ok'):
            raise RuntimeError(reply)
        return reply

    def generate(self, allow_error=False, **request):
        started = time.monotonic()
        self.send(request)
        text, reasoning, calls, events = [], [], [], []
        first_text = None
        while True:
            event = self.read()
            events.append(event)
            failed = event.get('ok') is False or event.get('event') == 'error'
            if failed and not allow_error:
                raise RuntimeError(event)
            kind = event.get('event')
            if kind == 'text_delta':
                if first_text is None:
                    first_text = time.monotonic() - started
                text.append(event['text'])
            elif kind == 'reasoning_delta':
                reasoning.append(event['text'])
            elif kind == 'toolcall_end':
                calls.append(event)
            if kind == 'done' or failed:
                return {'text': ''.join(text), 'reasoning': ''.join(reasoning), 'calls': calls,
                        'error': event if failed else None,
                        'done': event, 'first_text_seconds': first_text,
                        'wall_seconds': time.monotonic() - started, 'events': events}

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
