"""Small helpers shared by the model-backed vision checks."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import time


def check(condition, message="check failed"):
    if not condition:
        raise AssertionError(message)


def fresh_output(path):
    """Never reuse a state/cache directory or a previous successful report."""
    path = path.resolve()
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"--output must not already exist: {path}") from None
    return path


def write_report(output, report):
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


class Wire:
    def __init__(self, output, report):
        self.output = output
        self.report = report
        self.proc = None
        self.log = None
        self.pending = b""
        self.starts = 0

    def start(self, command):
        check(self.proc is None, "server already running")
        self.pending = b""
        self.log = (self.output / f"server-{self.starts}.log").open("wb")
        self.starts += 1
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def stop(self):
        if self.proc is None:
            return
        status = Path(f"/proc/{self.proc.pid}/status")
        if status.exists():
            fields = dict(line.split(":", 1) for line in status.read_text().splitlines()
                          if ":" in line)
            self.report.setdefault("server_memory", []).append({
                key: fields[key].strip() for key in ("VmPeak", "VmHWM", "VmRSS", "VmSwap")
                if key in fields})
        self.proc.stdin.close()
        code = self.proc.wait(timeout=120)
        self.proc.stdout.close()
        self.log.close()
        self.proc = None
        check(code == 0, f"engine exited with status {code}")

    def abort(self):
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            if not self.proc.stdin.closed:
                try:
                    self.proc.stdin.close()
                except BrokenPipeError:
                    pass
            self.proc.stdout.close()
            self.log.close()
            self.proc = None

    def send(self, request):
        self.proc.stdin.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        self.proc.stdin.flush()

    def read(self):
        deadline = time.monotonic() + 600
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            while b"\n" not in self.pending:
                check(selector.select(max(0, deadline - time.monotonic())), "response timeout")
                part = os.read(self.proc.stdout.fileno(), 1048576)
                check(part, "engine closed output")
                self.pending += part
        line, self.pending = self.pending.split(b"\n", 1)
        return json.loads(line)

    def call(self, request, ok=True):
        self.send(request)
        reply = self.read()
        check(reply.get("ok") is ok, (request, reply))
        return reply
