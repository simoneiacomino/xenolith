#!/usr/bin/env python3
"""Opt-in real-model installation/client smoke; never part of unittest discovery.

Requires Linux, a prepared distribution containing all three components, supported
GPU/runtime, >=21 GiB available memory and no current Xenolith weight owner.
Invokes a real loopback curl | sh prepare-only installation, then explicitly runs
serial inference. Never uses XENOLITH_MODEL or creates client inference settings.
All results remain in work-dir.
"""
import argparse
import fcntl
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import tempfile
import threading
import time
import zlib
import secrets


MIN_AVAILABLE = 21 * 1024 ** 3


def start_identity(pid):
    """Linux starttime field distinguishes an owned process from a reused PID."""
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing')


def assert_lock_free(lock):
    # A missing file is free. Do not create an alternate lock under the test HOME.
    if not lock.exists():
        return
    with lock.open('rb') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Existing weight owner holds {lock}; do not stop it') from exc
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def tiny_png(path):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    pixels = b''.join(b'\0' + bytes((30, 100, 220)) * 64 for _ in range(64))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 64, 64, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


class Runner:
    def __init__(self, args):
        self.args = args
        self.work = args.work_dir.resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        if any(self.work.iterdir()):
            raise RuntimeError('--work-dir must be empty; use a fresh directory to retain previous evidence')
        self.runtime = Path(tempfile.mkdtemp(prefix='xe-smoke-', dir='/tmp'))
        self.socket = self.runtime / 'engine.sock'
        self.owned = {}
        inherited_state = os.environ.get('XDG_STATE_HOME', '')
        self.global_state = Path(inherited_state if inherited_state.startswith('/')
                                 else str(Path.home() / '.local/state'))
        self.lock = self.global_state / 'xenolith/engine.lock'
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('XENOLITH_', 'XE_', 'XENOPI_', 'PI_', 'XDG_', 'CODEX_', 'OPENAI_'))}
        self.env.update(HOME=str(self.work / 'home'), XDG_CONFIG_HOME=str(self.work / 'config'),
                        XDG_CACHE_HOME=str(self.work / 'cache'), XDG_DATA_HOME=str(self.work / 'data'),
                        XDG_STATE_HOME=str(self.global_state), XDG_RUNTIME_DIR=str(self.runtime))
        self.env['NO_PROXY'] = '127.0.0.1,localhost'
        self.env['no_proxy'] = self.env['NO_PROXY']
        self.bin = self.work / 'home/.local/bin'
        self.env['PATH'] = str(self.bin) + ':' + self.env['PATH']
        self.engine = self.bin / 'xenolith'
        self.config = self.work / 'config/xenolith/config.json'
        self.cwd = self.work / 'project'
        self.cwd.mkdir()
        self.report = {'schema_version': 1, 'kind': 'real-model-smoke', 'status': 'running',
                       'distribution': str(args.distribution), 'model': str(args.model),
                       'projector': str(args.mmproj) if args.mmproj else None,
                       'global_weight_lock': str(self.lock), 'private_socket': str(self.socket),
                       'installation_inference': 'not run: install and rerun use --skip-verify (prepare-only)',
                       'inference_checks': 'explicit chat and client generation checks after installation',
                       'steps': [], 'cleanup': []}
        self.save()

    def save(self):
        (self.work / 'report.json').write_text(json.dumps(self.report, indent=2) + '\n')

    def discover_own_engines(self):
        if not self.engine.exists():
            return
        expected = self.engine.resolve()
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                argv = (entry / 'cmdline').read_bytes().split(b'\0')
                wanted = str(self.socket).encode()
                if b'serve' not in argv or b'--socket' not in argv or wanted not in argv:
                    continue
                if argv[argv.index(b'--socket') + 1] != wanted or (entry / 'exe').resolve() != expected:
                    continue
                identity = start_identity(pid)
                if identity and pid not in self.owned:
                    fd = os.pidfd_open(pid)
                    if start_identity(pid) != identity:
                        os.close(fd)
                        continue
                    self.owned[pid] = (identity, fd)
                    self.report['cleanup'].append({'event': 'identified_own_engine', 'pid': pid,
                                                   'starttime': identity, 'socket': str(self.socket)})
            except (OSError, IndexError):
                continue

    def stop_own_engines(self):
        # Catch a detached client spawn that was starting when its parent failed.
        for _ in range(4):
            self.discover_own_engines()
            time.sleep(0.1)
        for pid, (identity, fd) in list(self.owned.items()):
            try:
                if start_identity(pid) == identity:
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
                    deadline = time.monotonic() + 15
                    while start_identity(pid) == identity and time.monotonic() < deadline:
                        time.sleep(0.1)
                    if start_identity(pid) == identity:
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                        deadline = time.monotonic() + 5
                        while start_identity(pid) == identity and time.monotonic() < deadline:
                            time.sleep(0.1)
                    if start_identity(pid) == identity:
                        raise RuntimeError(f'Owned engine {pid} did not stop; refusing next model load')
                    self.report['cleanup'].append({'event': 'stopped_own_engine', 'pid': pid,
                                                  'starttime': identity})
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)
                del self.owned[pid]
        self.save()
        assert_lock_free(self.lock)

    def preflight(self, stage):
        assert_lock_free(self.lock)
        available = memory_available()
        self.report.setdefault('preflights', []).append({'stage': stage, 'available_bytes': available,
                                                       'minimum_bytes': MIN_AVAILABLE})
        self.save()
        if available < MIN_AVAILABLE:
            raise RuntimeError(f'{stage}: only {available / 1024**3:.2f} GiB available; need >=21 GiB')

    def run(self, name, argv, text=None, stream=None, timeout=600):
        out_path, err_path = self.work / f'{name}.stdout', self.work / f'{name}.stderr'
        started = time.monotonic()
        step = {'name': name, 'argv': list(map(str, argv)), 'stdout': out_path.name, 'stderr': err_path.name}
        self.report['steps'].append(step)
        self.save()
        with out_path.open('wb') as out, err_path.open('wb') as err:
            proc = subprocess.Popen(list(map(str, argv)), cwd=self.cwd, env=self.env,
                                    stdin=stream if stream is not None else subprocess.PIPE,
                                    stdout=out, stderr=err, start_new_session=True)
            try:
                if stream is None:
                    proc.stdin.write((text or '').encode())
                    proc.stdin.close()
                while proc.poll() is None:
                    self.discover_own_engines()
                    if time.monotonic() - started > timeout:
                        raise TimeoutError(f'{name} exceeded {timeout}s')
                    time.sleep(0.1)
                step['returncode'] = proc.returncode
                if proc.returncode:
                    raise RuntimeError(f'{name} failed ({proc.returncode}); see {err_path}')
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=10)
                raise
            finally:
                step['elapsed_seconds'] = round(time.monotonic() - started, 3)
                self.save()
        return out_path.read_text(errors='replace')

    def install(self):
        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        handler = functools.partial(Quiet, directory=str(self.args.distribution))
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        manifest_hash = hashlib.sha256((self.args.distribution / 'manifest.tsv').read_bytes()).hexdigest()
        argv = ['sh', '-s', '--', '--source', url, '--yes', '--use-packages', '--interface', 'all',
                '--model', str(self.args.model), '--manifest-sha256', manifest_hash, '--skip-verify']
        if self.args.mmproj:
            argv += ['--mmproj', str(self.args.mmproj)]
        if self.args.allow_unverified:
            argv += ['--allow-unverified']
        def piped_install(name, command):
            with (self.work / f'{name}-curl.stderr').open('wb') as err:
                curl = subprocess.Popen(['curl', '-fsSL', '--noproxy', '127.0.0.1', url + '/install.sh'],
                                        env=self.env, stdout=subprocess.PIPE, stderr=err)
                try:
                    output = self.run(name, command, stream=curl.stdout)
                    curl.stdout.close()
                    if curl.wait(timeout=20):
                        raise RuntimeError('curl fetching install.sh failed')
                    return output
                finally:
                    if curl.poll() is None:
                        curl.terminate()
                        curl.wait(timeout=10)
        try:
            piped_install('install', argv)
            # Isolate operational data while retaining the real user's weight lock.
            config = json.loads(self.config.read_text())
            config.update(socket=str(self.socket), state_dir=str(self.work / 'engine-state'),
                          cache_dir=str(self.work / 'engine-cache'))
            self.config.write_text(json.dumps(config, indent=2) + '\n')
            self.report['native_config_isolation_overrides'] = {k: config[k] for k in ('socket', 'state_dir', 'cache_dir')}
            before = hashlib.sha256(self.config.read_bytes()).hexdigest()
            piped_install('install-rerun', argv)
            after = hashlib.sha256(self.config.read_bytes()).hexdigest()
            self.report['installer_rerun'] = {'config_sha256_before': before, 'config_sha256_after': after,
                                              'same_model_and_interface_options': True}
            if before != after:
                raise RuntimeError('Installer rerun changed the existing native configuration')
            for path in (self.work / 'home/.xenopi/agent/xenolith.json', self.work / 'home/.codexe/config.toml',
                         self.work / 'home/.codex/config.toml'):
                if path.exists():
                    raise RuntimeError(f'Installer unexpectedly created per-client inference configuration: {path}')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        inspect = json.loads(self.run('inspect', [self.engine, 'config', 'inspect', '--json']))
        if inspect['engine_lock'] != str(self.lock) or inspect['endpoint']['path'] != str(self.socket):
            raise RuntimeError('Native config does not retain the expected global lock/private socket')
        self.report['inspect'] = inspect
        # Engine config inspect above validates its loader/API. Its current
        # standalone --help intentionally exits 1, unlike the clients.
        for name in ('xenopi', 'codexe'):
            self.run(f'{name}-help', [self.bin / name, '--help'], timeout=30)

    def check_codexe(self, text, final):
        events = [json.loads(line) for line in text.splitlines() if line.strip()]
        if any(event.get('type') in ('error', 'turn.failed') or
               event.get('item', {}).get('type') == 'error' for event in events):
            raise RuntimeError('CodeXe emitted an error/failed turn')
        if not any(event.get('type') == 'turn.completed' for event in events):
            raise RuntimeError('CodeXe did not complete a turn')
        if not final.is_file() or not final.read_text().strip():
            raise RuntimeError('CodeXe final response is empty')
        threads = [event['thread_id'] for event in events if event.get('type') == 'thread.started']
        if len(threads) != 1:
            raise RuntimeError('CodeXe did not identify exactly one conversation')
        return threads[0]

    def inference(self):
        image = None
        if self.args.mmproj:
            image = self.args.image or self.work / 'tiny-blue.png'
            if not self.args.image:
                tiny_png(image)
        first = ('Describe this image briefly in one sentence.' if image else
                 'Remember the word cedar. Say hello in one short sentence. Do not use tools.')
        followup = 'Reply in one short sentence about what I just asked. Do not use tools.'
        self.preflight('internal-chat')
        try:
            argv = [self.engine, 'chat', '--temp', '0', '-n', '96']
            if image:
                argv += ['--image', image]
            text = self.run('chat-two-turns', argv, text=first + '\n' + followup + '\n/exit\n')
            if len(re.findall(r'\[[1-9][0-9]* tokens \|', text)) < 2:
                raise RuntimeError('Internal chat lacks two nonempty generated turns')
        finally:
            self.stop_own_engines()
        self.preflight('xenopi')
        try:
            session = self.work / 'xenopi-session.jsonl'
            base = [self.bin / 'xenopi', '--print', '--thinking', 'off', '--no-tools', '--session', session]
            argv = base + ([f'@{image}'] if image else []) + [first]
            if not self.run('xenopi-first', argv).strip():
                raise RuntimeError('XenoPi first reply is empty')
            if not session.is_file():
                raise RuntimeError('XenoPi did not persist its session')
            # Restart the own engine to exercise the persisted native path too.
            self.stop_own_engines()
            self.preflight('xenopi-resume')
            if not self.run('xenopi-resume', base + [followup]).strip():
                raise RuntimeError('XenoPi resumed reply is empty')
        finally:
            self.stop_own_engines()
        self.preflight('codexe')
        try:
            base = [self.bin / 'codexe', 'exec', '--skip-git-repo-check', '--sandbox', 'read-only', '--json']
            final = self.work / 'codexe-first.final.txt'
            # CodeXe's shared --image accepts multiple paths: terminate options
            # so the natural-language prompt cannot become another image path.
            argv = base + ['--output-last-message', final] + (['--image', image] if image else []) + ['--', first]
            first_thread = self.check_codexe(self.run('codexe-first', argv), final)
            self.stop_own_engines()
            self.preflight('codexe-resume')
            final = self.work / 'codexe-resume.final.txt'
            resumed_thread = self.check_codexe(self.run('codexe-resume', base + ['--output-last-message', final,
                                                               'resume', '--last', '--', followup]), final)
            if resumed_thread != first_thread:
                raise RuntimeError('CodeXe resumed a different conversation')
        finally:
            self.stop_own_engines()

    def tools(self):
        """Require an actual tool result, not just a generated claim of success."""
        self.preflight('codexe-tools')
        nonce = 'xenolith-tool-check-' + secrets.token_hex(12)
        fixture = self.cwd / 'release-control.txt'
        fixture.write_text(nonce + '\n')
        final = self.work / 'codexe-tools.final.txt'
        try:
            prompt = ('Use a tool to read release-control.txt in the current directory. '
                      'Return its exact contents. Do not guess the contents.')
            text = self.run('codexe-tools', [self.bin / 'codexe', 'exec',
                            '--skip-git-repo-check', '--sandbox', 'read-only', '--json',
                            '--output-last-message', final, '--', prompt])
            thread = self.check_codexe(text, final)
            events = [json.loads(line) for line in text.splitlines() if line.strip()]
            completed = [event.get('item', {}) for event in events
                         if event.get('type') == 'item.completed']
            calls = [item for item in completed if item.get('type') in
                     ('command_execution', 'mcp_tool_call') and item.get('status') == 'completed']
            if not calls or nonce not in final.read_text():
                raise RuntimeError('CodeXe must complete a real tool call and report the unread nonce')
            self.report['tool_acceptance'] = {'thread_id': thread, 'fixture': fixture.name,
                                              'nonce': nonce, 'completed_calls': len(calls)}
            self.save()
        finally:
            self.stop_own_engines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distribution', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--mmproj', type=Path)
    parser.add_argument('--image', type=Path, help='Optional image; requires --mmproj; otherwise a tiny PNG is generated')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--allow-unverified', action='store_true')
    parser.add_argument('--check-tools', action='store_true',
                        help='Also require a real CodeXe tool call to read a fresh local nonce')
    args = parser.parse_args()
    for name in ('distribution', 'model', 'mmproj', 'image'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve(strict=True))
    if args.image and not args.mmproj:
        parser.error('--image requires --mmproj')
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        parser.error('Linux Python with pidfd support is required for safe owned-engine cleanup')
    runner = Runner(args)
    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        runner.preflight('before-install')
        runner.install()
        runner.inference()
        if args.check_tools:
            runner.tools()
        runner.report['status'] = 'passed'
    except BaseException as exc:
        runner.report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            runner.stop_own_engines()
        except BaseException as exc:
            runner.report.update(status='failed', cleanup_error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            runner.save()
            print(f'Report: {runner.work / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
