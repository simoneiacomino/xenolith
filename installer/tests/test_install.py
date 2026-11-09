"""Installer integration tests with explicitly simulated hardware/engine.

Actual curl/tar/checksums/filesystem/HTTP range requests are exercised. These
tests do not establish inference or GPU compatibility; real-model smoke is separate.
"""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import pty
import select
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "install.sh"

FAKE_ENGINE = b'''#!/usr/bin/env python3
import json, os, pathlib, sys
a=sys.argv[1:]
if os.environ.get('FIXTURE_CALLS'):
 with pathlib.Path(os.environ['FIXTURE_CALLS']).open('a') as out: out.write(json.dumps({'command':'xenolith','args':a})+'\\n')
def value(flag, default=None):
 return a[a.index(flag)+1] if flag in a else default
p=pathlib.Path(value('--config', str(pathlib.Path(os.environ['HOME'])/'.config/xenolith/config.json')))
if a==['--help']: print('check-device config init inspect')
elif a[:2]==['config','inspect']:
 if p.exists(): json.loads(p.read_text())
 print(json.dumps({'schema_version':1,'config_path':str(p),'config_present':p.exists()}))
elif a[:2]==['config','init']:
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f: json.dump({'schema_version':1,'model':value('--model'),'mmproj':value('--mmproj'),'context':int(value('--ctx'))},f)
elif a==['check-device']:
 if os.environ.get('FIXTURE_RUNTIME_MISSING') and not pathlib.Path(os.environ['FIXTURE_RUNTIME_READY']).exists():
  print('fixture: Level Zero runtime missing',file=sys.stderr); sys.exit(1)
 print('fixture: Level Zero device ready')
elif a[0]=='chat':
 print('Simulated hello!\\n[2 tokens | avg 1.0 tps]')
else: sys.exit(1)
'''


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="installer-test-")
        self.root = Path(self.temp.name)
        self.source = self.root / "distribution"
        self.source.mkdir()
        self.home = self.root / "user home"
        self.home.mkdir()
        self.probe = self.root / "probe"
        self.write("probe/etc/os-release", 'PRETTY_NAME="Fixture Linux"\n')
        self.write("probe/proc/cpuinfo", "flags : avx2 fma f16c avx_vnni\n")
        self.write("probe/proc/meminfo", "MemTotal: 33554432 kB\nMemAvailable: 26214400 kB\n")
        self.write("probe/sys/class/drm/card0/device/vendor", "0x8086\n")
        self.write("probe/sys/class/drm/card0/device/device", "0xa7a0\n")
        self.model = self.root / "existing model.gguf"
        self.model.write_bytes(b"GGUF" + b"fixture" * 200)
        self.records = []
        self.add("gpu-matrix", b"a7a0\ttested\n46a0\tuntested\n", "gpu.tsv")
        self.add("requirements", b"cpu_flags\tavx2\nglibc\t2.17\n", "requirements.tsv")
        self.add_archive("engine", {"bin/xenolith": FAKE_ENGINE})
        for client in ("xenopi", "codexe"):
            files = {f"bin/{client}": b"#!/bin/sh\necho fixture-client-1\n"}
            if client == 'codexe':
                for helper in ('bin/codex-code-mode-host', 'codex-path/rg',
                               'codex-resources/bwrap'):
                    files[helper] = b"#!/bin/sh\necho fixture-helper\n"
                files['bin/codex-exec-server'] = b"#!/bin/sh\necho 'codex-exec-server fixture help'\n"
            self.add_archive(client, files)
        self.add("model", self.model.read_bytes(), "model.gguf")
        self.add("projector", b"GGUFprojector", "projector.gguf")
        self.manifest()
        self.env = os.environ.copy()
        for key in list(self.env):
            if key.startswith(("XDG_", "XENOLITH_", "XE_")):
                self.env.pop(key)
        self.env.update(HOME=str(self.home), XENOLITH_INSTALL_TEST_ROOT=str(self.probe),
                        FIXTURE_CALLS=str(self.root / 'calls.jsonl'))
        # Expose only fixture dependencies, never installed user components (even
        # those in system directories). Tests prepend their own reuse candidates.
        commands = self.root / 'fixture commands'
        commands.mkdir()
        for name in ('sh', 'curl', 'tar', 'gzip', 'sha256sum', 'awk', 'sed', 'wc',
                     'df', 'uname', 'mktemp', 'readlink', 'timeout', 'tr', 'grep',
                     'dd', 'getconf', 'cmp', 'id', 'env', 'rm', 'rmdir', 'mkdir',
                     'cp', 'mv', 'ln', 'cat', 'dirname', 'basename', 'touch'):
            executable = shutil.which(name, path=os.defpath)
            if executable is None:
                raise RuntimeError(f'Required fixture utility missing: {name}')
            (commands / name).symlink_to(executable)
        (commands / 'python3').symlink_to(sys.executable)
        # Keep the selected runtime, including actions/setup-node's Node 26.
        node = shutil.which('node')
        if node:
            (commands / 'node').symlink_to(node)
        self.env['PATH'] = str(commands)

    def calls(self):
        trace = self.root / 'calls.jsonl'
        return [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, text):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def add(self, component, data, filename):
        (self.source / filename).write_bytes(data)
        self.records.append([component, "test-1", filename, str(len(data)), hashlib.sha256(data).hexdigest()])

    def add_archive(self, component, files):
        if component == 'codexe':
            files = dict(files)
            if 'bin/codexe' in files: files['bin/codex'] = files.pop('bin/codexe')
            files['codex-package.json'] = b'{"xenolith":{"nativeConfig":1}}'
            # Explicit activation fixture. The actual upstream-derived installer
            # has its own lifecycle/checksum/ownership tests and real bundle smoke.
            files['install/install.sh'] = b'''#!/bin/sh
set -eu
shift
while [ "$#" -gt 0 ]; do
 key=$1; value=$2; shift 2
 case "$key" in --package-dir) package=$value;; --prefix) prefix=$value;; --bin-dir) bindir=$value;; *) exit 2;; esac
done
root=$prefix/lib/codexe/standalone
mkdir -p "$root/releases/fixture" "$bindir"
cp -R "$package/." "$root/releases/fixture/"
printf '%s\\n' codexe-local-v1 > "$root/.codexe-local"
ln -sfn "$root/releases/fixture" "$root/current"
ln -sfn "$root/current/bin/codex" "$bindir/codexe"
'''
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as out:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.mode = 0o755
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))
        self.add(component, buf.getvalue(), component + ".tar.gz")

    def manifest(self):
        (self.source / "manifest.tsv").write_text("# xenolith-distribution-v1\n" + "".join("\t".join(r) + "\n" for r in self.records))

    def install(self, *args, source=None, stdin=None):
        argv = ["sh", str(SCRIPT), "--source", str(source or self.source), "--yes", *args]
        if stdin is not None:
            argv = ["sh", "-s", "--", *argv[2:]]
        return subprocess.run(argv, input=stdin, env=self.env, text=True, capture_output=True,
                              start_new_session=True, timeout=30)

    @property
    def config(self):
        return self.home / ".config/xenolith/config.json"

    def prepared_script(self):
        digest = hashlib.sha256((self.source / 'manifest.tsv').read_bytes()).hexdigest()
        return SCRIPT.read_text().replace("DEFAULT_SOURCE=''", 'DEFAULT_SOURCE=' + shlex.quote(str(self.source)), 1).replace(
            "DEFAULT_MANIFEST_SHA256=''", 'DEFAULT_MANIFEST_SHA256=' + shlex.quote(digest), 1)

    def test_ambient_components_are_not_exposed_to_fixture(self):
        poison = self.root / 'ambient commands'
        poison.mkdir()
        markers = self.root / 'ambient calls'
        markers.mkdir()
        for name in ('xenolith', 'xenopi', 'codexe'):
            command = poison / name
            command.write_text('#!/bin/sh\nprintf called > ' +
                               shlex.quote(str(markers / name)) + '\nexit 93\n')
            command.chmod(0o755)
        fixture = InstallerTest()
        with mock.patch.dict(os.environ, PATH=str(poison) + os.pathsep + os.environ['PATH']):
            fixture.setUp()
        self.addCleanup(fixture.tearDown)
        result = fixture.install('--model', str(fixture.model), '--interface', 'all', '--skip-verify')
        self.assertEqual(list(markers.iterdir()), [], 'ambient commands executed: ' + result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ('xenolith', 'xenopi', 'codexe'):
            self.assertTrue((fixture.home / '.local/bin' / name).is_symlink())

    def test_prepared_installer_uses_defaults_through_pipe(self):
        script = self.prepared_script()
        args = ['sh', '-s', '--', '--yes', '--model', str(self.model), '--skip-verify']
        first = subprocess.run(args, input=script, env=self.env, text=True,
                               capture_output=True, start_new_session=True, timeout=30)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        before = self.config.read_bytes()
        with (self.source / 'manifest.tsv').open('a') as out:
            out.write('# fixture changed after installer preparation\n')
        refused = subprocess.run(args, input=script, env=self.env, text=True,
                                 capture_output=True, start_new_session=True, timeout=30)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('Manifest checksum mismatch', refused.stderr)
        self.assertEqual(self.config.read_bytes(), before)

    def test_prepared_source_override_and_explicit_pin_precedence(self):
        script = self.prepared_script()
        # Same component records, different manifest identity and source path.
        mirror = self.root / 'mirror'
        import shutil
        shutil.copytree(self.source, mirror)
        with (mirror / 'manifest.tsv').open('a') as out:
            out.write('# independent candidate\n')
        args = ['sh', '-s', '--', '--yes', '--model', str(self.model), '--skip-verify']
        for flags, env in [(['--source', str(mirror)], self.env),
                           ([], {**self.env, 'XENOLITH_DISTRIBUTION_SOURCE': str(mirror)})]:
            result = subprocess.run(args + flags, input=script, env=env, text=True,
                                    capture_output=True, start_new_session=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        pinned = subprocess.run(args + ['--source', str(mirror), '--manifest-sha256', '0' * 64],
                                input=script, env=self.env, text=True, capture_output=True,
                                start_new_session=True, timeout=30)
        self.assertNotEqual(pinned.returncode, 0)
        self.assertIn('Manifest checksum mismatch', pinned.stderr)

    def test_clean_existing_model_and_rerun_preserves_config(self):
        result = self.install("--model", str(self.model), "--skip-verify", "--interface", "xenopi")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        before = self.config.read_bytes()
        self.assertEqual(json.loads(before)["model"], str(self.model))
        self.assertFalse((self.home / ".local/bin/codexe").exists())
        same = self.install("--model", str(self.model), "--skip-verify", "--interface", "xenopi")
        self.assertEqual(same.returncode, 0, same.stdout + same.stderr)
        self.assertEqual(self.config.read_bytes(), before)
        again = self.install("--skip-verify", "--interface", "all")
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertTrue((self.home / ".local/bin/codexe").is_symlink())
        self.assertNotIn("NOT VERIFIED", again.stdout)
        self.assertEqual(len(list((self.home / ".local/share/xenolith/install/releases").iterdir())), 2)
        self.assertFalse((self.home / ".profile").exists())

    def test_pipe_input_and_noninteractive_model_required(self):
        result = self.install("--model", str(self.model), "--skip-verify", stdin=SCRIPT.read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.config.unlink()
        missing = self.install("--skip-verify")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("supply the corresponding option", missing.stderr)

    def run_pipe_tty(self, options, prompts, expected_exit=0):
        # Keep the supervising shell alive while the installer handles Ctrl+C.
        # A caught signal is reset for children, unlike an ignored signal.
        command = "trap ':' INT; cat -- " + shlex.quote(str(SCRIPT)) + ' | sh -s -- --source ' + shlex.quote(str(self.source)) + ' ' + shlex.join(options)
        pid, master = pty.fork()
        if pid == 0:
            os.execvpe('sh', ['sh', '-c', command], self.env)
        transcript = b''
        answered = 0
        status = None
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        data = b''  # PTY EIO signals that the child closed it.
                    transcript += data
                    while answered < len(prompts) and prompts[answered][0].encode() in transcript:
                        reply = prompts[answered][1]
                        os.write(master, reply if isinstance(reply, bytes) else (reply + '\n').encode())
                        answered += 1
                waited, child_status = os.waitpid(pid, os.WNOHANG)
                if waited:
                    status = child_status
                    # Drain final diagnostics written immediately before exit.
                    while select.select([master], [], [], 0)[0]:
                        try:
                            data = os.read(master, 65536)
                        except OSError:
                            break
                        if not data:
                            break
                        transcript += data
                    break
            output = transcript.decode(errors='replace')
            self.assertIsNotNone(status, 'PTY installer timed out:\n' + output)
            self.assertEqual(os.waitstatus_to_exitcode(status), expected_exit, output)
            self.assertEqual(answered, len(prompts), output)
            return output
        finally:
            if status is None:
                # pty.fork created this private session/process group.
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.waitpid(pid, 0)
            os.close(master)

    def test_ctrl_c_cleans_stage_and_lock_even_when_repeated(self):
        commands = self.root / 'interrupt commands'
        commands.mkdir()
        engine = commands / 'fixture-engine'
        delayed_check = """
 if os.environ.get('FIXTURE_INTERRUPT'):
  import signal, time
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  with open('/dev/tty', 'w') as tty: print('FOREGROUND READY', file=tty, flush=True)
  time.sleep(0.6)
  sys.exit(0)
"""
        engine.write_text(FAKE_ENGINE.decode().replace("elif a==['check-device']:",
                                                       "elif a==['check-device']:" + delayed_check))
        engine.chmod(0o755)
        slow_rm = commands / 'rm'
        slow_rm.write_text("""#!/usr/bin/env python3
import os, sys, time
if os.environ.get('FIXTURE_SLOW_CLEANUP'):
 with open('/dev/tty', 'w') as tty: print('CLEANUP READY', file=tty, flush=True)
 time.sleep(0.6)
os.execv('/bin/rm', ['rm', *sys.argv[1:]])
""")
        slow_rm.chmod(0o755)
        self.env['PATH'] = str(commands) + ':' + self.env['PATH']
        prefix = self.home / '.local/share/xenolith/install'
        options = ['--yes', '--model', str(self.model), '--engine-bin', str(engine),
                   '--no-modify-path']
        for repeated in (False, True):
            with self.subTest(repeated=repeated):
                self.env.update(FIXTURE_INTERRUPT='1', FIXTURE_SLOW_CLEANUP='1')
                prompts = [('FOREGROUND READY', b'\x03')]
                if repeated:
                    prompts.append(('CLEANUP READY', b'\x03'))
                self.run_pipe_tty(options, prompts, expected_exit=130)
                self.assertFalse((prefix / '.install-lock').exists())
                self.assertEqual(list(prefix.glob('.stage.*')), [])
                self.env.pop('FIXTURE_INTERRUPT')
                self.env.pop('FIXTURE_SLOW_CLEANUP')
                retry = self.install(*options, '--skip-verify')
                self.assertEqual(retry.returncode, 0, retry.stdout + retry.stderr)
                self.assertFalse((prefix / '.install-lock').exists())
                self.assertEqual(list(prefix.glob('.stage.*')), [])

    def test_pipe_source_prompts_use_controlling_tty(self):
        output = self.run_pipe_tty(['--skip-verify', '--no-modify-path'], [
            ('Interface: chat, xenopi, codexe, all [chat]: ', 'chat'),
            ('Continue installation? (yes/no) [no]: ', 'yes'),
            ('Existing GGUF absolute path, or download [download]: ', str(self.model)),
            ('Enable images (downloads the matching 1.19 GB projector)? yes/no [no]: ', 'no'),
        ])
        self.assertEqual(json.loads(self.config.read_text())['model'], str(self.model))
        self.assertNotIn('NOT VERIFIED', output)
        self.assertFalse(any(call['args'][:1] == ['chat'] for call in self.calls()))
        self.assertIn('Commands for this and later sessions:', output)
        self.assertNotIn('Starting chat.', output)
        self.assertFalse((self.home / '.local/bin/xenopi').exists())

    def test_selected_interface_receives_terminal_and_configuration(self):
        marker = self.root / 'launch.json'
        # Real PTY and shell handoff; simulated frontends deliberately read one
        # line from the terminal and record configuration/lock state.
        handoff = r"""
if os.environ.get('FIXTURE_LAUNCH') and sys.stdin.isatty():
 assert all(os.isatty(fd) for fd in (0,1,2))
 assert not (pathlib.Path(os.environ['HOME'])/'.local/share/xenolith/install/.install-lock').exists()
 print('FRONTEND INPUT: ',end='',flush=True)
 line=input()
 pathlib.Path(os.environ['FIXTURE_LAUNCH']).write_text(json.dumps({'argv':sys.argv,'line':line,'config':os.environ.get('XENOLITH_CONFIG')}))
 sys.exit(0)
"""
        engine = FAKE_ENGINE.decode().replace("def value(flag, default=None):", handoff+"\ndef value(flag, default=None):")
        client_trace = """
with pathlib.Path(os.environ['FIXTURE_CALLS']).open('a') as out:
 out.write(json.dumps({'command':pathlib.Path(sys.argv[0]).name,'args':sys.argv[1:]})+'\\n')
"""
        client = ('#!/usr/bin/env python3\nimport os,sys,pathlib,json\n'+client_trace+handoff+"\nprint('fixture-client')\n").encode()
        self.records = [r for r in self.records if r[0] not in ('engine','xenopi','codexe')]
        self.add_archive('engine', {'bin/xenolith': engine.encode()})
        self.add_archive('xenopi', {'bin/xenopi': client})
        files = {'bin/codexe': client}
        for helper in ('bin/codex-code-mode-host','codex-path/rg','codex-resources/bwrap',
                       'bin/codex-exec-server'):
            files[helper] = b'#!/bin/sh\nexit 0\n'
        self.add_archive('codexe',files); self.manifest()
        self.env['FIXTURE_LAUNCH'] = str(marker)
        config = self.home / 'custom config.json'
        for selected in ('chat','xenopi','codexe','all'):
            with self.subTest(interface=selected):
                marker.unlink(missing_ok=True)
                (self.root / 'calls.jsonl').unlink(missing_ok=True)
                prompts = []
                if selected == 'all': prompts.append(('Start now: chat, xenopi, codexe [chat]: ', 'codexe'))
                prompts.append(('FRONTEND INPUT: ', 'hello from terminal'))
                output = self.run_pipe_tty(['--yes','--model',str(self.model),'--config',str(config),
                                           '--interface',selected,'--no-modify-path'],prompts)
                data = json.loads(marker.read_text())
                expected = 'codexe' if selected == 'all' else selected
                self.assertEqual(Path(data['argv'][0]).name, 'xenolith' if expected == 'chat' else expected)
                self.assertEqual(data['line'], 'hello from terminal')
                if expected == 'chat': self.assertEqual(data['argv'][1:], ['chat','--config',str(config)])
                else: self.assertEqual(data['config'],str(config))
                calls = self.calls()
                chats = [call for call in calls if call['command'] == 'xenolith' and call['args'][:1] == ['chat']]
                frontends = [call for call in calls if call['command'] in ('xenopi','codexe') and not call['args']]
                self.assertEqual(len(chats), 1 if expected == 'chat' else 0, calls)
                self.assertEqual(frontends, [] if expected == 'chat' else [{'command':expected,'args':[]}], calls)
                self.assertNotIn('Real engine response verified.',output)
                self.assertFalse(list((self.home / '.local/share/xenolith/install').glob('last-verification.*')))


    def test_noninteractive_default_prepares_without_inference(self):
        result = self.install('--model', str(self.model))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.config.exists())
        self.assertFalse(any(call['args'][:1] == ['chat'] for call in self.calls()), self.calls())
        self.assertNotIn('Starting ', result.stdout)
        self.assertFalse(list((self.home / '.local/share/xenolith/install').glob('last-verification.*')))

    def test_path_setup_is_opt_in_and_idempotent_with_quoted_directory(self):
        self.env.update(SHELL='/bin/bash')
        profile = self.home / '.bashrc'
        original = '# existing user settings\nexport KEEP_ME=yes\n'
        profile.write_text(original)
        first = self.install('--model', str(self.model), '--skip-verify')
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(profile.read_text(), original)  # --yes is not PATH consent.
        bindir = self.home / "User's commands"
        for attempt in range(2):
            result = self.install('--skip-verify', '--bin-dir', str(bindir), '--add-to-path')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            text = profile.read_text()
            self.assertTrue(text.startswith(original))
            self.assertEqual(text.count('# >>> xenolith PATH >>>'), 1)
            if attempt == 0: before = text
            else: self.assertEqual(text, before)
        self.assertEqual(profile.with_name('.bashrc.xenolith.bak').read_text(), original)
        shell = subprocess.run(['/bin/bash', '-c', '. "$HOME/.bashrc"; command -v xenolith'],
                               env=self.env, capture_output=True, text=True, check=True)
        self.assertEqual(shell.stdout.strip(), str(bindir / 'xenolith'))

    def test_explicit_path_setup_even_if_directory_is_in_current_path(self):
        bindir = self.home / '.local/bin'
        self.env.update(SHELL='/bin/bash', PATH=str(bindir) + ':' + self.env['PATH'])
        result = self.install('--model', str(self.model), '--skip-verify', '--add-to-path')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.home / '.bashrc').is_file())

    def test_zdotdir_and_symlink_profile_preservation(self):
        self.env.update(SHELL='/bin/zsh', ZDOTDIR=str(self.home / 'zsh config'))
        result = self.install('--model', str(self.model), '--skip-verify', '--add-to-path')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        profile = Path(self.env['ZDOTDIR']) / '.zshrc'
        self.assertIn('# >>> xenolith PATH >>>', profile.read_text())
        target = self.home / 'managed-dotfile'
        target.write_text('# user-managed symlink target\n')
        profile.unlink(); profile.symlink_to(target)
        again = self.install('--skip-verify', '--add-to-path')
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertTrue(profile.is_symlink())
        self.assertEqual(target.read_text(), '# user-managed symlink target\n')

    def test_runtime_install_requires_specific_consent_and_rechecks(self):
        self.write('probe/etc/os-release', 'ID=arch\nPRETTY_NAME="Fixture Arch"\n')
        ready = self.root / 'runtime-ready'
        calls = self.root / 'package-calls'
        commands = self.root / 'commands'; commands.mkdir()
        pm = commands / 'pacman'
        pm.write_text(f'''#!{sys.executable}
import json,sys,pathlib
with pathlib.Path({str(calls)!r}).open('a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='-Q': sys.exit(1)
assert sys.argv[1:]==['-S','--needed','--noconfirm','level-zero-loader','intel-compute-runtime']
pathlib.Path({str(ready)!r}).touch()
''')
        pm.chmod(0o755)
        sudo = commands / 'sudo'
        sudo.write_text('#!/bin/sh\n[ "$1" != -- ] || shift\nexec "$@"\n')
        sudo.chmod(0o755)
        self.env.update(PATH=str(commands)+':'+self.env['PATH'], FIXTURE_RUNTIME_MISSING='1',
                        FIXTURE_RUNTIME_READY=str(ready))
        refused = self.install('--model', str(self.model), '--skip-verify')
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('--install-runtime --yes', refused.stderr)
        self.assertFalse(ready.exists())
        self.assertFalse(self.config.exists())
        accepted = self.install('--model', str(self.model), '--skip-verify', '--install-runtime')
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertTrue(ready.exists())
        self.assertTrue(self.config.exists())

    def test_ubuntu_refreshes_package_index_only_after_consent(self):
        self.write('probe/etc/os-release', 'ID=ubuntu\nVERSION_ID="24.04"\n')
        ready = self.root / 'runtime-ready'
        calls = self.root / 'apt-calls'
        commands = self.root / 'commands'; commands.mkdir()
        scripts = {
            'dpkg-query': '#!/bin/sh\nexit 1\n',
            'sudo': '#!/bin/sh\n[ "$1" != -- ] || shift\nexec "$@"\n',
            'apt-get': f'''#!{sys.executable}
import json,sys,pathlib
with pathlib.Path({str(calls)!r}).open('a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='install': pathlib.Path({str(ready)!r}).touch()
''',
        }
        for name, script in scripts.items():
            command = commands / name; command.write_text(script); command.chmod(0o755)
        self.env.update(PATH=str(commands)+':'+self.env['PATH'], FIXTURE_RUNTIME_MISSING='1',
                        FIXTURE_RUNTIME_READY=str(ready))
        refused = self.install('--model', str(self.model), '--skip-verify')
        self.assertNotEqual(refused.returncode, 0)
        self.assertFalse(calls.exists())
        accepted = self.install('--model', str(self.model), '--skip-verify', '--install-runtime')
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertEqual([json.loads(line) for line in calls.read_text().splitlines()], [
            ['update'], ['install', '--no-install-recommends', '-y', 'libze1', 'libze-intel-gpu1']])
        self.assertTrue(self.config.exists())

    def xenopi_runtime_fixture(self, *, repair=True):
        self.write('probe/etc/os-release', 'ID=ubuntu\nVERSION_ID="24.04"\n')
        ready = self.root / 'xenopi-runtime-ready'
        calls = self.root / 'xenopi-apt-calls'
        loader = f'''#!{sys.executable}
import os,pathlib,sys
if not pathlib.Path({str(ready)!r}).exists():
 print(os.environ.get('FIXTURE_XENOPI_ERROR', 'node: error while loading shared libraries: libatomic.so.1: cannot open shared object file: No such file or directory'), file=sys.stderr)
 sys.exit(int(os.environ.get('FIXTURE_XENOPI_STATUS', '127')))
print('fixture-xenopi-1')
'''
        self.records = [r for r in self.records if r[0] != 'xenopi']
        self.add_archive('xenopi', {'bin/xenopi': loader.encode()}); self.manifest()
        commands = self.root / 'runtime commands'; commands.mkdir()
        scripts = {
            'dpkg-query': '#!/bin/sh\nexit 1\n',
            'sudo': '#!/bin/sh\n[ "$1" != -- ] || shift\nexec "$@"\n',
            'apt-get': f'''#!{sys.executable}
import json,sys,pathlib
with pathlib.Path({str(calls)!r}).open('a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')
assert sys.argv[1:] in [['update'], ['install','--no-install-recommends','-y','libatomic1']]
if sys.argv[1]=='install' and {repair!r}: pathlib.Path({str(ready)!r}).touch()
''',
        }
        for name, script in scripts.items():
            command = commands / name; command.write_text(script); command.chmod(0o755)
        self.env['PATH'] = str(commands)+':'+self.env['PATH']
        return ready, calls, loader

    def test_xenopi_libatomic_requires_consent_and_checks_before_activation(self):
        ready, calls, _ = self.xenopi_runtime_fixture()
        for interface in ('chat', 'codexe'):
            result = self.install('--model', str(self.model), '--interface', interface,
                                  '--skip-verify', '--install-runtime')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(calls.exists())
        config = self.config.read_bytes()
        refused = self.install('--interface', 'xenopi', '--skip-verify')
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('--install-runtime --yes', refused.stderr)
        self.assertFalse(calls.exists())
        self.assertFalse((self.home / '.local/bin/xenopi').is_symlink())
        self.assertEqual(self.config.read_bytes(), config)
        accepted = self.install('--interface', 'all', '--skip-verify', '--install-runtime')
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertTrue(ready.exists())
        expected = [['update'], ['install', '--no-install-recommends', '-y', 'libatomic1']]
        self.assertEqual([json.loads(line) for line in calls.read_text().splitlines()], expected)
        again = self.install('--interface', 'xenopi', '--skip-verify')
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertEqual([json.loads(line) for line in calls.read_text().splitlines()], expected)
        self.assertEqual(self.config.read_bytes(), config)

    def test_xenopi_runtime_repair_must_succeed_before_activation(self):
        _, calls, _ = self.xenopi_runtime_fixture(repair=False)
        result = self.install('--model', str(self.model), '--interface', 'xenopi',
                              '--skip-verify', '--install-runtime')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('XenoPi runtime remains incomplete', result.stderr)
        self.assertTrue(calls.exists())
        self.assertFalse((self.home / '.local/bin/xenopi').is_symlink())
        self.assertFalse(self.config.exists())

    def test_xenopi_other_startup_errors_do_not_install_packages(self):
        _, calls, _ = self.xenopi_runtime_fixture()
        for diagnostic, status in (
            ('fixture: unsupported Node runtime', '1'),
            ('node: error while loading shared libraries: libstdc++.so.6: missing', '127'),
            ('fixture: libatomic.so.1 mentioned in application error', '1'),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.env.update(FIXTURE_XENOPI_ERROR=diagnostic, FIXTURE_XENOPI_STATUS=status)
                result = self.install('--model', str(self.model), '--interface', 'xenopi',
                                      '--skip-verify', '--install-runtime')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse(calls.exists())
                self.assertFalse((self.home / '.local/bin/xenopi').is_symlink())

    def test_reused_xenopi_node_can_repair_libatomic_before_capability_check(self):
        ready, calls, loader = self.xenopi_runtime_fixture()
        actual_node = shutil.which('node', path=self.env['PATH'])
        commands, app, _ = self.existing_components()
        node = commands / 'node'
        node.write_text(loader.replace("print('fixture-xenopi-1')",
                        f'os.execv({actual_node!r}, [{actual_node!r}, *sys.argv[1:]])'))
        node.chmod(0o755)
        before = (app / 'package.json').read_bytes()
        link = (commands / 'xenopi').readlink()
        (self.source / 'xenopi.tar.gz').write_bytes(b'must not download reused client')
        refused = self.install('--model', str(self.model), '--interface', 'xenopi', '--skip-verify')
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('--install-runtime --yes', refused.stderr)
        self.assertFalse(calls.exists())
        accepted = self.install('--model', str(self.model), '--interface', 'xenopi',
                                '--skip-verify', '--install-runtime')
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertTrue(ready.exists())
        self.assertEqual((app / 'package.json').read_bytes(), before)
        self.assertEqual((commands / 'xenopi').readlink(), link)

    def test_arch_xenopi_repairs_only_libatomic(self):
        ready, calls, _ = self.xenopi_runtime_fixture()
        self.write('probe/etc/os-release', 'ID=arch\n')
        pm = self.root / 'runtime commands/pacman'
        pm.write_text(f'''#!{sys.executable}
import json,sys,pathlib
if sys.argv[1:] == ['-Q', 'libatomic']: sys.exit(1)
assert sys.argv[1:]==['-S','--needed','--noconfirm','libatomic']
pathlib.Path({str(calls)!r}).write_text(json.dumps(sys.argv[1:]))
pathlib.Path({str(ready)!r}).touch()
''')
        pm.chmod(0o755)
        result = self.install('--model', str(self.model), '--interface', 'xenopi',
                              '--skip-verify', '--install-runtime')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(calls.read_text()), ['-S', '--needed', '--noconfirm', 'libatomic'])

    def test_codexe_helper_runtime_is_checked_and_repaired_only_when_selected(self):
        self.write('probe/etc/os-release', 'ID=ubuntu\nVERSION_ID="24.04"\n')
        ready = self.root / 'client-runtime-ready'
        calls = self.root / 'client-apt-calls'
        self.records = [r for r in self.records if r[0] != 'codexe']
        files = {name: b'#!/bin/sh\nexit 0\n' for name in (
            'bin/codexe', 'bin/codex-code-mode-host', 'codex-path/rg', 'codex-resources/bwrap')}
        files['bin/codex-exec-server'] = (
            f'#!{sys.executable}\nimport pathlib,sys\nsys.exit(0 if pathlib.Path({str(ready)!r}).exists() else 1)\n'.encode())
        self.add_archive('codexe', files); self.manifest()
        commands = self.root / 'commands'; commands.mkdir()
        scripts = {
            'dpkg-query': '#!/bin/sh\nexit 1\n',
            'sudo': '#!/bin/sh\n[ "$1" != -- ] || shift\nexec "$@"\n',
            'apt-get': f'''#!{sys.executable}
import json,sys,pathlib
with pathlib.Path({str(calls)!r}).open('a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='install': pathlib.Path({str(ready)!r}).touch()
''',
        }
        for name, script in scripts.items():
            command = commands / name; command.write_text(script); command.chmod(0o755)
        self.env['PATH'] = str(commands)+':'+self.env['PATH']
        chat = self.install('--model', str(self.model), '--skip-verify', '--install-runtime')
        self.assertEqual(chat.returncode, 0, chat.stdout + chat.stderr)
        self.assertFalse(calls.exists())
        refused = self.install('--skip-verify', '--interface', 'codexe')
        self.assertNotEqual(refused.returncode, 0)
        self.assertFalse(calls.exists())
        accepted = self.install('--skip-verify', '--interface', 'codexe', '--install-runtime')
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertEqual([json.loads(line) for line in calls.read_text().splitlines()], [
            ['update'], ['install', '--no-install-recommends', '-y', 'libssl3t64', 'libcap2', 'libtinfo6']])

    def test_codexe_missing_helpers_are_rejected_before_remediation_or_activation(self):
        self.write('probe/etc/os-release', 'ID=ubuntu\nVERSION_ID="24.04"\n')
        commands = self.root / 'commands'
        commands.mkdir()
        calls = self.root / 'unexpected-package-manager-call'
        for name in ('apt-get', 'sudo', 'dpkg-query'):
            executable = commands / name
            executable.write_text(f'#!/bin/sh\ntouch "{calls}"\nexit 1\n')
            executable.chmod(0o755)
        self.env['PATH'] = str(commands) + ':' + self.env['PATH']
        helpers = ('bin/codex-code-mode-host', 'bin/codex-exec-server',
                   'codex-path/rg', 'codex-resources/bwrap')
        for missing in helpers:
            with self.subTest(missing=missing):
                self.records = [r for r in self.records if r[0] != 'codexe']
                files = {name: b'#!/bin/sh\nexit 0\n' for name in ('bin/codexe', *helpers)
                         if name != missing}
                self.add_archive('codexe', files)
                self.manifest()
                result = self.install('--model', str(self.model), '--skip-verify',
                                      '--interface', 'codexe', '--install-runtime')
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('missing required helper: ' + missing, result.stderr)
                self.assertFalse(calls.exists(), 'A missing bundled helper cannot be repaired by apt')
                self.assertFalse((self.home / '.local/bin/codexe').exists())
                standalone = self.home / '.local/lib/codexe/standalone'
                self.assertFalse((standalone / '.codexe-local').exists())

    def test_pci_gpu_is_detected_before_drm_driver_is_ready(self):
        import shutil
        shutil.rmtree(self.probe / 'sys/class/drm')
        self.write('probe/sys/bus/pci/devices/0000:00:02.0/vendor', '0x8086\n')
        self.write('probe/sys/bus/pci/devices/0000:00:02.0/device', '0xa7a0\n')
        self.write('probe/sys/bus/pci/devices/0000:00:02.0/class', '0x030000\n')
        result = self.install('--check-only')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('8086:a7a0: tested', result.stdout)

    def test_client_glibc_requirement_applies_only_when_selected(self):
        self.records = [r for r in self.records if r[0] != 'requirements']
        self.add('requirements', b'cpu_flags\tavx2\nglibc\t2.17\nxenopi_glibc\t99.0\n', 'requirements.tsv')
        self.manifest()
        chat = self.install('--model', str(self.model), '--skip-verify', '--interface', 'chat')
        self.assertEqual(chat.returncode, 0, chat.stdout + chat.stderr)
        client = self.install('--skip-verify', '--interface', 'xenopi')
        self.assertNotEqual(client.returncode, 0)
        self.assertIn('xenopi package requires glibc 99.0', client.stderr)
        self.assertFalse((self.home / '.local/bin/xenopi').exists())

    def test_check_only_checks_selected_client_glibc_without_installing(self):
        self.records = [r for r in self.records if r[0] != 'requirements']
        self.add('requirements', b'cpu_flags\tavx2\nglibc\t2.17\nxenopi_glibc\t99.0\ncodexe_glibc\t99.0\n', 'requirements.tsv')
        self.manifest()
        chat = self.install('--check-only', '--interface', 'chat')
        self.assertEqual(chat.returncode, 0, chat.stdout + chat.stderr)
        for selected in ('xenopi', 'codexe', 'all'):
            with self.subTest(interface=selected):
                result = self.install('--check-only', '--interface', selected)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('requires glibc 99.0', result.stderr)
        self.assertFalse(self.config.exists())
        self.assertEqual(list((self.home / '.local/bin').iterdir()), [])
        self.assertEqual(list((self.home / '.local/share/xenolith/install/releases').iterdir()), [])
        self.assertFalse(self.calls())

    def test_relative_xdg_paths_use_home_fallbacks(self):
        self.env.update(XDG_CONFIG_HOME='relative-config', XDG_DATA_HOME='relative-data')
        result = self.install('--download-model', '--skip-verify')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.config.is_file())
        expected_model = self.home / '.local/share/xenolith/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'
        self.assertEqual(json.loads(self.config.read_text())['model'], str(expected_model))
        self.assertEqual(expected_model.read_bytes(), self.model.read_bytes())
        self.assertTrue((self.home / '.local/share/xenolith/install/releases').is_dir())
        command = next(line for line in result.stdout.splitlines() if line.startswith('  '))
        self.assertNotIn('--config', command)

    def test_absolute_xdg_paths_and_explicit_config_keep_precedence(self):
        config_home = self.home / 'xdg config'
        data_home = self.home / 'xdg data'
        self.env.update(XDG_CONFIG_HOME=str(config_home), XDG_DATA_HOME=str(data_home))
        result = self.install('--model', str(self.model), '--skip-verify')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        config = config_home / 'xenolith/config.json'
        self.assertTrue(config.is_file())
        self.assertTrue((data_home / 'xenolith/install/releases').is_dir())
        before = config.read_bytes()
        selected = self.home / 'selected.json'
        self.env['XENOLITH_CONFIG'] = str(self.home / 'unused.json')
        result = self.install('--config', str(selected), '--model', str(self.model), '--skip-verify')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(selected.is_file())
        self.assertEqual(config.read_bytes(), before)
        self.assertFalse(Path(self.env['XENOLITH_CONFIG']).exists())

    def existing_components(self):
        commands = self.root / 'existing commands'; commands.mkdir()
        engine = commands / 'xenolith'; engine.write_bytes(FAKE_ENGINE); engine.chmod(0o755)
        app = self.root / 'existing npm app'
        metadata = {'name':'xenopi','version':'0.2.0','bin':{'xenopi':'dist/src/bin/xenopi.js'},
                    'piConfig':{'name':'xenopi','configDir':'.xenopi'},'xenolith':{'nativeConfig':1}}
        app.mkdir(); (app / 'package.json').write_text(json.dumps(metadata))
        for rel in ('bin/xenopi.js','extensions/provider.js','wire/discovery.js','wire/client.js'):
            file = app / 'dist/src' / rel; file.parent.mkdir(parents=True,exist_ok=True)
            file.write_text('#!/usr/bin/env node\nconsole.log("0.2.0");\n'); file.chmod(0o755)
        (commands / 'xenopi').symlink_to(app / 'dist/src/bin/xenopi.js')
        package = self.root / 'existing codexe package'; package.mkdir()
        with tarfile.open(self.source / 'codexe.tar.gz') as archive:
            archive.extractall(package,filter='data')
        (package / 'codex-package.json').write_text(json.dumps({'xenolith':{'nativeConfig':1}}, indent=2))
        (commands / 'codexe').symlink_to(package / 'bin/codex')
        self.env['PATH'] = str(commands)+':'+self.env['PATH']
        return commands, app, package

    def test_reuses_manual_and_npm_installs_without_replacing_or_downloading(self):
        commands, app, package = self.existing_components()
        original_engine = (commands / 'xenolith').read_bytes()
        original_links = [(commands / name).readlink() for name in ('xenopi','codexe')]
        # The downloaded artifacts' ABI/CPU minimum must not reject a compatible
        # existing build. Their archives are deliberately corrupt to catch fetches.
        self.records = [r for r in self.records if r[0] != 'requirements']
        self.add('requirements', b'cpu_flags\tmissing_package_flag\nglibc\t99.0\nxenopi_glibc\t99.0\ncodexe_glibc\t99.0\n', 'requirements.tsv')
        self.manifest()
        for name in ('engine','xenopi','codexe'): (self.source / (name+'.tar.gz')).write_bytes(b'not downloaded')
        result = self.install('--model',str(self.model),'--interface','all','--skip-verify')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        config = self.config.read_bytes()
        again = self.install('--interface','all','--skip-verify')
        self.assertEqual(again.returncode,0,again.stdout+again.stderr)
        self.assertEqual(self.config.read_bytes(),config)
        self.assertEqual((commands / 'xenolith').read_bytes(),original_engine)
        self.assertEqual([(commands / name).readlink() for name in ('xenopi','codexe')],original_links)
        self.assertEqual(list((self.home / '.local/share/xenolith/install/releases').iterdir()),[])
        self.assertEqual(list((self.home / '.local/share/xenolith/install/downloads').glob('*.tar.gz')),[])
        self.assertFalse((self.home / '.local/bin/xenolith').exists())
        self.assertIn('XENOLITH_BIN=',result.stdout)

    def test_existing_client_without_capability_is_rejected_unchanged(self):
        commands, app, package = self.existing_components()
        for selected, path in (('xenopi',app/'package.json'), ('codexe',package/'codex-package.json')):
            with self.subTest(client=selected):
                data=json.loads(path.read_text()); data.pop('xenolith'); path.write_text(json.dumps(data))
                before=path.read_bytes()
                result=self.install('--model',str(self.model),'--interface',selected,'--skip-verify')
                self.assertNotEqual(result.returncode,0)
                self.assertIn('left untouched',result.stderr)
                self.assertEqual(path.read_bytes(),before)
                self.assertFalse(self.config.exists())
                data['xenolith']={'nativeConfig':1}; path.write_text(json.dumps(data))

    def test_reused_npm_client_preserves_original_node_resolution(self):
        commands, app, package = self.existing_components()
        bindir = self.home / '.local/bin'; bindir.mkdir(parents=True)
        marker = self.root / 'wrong-node-used'
        node = bindir / 'node'
        node.write_text('#!/bin/sh\nprintf poisoned > "$POISON_NODE_MARKER"\nexit 42\n')
        node.chmod(0o755)
        self.env['POISON_NODE_MARKER'] = str(marker)
        self.assertNotIn(str(bindir), self.env['PATH'].split(':'))
        before = subprocess.run([str(commands / 'xenopi'), '--version'], env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(before.returncode, 0, before.stderr)
        result = self.install('--model', str(self.model), '--interface', 'xenopi', '--skip-verify')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        printed = [line.strip() for line in result.stdout.splitlines() if line.startswith('  ')][-1]
        copied = subprocess.run(['sh', '-c', printed], env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(copied.returncode, 0, copied.stderr)
        self.assertFalse(marker.exists(), 'installation and printed command must preserve the existing Node selection')

    def test_printed_default_config_overrides_inherited_config_selection(self):
        client = b'''#!/usr/bin/env python3
import os,sys
if '--version' in sys.argv: print('fixture-client'); sys.exit(0)
assert os.environ.get('XENOLITH_CONFIG') == os.environ['EXPECTED_CONFIG'], repr(os.environ.get('XENOLITH_CONFIG'))
print('selected default configuration')
'''
        self.records = [r for r in self.records if r[0] != 'xenopi']
        self.add_archive('xenopi', {'bin/xenopi': client}); self.manifest()
        for inherited in (str(self.root / 'old-custom-config.json'), ''):
            with self.subTest(inherited=inherited):
                self.env['XENOLITH_CONFIG'] = inherited
                result = self.install('--model', str(self.model), '--config', str(self.config),
                                      '--interface', 'xenopi', '--skip-verify', '--use-packages')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                printed = [line.strip() for line in result.stdout.splitlines() if line.startswith('  ')]
                self.assertIn('--config', printed[0], 'engine command must freeze explicit default selection too')
                copied = subprocess.run(['sh', '-c', printed[-1]],
                                        env={**self.env, 'EXPECTED_CONFIG': str(self.config)},
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(copied.returncode, 0, printed[-1] + '\n' + copied.stderr)

    def test_explicit_engine_outside_path_and_incompatible_engine(self):
        engine=self.root / 'custom engine'; engine.write_bytes(FAKE_ENGINE); engine.chmod(0o755)
        result=self.install('--engine-bin',str(engine),'--model',str(self.model),'--skip-verify')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertFalse((self.home / '.local/bin/xenolith').exists())
        self.assertIn(str(engine),result.stdout)
        engine.write_text('#!/bin/sh\nexit 0\n')
        invalid=self.install('--engine-bin',str(engine),'--skip-verify')
        self.assertNotEqual(invalid.returncode,0)
        self.assertIn('lacks check-device',invalid.stderr)

    def test_reused_engine_missing_loader_requires_consent_then_rechecks_capability(self):
        self.write('probe/etc/os-release', 'ID=arch\nPRETTY_NAME="Fixture Arch"\n')
        ready = self.root / 'loader-ready'
        calls = self.root / 'loader-package-calls'
        commands = self.root / 'loader commands'; commands.mkdir()
        engine = commands / 'xenolith'
        loader_guard = (
            f'if not pathlib.Path({str(ready)!r}).exists():\n'
            ' print("xenolith: error while loading shared libraries: libze_loader.so.1: cannot open shared object file: No such file or directory", file=sys.stderr)\n'
            ' sys.exit(127)\n'
        )
        engine.write_text(FAKE_ENGINE.decode().replace('a=sys.argv[1:]\n', loader_guard + 'a=sys.argv[1:]\n'))
        engine.chmod(0o755)
        original = engine.read_bytes()
        pacman = commands / 'pacman'
        pacman.write_text(f'''#!{sys.executable}
import json,pathlib,sys
with pathlib.Path({str(calls)!r}).open('a') as out: out.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='-Q': sys.exit(1)
assert sys.argv[1:]==['-S','--needed','--noconfirm','level-zero-loader','intel-compute-runtime']
pathlib.Path({str(ready)!r}).touch()
''')
        pacman.chmod(0o755)
        sudo = commands / 'sudo'
        sudo.write_text('#!/bin/sh\n[ "$1" != -- ] || shift\nexec "$@"\n')
        sudo.chmod(0o755)
        self.env['PATH'] = str(commands) + ':' + self.env['PATH']
        (self.source / 'engine.tar.gz').write_bytes(b'reused engine must not be downloaded')
        refused = self.install('--model', str(self.model), '--skip-verify')
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('--install-runtime --yes', refused.stderr)
        self.assertFalse(ready.exists())
        self.assertFalse(self.config.exists())
        accepted = self.install('--model', str(self.model), '--skip-verify', '--install-runtime')
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertTrue(ready.exists())
        self.assertTrue(self.config.exists())
        self.assertEqual(engine.read_bytes(), original)
        self.assertFalse((self.home / '.local/bin/xenolith').exists())
        commands_seen = [json.loads(line) for line in calls.read_text().splitlines()]
        self.assertEqual(sum(args[0] == '-S' for args in commands_seen), 1)
        before = calls.read_bytes()
        engine.write_text('#!/bin/sh\necho "Usage: old xenolith" >&2\nexit 1\n')
        old = self.install('--skip-verify', '--install-runtime')
        self.assertNotEqual(old.returncode, 0)
        self.assertIn('lacks check-device', old.stderr)
        self.assertEqual(calls.read_bytes(), before, 'old CLI must not trigger runtime package operations')

    def test_checksum_failure_no_engine_installed(self):
        (self.source / "engine.tar.gz").write_bytes(b"corrupt")
        result = self.install("--model", str(self.model), "--skip-verify")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checksum/size mismatch", result.stderr)
        self.assertFalse((self.home / ".local/bin/xenolith").exists())

    def test_existing_command_and_config_not_overwritten(self):
        binary = self.home / ".local/bin/xenolith"
        binary.parent.mkdir(parents=True)
        binary.write_text("existing user command")
        result = self.install("--model", str(self.model), "--skip-verify")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(binary.read_text(), "existing user command")

    def test_unsupported_and_untested_gpu(self):
        self.write("probe/sys/class/drm/card0/device/device", "0xffff\n")
        result = self.install("--check-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no enabled Intel", result.stderr)
        self.write("probe/sys/class/drm/card0/device/device", "0x46a0\n")
        check = self.install("--check-only")
        self.assertEqual(check.returncode, 0, check.stderr)
        rejected = self.install("--model", str(self.model), "--skip-verify")
        self.assertNotEqual(rejected.returncode, 0)
        accepted = self.install("--model", str(self.model), "--skip-verify", "--allow-unverified")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_fixture_downloads_model_and_projector_without_inference(self):
        result = self.install("--download-model", "--vision", "--interface", "all")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(self.config.read_text())["mmproj"].endswith("mmproj-F16.gguf"))
        self.assertIn("SIMULATED DEVICE EVIDENCE", result.stdout)
        self.assertFalse(any(call['args'][:1] == ['chat'] for call in self.calls()), self.calls())

    def test_changed_model_does_not_overwrite_config(self):
        first = self.install("--model", str(self.model), "--skip-verify")
        self.assertEqual(first.returncode, 0, first.stderr)
        before = self.config.read_bytes()
        second = self.install("--download-model", "--skip-verify")
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("ignored for this existing config", second.stdout)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse((self.home / '.local/share/xenolith/models').exists())

    def test_printed_commands_run_with_custom_paths_without_path_setup(self):
        # Execute exactly what a user copies, in a new shell with no installer
        # PATH mutation. The fixture checks the client-to-engine handoff only.
        client = b'''#!/usr/bin/env python3
import os, pathlib, shutil, sys
if '--version' in sys.argv:
 print('fixture-client-1'); sys.exit(0)
config=os.environ.get('XENOLITH_CONFIG')
assert config == os.environ['EXPECTED_CONFIG'], repr(config)
engine=os.environ.get('XENOLITH_BIN', 'xenolith')
resolved=shutil.which(engine)
assert resolved and pathlib.Path(resolved).is_file(), repr(engine)
assert resolved == os.environ['EXPECTED_ENGINE'], repr(resolved)
print('client handoff verified')
'''
        self.records = [r for r in self.records if r[0] not in ('xenopi', 'codexe')]
        for name in ('xenopi', 'codexe'):
            files = {f'bin/{name}': client}
            if name == 'codexe':
                for helper in ('bin/codex-code-mode-host', 'codex-path/rg',
                               'codex-resources/bwrap', 'bin/codex-exec-server'):
                    files[helper] = b'#!/bin/sh\nexit 0\n'
            self.add_archive(name, files)
        self.manifest()
        config = self.home / "User's config/native config.json"
        bindir = self.home / "User's commands"
        shadow = self.root / 'earlier commands'
        shadow.mkdir()
        wrong_engine = shadow / 'xenolith'
        wrong_engine.write_text('#!/bin/sh\nexit 88\n')
        wrong_engine.chmod(0o755)
        self.env['PATH'] = str(shadow) + ':' + str(bindir) + ':' + self.env['PATH']
        result = self.install('--model', str(self.model), '--skip-verify',
                              '--interface', 'all', '--config', str(config),
                              '--bin-dir', str(bindir), '--use-packages')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = [line.strip() for line in result.stdout.splitlines() if line.startswith('  ')]
        self.assertEqual(len(commands), 3, result.stdout)
        env = {**self.env, 'EXPECTED_CONFIG': str(config), 'EXPECTED_ENGINE': str(bindir / 'xenolith')}
        for command in commands:
            with self.subTest(command=command):
                copied = subprocess.run(['sh', '-c', command], env=env, text=True,
                                        capture_output=True, timeout=10)
                self.assertEqual(copied.returncode, 0, command + '\n' + copied.stderr)

    def test_invalid_existing_config_preserved_before_download(self):
        self.config.parent.mkdir(parents=True)
        self.config.write_bytes(b'{broken user configuration')
        before = self.config.read_bytes()
        result = self.install('--download-model', '--skip-verify', '--interface', 'all')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse((self.home / '.local/share/xenolith/models').exists())
        self.assertFalse((self.home / '.local/bin/xenopi').exists())
        self.assertFalse((self.home / '.local/bin/codexe').exists())

    def test_existing_selected_client_is_not_overwritten(self):
        binary = self.home / '.local/bin/xenopi'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'existing user client')
        result = self.install('--model', str(self.model), '--skip-verify', '--interface', 'xenopi')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(binary.read_bytes(), b'existing user client')
        self.assertFalse(self.config.exists())

    def test_tar_traversal_rejected(self):
        self.records = [r for r in self.records if r[0] != "engine"]
        self.add_archive("engine", {"../escape": b"bad", "bin/xenolith": FAKE_ENGINE})
        self.manifest()
        result = self.install("--model", str(self.model), "--skip-verify")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsafe archive", result.stderr)

    def test_interrupted_download_resumes_http_range(self):
        folder = self.source
        ranges = []
        interrupt = [True]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                p = folder / self.path.lstrip("/")
                if not p.is_file():
                    self.send_error(404)
                    return
                data = p.read_bytes()
                offset = 0
                if self.headers.get("Range"):
                    ranges.append(self.headers["Range"])
                    offset = int(self.headers["Range"].split("=")[1].split("-")[0])
                self.send_response(206 if offset else 200)
                self.send_header("Content-Length", str(len(data) - offset))
                if offset:
                    self.send_header("Content-Range", f"bytes {offset}-{len(data)-1}/{len(data)}")
                self.end_headers()
                if p.name == "model.gguf" and interrupt[0]:
                    interrupt[0] = False
                    self.wfile.write(data[:128])
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    self.wfile.write(data[offset:])

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            source = f"http://127.0.0.1:{server.server_port}"
            first = self.install("--download-model", "--skip-verify", source=source)
            self.assertNotEqual(first.returncode, 0)
            partial = self.home / ".local/share/xenolith/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf.part"
            self.assertEqual(partial.stat().st_size, 128)
            second = self.install("--download-model", "--skip-verify", source=source)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("bytes=128-", ranges)
            self.assertFalse(partial.exists())
            installed = Path(json.loads(self.config.read_text())["model"])
            self.assertEqual(installed.read_bytes(), self.model.read_bytes())
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
