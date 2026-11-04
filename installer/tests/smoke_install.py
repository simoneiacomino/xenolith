#!/usr/bin/env python3
"""Real Linux package installation/reuse smoke; never loads model weights."""
import argparse
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--distribution', type=Path, required=True)
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--mmproj', type=Path)
p.add_argument('--existing-xenopi', type=Path, help='Optional npm/source command to verify reuse')
p.add_argument('--use-default-source', action='store_true',
               help='Exercise the prepared installer without --source or a command-line manifest pin')
a = p.parse_args()
dist = a.distribution.resolve()
work = a.work_dir.resolve()
work.mkdir(parents=True, exist_ok=True)
if any(work.iterdir()):
    raise RuntimeError('Use a fresh empty work-dir to retain previous evidence')
home = work / 'home'; home.mkdir()
profile = home / '.bashrc'; profile.write_text('# existing user preferences\n')
env = {k: v for k, v in os.environ.items() if not k.startswith(
    ('XENOLITH_', 'XE_', 'XENOPI_', 'PI_', 'XDG_', 'CODEX_', 'OPENAI_'))}
# Preserve the real global weight-lock location even though this test never
# requests inference. An accidental client startup cannot create a second owner.
state = os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state'))
env.update(HOME=str(home), SHELL='/bin/bash', XDG_STATE_HOME=state,
           NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
report = {'kind': 'real-install-without-model-load', 'status': 'running',
          'inference': 'NOT RUN: --skip-verify selects prepare-only mode without interface launch', 'steps': [],
          'uses_prepared_defaults': a.use_default_source,
          'manifest_sha256': hashlib.sha256((dist / 'manifest.tsv').read_bytes()).hexdigest()}
def save():
    (work / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
def run(name, argv):
    step = {'name': name, 'argv': argv}; report['steps'].append(step); save()
    with (work / (name+'.stdout')).open('w') as out, (work / (name+'.stderr')).open('w') as err:
        result = subprocess.run(argv, env=env, cwd=work, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, timeout=240, start_new_session=True)
    step['returncode'] = result.returncode; save()
    if result.returncode:
        raise RuntimeError(f'{name} failed; inspect retained logs')
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        with (work / 'http.log').open('a') as f: f.write((fmt % args)+'\n')
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(dist)))
thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
url = f'http://127.0.0.1:{server.server_port}'
args = ['sh','-s','--']
if not a.use_default_source:
    args += ['--source',url,'--manifest-sha256',report['manifest_sha256']]
args += ['--yes','--interface','all','--model',str(a.model.resolve()),
        '--skip-verify','--add-to-path','--use-packages']
if a.mmproj: args += ['--mmproj',str(a.mmproj.resolve())]
command = 'curl -fsSL '+shlex.quote(url+'/install.sh')+' | '+shlex.join(args)
try:
    run('clean-curl-pipe-install', ['bash','-o','pipefail','-c',command])
    config = home / '.config/xenolith/config.json'
    original_config, original_profile = config.read_bytes(), profile.read_bytes()
    run('repeat-curl-pipe-install', ['bash','-o','pipefail','-c',command])
    assert config.read_bytes() == original_config
    assert profile.read_bytes() == original_profile
    assert (home / '.bashrc.xenolith.bak').read_text() == '# existing user preferences\n'
    run('ordinary-shell-commands', ['bash','-c',
        '. "$HOME/.bashrc"; set -e; command -v xenolith xenopi codexe; '
        'xenolith check-device; xenolith config inspect --json; xenopi --version; codexe --version'])
    run('xenopi-help', [str(home / '.local/bin/xenopi'),'--help'])
    run('codexe-help', [str(home / '.local/bin/codexe'),'--help'])
    # Reuse the same installed commands from another installer storage/prefix:
    # no copies or archive downloads, and the same native config is retained.
    reuse_store = work / 'reuse-storage'
    reuse_args = ['sh',str(dist/'install.sh'),'--source',str(dist),'--yes',
                  '--interface','all','--skip-verify','--no-modify-path',
                  '--prefix',str(reuse_store),'--bin-dir',str(work/'unused-bin'),
                  '--engine-bin',str(home/'.local/bin/xenolith'),
                  '--xenopi-bin',str(home/'.local/bin/xenopi'),
                  '--codexe-bin',str(home/'.local/bin/codexe')]
    run('reuse-installed-components',reuse_args)
    assert config.read_bytes() == original_config
    assert not list((reuse_store/'downloads').glob('*.tar.gz'))
    assert not list((reuse_store/'releases').iterdir())
    assert not list((work/'unused-bin').iterdir())
    if a.existing_xenopi:
        npm_args = list(reuse_args)
        # Keep the command alias (resolving the symlink would change argv[0]).
        npm_args[npm_args.index('--xenopi-bin')+1] = str(a.existing_xenopi.absolute())
        run('reuse-npm-xenopi',npm_args)
        assert config.read_bytes() == original_config
    assert not list(home.rglob('*.gguf')), 'Existing model should not be copied'
    report.update(status='PASS', config_sha256=hashlib.sha256(original_config).hexdigest(),
                  profile_sha256=hashlib.sha256(original_profile).hexdigest())
except BaseException as exc:
    report.update(status='FAIL', error=str(exc)); raise
finally:
    save(); server.shutdown(); server.server_close(); thread.join()
