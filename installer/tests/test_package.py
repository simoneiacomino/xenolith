"""Metadata assembly uses unchanged component archives without executing them."""
import functools
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import threading
import test_install as install_fixtures

PACKAGER = Path(__file__).resolve().parents[1] / 'tools/prepare_distribution.py'


class PackagingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='distribution-assembly-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'metadata source'
        self.repo.mkdir()
        (self.repo / 'xenolith.c').write_text('''
static xe_gpu_status xe_gpu_classify(int device) {
 switch (device) { case 0xa7a0: return XE_GPU_TESTED;
 case 0x46a0: return XE_GPU_UNTESTED; }
}
void *xe_vision_alloc_shared;
''')
        vision = self.repo / 'tests/vision'
        vision.mkdir(parents=True)
        (vision / 'manifest.json').write_text(json.dumps({
            'model_revision': 'fixture', 'artifacts': [
                {'url': 'https://example.invalid/' + name, 'size_bytes': 4, 'sha256': '0' * 64}
                for name in ('model', 'projector')]}))
        self.payload = Path(shutil.which('true')).read_bytes()
        self.archives = {}
        self.write_component('engine', {'bin/xenolith': (self.payload, 0o755), 'LICENSE.md': (b'license', 0o644)})
        self.write_component('xenopi', {'bin/xenopi': (b'#!/bin/sh\nexit 99\n', 0o755),
            'node/bin/node': (self.payload, 0o755), 'lib/xenopi/package.json': (b'{"name":"xenopi","version":"2"}', 0o644),
            'lib/xenopi/dist/src/bin/xenopi.js': (b'// fixture', 0o644)})
        self.codexe_files = {'bin/codex': (self.payload, 0o755),
            'install/install.sh': (b'#!/bin/sh\nexit 99\n', 0o755),
            'codex-package.json': (b'{"version":"3"}', 0o644),
            'codexe-package_SHA256SUMS': (b'fixture checksums', 0o644)}
        self.write_component('codexe', self.codexe_files)

    def write_component(self, name, files, extras=()):
        archive = self.root / f'{name}-fixture.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for relative, (data, mode) in files.items():
                info = tarfile.TarInfo(relative)
                info.mode, info.size = mode, len(data)
                tar.addfile(info, io.BytesIO(data))
            for info, data in extras:
                tar.addfile(info, io.BytesIO(data) if data is not None else None)
        self.archives[name] = archive
        return archive

    def command(self, output, components=('engine',), extra=()):
        command = [sys.executable, str(PACKAGER), '--engine-repo', str(self.repo),
            '--distribution-version', 'candidate-1', '--cpu-flags', 'avx2 fma', '--output', str(output)]
        for name in components:
            version = {'engine': '1', 'xenopi': '2', 'codexe': '3'}[name]
            repo, prefix = {'engine': ('simoneiacomino/xenolith', 'v'),
                'xenopi': ('simoneiacomino/xenopi', 'v'), 'codexe': ('crrisci/codexe', 'codexe-v')}[name]
            command += [f'--{name}-archive', str(self.archives[name]), f'--{name}-version', version,
                f'--{name}-url', f'https://github.com/{repo}/releases/download/{prefix}{version}/{self.archives[name].name}']
        return subprocess.run(command + list(extra), text=True, capture_output=True, timeout=30)

    def test_three_prebuilt_components_produce_only_metadata_and_preserve_inputs(self):
        output = self.root / 'distribution'
        originals = {name: archive.read_bytes() for name, archive in self.archives.items()}
        result = self.command(output, ('engine', 'xenopi', 'codexe'), extra=['--default-source', 'https://xenolith.it/distributions/candidate-1'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in output.iterdir()}, {'manifest.tsv', 'install.sh', 'requirements.tsv',
            'gpu-matrix.tsv', 'elf-requirements.json', 'provenance.json', 'SHA256SUMS'})
        records = {row[0]: row for row in (line.split('\t') for line in (output / 'manifest.tsv').read_text().splitlines()[1:])}
        provenance = json.loads((output / 'provenance.json').read_text())
        for name, data in originals.items():
            self.assertEqual(self.archives[name].read_bytes(), data)
            self.assertEqual(records[name][4], hashlib.sha256(data).hexdigest())
            self.assertEqual(int(records[name][3]), len(data))
            self.assertTrue(records[name][2].startswith('https://github.com/'))
            self.assertEqual(provenance['components'][name]['archive_source_match'], 'not-verified')
        self.assertNotIn(str(self.root), (output / 'manifest.tsv').read_text())
        self.assertNotIn(str(self.root), (output / 'provenance.json').read_text())
        abi = json.loads((output / 'elf-requirements.json').read_text())
        self.assertIn('libc.so.6', abi['engine']['binaries']['bin/xenolith']['needed'])
        self.assertIn('node/bin/node', abi['xenopi']['binaries'])
        self.assertIn('bin/codex', abi['codexe']['binaries'])
        subprocess.run(['sha256sum', '--check', 'SHA256SUMS'], cwd=output, check=True, capture_output=True)

    def test_invalid_archive_layouts_fail_before_output_writes(self):
        for label, name, member_type, link in [
            ('absolute', '/escape', tarfile.REGTYPE, ''), ('traversal', '../escape', tarfile.REGTYPE, ''),
            ('dot', './escape', tarfile.REGTYPE, ''), ('backslash', 'a\\b', tarfile.REGTYPE, ''),
            ('duplicate', 'bin/xenolith', tarfile.REGTYPE, ''), ('symlink', 'link', tarfile.SYMTYPE, 'bin/xenolith'),
            ('hardlink', 'link', tarfile.LNKTYPE, 'bin/xenolith'), ('fifo', 'fifo', tarfile.FIFOTYPE, '')]:
            with self.subTest(label=label):
                info = tarfile.TarInfo(name)
                info.type, info.linkname = member_type, link
                info.size = 1 if info.isfile() else 0
                self.write_component('engine', {'bin/xenolith': (self.payload, 0o755)}, [(info, b'x' if info.isfile() else None)])
                output = self.root / ('rejected-' + label)
                result = self.command(output)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists(), result.stdout + result.stderr)

    def test_each_component_enforces_glibc_gate_on_actual_payload(self):
        for component in ('engine', 'xenopi', 'codexe'):
            with self.subTest(component=component):
                output = self.root / ('abi-' + component)
                # For client-specific gating use a script engine in explicit test mode.
                if component != 'engine':
                    self.write_component('engine', {'bin/xenolith': (b'#!/bin/sh\nexit 99\n', 0o755)})
                names = ('engine',) if component == 'engine' else ('engine', component)
                result = self.command(output, names, ['--max-glibc', '0.1', '--development-components'])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(component + ' requires glibc', result.stderr)
                self.assertFalse(output.exists())

    def test_codexe_metadata_version_must_match_and_layout_is_required(self):
        for files in ({**self.codexe_files, 'codex-package.json': (b'{"version":"wrong"}', 0o644)},
                      {k: v for k, v in self.codexe_files.items() if k != 'install/install.sh'},
                      {'wrapper/' + k: v for k, v in self.codexe_files.items()}):
            self.write_component('codexe', files)
            result = self.command(self.root / 'bad-codexe', ('engine', 'codexe'))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((self.root / 'bad-codexe').exists())

    def test_xenopi_metadata_identity_must_match_before_output_writes(self):
        for metadata in ([1], 'xenopi', 7, None, {}, {'version': '2'},
                         {'name': 'unrelated', 'version': '2'},
                         {'name': 'xenopi', 'version': '99.9.9'},
                         {'name': 'xenopi', 'version': 2}):
            with self.subTest(metadata=metadata):
                self.write_component('xenopi', {'bin/xenopi': (b'#!/bin/sh\nexit 99\n', 0o755),
                    'node/bin/node': (self.payload, 0o755),
                    'lib/xenopi/package.json': (json.dumps(metadata).encode(), 0o644),
                    'lib/xenopi/dist/src/bin/xenopi.js': (b'// fixture', 0o644)})
                output = self.root / 'bad-xenopi'
                result = self.command(output, ('engine', 'xenopi'))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('XenoPi metadata', result.stderr)
                self.assertFalse(output.exists())

    def test_development_absolute_urls_are_rejected_with_supported_setup(self):
        output = self.root / 'absolute-component-location'
        result = self.command(output, extra=['--development-components', '--engine-url', str(self.archives['engine'])])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('serve the archive over loopback HTTP', result.stderr)
        self.assertFalse(output.exists())

    def test_development_loopback_component_download_installs_from_metadata_only_bundle(self):
        fixture = install_fixtures.InstallerTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.write_component('engine', {'bin/xenolith': (install_fixtures.FAKE_ENGINE, 0o755)})
        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args): pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
            functools.partial(Handler, directory=str(self.root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def close():
            server.shutdown(); server.server_close(); thread.join()
        self.addCleanup(close)
        url = f'http://127.0.0.1:{server.server_port}/{self.archives["engine"].name}'
        output = self.root / 'loopback-metadata'
        result = self.command(output, extra=['--development-components', '--engine-url', url,
                                            '--default-source', str(output)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(list(output.glob('*.tar.gz')))
        result = subprocess.run(['sh', str(output / 'install.sh'), '--yes', '--interface', 'chat',
            '--model', str(fixture.model), '--skip-verify', '--no-modify-path', '--use-packages'],
            env=fixture.env, text=True, capture_output=True, timeout=30, start_new_session=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((fixture.home / '.local/bin/xenolith').is_file())
        self.assertTrue(fixture.config.is_file())
        self.assertIn('Commands for this and later sessions:', result.stdout)
        self.assertFalse(any(call['args'][:1] == ['chat'] for call in fixture.calls()))

    def test_canonical_codexe_shell_installer_need_not_be_executable(self):
        files = {**self.codexe_files, 'install/install.sh': (b'#!/bin/sh\nexit 99\n', 0o644)}
        self.write_component('codexe', files)
        result = self.command(self.root / 'nonexecutable-shell-installer', ('engine', 'codexe'))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_scripts_rejected_as_release_binaries_but_allowed_explicitly_for_fixtures(self):
        self.write_component('engine', {'bin/xenolith': (b'#!/bin/sh\nexit 99\n', 0o755)})
        output = self.root / 'script-release'
        result = self.command(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Linux x86-64 ELF', result.stderr)
        self.assertFalse(output.exists())
        fixture = self.root / 'script-fixture'
        result = self.command(fixture, extra=['--development-components', '--engine-url', 'http://127.0.0.1:8123/engine-fixture.tar.gz'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads((fixture / 'provenance.json').read_text())['development_components'])

    def test_invalid_versions_urls_optional_triples_and_outputs_are_rejected(self):
        for flag, value in [('--distribution-version', '../escape'), ('--engine-version', 'x/y'),
            ('--engine-url', 'https://github.com/wrong/repo/releases/download/v1/engine-fixture.tar.gz'),
            ('--engine-url', 'https://github.com/simoneiacomino/xenolith/releases/download/latest/engine-fixture.tar.gz'),
            ('--engine-url', 'https://github.com/simoneiacomino/xenolith/releases/download/v2/engine-fixture.tar.gz'),
            ('--engine-url', 'https://github.com/simoneiacomino/xenolith/releases/download/v1/wrong.tar.gz'),
            ('--engine-url', '/tmp/component.tar.gz'), ('--codexe-version', '3'), ('--cpu-flags', 'avx2\nx'),
            ('--max-glibc', '2.39\nx')]:
            with self.subTest(flag=flag, value=value):
                output = self.root / 'invalid'
                self.assertNotEqual(self.command(output, extra=[flag, value]).returncode, 0)
                self.assertFalse(output.exists())
        sentinel = self.root / 'previous'
        sentinel.mkdir()
        (sentinel / 'keep').write_text('unchanged')
        for output in (self.repo, self.repo / 'nested', self.root, sentinel):
            self.assertNotEqual(self.command(output).returncode, 0)
        self.assertEqual((sentinel / 'keep').read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main()
