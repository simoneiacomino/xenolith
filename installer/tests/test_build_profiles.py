"""Exercise build recipes without Docker, Cargo, make, network or GPU access."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def load_builder():
    with patch.object(sys, 'path', [str(TOOLS), *sys.path]):
        spec = importlib.util.spec_from_file_location('profile_builder', TOOLS / 'build_linux.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class BuildProfiles(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='build-profile-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.out = self.root / 'output'
        self.tool = load_builder()

    def recipe(self, component='engine', extra=()):
        argv = ['build_linux.py', '--component', component, '--source', str(self.source),
                '--output', str(self.out), *extra]
        def metadata(command, **kwargs):
            if command[0] == 'git':
                return 'a' * 40 + '\n'
            return json.dumps([{'Id': 'sha256:fixture', 'RepoDigests': ['fixture@sha256:pinned']}])
        with patch.object(sys, 'argv', argv), patch.object(self.tool, 'run') as run, \
             patch.object(self.tool.subprocess, 'check_output', side_effect=metadata), \
             contextlib.redirect_stdout(io.StringIO()):
            self.tool.main()
        return (self.out / 'build.sh').read_text(), json.loads(
            (self.out / 'build-provenance.json').read_text()), run.call_args_list

    def test_engine_snapshot_does_not_depend_on_relocated_installer(self):
        (self.source / 'vendor').mkdir()
        (self.source / 'LICENSE.md').write_text('fixture')
        (self.source / 'xenolith.c').write_text('fixture')
        (self.source / 'installer').mkdir()
        (self.source / 'installer/private-fixture').write_text('must not enter engine snapshot')
        _, provenance, calls = self.recipe('engine', ('--build-image',))
        self.assertEqual(provenance['component'], 'engine')
        self.assertNotIn('cargo_profile', provenance)
        script = (self.out / 'build.sh').read_text()
        self.assertIn('BASE_CFLAGS=' + self.tool.ENGINE_CFLAGS, script)
        self.assertEqual(provenance['cpu_flags'], self.tool.ENGINE_CPU_FLAGS)
        self.assertFalse(provenance['model_loaded'])
        self.assertEqual(provenance['build_script_sha256'], hashlib.sha256(script.encode()).hexdigest())
        command = calls[-1].args[0]
        self.assertEqual(command.count('--mount'), 1)
        for forbidden in ('CARGO', 'RUSTUP', '--device', '--gpus', '--privileged', '/var/run/docker.sock', '/dev/dri'):
            self.assertNotIn(forbidden, ' '.join(map(str, command)))
        self.assertFalse((self.out / 'source/installer').exists())
        self.assertTrue((self.out / 'source/xenolith.c').is_file())
        self.assertIn(TOOLS.parent / 'packaging/Dockerfile.ubuntu24', calls[0].args[0])

    def test_client_build_options_are_rejected_before_output_creation(self):
        for component, extra in [('codexe', ()), ('engine', ('--cargo-profile', 'release'))]:
            with self.subTest(component=component), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.recipe(component, extra)
                self.assertEqual(raised.exception.code, 2)
                self.assertFalse(self.out.exists())

    def test_docker_image_contains_only_engine_toolchain(self):
        recipe = (TOOLS.parent / 'packaging/Dockerfile.ubuntu24').read_text()
        self.assertIn('libze-dev intel-ocloc', recipe)
        for forbidden in ('rust:', 'cargo', 'rustup', 'libasound', 'protobuf', 'libssl-dev', 'libcap-dev'):
            self.assertNotIn(forbidden, recipe)


if __name__ == '__main__':
    unittest.main()
