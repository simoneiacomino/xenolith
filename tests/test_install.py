"""Install the built command into temporary prefixes, never host locations."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BINARY = Path(sys.argv.pop(1) if len(sys.argv) > 1 else './xenolith').resolve()
REPO = Path(__file__).resolve().parents[1]


class Install(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='xenolith-install-test-')
        self.root = Path(self.temp.name)
        self.env = {**os.environ, 'HOME': str(self.root / 'home'),
                    'XDG_CONFIG_HOME': str(self.root / 'config')}
        for key in ('XENOLITH_CONFIG', 'MAKEFLAGS', 'MFLAGS', 'PREFIX', 'DESTDIR', 'bindir'):
            self.env.pop(key, None)

    def tearDown(self):
        self.temp.cleanup()

    def install(self, prefix, destdir=''):
        return subprocess.run(['make', '--no-print-directory', 'install', f'PREFIX={prefix}', f'DESTDIR={destdir}'],
                              cwd=REPO, env=self.env, capture_output=True, text=True, timeout=60)

    def test_user_install_is_on_explicit_path_without_configuration(self):
        prefix = self.root / 'user prefix'
        unrelated = prefix / 'bin/unrelated'
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text('keep me')
        for _ in range(2):
            result = self.install(prefix)
            self.assertEqual(result.returncode, 0, result.stderr)
        binary = prefix / 'bin/xenolith'
        self.assertEqual(binary.read_bytes(), BINARY.read_bytes())
        self.assertEqual(binary.stat().st_mode & 0o777, 0o755)
        self.assertEqual(unrelated.read_text(), 'keep me')
        env = {**self.env, 'PATH': str(prefix / 'bin') + ':' + self.env.get('PATH', '')}
        check = subprocess.run(['xenolith', 'config', 'inspect', '--json'], env=env,
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertFalse((self.root / 'config').exists())
        self.assertFalse((self.root / 'home').exists())

    def test_staged_system_install_respects_destdir(self):
        stage = self.root / 'package stage'
        result = self.install('/usr', stage)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((stage / 'usr/bin/xenolith').read_bytes(), BINARY.read_bytes())
        self.assertEqual([str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()], ['usr/bin/xenolith'])

    def test_default_install_uses_user_home(self):
        result = subprocess.run(['make', '--no-print-directory', 'install'], cwd=REPO,
                                env=self.env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / 'home/.local/bin/xenolith').read_bytes(), BINARY.read_bytes())
        self.assertFalse((self.root / 'config').exists())

    def test_unrelated_symlink_target_is_preserved(self):
        prefix = self.root / 'prefix'
        (prefix / 'bin').mkdir(parents=True)
        other = self.root / 'other'
        other.write_text('do not overwrite')
        (prefix / 'bin/xenolith').symlink_to(other)
        result = self.install(prefix)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(other.read_text(), 'do not overwrite')
        self.assertTrue((prefix / 'bin/xenolith').is_symlink())

    def test_directory_at_command_path_is_preserved(self):
        prefix = self.root / 'prefix'
        (prefix / 'bin/xenolith').mkdir(parents=True)
        result = self.install(prefix)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((prefix / 'bin/xenolith').is_dir())


if __name__ == '__main__':
    unittest.main()
