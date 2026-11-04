"""Actual CLI with mocked Level Zero: no GPU or model is touched."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

BINARY = Path(sys.argv.pop(1) if len(sys.argv) > 1 else './xenolith').resolve()


class DeviceCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='xenolith-probe-test-')
        cls.root = Path(cls.temp.name)
        cls.loader = cls.root / 'probe.so'
        source = Path(__file__).parent / 'fixtures/level_zero_probe.c'
        subprocess.run([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-Wall', '-Wextra',
                        '-shared', '-fPIC', str(source), '-o', str(cls.loader)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def probe(self, scenario='tested', args=('check-device',)):
        trace = self.root / 'trace'
        trace.unlink(missing_ok=True)
        broken = self.root / 'invalid-config.json'
        broken.write_text('not JSON')
        env = {**os.environ, 'LD_PRELOAD': str(self.loader), 'XE_PROBE_TEST_CASE': scenario,
               'XE_PROBE_TEST_TRACE': str(trace), 'XENOLITH_CONFIG': str(broken),
               'HOME': str(self.root / 'home-not-created'), 'XE_WORKERS': 'invalid',
               'XDG_STATE_HOME': str(self.root / 'state-not-created')}
        result = subprocess.run([str(BINARY), *args], env=env, text=True, capture_output=True, timeout=10)
        self.assertFalse((self.root / 'home-not-created').exists())
        self.assertFalse((self.root / 'state-not-created').exists())
        self.assertEqual(broken.read_text(), 'not JSON')
        self.assertNotIn('FORBIDDEN', trace.read_text() if trace.exists() else '')
        return result, trace.read_text().splitlines() if trace.exists() else []

    def test_tested_device_bypasses_config_and_cpu_options(self):
        result, calls = self.probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('8086:a7a0', result.stdout)
        self.assertIn('tested', result.stdout)
        self.assertIn('runtime access: ready', result.stdout)
        self.assertEqual(calls, ['init', 'driver-count', 'driver', 'device-count', 'device', 'properties', 'status'])

    def test_untested_enabled_device_is_reachable(self):
        result, _ = self.probe('untested')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('enabled, not yet verified', result.stdout)

    def test_runtime_and_access_errors_are_specific(self):
        for case, diagnostic in [('unavailable', 'runtime unavailable'), ('no-drivers', 'no GPU driver'),
                                 ('driver-error', 'dependency unavailable'), ('no-devices', 'no GPU device'),
                                 ('device-error', 'access denied'), ('properties-error', 'device lost'),
                                 ('permissions', 'access denied'), ('lost', 'device lost')]:
            with self.subTest(case=case):
                result, _ = self.probe(case)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(diagnostic, result.stderr)

    def test_unsupported_device_or_vendor_fails(self):
        for case in ['unknown', 'wrong-vendor', 'multiple']:
            with self.subTest(case=case):
                result, calls = self.probe(case)
                self.assertEqual(result.returncode, 1)
                self.assertIn('no compatible GPU selected', result.stderr)
                self.assertNotIn('status', calls)

    def test_syntax_errors_do_not_initialize_runtime(self):
        for args in [('check-device', '--check'), ('check-device', '--json'),
                     ('check-device', '--config', '/tmp/unused'), ('devices', '--check')]:
            with self.subTest(args=args):
                result, calls = self.probe(args=args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(calls)

    def test_config_inspect_still_does_not_initialize_runtime(self):
        result, calls = self.probe(args=('config', 'inspect', '--json', '--config', str(self.root / 'missing.json')))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
