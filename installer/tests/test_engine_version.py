"""Strict SemVer rules for engine component releases."""
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import engine_version


class EngineVersionTest(unittest.TestCase):
    def test_accepts_semver_and_detects_any_prerelease(self):
        for value in ('0.0.0', '1.2.3', '1.2.3-dev.1', '1.2.3-alpha.0'):
            with self.subTest(value=value):
                self.assertEqual(engine_version.validate(value), value)
        self.assertFalse(engine_version.is_prerelease('1.2.3'))
        self.assertTrue(engine_version.is_prerelease('1.2.3-dev.1'))
        self.assertTrue(engine_version.is_prerelease('1.2.3-rc.2'))

    def test_rejects_invalid_core_and_numeric_prerelease_leading_zeroes(self):
        for value in ('1', '1.2', '01.2.3', '1.02.3', '1.2.03', '1.2.3-',
                      '1.2.3-alpha.01', '1.2.3-00'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                engine_version.validate(value)
        with self.assertRaisesRegex(ValueError, 'without build metadata'):
            engine_version.validate('1.2.3+build.1')

if __name__ == '__main__':
    unittest.main()
