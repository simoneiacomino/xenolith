"""Acquisition helpers stay inert until an explicit call is made."""

from pathlib import Path
import sys
import os
import io
import contextlib
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality" / "src"))
from xe_quality import acquisition


class AcquisitionTests(unittest.TestCase):
    def test_pinned_sources_are_explicit_and_file_writes_are_idempotent(self):
        self.assertEqual(
            "b08601e04326c79dfdd32d625aee71d232d685c3",
            acquisition.PINNED["wikitext2_raw"]["revision"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "source.jsonl"
            first = acquisition.save(target, b'{"fixture":true}\n')
            second = acquisition.save(target, b'{"fixture":true}\n')
            self.assertEqual(first, second)
            with self.assertRaises(RuntimeError):
                acquisition.save(target, b"different\n")

    def test_module_help_is_inert_and_output_is_explicit(self):
        env = dict(os.environ)
        package_root = str(ROOT / "tools" / "quality" / "src")
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            help_result = subprocess.run(
                [sys.executable, "-m", "xe_quality.acquisition", "--help"],
                cwd=work,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, help_result.returncode)
            self.assertIn("--output", help_result.stdout)
            missing = subprocess.run(
                [sys.executable, "-m", "xe_quality.acquisition"],
                cwd=work,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(2, missing.returncode)
            self.assertIn("required", missing.stderr)
            destination = work / "acquired"
            with (
                mock.patch.object(acquisition, "acquire_wikitext"),
                mock.patch.object(acquisition, "acquire_ifeval"),
                mock.patch.object(acquisition, "acquire_humaneval"),
                mock.patch.object(acquisition, "acquire_belebele"),
                mock.patch.object(acquisition, "acquire_ruler"),
                mock.patch.object(acquisition, "acquire_local"),
            ):
                self.assertEqual(0, acquisition.main(["--output", str(destination)]))
            self.assertTrue((destination / "pinned-source-manifest.json").is_file())
            self.assertFalse(
                (
                    ROOT
                    / "tools"
                    / "quality"
                    / "src"
                    / "xe_quality"
                    / "pinned-source-manifest.json"
                ).exists()
            )
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                acquisition.main(["--output", str(destination)])
            self.assertEqual(2, error.exception.code)


if __name__ == "__main__":
    unittest.main()
