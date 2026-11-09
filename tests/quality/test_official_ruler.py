"""Small synthetic checks for the optional official RULER generator adapter."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality" / "src"))
from xe_quality import run_official_ruler as ruler


class OfficialRulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_help_is_inert_and_reports_required_explicit_paths(self):
        output = self.root / "must-not-exist"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "xe_quality.run_official_ruler",
                "--help",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": str(ROOT / "tools" / "quality" / "src")},
        )
        self.assertEqual(0, result.returncode, result.stderr.decode())
        help_text = result.stdout.decode()
        for required in ("--source", "--tokenizer", "--model", "--output"):
            self.assertIn(required, help_text)
        self.assertFalse(output.exists())

    def test_source_git_pin_and_tracked_python_bytes_are_enforced(self):
        source = self.root / "source"
        source.mkdir()
        committed = {}
        for relative in ruler.EXECUTED_PYTHON:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = ("# fixture " + relative + "\n").encode()
            path.write_bytes(payload)
            committed[relative] = payload

        def git_side_effect(_source, *args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(source.resolve()).encode()
            if args == ("rev-parse", "HEAD"):
                return b"0" * 40
            if args[:2] == ("ls-files", "--error-unmatch"):
                return args[-1].encode()
            raise AssertionError(args)

        with mock.patch.object(ruler, "_git", side_effect=git_side_effect):
            with self.assertRaisesRegex(ruler.RulerGenerationError, "HEAD must be"):
                ruler.verify_upstream_source(source)

        def correct_git(_source, *args):
            if args == ("rev-parse", "--show-toplevel"):
                return str(source.resolve()).encode()
            if args == ("rev-parse", "HEAD"):
                return ruler.UPSTREAM_COMMIT.encode()
            if args[:2] == ("ls-files", "--error-unmatch"):
                return args[-1].encode()
            raise AssertionError(args)

        with (
            mock.patch.object(ruler, "_git", side_effect=correct_git),
            mock.patch.object(
                ruler, "_git_show", side_effect=lambda _source, relative: committed[relative]
            ),
        ):
            pins = ruler.verify_upstream_source(source)
            self.assertEqual(set(ruler.EXECUTED_PYTHON), set(pins))
            changed = source / ruler.EXECUTED_PYTHON[0]
            changed.write_bytes(changed.read_bytes() + b"# dirty\n")
            with self.assertRaisesRegex(ruler.RulerGenerationError, "differs from pinned Git bytes"):
                ruler.verify_upstream_source(source)

    def test_fake_tokenizer_adapter_passes_explicit_model_and_text_bytes(self):
        tokenizer = self.root / "fake-tokenizer"
        model = self.root / "fake-model.gguf"
        model.write_bytes(b"fixture model bytes")
        tokenizer.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "assert pathlib.Path(sys.argv[1]).read_bytes() == b'fixture model bytes'\n"
            "data = pathlib.Path(sys.argv[2]).read_bytes()\n"
            "print(','.join(str(x) for x in data))\n",
            encoding="utf-8",
        )
        tokenizer.chmod(0o755)
        temp_root = self.root / "work"
        temp_root.mkdir()
        adapter = ruler._GGUFTokenizer(tokenizer, model, temp_root)
        self.assertEqual([65, 32, 66], adapter.text_to_tokens("A B"))
        self.assertEqual([], list(temp_root.iterdir()))
        with self.assertRaisesRegex(ruler.RulerGenerationError, "detokenization is unsupported"):
            adapter.tokens_to_text([65])

    def run_broad_except_worker(self, tokenizer, model, max_calls):
        script = self.root / "broad-except-worker.py"
        script.write_text(
            "from pathlib import Path\n"
            "from xe_quality.run_official_ruler import _GGUFTokenizer\n"
            f"adapter = _GGUFTokenizer(Path({str(tokenizer)!r}), Path({str(model)!r}), Path({str(self.root)!r}))\n"
            f"adapter.max_calls = {max_calls}\n"
            "while True:\n"
            "    try:\n"
            "        adapter.text_to_tokens('text')\n"
            "    except:\n"
            "        continue\n",
            encoding="utf-8",
        )
        return subprocess.run(
            [sys.executable, str(script)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            env={**os.environ, "PYTHONPATH": str(ROOT / "tools" / "quality" / "src")},
        )

    def test_tokenizer_failure_and_impossible_retry_budget_abort_broad_except(self):
        model = self.root / "model.gguf"
        model.write_bytes(b"fake model")
        tokenizer = self.root / "bad-tokenizer"
        tokenizer.write_text("#!/bin/sh\necho controlled-failure >&2\nexit 7\n", encoding="utf-8")
        tokenizer.chmod(0o755)
        failed = self.run_broad_except_worker(tokenizer, model, 4)
        self.assertEqual(75, failed.returncode)
        self.assertIn(b"tokenizer exited 7", failed.stderr)
        self.assertIn(b"XE_QUALITY_RULER_ABORT", failed.stderr)

        good_tokenizer = self.root / "good-tokenizer"
        good_tokenizer.write_text("#!/bin/sh\necho 1\n", encoding="utf-8")
        good_tokenizer.chmod(0o755)
        budget = self.run_broad_except_worker(good_tokenizer, model, 2)
        self.assertEqual(75, budget.returncode)
        self.assertIn(b"call budget exceeded (2)", budget.stderr)

    def test_worker_uses_isolated_python_and_sanitized_python_environment(self):
        args = tuple(self.root / name for name in ("source", "tokenizer", "model", "output"))
        fake_result = SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
        with (
            mock.patch.object(ruler.sys, "executable", "/venv/bin/python"),
            mock.patch.object(ruler.subprocess, "run", return_value=fake_result) as run,
            mock.patch.dict(os.environ, {"PYTHONPATH": "/poison", "PYTHONHOME": "/also-poison"}),
        ):
            ruler._run_worker(*args, length=512, samples=2, seed=3)
        command = run.call_args.args[0]
        self.assertEqual(["/venv/bin/python", "-I"], command[:2])
        self.assertLess(run.call_args.kwargs["timeout"], 10 * 60)
        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])
        self.assertNotIn("PYTHONHOME", run.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
