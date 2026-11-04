#!/usr/bin/env python3
"""Real CLI configuration checks; never opens model weights or a GPU."""
import concurrent.futures
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

BINARY = str(Path(sys.argv.pop(1) if len(sys.argv) > 1 else "./xenolith").resolve())


class ConfigCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="xe-config-")
        self.home = Path(self.tmp.name)
        self.config = self.home / ".config/xenolith/config.json"
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("XDG_", "XENOLITH_", "XE_"))}
        self.env["HOME"] = str(self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args, success=True, env=None):
        result = subprocess.run([BINARY, *args], env=env or self.env,
                                stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=5)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result

    def inspect(self, *args, env=None):
        return json.loads(self.cli("config", "inspect", "--json", *args, env=env).stdout)

    def test_missing_config_is_read_only(self):
        before = list(self.home.rglob("*"))
        data = self.inspect()
        self.assertFalse(data["config_present"])
        self.assertEqual(data["config_path"], str(self.config))
        self.assertEqual(data["endpoint"]["transport"], "unix")
        self.assertEqual(data["endpoint"]["path"], str(self.home / ".local/state/xenolith/wire.sock"))
        self.assertEqual(before, list(self.home.rglob("*")))

    def test_init_inspect_and_no_clobber(self):
        model = self.home / 'models/quoted " model.gguf'
        self.cli("config", "init", "--model", str(model), "--ctx", "65536",
                 "--mmproj", "/models/projector.gguf", "--cache-budget", "4GiB")
        raw = self.config.read_bytes()
        settings = json.loads(raw)
        self.assertEqual(settings["model"], str(model))
        self.assertEqual(settings["cache_budget"], 4294967296)
        self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)
        self.assertTrue(self.inspect()["config_present"])
        self.cli("config", "init", "--model", "/other.gguf", success=False)
        self.assertEqual(self.config.read_bytes(), raw)
        self.assertFalse(list(self.config.parent.glob("*.tmp.*")))

    def test_custom_paths_and_global_lock(self):
        self.cli("config", "init", "--model", "/missing/model.gguf",
                 "--socket", str(self.home / "custom/wire.sock"),
                 "--state", str(self.home / "custom/state"),
                 "--cache", str(self.home / "custom/cache"))
        data = self.inspect()
        self.assertEqual(data["state_dir"], str(self.home / "custom/state"))
        self.assertEqual(data["log_path"], str(self.home / "custom/state/service.log"))
        self.assertEqual(data["engine_lock"], str(self.home / ".local/state/xenolith/engine.lock"))
        self.assertFalse((self.home / "custom").exists())
        changed = self.inspect("--socket", str(self.home / "other.sock"), "--state", "/override/state")
        self.assertEqual(changed["endpoint"]["path"], str(self.home / "other.sock"))
        self.assertEqual(changed["state_dir"], "/override/state")
        self.assertEqual(changed["engine_lock"], data["engine_lock"])

    def test_xdg_runtime_fallback_does_not_create(self):
        runtime = self.home / "runtime"
        runtime.mkdir()
        env = dict(self.env, XDG_RUNTIME_DIR=str(runtime))
        self.assertEqual(self.inspect(env=env)["endpoint"]["path"], str(runtime / "xenolith/wire.sock"))
        self.assertFalse((runtime / "xenolith").exists())
        (runtime / "xenolith").write_text("blocker")
        self.assertEqual(self.inspect(env=env)["endpoint"]["path"], str(self.home / ".local/state/xenolith/wire.sock"))

    def test_selectors_and_errors(self):
        alternative = self.home / "alternative.json"
        env = dict(self.env, XENOLITH_CONFIG=str(alternative))
        self.assertEqual(self.inspect(env=env)["config_path"], str(alternative))
        self.assertEqual(self.inspect("--config", str(self.config), env=env)["config_path"], str(self.config))
        alternative.write_text('{"schema_version": 3}')
        self.assertIn("schema_version", self.cli("config", "inspect", "--json", success=False, env=env).stderr)
        alternative.write_text("broken")
        self.assertIn(str(alternative), self.cli("config", "inspect", "--json", success=False, env=env).stderr)
        self.cli("config", "inspect", "--json", "--config", "relative", success=False)
        self.cli("config", "inspect", "--json", "--model", "/unexpected", success=False)
        self.cli("config", "init", "--model", "/model", "--mmproj", "/projector", "--ctx", "4096", success=False)

    def test_atomic_concurrent_init(self):
        def attempt(i):
            return subprocess.run([BINARY, "config", "init", "--model", f"/model-{i}.gguf"],
                                  env=self.env, capture_output=True, timeout=5).returncode
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(attempt, range(4)))
        self.assertEqual(results.count(0), 1)
        self.assertIn(json.loads(self.config.read_text())["model"], [f"/model-{i}.gguf" for i in range(4)])
        self.assertFalse(list(self.config.parent.glob("*.tmp.*")))

    def test_missing_model_without_terminal_is_actionable(self):
        result = subprocess.run([BINARY, "config", "init"], env=self.env,
                                stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=5, start_new_session=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--model", result.stderr)
        self.assertFalse(self.config.exists())

    def test_symlink_and_nonregular_are_not_replaced_or_read(self):
        self.config.parent.mkdir(parents=True)
        self.config.symlink_to(self.home / "absent.json")
        self.cli("config", "init", "--model", "/model.gguf", success=False)
        self.assertTrue(self.config.is_symlink())
        self.config.unlink()
        os.mkfifo(self.config)
        self.assertIn("regular file", self.cli("config", "inspect", "--json", success=False).stderr)

    def test_command_defaults_and_literal_option_values(self):
        model = str(self.home / "missing.gguf")
        self.cli("config", "init", "--model", model, "--ctx", "64")
        # A vocabulary-only command must not inherit --ctx. Prompt values that
        # look like selectors remain literal, wherever a real selector occurs.
        for command in ("run", "tokenize"):
            for value in ("--config", "--no-config", "--ctx", "--image"):
                for selector in ((), ("--config", str(self.config))):
                    result = self.cli(command, *selector, "-p", value, success=False)
                    self.assertIn(model, result.stderr)
                    self.assertNotIn("Usage:", result.stderr)
        for args in (("--no-config", "--config", str(self.config)),
                     ("--config", str(self.config), "--no-config"),
                     ("--no-config", "--no-config"),
                     ("--config", str(self.config), "--config", str(self.config))):
            result = self.cli("chat", *args, success=False)
            self.assertIn("--config", result.stderr)
        self.config.write_text("invalid")
        result = self.cli("run", model, "-p", "--config", "--no-config", success=False)
        self.assertIn(model, result.stderr)
        self.assertNotIn("configuration", result.stderr)
        result = self.cli("tokenize", model, "--detok", "--no-config", success=False)
        self.assertIn(model, result.stderr)
        self.assertNotIn("configuration", result.stderr)

    def test_projector_inheritance_and_context_override(self):
        model = str(self.home / "missing.gguf")
        image = str(self.home / "missing.png")
        self.cli("config", "init", "--model", model, "--mmproj", "/projector.gguf", "--ctx", "65536")
        # Text-only commands do not inherit the configured projector.
        for command in ("chat", "run"):
            prompt = ("-p", "hello") if command == "run" else ()
            result = self.cli(command, *prompt, "--ctx", "64", success=False)
            self.assertIn(model, result.stderr)
            self.assertNotIn("vision", result.stderr)
            result = self.cli(command, *prompt, "--image", image, "--ctx", "64", success=False)
            self.assertIn("vision", result.stderr)
            result = self.cli(command, *prompt, "--image", image, success=False)
            self.assertIn("cannot open image", result.stderr)
        for command in ("wire", "serve", "vision-encode"):
            options = ("--image", image, "--vision-dump", str(self.home / "dump")) if command == "vision-encode" else ()
            result = self.cli(command, "/other.gguf", *options, success=False)
            self.assertIn("explicit model differs", result.stderr)
            result = self.cli(command, "/other.gguf", *options, "--mmproj", "/other-projector.gguf", success=False)
            self.assertNotIn("explicit model differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
