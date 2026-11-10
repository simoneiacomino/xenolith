from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from xe_quality.reference import source_pin
from xe_quality.reference.hf_source_pin import verify_hf_source


class GoogleSourcePinTests(unittest.TestCase):
    def test_source_pin_checks_executed_bytes_and_license(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "google"
            relative = "gemma/gm/nn/gemma4/_modules.py"
            file_path = source / relative
            file_path.parent.mkdir(parents=True)
            file_path.write_bytes(b"pinned module")
            license_data = b"Apache License\nVersion 2.0"
            (source / "LICENSE").write_bytes(license_data)
            pin = source_pin.GOOGLE_PIN
            source_pin.GOOGLE_PIN = {
                "files": {relative: hashlib.sha256(b"pinned module").hexdigest()},
                "license": {
                    "path": "LICENSE",
                    "sha256": hashlib.sha256(license_data).hexdigest(),
                },
            }
            try:
                self.assertEqual(
                    source_pin.verify_google_source(source)[relative],
                    hashlib.sha256(b"pinned module").hexdigest(),
                )
                file_path.write_bytes(b"changed module")
                with self.assertRaisesRegex(ValueError, "pin mismatch"):
                    source_pin.verify_google_source(source)
            finally:
                source_pin.GOOGLE_PIN = pin

    def test_source_pin_requires_license(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "google"
            source.mkdir()
            data = b"source"
            (source / "module.py").write_bytes(data)
            pin = source_pin.GOOGLE_PIN
            source_pin.GOOGLE_PIN = {
                "files": {"module.py": hashlib.sha256(data).hexdigest()},
                "license": {"path": "LICENSE", "sha256": "0" * 64},
            }
            try:
                with self.assertRaisesRegex(ValueError, "LICENSE"):
                    source_pin.verify_google_source(source)
            finally:
                source_pin.GOOGLE_PIN = pin

    def test_hf_configuration_hash_is_checked_before_optional_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "changed-config.json"
            config.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration hash"):
                verify_hf_source(config)

    def test_module_identity_works_without_opening_gguf(self):
        try:
            import jax  # noqa: F401
            import flax  # noqa: F401
        except ImportError as error:
            self.skipTest(f"optional Google runtime unavailable: {error}")
        from xe_quality.reference import stream

        files = {"gemma/gm/nn/gemma4/module.py": b"source"}
        license_data = b"Apache License\nVersion 2.0"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            for relative, data in files.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (source / "LICENSE").write_bytes(license_data)
            model = root / "opaque-model.gguf"
            model.write_bytes(b"not a GGUF; identity helper must not parse it")
            original = source_pin.GOOGLE_PIN
            source_pin.GOOGLE_PIN = {
                "files": {
                    name: hashlib.sha256(data).hexdigest()
                    for name, data in files.items()
                },
                "license": {
                    "path": "LICENSE",
                    "sha256": hashlib.sha256(license_data).hexdigest(),
                },
            }
            try:
                identity = stream.code_identity(model, source, precision="fp64")
                self.assertEqual(
                    identity["weights_sha256"],
                    hashlib.sha256(model.read_bytes()).hexdigest(),
                )
                self.assertIn("runtime_source_and_binary_sha256", identity["runtime"])
                self.assertIn("runtime_info_code_sha256", identity)
            finally:
                source_pin.GOOGLE_PIN = original


if __name__ == "__main__":
    unittest.main()
