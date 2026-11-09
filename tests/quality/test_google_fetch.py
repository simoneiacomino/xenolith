from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xe_quality.reference import fetch_google, source_pin


class GoogleFetchTests(unittest.TestCase):
    def test_pinned_archive_extract_is_explicit_and_verified(self):
        payloads = {
            "gemma/gm/nn/gemma4/test_module.py": b"pinned bytes",
            "LICENSE": b"Apache License\nVersion 2.0",
        }
        pin = {
            "repository": "https://example.invalid/gemma",
            "commit": "abc123",
            "files": {
                "gemma/gm/nn/gemma4/test_module.py": hashlib.sha256(
                    payloads["gemma/gm/nn/gemma4/test_module.py"]
                ).hexdigest()
            },
            "license": {
                "path": "LICENSE",
                "spdx": "Apache-2.0",
                "sha256": hashlib.sha256(payloads["LICENSE"]).hexdigest(),
            },
            "notice_files": [],
        }
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            for relative, data in payloads.items():
                info = tarfile.TarInfo(f"gemma-abc123/{relative}")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            extra = b"ignored unpinned file"
            info = tarfile.TarInfo("gemma-abc123/unpinned.txt")
            info.size = len(extra)
            archive.addfile(info, io.BytesIO(extra))

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self, _limit):
                return archive_bytes.getvalue()

        old_fetch_pin, old_check_pin = fetch_google.GOOGLE_PIN, source_pin.GOOGLE_PIN
        fetch_google.GOOGLE_PIN = source_pin.GOOGLE_PIN = pin
        try:
            with (
                tempfile.TemporaryDirectory() as temp,
                patch.object(fetch_google, "urlopen", return_value=Response()),
            ):
                result = fetch_google.fetch(Path(temp))
                self.assertEqual(result.name, "gemma-abc123")
                self.assertEqual(
                    (result / "gemma/gm/nn/gemma4/test_module.py").read_bytes(),
                    payloads["gemma/gm/nn/gemma4/test_module.py"],
                )
                self.assertFalse((result / "unpinned.txt").exists())
                self.assertTrue((result / "SOURCE-PROVENANCE.json").is_file())
                self.assertEqual(fetch_google.fetch(Path(temp)), result)
        finally:
            fetch_google.GOOGLE_PIN, source_pin.GOOGLE_PIN = (
                old_fetch_pin,
                old_check_pin,
            )


if __name__ == "__main__":
    unittest.main()
