"""Verification helpers for the immutable upstream Google source cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from . import GOOGLE_PIN


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_google_source(source: str | Path) -> dict[str, str]:
    """Verify every executed Google file against the checked-in commit pin."""
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Google source cache not found: {root}; run fetch-google"
        )
    actual: dict[str, str] = {}
    for relative, expected in GOOGLE_PIN["files"].items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Pinned Google source file is missing: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"Google source pin mismatch for {relative}: {digest}")
        actual[relative] = digest
    license_info = GOOGLE_PIN["license"]
    license_path = root / license_info["path"]
    if (
        not license_path.is_file()
        or sha256_file(license_path) != license_info["sha256"]
    ):
        raise ValueError(
            "Google upstream LICENSE is missing or its pinned hash differs"
        )
    if b"Apache License" not in license_path.read_bytes()[:2048]:
        raise ValueError("Google upstream LICENSE is not recognized as Apache-2.0")
    for relative, expected in GOOGLE_PIN.get("notices", {}).items():
        if not (root / relative).is_file() or sha256_file(root / relative) != expected:
            raise ValueError(
                f"Google upstream notice is missing or changed: {relative}"
            )
    return actual
