"""Verify the independently maintained Transformers source/config pins."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

_PACKAGE = Path(__file__).parent
_PIN = json.loads((_PACKAGE / "HF-PIN.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hf_source(config_path: str | Path | None = None) -> dict[str, str]:
    config = _PIN["config"]
    selected_config = (
        Path(config_path) if config_path is not None else _PACKAGE / config["path"]
    )
    if not selected_config.is_file() or _sha256(selected_config) != config["sha256"]:
        raise ValueError("Selected HF configuration hash differs from HF-PIN.json")
    verified = {str(selected_config.resolve()): config["sha256"]}
    for filename, expected in _PIN["transformers_modules"].items():
        module = f"transformers.models.gemma4.{filename[:-3]}"
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ImportError(
                f"Pinned HF qualification module is unavailable: {module}"
            )
        path = Path(spec.origin)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"HF source pin mismatch for {filename}: {actual}")
        verified[filename] = actual
    return verified
