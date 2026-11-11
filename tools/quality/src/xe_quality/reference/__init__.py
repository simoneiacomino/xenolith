"""Pinned Google Gemma 4 and optional HF qualification adapters."""

from __future__ import annotations

import json
from pathlib import Path

GOOGLE_PIN_PATH = Path(__file__).with_name("GOOGLE-PIN.json")
GOOGLE_PIN = json.loads(GOOGLE_PIN_PATH.read_text(encoding="utf-8"))

__all__ = ["GOOGLE_PIN", "GOOGLE_PIN_PATH"]
