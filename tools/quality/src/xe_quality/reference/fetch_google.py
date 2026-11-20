"""Explicitly acquire the pinned Google source subset into an external cache."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from . import GOOGLE_PIN
from .source_pin import verify_google_source

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024


def fetch(cache: Path) -> Path:
    cache = cache.expanduser().resolve()
    target = cache / f"gemma-{GOOGLE_PIN['commit']}"
    if target.exists():
        verify_google_source(target)
        return target
    cache.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/google-deepmind/gemma/archive/{GOOGLE_PIN['commit']}.tar.gz"
    with urlopen(url, timeout=60) as response:
        archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"Pinned Google source archive exceeds {MAX_ARCHIVE_BYTES} bytes"
        )
    with tempfile.TemporaryDirectory(prefix="xe-google-source-", dir=cache) as temp:
        stage = Path(temp) / target.name
        stage.mkdir()
        wanted = set(GOOGLE_PIN["files"]) | {"LICENSE"}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue
                relative = Path(*parts[1:]).as_posix()
                if relative not in wanted:
                    continue
                if not member.isfile():
                    raise ValueError(
                        f"Unexpected non-file pinned archive member: {relative}"
                    )
                output = stage / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(
                        f"Could not read pinned archive member: {relative}"
                    )
                output.write_bytes(source.read())
        verify_google_source(stage)
        (stage / "SOURCE-PROVENANCE.json").write_text(
            json.dumps(
                {
                    "repository": GOOGLE_PIN["repository"],
                    "commit": GOOGLE_PIN["commit"],
                    "archive_url": url,
                    "verified_files": GOOGLE_PIN["files"],
                    "license": GOOGLE_PIN["license"],
                    "notice_files": GOOGLE_PIN.get("notice_files", []),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            stage.rename(target)
        except FileExistsError:
            verify_google_source(target)
        return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="external cache directory (nothing downloads at import/test time)",
    )
    args = parser.parse_args()
    print(fetch(args.cache))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
