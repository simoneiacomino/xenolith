"""Build the native session driver from an explicitly selected Xenolith tree.

The reference archive is independent of this receipt: rebuilding the candidate
never regenerates or changes the reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import shlex
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(argv, cwd):
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def source_files(source):
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).split(b"\0")
    selected = {
        source / os.fsdecode(p)
        for p in paths
        if p
        and (
            Path(os.fsdecode(p)).suffix in {".c", ".h", ".cl"}
            or os.fsdecode(p) == "Makefile"
        )
    }
    # Include not-yet-committed driver and diagnostic headers in development.
    selected.update((source / "tools/quality/native").glob("*.c"))
    selected.update((source / "tests/support").glob("*.h"))
    return sorted(p for p in selected if p.is_file())


def prepare(source, model, output, *, make="make", jobs=2):
    source, model, output = (
        Path(source).resolve(),
        Path(model).resolve(),
        Path(output).resolve(),
    )
    if output.exists():
        raise ValueError("Backend receipt already exists; use a new output path")
    if not (source / "xenolith.c").is_file() or not model.is_file():
        raise ValueError("Provide the Xenolith source tree and an existing GGUF")
    if not 1 <= jobs <= 256:
        raise ValueError("jobs must be between 1 and 256")
    output.parent.mkdir(parents=True, exist_ok=True)
    commit_before = command(["git", "rev-parse", "HEAD"], source).strip()
    diff_before = subprocess.check_output(
        ["git", "diff", "HEAD", "--binary"], cwd=source
    )
    build_info = command([make, "--no-print-directory", "quality-build-info"], source)
    before = {str(p): sha(p) for p in source_files(source)}
    started = time.monotonic()
    argv = [make, "-B", "-j", str(jobs), "quality-driver"]
    log = output.with_suffix(output.suffix + ".build.log")
    with log.open("xb") as stream:
        subprocess.run(
            argv, cwd=source, stdout=stream, stderr=subprocess.STDOUT, check=True
        )
    after = {str(p): sha(p) for p in source_files(source)}
    if (
        after != before
        or command(["git", "rev-parse", "HEAD"], source).strip() != commit_before
        or subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=source)
        != diff_before
    ):
        raise ValueError(
            "Source changed during build; keep log and rebuild to a new receipt"
        )
    driver = source / "tools/quality/native/xenolith_driver"
    pins = dict(after)
    pins[str(driver)] = sha(driver)
    pins[str(source / "xenolith_gpu.spv")] = sha(source / "xenolith_gpu.spv")
    libraries = command(["ldd", str(driver)], source)
    for line in libraries.splitlines():
        for part in line.split():
            if part.startswith("/") and Path(part).is_file():
                pins[part] = sha(part)
    diff = diff_before
    compiler = next(
        line[3:] for line in build_info.splitlines() if line.startswith("CC=")
    )
    diff_path = output.with_suffix(output.suffix + ".diff")
    with diff_path.open("xb") as stream:
        stream.write(diff)
    metadata = {
        "schema": "xenolith-quality-backend-v1",
        "backend": "Xenolith current build",
        "weights_sha256": sha(model),
        "driver": str(driver),
        "driver_sha256": sha(driver),
        "input_pins": pins,
        "source_commit": commit_before,
        "source_status": command(["git", "status", "--porcelain"], source),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "build_argv": argv,
        "build_log_sha256": sha(log),
        "build_options": build_info,
        "build_seconds": time.monotonic() - started,
        "compiler": command([*shlex.split(compiler), "--version"], source),
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "linked_libraries": libraries,
        },
        "precision": "Engine arithmetic; F16 KV; native session API; eight CPU workers",
        "cache": "Each case resets its own session; chunks are explicitly declared",
        "logits": "Complete post-softcap pre-sampler vocabulary; no sampling",
    }
    # Never replace an existing receipt, including a concurrent creator's file.
    with output.open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Current Xenolith checkout")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="New backend receipt JSON")
    parser.add_argument("--make", default="make")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        data = prepare(
            args.source, args.model, args.output, make=args.make, jobs=args.jobs
        )
        print(
            json.dumps(
                {
                    "outcome": "BUILT",
                    "driver": data["driver"],
                    "driver_sha256": data["driver_sha256"],
                }
            )
        )
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"outcome": "ERROR", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
