#!/usr/bin/env python3
"""Explicit Linux x86_64 / Python 3.12 installation from hashed profile locks.

Creates a NEW virtual environment. Never called by engine builds or tests.
Run this script with the Python interpreter that should create the environment.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["compare", "reference", "qualification", "corpus"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        sys.version_info[:2] != (3, 12)
        or sys.platform != "linux"
        or platform.machine() != "x86_64"
    ):
        parser.error(
            "Tested locks require Linux x86_64 and Python 3.12; use Python 3.12 to run this helper. Other platforms may install the pyproject dependencies explicitly and must qualify their own runtime."
        )
    output = args.output.resolve()
    if output.exists():
        parser.error("Output must not exist; choose a new virtual environment path.")
    package = Path(__file__).resolve().parent
    lock = package / "locks" / (args.profile + "-py312-linux-x86_64.txt")
    venv.EnvBuilder(with_pip=True).create(output)
    python = output / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary=:all:",
            "-r",
            str(lock),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            str(package),
        ],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    subprocess.run(
        [str(python), "-c", "import xe_quality, numpy; print(numpy.__version__)"],
        check=True,
    )
    if args.profile == "reference":
        subprocess.run(
            [
                str(python),
                "-c",
                'import jax,flax; assert jax.default_backend()=="cpu"; print(jax.__version__, flax.__version__)',
            ],
            check=True,
            env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"},
        )
    if args.profile == "qualification":
        subprocess.run(
            [
                str(python),
                "-c",
                "import torch,transformers; print(torch.__version__,transformers.__version__)",
            ],
            check=True,
        )
    if args.profile == "corpus":
        subprocess.run(
            [str(python), "-c", "import nltk,wonderwords,tqdm; print(nltk.__version__,tqdm.__version__)"],
            check=True,
        )
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (output / "QUALITY-INSTALL.json").write_text(
        json.dumps(
            {
                "profile": args.profile,
                "python": sys.version,
                "platform": platform.platform(),
                "lock_sha256": __import__("hashlib")
                .sha256(lock.read_bytes())
                .hexdigest(),
                "freeze": freeze,
            },
            indent=2,
        )
        + "\n"
    )
    print("Installed", args.profile, "environment at", output)
    print("Installation does not qualify a reference or accept a candidate.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.SubprocessError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(2)
