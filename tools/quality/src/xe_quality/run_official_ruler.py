"""Generate a small official RULER NIAH noise corpus using an exact GGUF tokenizer.

This creates prompt data only. It does not run model inference or claim RULER task
scores. The upstream source, GGUF model, tokenizer executable, and output directory
must all be supplied explicitly; no source or model is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
import types
from typing import Sequence


UPSTREAM_COMMIT = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
MAX_TOKENIZER_CALLS = 128
WORKER_TIMEOUT_SECONDS = 8 * 60
EXECUTED_PYTHON = (
    "scripts/data/synthetic/niah.py",
    "scripts/data/synthetic/constants.py",
    "scripts/data/manifest_utils.py",
)
OPTIONAL_DEPENDENCIES = {
    "nltk": "3.9.1",
    "wonderwords": "2.2.0",
    "tqdm": "4.67.1",
}


class RulerGenerationError(ValueError):
    """The upstream source, optional runtime, or generated output is invalid."""


def _abort_upstream(reason: str) -> None:
    """Stop the isolated worker despite NIAH's bare-except retry loop."""
    # The pinned upstream uses `except:` around token counting, which catches
    # BaseException too. A raised sentinel would be swallowed, so terminate only
    # this disposable worker with a recognizable nonzero status.
    os.write(2, ("XE_QUALITY_RULER_ABORT: " + reason + "\n").encode("utf-8", "replace"))
    os._exit(75)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _git(source: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RulerGenerationError(
            f"cannot verify RULER Git source at {source}: {exc}"
        ) from exc
    return result.stdout.rstrip(b"\n")


def _git_show(source: Path, relative: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(source), "show", f"HEAD:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RulerGenerationError(
            f"cannot read committed RULER Python source {relative}: {exc}"
        ) from exc


def verify_upstream_source(source: str | Path) -> dict[str, str]:
    """Require the exact RULER revision and byte-identical tracked Python closure."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise RulerGenerationError(f"RULER source checkout does not exist: {source}")
    try:
        top = Path(os.fsdecode(_git(source, "rev-parse", "--show-toplevel"))).resolve()
        head = os.fsdecode(_git(source, "rev-parse", "HEAD")).strip()
    except (OSError, ValueError) as exc:
        raise RulerGenerationError(f"invalid RULER source checkout: {exc}") from exc
    if top != source:
        raise RulerGenerationError(
            f"--source must name the RULER Git checkout root ({top})"
        )
    if head != UPSTREAM_COMMIT:
        raise RulerGenerationError(
            f"RULER source HEAD must be {UPSTREAM_COMMIT}, got {head}"
        )

    pins: dict[str, str] = {}
    for relative in EXECUTED_PYTHON:
        tracked = _git(source, "ls-files", "--error-unmatch", "--", relative)
        if tracked.decode("utf-8") != relative:
            raise RulerGenerationError(f"RULER Python source is not tracked: {relative}")
        path = source / relative
        current = source
        linked_parent = False
        for part in PurePosixPath(relative).parts:
            current = current / part
            linked_parent = linked_parent or current.is_symlink()
        if linked_parent or not path.is_file():
            raise RulerGenerationError(f"RULER Python source is missing or linked: {relative}")
        working = path.read_bytes()
        committed = _git_show(source, relative)
        if working != committed:
            raise RulerGenerationError(
                f"RULER executed Python source differs from pinned Git bytes: {relative}"
            )
        pins[relative] = hashlib.sha256(working).hexdigest()
    return pins


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    missing = []
    for package, expected in OPTIONAL_DEPENDENCIES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"{package}=={expected}")
            continue
        if actual != expected:
            raise RulerGenerationError(
                f"{package} must be {expected}, found {actual}; use the quality corpus profile"
            )
        versions[package] = actual
    if missing:
        raise RulerGenerationError(
            "official RULER generation needs optional packages "
            + ", ".join(missing)
            + "; install tools/quality with the corpus profile"
        )
    try:
        versions["numpy"] = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RulerGenerationError(
            "official RULER generation needs NumPy; install tools/quality with the corpus profile"
        ) from exc
    return versions


def _regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise RulerGenerationError(f"{label} must be a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise RulerGenerationError(f"{label} must be executable: {path}")
    return path


class _GGUFTokenizer:
    """Explicit adapter for the tokenizer CLI used by the public panel tool."""

    def __init__(self, executable: Path, model: Path, temp_root: Path):
        self.executable = executable
        self.model = model
        self.temp_root = temp_root
        self.calls = 0
        self.max_calls = MAX_TOKENIZER_CALLS

    def text_to_tokens(self, text: str) -> list[int]:
        self.calls += 1
        if self.calls > self.max_calls:
            _abort_upstream(
                f"RULER tokenizer call budget exceeded ({self.max_calls}); "
                "generation may be impossible at the selected length"
            )
        with tempfile.TemporaryDirectory(prefix="ruler-tokenizer-", dir=self.temp_root) as tmp:
            text_path = Path(tmp) / "text.txt"
            text_path.write_bytes(text.encode("utf-8"))
            try:
                result = subprocess.run(
                    [str(self.executable), str(self.model), str(text_path)],
                    check=False,
                    timeout=60,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.TimeoutExpired:
                _abort_upstream("GGUF tokenizer timed out after 60 seconds")
            except OSError as exc:
                _abort_upstream(f"cannot run GGUF tokenizer: {exc}")
            if result.returncode:
                stderr = result.stderr.decode("utf-8", "replace")[-2000:]
                _abort_upstream(
                    f"GGUF tokenizer exited {result.returncode}: {stderr}"
                )
            try:
                output = result.stdout.decode("ascii").strip()
                tokens = [int(part) for part in output.split(",")]
            except (UnicodeDecodeError, ValueError):
                _abort_upstream("GGUF tokenizer must emit comma-separated integer IDs")
            if not output or any(token < 0 for token in tokens):
                _abort_upstream("GGUF tokenizer emitted empty or negative IDs")
            return tokens

    def tokens_to_text(self, tokens: Sequence[int]) -> str:
        raise RulerGenerationError(
            "official NIAH noise generation requires token counts only; detokenization is unsupported"
        )


def _worker(source: Path, tokenizer: Path, model: Path, output: Path, length: int, samples: int, seed: int) -> None:
    """Run the pinned script in a child process with only the explicit adapter."""
    synthetic = source / "scripts/data/synthetic"
    data_scripts = source / "scripts/data"
    sys.path.insert(0, str(data_scripts))
    sys.path.insert(0, str(synthetic))

    module = types.ModuleType("tokenizer")

    def select_tokenizer(kind, path):
        if kind != "gguf" or Path(path).resolve() != model:
            raise RulerGenerationError("upstream tokenizer request differs from explicit GGUF inputs")
        return _GGUFTokenizer(tokenizer, model, output)

    module.select_tokenizer = select_tokenizer
    sys.modules["tokenizer"] = module

    import runpy

    constants = runpy.run_path(str(synthetic / "constants.py"))
    task = constants["TASKS"]["niah"]
    template = task["template"] + task["answer_prefix"]
    sys.argv = [
        str(synthetic / "niah.py"),
        "--save_dir", str(output),
        "--save_name", "niah_single_1",
        "--subset", "validation",
        "--tokenizer_path", str(model),
        "--tokenizer_type", "gguf",
        "--max_seq_length", str(length),
        "--tokens_to_generate", str(task["tokens_to_generate"]),
        "--num_samples", str(samples),
        "--random_seed", str(seed),
        "--template", template,
        "--type_haystack", "noise",
        "--type_needle_k", "words",
        "--type_needle_v", "numbers",
        "--num_needle_k", "1",
        "--num_needle_v", "1",
        "--num_needle_q", "1",
    ]
    runpy.run_path(str(synthetic / "niah.py"), run_name="__main__")


def _run_worker(source: Path, tokenizer: Path, model: Path, output: Path, length: int, samples: int, seed: int) -> None:
    command = [
        sys.executable,
        "-I",
        "-m", "xe_quality.run_official_ruler",
        "--_worker",
        "--source", str(source),
        "--tokenizer", str(tokenizer),
        "--model", str(model),
        "--output", str(output),
        "--length", str(length),
        "--num-samples", str(samples),
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=WORKER_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RulerGenerationError(
            f"official NIAH generator exceeded {WORKER_TIMEOUT_SECONDS} seconds"
        ) from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace")[-4000:]
        raise RulerGenerationError(
            f"official NIAH child exited {result.returncode}: {detail}"
        )


def _wordlist_pins() -> dict[str, str]:
    try:
        import wonderwords
    except ImportError as exc:
        raise RulerGenerationError(
            "official RULER generation needs wonderwords==2.2.0; install the quality corpus profile"
        ) from exc
    package_root = Path(wonderwords.__file__).resolve().parent
    pins = {}
    for name in ("nounlist.txt", "adjectivelist.txt"):
        matches = list(package_root.rglob(name))
        if len(matches) != 1 or not matches[0].is_file() or matches[0].is_symlink():
            raise RulerGenerationError(f"wonderwords word list is missing or linked: {name}")
        path = matches[0]
        pins[path.relative_to(package_root).as_posix()] = file_sha256(path)
    return pins


def generate(
    source: str | Path,
    tokenizer: str | Path,
    model: str | Path,
    output: str | Path,
    *,
    length: int = 512,
    num_samples: int = 2,
    seed: int = 20261104,
) -> dict:
    if type(length) is not int or not 256 <= length <= 32768:
        raise RulerGenerationError("length must be in the explicit range 256..32768 tokens")
    if type(num_samples) is not int or not 2 <= num_samples <= 16:
        raise RulerGenerationError("num_samples must be in the explicit range 2..16")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise RulerGenerationError("seed must be an integer in range 0..2^32-1")

    source_path = Path(source).resolve(strict=True)
    source_pins = verify_upstream_source(source_path)
    tokenizer_path = _regular_file(Path(tokenizer), "tokenizer executable", executable=True)
    model_path = _regular_file(Path(model), "GGUF model")
    dependencies = dependency_versions()
    wordlist_pins = _wordlist_pins()
    tokenizer_sha256 = file_sha256(tokenizer_path)
    model_sha256 = file_sha256(model_path)

    output_path = Path(output).absolute()
    if output_path.exists() or output_path.is_symlink():
        raise RulerGenerationError(f"output must be a new directory: {output_path}")
    for ancestor in output_path.parents:
        if ancestor.is_symlink():
            raise RulerGenerationError(f"output path contains a symlink: {ancestor}")
    normalized_output = output_path.resolve()
    if (
        output_path in (source_path, tokenizer_path, model_path)
        or normalized_output.is_relative_to(source_path)
    ):
        raise RulerGenerationError(
            "output must be a new directory outside the source, model, and tokenizer"
        )

    output_path.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        _run_worker(
            source_path, tokenizer_path, model_path, output_path,
            length, num_samples, seed,
        )
        if verify_upstream_source(source_path) != source_pins:
            raise RulerGenerationError("RULER executed Python bytes changed during generation")
        if file_sha256(tokenizer_path) != tokenizer_sha256:
            raise RulerGenerationError("tokenizer executable changed during generation")
        if file_sha256(model_path) != model_sha256:
            raise RulerGenerationError("GGUF model changed during generation")
        if _wordlist_pins() != wordlist_pins:
            raise RulerGenerationError("wonderwords word lists changed during generation")
        generated = sorted(output_path.rglob("*.jsonl"))
        if not generated:
            raise RulerGenerationError("official NIAH generator produced no JSONL files")
        output_pins = {
            path.relative_to(output_path).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in generated
        }
        report = {
            "schema": "xe-quality-official-ruler-generation-v1",
            "status": "official NIAH noise prompt data generated; no benchmark score",
            "upstream_commit": UPSTREAM_COMMIT,
            "executed_python_sha256": source_pins,
            "source_pins": source_pins,
            "task": "niah_single_1",
            "subset": "validation",
            "haystack": "noise",
            "num_samples": num_samples,
            "seed": seed,
            "max_sequence_length": length,
            "tokens_to_generate": 128,
            "tokenizer_executable_sha256": tokenizer_sha256,
            "model_sha256": model_sha256,
            "dependencies": dependencies,
            "wordlist_sha256": wordlist_pins,
            "output_pins": output_pins,
            "elapsed_seconds": time.monotonic() - start,
            "limits": "Official NIAH noise subset input generation only; not full RULER generation/evaluation and no task score claim.",
        }
        receipt = output_path / "GENERATION.json"
        with receipt.open("xb") as stream:
            stream.write(json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
        return report
    except BaseException:
        shutil.rmtree(output_path)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="pinned RULER Git checkout root")
    parser.add_argument("--tokenizer", required=True, type=Path, help="GGUF tokenizer executable")
    parser.add_argument("--model", required=True, type=Path, help="exact GGUF model file")
    parser.add_argument("--output", required=True, type=Path, help="new directory for JSONL and receipt")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._worker:
        _worker(
            args.source.resolve(strict=True),
            args.tokenizer.resolve(strict=True),
            args.model.resolve(strict=True),
            args.output.resolve(strict=True),
            args.length,
            args.num_samples,
            args.seed,
        )
        return 0
    report = generate(
        args.source,
        args.tokenizer,
        args.model,
        args.output,
        length=args.length,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RulerGenerationError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
