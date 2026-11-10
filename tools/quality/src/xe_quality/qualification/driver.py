"""Run pinned Google and independent HF controls in their own environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import numpy as np

from xe_quality.reference import GOOGLE_PIN
from xe_quality.reference.source_pin import verify_google_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_files(root: Path) -> dict[str, dict[str, object]]:
    """Pin every regular qualification artifact under its output directory."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"qualification output is not a directory: {root}")
    evidence = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"qualification evidence contains a symlink: {path}")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"qualification evidence is not a regular file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative == "QUALIFICATION.json":
            continue
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError(
                f"qualification evidence changed while hashing: {relative}"
            )
        evidence[relative] = {"bytes": after.st_size, "sha256": digest}
    return evidence


_PACKAGE_PROBE = r"""
import hashlib, importlib.util, json, pathlib, sys
required_modules = json.loads(sys.argv[1])
spec = importlib.util.find_spec("xe_quality")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("installed xe_quality package is not discoverable")
root = pathlib.Path(next(iter(spec.submodule_search_locations)))
prefix = pathlib.Path(sys.prefix)
if sys.prefix == sys.base_prefix:
    raise RuntimeError("selected interpreter is not an isolated virtual environment")
try:
    root.resolve().relative_to(prefix.resolve())
except ValueError as exc:
    raise RuntimeError("xe_quality is not installed inside the selected environment") from exc
files = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink() and path.suffix in (".py", ".json"):
        raise RuntimeError("maintained package files may not be symlinks")
    if path.is_file() and path.suffix in (".py", ".json"):
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
canonical = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing_modules:
    print(json.dumps({"schema": "xe-quality-package-probe-v1",
                      "missing_modules": missing_modules}, sort_keys=True))
    sys.exit(3)
print(json.dumps({
    "schema": "xe-quality-package-probe-v1",
    "python_executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "package_root": str(root),
    "package_files": files,
    "package_sha256": hashlib.sha256(canonical).hexdigest(),
}, sort_keys=True))
"""


def _package_manifest(package_root: Path) -> dict:
    """Hash all maintained Python/JSON files in the caller's public package."""
    root = package_root.expanduser().resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and path.suffix in (".py", ".json"):
            raise ValueError(f"maintained package file is a symlink: {path}")
        if path.is_file() and path.suffix in (".py", ".json"):
            files[path.relative_to(root).as_posix()] = _sha256(path)
    if not files:
        raise ValueError(f"no maintained Python/JSON package files under {root}")
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "package_root": str(root),
        "package_files": files,
        "package_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _clean_python_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def _probe_interpreter(
    python: Path,
    expected: dict,
    timeout: int = 30,
    required_modules: tuple[str, ...] = (),
) -> dict:
    """Locate and hash the installed package in an isolated selected venv."""
    selected = _interpreter_path(python)
    result = subprocess.run(
        [
            str(selected),
            "-I",
            "-c",
            _PACKAGE_PROBE,
            json.dumps(required_modules, separators=(",", ":")),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_clean_python_environment(),
        timeout=timeout,
    )
    if result.returncode:
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        missing = payload.get("missing_modules")
        if missing:
            reason = f"missing required modules: {', '.join(missing)}"
        else:
            detail = (result.stderr or result.stdout).strip()
            reason = detail[:1200] if detail else f"probe exited {result.returncode}"
        raise RuntimeError(f"preflight probe failed for {selected}: {reason}")
    probe = json.loads(result.stdout)
    if probe.get("schema") != "xe-quality-package-probe-v1":
        raise ValueError(f"invalid package probe response from {selected}")
    actual_files = probe.get("package_files", {})
    expected_files = expected["package_files"]
    if (
        actual_files != expected_files
        or probe.get("package_sha256") != expected["package_sha256"]
    ):
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        changed = sorted(
            name
            for name in set(expected_files) & set(actual_files)
            if expected_files[name] != actual_files[name]
        )
        raise ValueError(
            f"selected package differs at {selected}: missing={missing[:8]}, "
            f"extra={extra[:8]}, changed={changed[:8]}"
        )
    probe["selected_executable"] = str(selected)
    probe["caller_package_sha256"] = expected["package_sha256"]
    return probe


def _run_worker(
    python: Path,
    kind: str,
    model: Path,
    input_path: Path,
    output: Path,
    source: Path,
    hf_config: Path,
    timeout: int,
    verified_probe: dict,
) -> dict:
    worker_output = output / kind
    selected = _interpreter_path(python)
    if verified_probe.get("selected_executable") != str(selected):
        raise ValueError(f"worker environment was not preflighted: {selected}")
    if verified_probe.get("package_sha256") != verified_probe.get(
        "caller_package_sha256"
    ):
        raise ValueError(f"worker package identity was not verified: {selected}")
    env = _clean_python_environment()
    env.update(
        {"JAX_PLATFORMS": "cpu", "OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "8"}
    )
    argv = [
        str(selected),
        "-I",
        "-m",
        "xe_quality.qualification.worker",
        "--kind",
        kind,
        "--model",
        str(model),
        "--input",
        str(input_path),
        "--output",
        str(worker_output),
    ]
    if kind == "google":
        argv += ["--source", str(source)]
    else:
        argv += ["--hf-config", str(hf_config)]
    started = time.monotonic()
    with (
        (output / f"{kind}.stdout.log").open("xb") as stdout,
        (output / f"{kind}.stderr.log").open("xb") as stderr,
    ):
        subprocess.run(
            argv, check=True, stdout=stdout, stderr=stderr, env=env, timeout=timeout
        )
    report = json.loads((worker_output / "WORKER.json").read_text(encoding="utf-8"))
    if report.get("worker_sha256") != _sha256(Path(__file__).with_name("worker.py")):
        raise ValueError(
            "Selected environment has a different qualification worker; install this public package in both environments"
        )
    report["process_elapsed_seconds"] = time.monotonic() - started
    return report


def _compare_routes(google_path: Path, hf_path: Path) -> dict:
    google = json.loads(google_path.read_text(encoding="utf-8"))
    hf = json.loads(hf_path.read_text(encoding="utf-8"))

    def by_layer(rows):
        result = {}
        for item in rows:
            layer = item.get("layer")
            choices = item.get("choices")
            if (
                type(layer) is not int
                or layer in result
                or not isinstance(choices, list)
            ):
                raise ValueError("Malformed or duplicate router trace layer")
            values = np.asarray(choices, dtype=np.int64)
            if values.ndim == 3 and values.shape[0] == 1:
                values = values[0]
            if values.ndim != 2:
                raise ValueError("Router trace choices must have shape [tokens, top_k]")
            result[layer] = values
        return result

    g, h = by_layer(google), by_layer(hf)
    if set(g) != set(range(30)) or set(h) != set(range(30)):
        raise ValueError("Router traces must contain all 30 layers")
    mismatches = []
    for layer in range(30):
        if g[layer].shape != h[layer].shape:
            mismatches.append(
                {
                    "layer": layer,
                    "shape_mismatch": [list(g[layer].shape), list(h[layer].shape)],
                }
            )
        else:
            count = int(np.count_nonzero(g[layer] != h[layer]))
            if count:
                mismatches.append(
                    {
                        "layer": layer,
                        "slot_mismatches": count,
                        "set_mismatch_rows": int(
                            np.count_nonzero(
                                np.any(
                                    np.sort(g[layer], axis=-1)
                                    != np.sort(h[layer], axis=-1),
                                    axis=-1,
                                )
                            )
                        ),
                    }
                )
    return {"layers": 30, "mismatches": mismatches, "passed": not mismatches}


def _compare_layers(google_dir: Path, hf_dir: Path, gate: float) -> list[dict]:
    rows = []
    for layer in range(30):
        name = f"layer-{layer:02d}.npy"
        google = np.load(google_dir / "layers" / name, allow_pickle=False).astype(
            np.float64
        )
        hf = np.load(hf_dir / name, allow_pickle=False).astype(np.float64)
        if (
            google.shape != hf.shape
            or not np.isfinite(google).all()
            or not np.isfinite(hf).all()
        ):
            raise ValueError(f"Layer {layer} shape or finite-value check failed")
        delta = google - hf
        scale = max(1.0, float(np.sqrt(np.mean(hf * hf))))
        maximum = float(np.max(np.abs(delta)))
        relative = float(np.linalg.norm(delta) / max(1.0, float(np.linalg.norm(hf))))
        rows.append(
            {
                "layer": layer,
                "shape": list(google.shape),
                "max_abs": maximum,
                "relative_l2": relative,
                "absolute_gate": gate * scale,
                "passed": maximum <= gate * scale,
            }
        )
    return rows


def _interpreter_path(path: Path) -> Path:
    """Keep the venv launcher path: resolving its symlink loses pyvenv.cfg."""
    return path.expanduser().absolute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="exact Gemma 4 GGUF weights"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON object with fixed `tokens` and selected `positions`",
    )
    parser.add_argument(
        "--google-source",
        type=Path,
        required=True,
        help="external cache returned by fetch-google",
    )
    parser.add_argument(
        "--google-python",
        type=Path,
        required=True,
        help="Python executable from the pinned JAX/Flax environment",
    )
    parser.add_argument(
        "--hf-python",
        type=Path,
        required=True,
        help="Python executable from the independent Torch/Transformers environment",
    )
    parser.add_argument(
        "--hf-config",
        type=Path,
        default=Path(__file__).parents[1] / "reference" / "HF-config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new evidence directory; existing paths are never overwritten",
    )
    parser.add_argument("--timeout-seconds", type=int, default=8 * 60 * 60)
    args = parser.parse_args(argv)
    model = args.model.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    source = args.google_source.expanduser().resolve()
    google_python = _interpreter_path(args.google_python)
    hf_python = _interpreter_path(args.hf_python)
    hf_config = args.hf_config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not model.is_file() or not input_path.is_file() or not hf_config.is_file():
        parser.error("model, fixed input JSON and HF config must all exist")
    for executable in (google_python, hf_python):
        if not executable.is_file():
            parser.error(f"Python executable does not exist: {executable}")
    request = json.loads(input_path.read_text(encoding="utf-8"))
    tokens, positions = request.get("tokens"), request.get("positions")
    if not isinstance(tokens, list) or not 2 <= len(tokens) <= 512:
        parser.error("fixed input `tokens` must contain between 2 and 512 IDs")
    if any(
        not isinstance(token, int) or isinstance(token, bool) or not 0 <= token < 262144
        for token in tokens
    ):
        parser.error("fixed input contains a non-integer or out-of-range token")
    if (
        not isinstance(positions, list)
        or not positions
        or any(
            not isinstance(pos, int)
            or isinstance(pos, bool)
            or not 0 <= pos < len(tokens)
            for pos in positions
        )
    ):
        parser.error("fixed input `positions` must select one or more token indices")
    if positions != sorted(set(positions)):
        parser.error("fixed input `positions` must be unique and sorted")
    if positions[-1] != len(tokens) - 1:
        parser.error("qualification must select the final token for cache continuation")
    verify_google_source(source)
    if output.exists():
        parser.error(f"evidence output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    model_sha = _sha256(model)
    input_file_sha = _sha256(input_path)
    input_protocol = "reset_full_history"
    input_spec = {
        "tokens": tokens,
        "positions": positions,
        "input_state_protocol": input_protocol,
    }
    input_spec_bytes = json.dumps(
        input_spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    input_spec_sha = hashlib.sha256(input_spec_bytes).hexdigest()
    input_copy = output / "INPUT.json"
    input_copy.write_bytes(input_path.read_bytes())
    caller_package = _package_manifest(Path(__file__).resolve().parents[1])
    preflight_path = output / "PREFLIGHT.json"
    preflight = {
        "schema": "xe-quality-reference-preflight-v1",
        "status": "running",
        "caller": caller_package,
        "children": {},
    }
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    failures = {}
    required_by_role = {
        "google": ("numpy", "jax", "jaxlib", "flax"),
        "hf": ("numpy", "torch", "transformers", "tokenizers", "safetensors"),
    }
    for role, executable in (("google", google_python), ("hf", hf_python)):
        try:
            probe = _probe_interpreter(
                executable, caller_package, required_modules=required_by_role[role]
            )
            preflight["children"][role] = {"status": "pass", **probe}
        except Exception as error:
            profile = "reference" if role == "google" else "qualification"
            setup = (
                "run tools/quality/setup_environment.py "
                f"--profile {profile} --output NEW_VENV"
            )
            failures[role] = f"{type(error).__name__}: {error}; {setup}"
            preflight["children"][role] = {
                "status": "fail",
                "selected_executable": str(executable),
                "error": failures[role],
            }
    if failures:
        preflight["status"] = "fail"
        preflight["failures"] = failures
        preflight_path.write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(f"selected Python environment preflight failed: {failures}")
    preflight["status"] = "pass"
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    google_report = _run_worker(
        google_python,
        "google",
        model,
        input_copy,
        output,
        source,
        hf_config,
        args.timeout_seconds,
        preflight["children"]["google"],
    )
    hf_report = _run_worker(
        hf_python,
        "hf",
        model,
        input_copy,
        output,
        source,
        hf_config,
        args.timeout_seconds,
        preflight["children"]["hf"],
    )
    if _sha256(model) != model_sha or _sha256(input_copy) != input_file_sha:
        raise ValueError("Model or fixed input changed during the qualification run")
    numerics_path = Path(__file__).parents[1] / "reference" / "NUMERICS.json"
    numerics = json.loads(numerics_path.read_text(encoding="utf-8"))
    layer_gate = numerics["qualification"]["fp64_layer_absolute"].split(" * ")[0]
    layer_gate_value = float(layer_gate)
    logits_gate = float(numerics["qualification"]["fp64_logits_absolute"])
    trajectory_gate = float(numerics["qualification"]["fp64_trajectory_relative_l2"])
    routes = _compare_routes(
        output / "google" / "ROUTES.json", output / "hf" / "ROUTES.json"
    )
    layers = _compare_layers(output / "google", output / "hf", layer_gate_value)
    google_logits = np.load(
        output / "google" / "logits.npy", allow_pickle=False
    ).astype(np.float64)
    hf_logits = np.load(output / "hf" / "logits.npy", allow_pickle=False).astype(
        np.float64
    )
    if (
        google_logits.shape != hf_logits.shape
        or not np.isfinite(google_logits).all()
        or not np.isfinite(hf_logits).all()
    ):
        raise ValueError("Logit shape or finite-value check failed")
    delta = google_logits - hf_logits
    logit_abs = float(np.max(np.abs(delta)))
    logit_relative = float(
        np.linalg.norm(delta) / max(1.0, float(np.linalg.norm(hf_logits)))
    )
    logits_passed = logit_abs <= logits_gate and logit_relative <= trajectory_gate
    google_ops = google_report["extra"]["operator_controls"]
    google_state = google_report["extra"]["state_control"]
    passed = (
        routes["passed"]
        and google_ops.get("passed") is True
        and google_state.get("status") == "pass"
        and google_report["extra"]["runtime_bridge"].get("status") == "pass"
        and all(row["passed"] for row in layers)
        and logits_passed
    )
    qualification = {
        "schema": "xe-quality-reference-qualification-v1",
        "procedure": {
            "driver_sha256": _sha256(Path(__file__)),
            "worker_sha256": _sha256(Path(__file__).with_name("worker.py")),
            "numerics_sha256": _sha256(numerics_path),
        },
        "status": "QUALIFIED_SCOPED" if passed else "FAIL",
        "meaning": "Empirical Google/HF FP64 adapter agreement for this exact model, inputs, sources and runtimes; not a global numerical proof.",
        "candidate_acceptance_policy": None,
        "model_sha256": model_sha,
        "weights_sha256": model_sha,
        "implementation": google_report["extra"]["code_identity"],
        "runtime": google_report["extra"]["code_identity"]["runtime"],
        "runtime_info": google_report["extra"]["runtime_info"],
        "maximum_supported_length": len(tokens),
        "supported_domains": [],
        "scope": {
            "schema": "xe-quality-reference-scope-v1",
            "domain_class": "fixed-token-control",
            "input_hash": input_spec_sha,
            "input_file_sha256": input_file_sha,
            "input_length": len(tokens),
            "positions": positions,
            "state_protocol": input_protocol,
            "input_state_protocol": input_protocol,
            "qualification_state_controls": ["full_prefill_cache_restore_reset_repeat"],
            "allowed_prefill_chunks": [
                [len(tokens)],
                [len(tokens) // 2, len(tokens) - len(tokens) // 2],
            ],
            "maximum_supported_length": len(tokens),
            "supported_domains": [],
        },
        "fixed_input_sha256": input_file_sha,
        "input_spec": input_spec,
        "input_spec_sha256": input_spec_sha,
        "tokens": len(tokens),
        "positions": positions,
        "google_pin": {
            "repository": GOOGLE_PIN["repository"],
            "commit": GOOGLE_PIN["commit"],
            "files": GOOGLE_PIN["files"],
            "license": GOOGLE_PIN["license"],
        },
        "google_worker": google_report,
        "hf_worker": hf_report,
        "numerical_gates": {
            "layer_max_abs": layer_gate_value,
            "logit_max_abs": logits_gate,
            "logit_relative_l2": trajectory_gate,
        },
        "router_agreement": routes,
        "operator_controls": google_ops,
        "state_control": google_state,
        "runtime_bridge": google_report["extra"]["runtime_bridge"],
        "layers": layers,
        "logits": {
            "shape": list(google_logits.shape),
            "max_abs": logit_abs,
            "relative_l2": logit_relative,
            "passed": logits_passed,
        },
        "limitations": [
            "The comparison uses the selected exact GGUF importer on both sides; independent import controls remain a separate requirement.",
            "The run qualifies only this model, fixed token histories, selected positions, source pins and runtime identities.",
            "A scoped reference comparison does not set acceptance thresholds for Xenolith candidate outputs.",
            "HF source hashes cover the pinned Gemma 4 modeling/config modules, not every framework binary.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    qualification["evidence_files"] = _evidence_files(output)
    (output / "QUALIFICATION.json").write_text(
        json.dumps(qualification, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": qualification["status"],
                "output": str(output / "QUALIFICATION.json"),
                "max_abs_logits": logit_abs,
                "max_layer_abs": max(x["max_abs"] for x in layers),
            }
        )
    )
    return 0 if passed else 1


def _entrypoint() -> int:
    try:
        return main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "error": type(error).__name__,
                    "message": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
