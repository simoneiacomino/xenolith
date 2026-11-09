"""Child-process entry point for xe_quality.qualification.driver."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from pathlib import PurePosixPath
import platform
import sys
import time

import numpy as np


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_HF_RUNTIME_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "numpy",
)


def _hf_runtime_inventory() -> dict:
    """Fingerprint HF runtime sources and native libraries without absolute paths."""
    distributions = {}
    for name in _HF_RUNTIME_DISTRIBUTIONS:
        distribution = importlib.metadata.distribution(name)
        files = {}
        for item in sorted(distribution.files or [], key=lambda value: str(value)):
            relative = str(item).replace("\\", "/")
            relative_path = PurePosixPath(relative)
            filename = relative_path.name
            if not (filename.endswith(".py") or ".so" in filename):
                continue
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"unsafe relative path in {name} distribution metadata: {relative}"
                )
            path = Path(distribution.locate_file(item))
            if not path.is_file():
                raise FileNotFoundError(
                    f"installed HF runtime file is missing: {name}/{relative}"
                )
            files[relative_path.as_posix()] = _hash(path)
        distributions[name] = {"version": distribution.version, "files": files}
    identity = {
        "schema": "xe-quality-hf-runtime-inventory-v1",
        "distributions": distributions,
    }
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {**identity, "fingerprint_sha256": hashlib.sha256(canonical).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("google", "hf"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--hf-config", type=Path)
    args = parser.parse_args()
    request = json.loads(args.input.read_text(encoding="utf-8"))
    tokens = np.asarray(request["tokens"])
    positions = request["positions"]
    if tokens.ndim != 1 or len(tokens) < 2 or len(tokens) > 512:
        raise ValueError("Qualification input must contain 2..512 exact token IDs")
    if not np.issubdtype(tokens.dtype, np.integer) or np.issubdtype(
        tokens.dtype, np.bool_
    ):
        raise ValueError("Qualification token IDs must be exact integers")
    if (tokens < 0).any() or (tokens >= 262144).any():
        raise ValueError("Qualification token ID outside the Gemma 4 vocabulary")
    if (
        not isinstance(positions, list)
        or not positions
        or any(
            not isinstance(p, int) or isinstance(p, bool) or not 0 <= p < len(tokens)
            for p in positions
        )
    ):
        raise ValueError("Select at least one valid output position")
    if positions != sorted(set(positions)):
        raise ValueError("Output positions must be unique and sorted")
    if positions != sorted(set(positions)) or positions[-1] != len(tokens) - 1:
        raise ValueError(
            "Qualification positions must be sorted, unique and include the final token"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.kind == "google":
        if args.source is None:
            raise ValueError("Google qualification requires the pinned source cache")
        from xe_quality.reference.stream import Reference
        from xe_quality.reference.runtime_info import runtime_info
        from xe_quality.reference.qualify_ops import run as qualify_ops

        ops_report = qualify_ops(args.source, args.output / "OPS.json")
        reference = Reference(args.model, args.source, precision="fp64")
        identity = reference.code_identity()
        exact_runtime = runtime_info()
        logits = reference.forward(
            tokens, positions=positions, trace_dir=args.output / "layers"
        )
        np.save(args.output / "logits.npy", logits, allow_pickle=False)
        (args.output / "ROUTES.json").write_text(
            json.dumps(reference.route_trace, separators=(",", ":")) + "\n"
        )
        route_trace_entries = len(reference.route_trace)
        split = len(tokens) // 2
        capacity = len(tokens)
        before_positions = [p for p in positions if p < split]
        after_positions = [p - split for p in positions if p >= split]
        before_logits = reference.forward(
            tokens[:split], positions=before_positions, cache=True, capacity=capacity
        )
        state_path = args.output / "state-cache.npz"
        reference.save_state(state_path)
        # First continuation consumes the in-memory cache; the second consumes
        # the serialized/restored bytes. Those continuations are the bitwise
        # restore gate. Whole-prefill versus chunked replay has its own measured
        # FP64 numerical allowance below.
        without_restore = reference.forward(
            tokens[split:], positions=after_positions, cache=True, capacity=capacity
        )
        reference.reset()
        reference.load_state(state_path, tokens[:split])
        after_logits = reference.forward(
            tokens[split:], positions=after_positions, cache=True, capacity=capacity
        )
        restore_bitwise = bool(np.array_equal(after_logits, without_restore))
        replay = {}
        for pos, row in zip(before_positions, before_logits):
            replay[pos] = row
        after_global_positions = [p for p in positions if p >= split]
        for pos, row in zip(after_global_positions, after_logits):
            replay[pos] = row
        replay_logits = np.stack([replay[p] for p in positions])
        chunked_abs = float(np.max(np.abs(replay_logits - logits)))
        state_gate = 1e-6
        reference.reset()
        repeated = reference.forward(tokens, positions=positions)
        reset_bitwise = bool(np.array_equal(repeated, logits))
        reset_abs = float(np.max(np.abs(repeated - logits)))
        state_pass = (
            restore_bitwise
            and reset_bitwise
            and chunked_abs <= state_gate
            and reset_abs <= state_gate
        )
        state_report = {
            "status": "pass" if state_pass else "fail",
            "schema_version": 1,
            "cache_dtype": "<f8",
            "logits_dtype": "<f8",
            "cache_restore_bitwise": restore_bitwise,
            "reset_repeat_bitwise": reset_bitwise,
            "whole_vs_chunked_max_abs": chunked_abs,
            "whole_vs_chunked_gate": state_gate,
            "max_abs_reset": reset_abs,
            "prefix_length": split,
            "capacity": capacity,
        }
        (args.output / "STATE64.json").write_text(
            json.dumps(state_report, indent=2, sort_keys=True) + "\n"
        )
        reference.close()
        runtime_post = runtime_info()
        identity_post = reference.code_identity(refresh=True)
        runtime_bridge_passed = (
            exact_runtime == runtime_post and identity == identity_post
        )
        extra = {
            "code_identity": identity,
            "code_identity_post": identity_post,
            "runtime_info": exact_runtime,
            "runtime_info_post": runtime_post,
            "runtime_bridge": {
                "kind": "same-process pre/post unchanged runtime",
                "status": "pass" if runtime_bridge_passed else "fail",
            },
            "route_trace_entries": route_trace_entries,
            "operator_controls": ops_report,
            "state_control": state_report,
        }
    else:
        if args.hf_config is None:
            raise ValueError("HF qualification requires the explicitly pinned config")
        from xe_quality.reference.hf_source_pin import verify_hf_source

        source_pins = verify_hf_source(args.hf_config)
        runtime_pre = _hf_runtime_inventory()
        runtime_path = args.output / "HF-RUNTIME.json"
        runtime_path.write_text(
            json.dumps(
                {
                    "schema": "xe-quality-hf-runtime-bridge-v1",
                    "status": "running",
                    "pre": runtime_pre,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        from xe_quality.reference.hf_control import run

        importer_path = Path(__file__).resolve().parents[1] / "gguf_import.py"
        final = run(
            args.model,
            args.hf_config,
            importer_path,
            tokens,
            positions,
            args.output,
            precision="fp64",
        )
        runtime_post = _hf_runtime_inventory()
        runtime_bridge = {
            "kind": "same-process pre/post HF source and native-library inventory",
            "status": (
                "pass"
                if runtime_pre["fingerprint_sha256"]
                == runtime_post["fingerprint_sha256"]
                else "fail"
            ),
            "pre_sha256": runtime_pre["fingerprint_sha256"],
            "post_sha256": runtime_post["fingerprint_sha256"],
        }
        runtime_document = {
            "schema": "xe-quality-hf-runtime-bridge-v1",
            "pre": runtime_pre,
            "post": runtime_post,
            "bridge": runtime_bridge,
        }
        runtime_path.write_text(
            json.dumps(runtime_document, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        if runtime_bridge["status"] != "pass":
            raise RuntimeError(
                "HF source/native runtime fingerprint changed during qualification"
            )

        package_dir = Path(__file__).resolve().parents[1] / "reference"
        adapter_paths = [
            package_dir / name
            for name in (
                "hf_control.py",
                "hf_precise.py",
                "hf_source_pin.py",
                "HF-PIN.json",
                "HF-config.json",
            )
        ]
        adapter_paths.append(importer_path)
        extra = {
            "source_pins": source_pins,
            "last_layer_shape": list(final.shape),
            "adapter_sha256": {path.name: _hash(path) for path in adapter_paths},
            "runtime": {
                "python": platform.python_version(),
                **{
                    name: entry["version"]
                    for name, entry in runtime_pre["distributions"].items()
                },
                "fingerprint_sha256": runtime_pre["fingerprint_sha256"],
            },
            "runtime_bridge": runtime_bridge,
            "runtime_inventory": "HF-RUNTIME.json",
        }
    report = {
        "schema": "xe-quality-reference-worker-v1",
        "worker_sha256": _hash(Path(__file__)),
        "kind": args.kind,
        "status": "completed",
        "model_sha256": _hash(args.model),
        "input_sha256": _hash(args.input),
        "tokens": len(tokens),
        "positions": positions,
        "precision": "FP64 parameters/activations/KV; exact GGUF decoding; natural router",
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": _hash(Path(sys.executable).resolve()),
        "elapsed_seconds": time.monotonic() - started,
        "extra": extra,
    }
    (args.output / "WORKER.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "kind": args.kind,
                "status": "completed",
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
