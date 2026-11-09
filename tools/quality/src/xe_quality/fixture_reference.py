"""Produce reusable Google FP64 fixture rows, without an HF dependency.

This explicitly trusts the pinned Google adapter as the operational reference;
it does not manufacture an independent qualification for new inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .artifacts import manifest_hash


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=invalid)


def write_json(path, value):
    # New evidence is never silently overwritten.
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def checked_file(root, record):
    from pathlib import PurePosixPath

    name = record["path"]
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or str(relative) != name):
        raise ValueError("unsafe artifact path")
    path = Path(root) / name
    for parent in [path, *path.parents]:
        if parent == Path(root).parent:
            break
        if parent.is_symlink():
            raise ValueError("symlink in fixture artifact path")
    if not path.is_file() or file_hash(path) != record["sha256"]:
        raise ValueError(f"missing or changed fixture artifact: {name}")
    return path


def load_corpus(path):
    import re

    path = Path(path)
    data = read_json(path)
    if data.get("schema") != "xe-quality-fixture-corpus-v1":
        raise ValueError("unsupported fixture corpus schema")
    if data.get("vocab_size") != 262144:
        raise ValueError("fixture corpus requires the complete Gemma4 vocabulary")
    if not re.fullmatch(r"[a-f0-9]{64}", data.get("weights_sha256", "")):
        raise ValueError("invalid weights identity")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("empty fixture corpus")
    seen, inputs = set(), {}
    for case in cases:
        key = case["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", key) or key in seen:
            raise ValueError("invalid or duplicate fixture case ID")
        seen.add(key)
        for name in ("group", "stratum"):
            if not isinstance(case.get(name), str) or not case[name]:
                raise ValueError("case grouping is required")
        spec = read_json(checked_file(path.parent, case["input"]))
        tokens, positions, chunks = (spec[k] for k in ("tokens", "positions", "chunks"))
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 262144
                or any(type(t) is not int or not 0 <= t < 262144 for t in tokens)):
            raise ValueError("invalid exact tokens")
        if (not isinstance(positions, list) or not positions
                or any(type(p) is not int or not 0 <= p < len(tokens) for p in positions)
                or positions != sorted(set(positions))):
            raise ValueError("invalid selected positions")
        if (not isinstance(chunks, list) or not chunks
                or any(type(n) is not int or n <= 0 for n in chunks)
                or sum(chunks) != len(tokens)
                or not set(p + 1 for p in positions) <= set(np.cumsum(chunks).tolist())):
            raise ValueError("positions must be declared chunk endpoints")
        if (type(spec.get("context")) is not int
                or not max(64, len(tokens)) <= spec["context"] <= 262144
                or spec.get("initial_state") != "empty"
                or spec.get("mode") != "prefill"):
            raise ValueError("unsupported context/state/mode contract")
        inputs[key] = spec
    for document in data.get("documents", []):
        checked_file(path.parent, document["text"])
    return data, inputs


def produce(corpus_path, model, google_source, output, max_forwards):
    corpus, inputs = load_corpus(corpus_path)
    if type(max_forwards) is not int or max_forwards < 1:
        raise ValueError("explicit positive forward budget required")
    planned = sum(len(spec["chunks"]) for spec in inputs.values())
    if planned > max_forwards:
        raise ValueError("reference forward budget is insufficient")
    if file_hash(model) != corpus["weights_sha256"]:
        raise ValueError("wrong GGUF")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Optional frameworks are imported only by this explicit producer command.
    import jax
    from .reference.stream import Reference

    if jax.default_backend() != "cpu":
        raise ValueError("this reference profile requires CPU")
    reference = Reference(model, google_source, precision="fp64")
    identity = reference.code_identity(refresh=True)
    write_json(output / "producer-internal.json", identity)
    files, forwards = {}, 0
    started = time.monotonic()
    try:
        for case in corpus["cases"]:
            key, spec = case["id"], inputs[case["id"]]
            reference.reset()
            rows, offset = [], 0
            for count in spec["chunks"]:
                positions = [p - offset for p in spec["positions"]
                             if offset <= p < offset + count]
                forwards += 1
                print(json.dumps({"event": "forward", "case": key,
                                  "request": forwards, "offset": offset,
                                  "tokens": count}), flush=True)
                rows.append(reference.forward(spec["tokens"][offset:offset+count],
                                              positions, cache=True,
                                              capacity=spec["context"]))
                offset += count
            z = np.concatenate(rows, axis=0)
            if z.shape != (len(spec["positions"]), 262144) or not np.isfinite(z).all():
                raise ValueError("invalid reference logits")
            dest = output / (key + ".npy")
            with dest.open("xb") as stream:
                np.save(stream, z.astype("<f8"), allow_pickle=False)
            files[key] = {"path": dest.name, "sha256": file_hash(dest)}
            print(json.dumps({"event": "case_complete", "case": key}), flush=True)
        if reference.code_identity(refresh=True) != identity:
            raise ValueError("producer identity changed during execution")
    finally:
        reference.close()
    # Only portable provenance belongs in the curated public fixture.
    pin = read_json(Path(__file__).with_name("reference") / "GOOGLE-PIN.json")
    producer = {"kind": "trusted-google-fp64", "upstream": pin,
                "precision": "FP64 weights/activations/KV/head; temperature 1; post-softcap",
                "independent_qualification": "Not asserted for this fixture corpus",
                "adapter_hashes": {p.name: file_hash(p) for p in
                                   (Path(__file__).with_name("reference")).glob("*.py")},
                "importer_sha256": file_hash(Path(__file__).with_name("gguf_import.py")),
                "generator_sha256": file_hash(__file__),
                "runtime_fingerprint": manifest_hash(identity["runtime"]),
                "runtime_versions": {k: v for k, v in identity["runtime"].items()
                                     if k in ("python", "jax", "jaxlib", "flax", "numpy", "backend")}}
    result = {"schema": "xe-quality-fixture-google-v1", "corpus_sha256": manifest_hash(corpus),
              "producer": producer, "files": files}
    write_json(output / "google.json", result)
    write_json(output / "execution.json", {"forwards": forwards,
               "seconds": time.monotonic() - started})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--google-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-forwards", type=int, required=True)
    args = parser.parse_args(argv)
    produce(args.corpus, args.model, args.google_source, args.output, args.max_forwards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
