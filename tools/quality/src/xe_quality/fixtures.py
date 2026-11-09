"""Verify, capture and compare public, fixed-corpus numerical fixtures.

PASS concerns this finite suite only. Reusing a public suite does not establish
statistical noninferiority on unseen documents or free-generation task quality.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from .artifacts import manifest_hash
from .fixture_reference import checked_file, file_hash, load_corpus, read_json, write_json
from .metrics import compare_logits


LEGACY_ACCEPTANCE = {
    "scope": "fixed-public-corpus", "margin": 0,
    "endpoints": ["mean_kl", "max_kl", "max_tv"],
    "direction": "Google||backend", "temperature": 1,
}
ACCEPTANCE = {
    "scope": "fixed-public-corpus", "margin": 0,
    "gate": "mean-family-kl",
    "aggregation": "equal-families/equal-document-groups/equal-observed-positions",
    "diagnostics": ["family_mean_kl", "family_max_kl", "family_max_tv", "position_kl", "position_tv"],
    "direction": "Google||backend", "temperature": 1,
}


def array(path, rows, dtype):
    z = np.load(path, allow_pickle=False, mmap_mode="r")
    if z.shape != (rows, 262144) or z.dtype != np.dtype(dtype) or not np.isfinite(z).all():
        raise ValueError(f"invalid complete-vocabulary array: {Path(path).name}")
    return z


def verify(root):
    root = Path(root).absolute()
    manifest = read_json(root / "manifest.json")
    digest = manifest.pop("content_sha256", None)
    if manifest.get("schema") != "xe-quality-fixtures-v1" or digest != manifest_hash(manifest):
        raise ValueError("fixture manifest identity mismatch")
    corpus_path = checked_file(root, manifest["corpus"])
    corpus, inputs = load_corpus(corpus_path)
    if manifest.get("reference", {}).get("kind") != "trusted-google-fp64":
        raise ValueError("explicit trusted Google reference required")
    if manifest.get("acceptance") not in (ACCEPTANCE, LEGACY_ACCEPTANCE):
        raise ValueError("unsupported fixture acceptance contract")
    keys = set(inputs)
    for role, dtype in (("google", "<f8"), ("llama", "<f4")):
        if set(manifest["arrays"][role]) != keys:
            raise ValueError("reference/baseline cases are missing or extra")
        for key, spec in inputs.items():
            array(checked_file(root, manifest["arrays"][role][key]), len(spec["positions"]), dtype)
    baseline = manifest.get("baseline", {})
    if (baseline.get("backend") != "llama.cpp CPU" or not baseline.get("source_commit")
            or baseline.get("verified_repetitions", 0) < 3
            or baseline.get("repeatability") != "bitwise"):
        raise ValueError("pinned repeatable llama baseline required")
    manifest["content_sha256"] = digest
    return manifest, corpus, inputs


def backend_receipt(path, model, expected_weights):
    backend = read_json(path)
    driver = Path(backend.get("driver", ""))
    if (backend.get("weights_sha256") != expected_weights
            or file_hash(model) != expected_weights):
        raise ValueError("backend GGUF mismatch")
    check_backend(backend)
    if file_hash(driver) != backend.get("driver_sha256"):
        raise ValueError("driver identity mismatch")
    return backend


def check_backend(backend):
    if not backend.get("backend") or not backend.get("input_pins"):
        raise ValueError("current-build backend receipt required")
    for name, digest in backend["input_pins"].items():
        if file_hash(name) != digest:
            raise ValueError("backend source/binary/runtime changed: " + name)
    if file_hash(backend["driver"]) != backend["driver_sha256"]:
        raise ValueError("driver changed during comparison")


def dispatch_lines(path):
    return [line.strip() for line in Path(path).read_text(errors="replace").splitlines()
            if any(field in line for field in
                   ("flash_attn", "Flash Attention", "flash attention",
                    "compute buffer size", "n_ctx ", "n_ctx_per_seq ",
                    "n_batch ", "n_ubatch ", "type_k =", "type_v ="))]


def verify_capture(folder, corpus, inputs):
    folder = Path(folder)
    receipt = read_json(folder / "capture.json")
    if (receipt.get("schema") != "xe-quality-fixture-capture-v1"
            or receipt.get("corpus_sha256") != manifest_hash(corpus)
            or set(receipt["files"]) != set(inputs)):
        raise ValueError("capture/corpus join mismatch")
    runs = receipt.get("runs")
    if (type(receipt.get("repetitions")) is not int or not isinstance(runs, list)
            or not 1 <= receipt["repetitions"] <= 10 or len(runs) != receipt["repetitions"]):
        raise ValueError("missing independent run inventory")
    paths, repeatable, observed_dispatch = set(), True, []
    for run in runs:
        if set(run) != set(inputs):
            raise ValueError("incomplete run inventory")
        run_dispatch = {}
        for key, record in run.items():
            if record["path"] in paths:
                raise ValueError("a run artifact cannot count as another execution")
            paths.add(record["path"])
            array(checked_file(folder, record), len(inputs[key]["positions"]), "<f4")
            repeatable &= record["sha256"] == receipt["files"][key]["sha256"]
            stderr = (folder / record["path"]).parent / "stderr.txt"
            if stderr.is_symlink():
                raise ValueError("symlink in retained dispatch logs")
            if stderr.is_file():
                run_dispatch[key] = dispatch_lines(stderr)
        observed_dispatch.append(run_dispatch)
    if runs[0] != receipt["files"] or (receipt.get("repeatable") and not repeatable):
        raise ValueError("capture repeatability claim contradicts its artifacts")
    # Derive effective dispatch from retained logs and check repeatability
    # without changing the original capture files.
    if any(observed_dispatch):
        if any(set(d) != set(inputs) for d in observed_dispatch):
            raise ValueError("incomplete retained dispatch logs")
        receipt["repeatable"] = receipt["repeatable"] and all(
            d == observed_dispatch[0] for d in observed_dispatch)
        receipt["effective_dispatch"] = observed_dispatch[0]
    return receipt


def capture(corpus_path, model, backend_path, output, repetitions=3, timeout=1800):
    if type(repetitions) is not int or not 1 <= repetitions <= 10 or timeout <= 0:
        raise ValueError("invalid repetition count/timeout")
    corpus, inputs = load_corpus(corpus_path)
    backend = backend_receipt(backend_path, model, corpus["weights_sha256"])
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "backend.json", backend)
    write_json(output / "corpus.json", corpus)
    files, repeatable, dispatch, runs = {}, True, {}, []
    started = time.monotonic()
    for repeat in range(repetitions):
        run = output / f"run-{repeat + 1}"
        run.mkdir()
        runs.append({})
        # Change order while keeping the input and numerical protocol fixed.
        cases = list(corpus["cases"])
        np.random.default_rng(610600 + repeat).shuffle(cases)
        for case in cases:
            key, spec = case["id"], inputs[case["id"]]
            dest = run / key
            dest.mkdir()
            tsv = dest / "input.tsv"
            tsv.write_text("\t".join([key, ",".join(map(str, spec["tokens"])),
                          ",".join(map(str, spec["positions"])),
                          ",".join(map(str, spec["chunks"]))]) + "\n")
            argv = [str(Path(backend["driver"]).resolve()), str(Path(model).resolve()),
                    str(tsv), str(dest), str(spec["context"]), spec["mode"]]
            with (dest / "stdout.txt").open("wb") as stdout, (dest / "stderr.txt").open("wb") as stderr:
                subprocess.run(argv, stdout=stdout, stderr=stderr, check=True, timeout=timeout)
            selected_dispatch = dispatch_lines(dest / "stderr.txt")
            if repeat == 0:
                dispatch[key] = selected_dispatch
            elif dispatch[key] != selected_dispatch:
                repeatable = False
            raw = dest / (key + ".bin")
            if raw.stat().st_size != len(spec["positions"]) * 262144 * 4:
                raise ValueError("driver returned an incomplete vocabulary/position set")
            z = np.fromfile(raw, dtype="<f4").reshape(-1, 262144)
            if not np.isfinite(z).all():
                raise ValueError("candidate returned nonfinite logits")
            target = dest / (key + ".npy")
            with target.open("xb") as stream:
                np.save(stream, z, allow_pickle=False)
            info = {"path": str(target.relative_to(output)), "sha256": file_hash(target)}
            runs[-1][key] = info
            if repeat == 0:
                files[key] = info
            elif info["sha256"] != files[key]["sha256"]:
                repeatable = False
            print(json.dumps({"event": "native_case", "repeat": repeat+1, "case": key}), flush=True)
        check_backend(backend)
    result = {"schema": "xe-quality-fixture-capture-v1", "corpus_sha256": manifest_hash(corpus),
              "backend": backend, "files": files, "runs": runs, "repetitions": repetitions,
              "repeatable": repeatable, "effective_dispatch": dispatch,
              "seconds": time.monotonic() - started}
    write_json(output / "capture.json", result)
    return result


def seal(corpus_path, google_dir, baseline_dir, output):
    corpus_path, google_dir, baseline_dir = map(Path, (corpus_path, google_dir, baseline_dir))
    corpus, inputs = load_corpus(corpus_path)
    google = read_json(google_dir / "google.json")
    baseline = verify_capture(baseline_dir, corpus, inputs)
    for receipt in (google, baseline):
        if receipt["corpus_sha256"] != manifest_hash(corpus) or set(receipt["files"]) != set(inputs):
            raise ValueError("producer/corpus join mismatch")
    if baseline["repetitions"] < 3 or not baseline["repeatable"]:
        raise ValueError("baseline needs three identical runs before sealing")
    backend = baseline["backend"]
    if backend["backend"] != "llama.cpp CPU" or not backend.get("public_profile"):
        raise ValueError("explicit llama production profile required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "corpus").mkdir()
    shutil.copyfile(corpus_path, output / "corpus/manifest.json")
    for item in [c["input"] for c in corpus["cases"]] + [d["text"] for d in corpus.get("documents", [])]:
        src = checked_file(corpus_path.parent, item)
        dst = output / "corpus" / item["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    arrays = {}
    for role, folder, receipt, dtype in (("google", google_dir, google, "<f8"),
                                         ("llama", baseline_dir, baseline, "<f4")):
        (output / role).mkdir()
        arrays[role] = {}
        for key, item in receipt["files"].items():
            src = checked_file(folder, item)
            array(src, len(inputs[key]["positions"]), dtype)
            dst = output / role / (key + ".npy")
            shutil.copyfile(src, dst)
            arrays[role][key] = {"path": str(dst.relative_to(output)), "sha256": file_hash(dst)}
    profile = dict(backend["public_profile"])
    profile.update(backend=backend["backend"], source_commit=backend["source_commit"],
                   verified_repetitions=baseline["repetitions"], repeatability="bitwise",
                   driver_sha256=backend["driver_sha256"],
                   effective_dispatch=baseline.get("effective_dispatch", {}))
    manifest = {"schema": "xe-quality-fixtures-v1", "corpus": {
        "path": "corpus/manifest.json", "sha256": file_hash(output / "corpus/manifest.json")},
        "reference": google["producer"], "baseline": profile, "arrays": arrays,
        "acceptance": ACCEPTANCE}
    manifest["content_sha256"] = manifest_hash(manifest)
    write_json(output / "manifest.json", manifest)
    verify(output)
    return manifest


def _metrics(g, x, l, extended=False):
    if not extended:
        gx = compare_logits(g, x, expected_vocab_size=262144)
        gl = compare_logits(g, l, expected_vocab_size=262144)
        return np.array([gx["kl_b_to_c_nats"], gl["kl_b_to_c_nats"], gx["tv"], gl["tv"]]).T
    def logs(z):
        z = np.asarray(z, dtype=np.longdouble)
        z = z - z.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    pg, px, pl = (logs(z) for z in (g, x, l))
    p = np.exp(pg)
    return np.array([(p*(pg-px)).sum(axis=-1), (p*(pg-pl)).sum(axis=-1),
                     np.abs(p-np.exp(px)).sum(axis=-1)/2,
                     np.abs(p-np.exp(pl)).sum(axis=-1)/2]).T


def _endpoints(rows):
    rows = np.asarray(rows)
    return np.array([rows[:, 0].mean()-rows[:, 1].mean(),
                     rows[:, 0].max()-rows[:, 1].max(),
                     rows[:, 2].max()-rows[:, 3].max()])


def _difference(a, b):
    return {"delta_x_minus_llama": float(a),
            "resolution_interval": [float(min(a, b)), float(max(a, b))]}


def _summarize(grouped, precise):
    """Equal weight for families, then document groups, then observed positions."""
    strata, family_means, precise_means = {}, [], []
    for stratum in sorted(set(k[0] for k in grouped)):
        keys = [k for k in grouped if k[0] == stratum]
        a = np.mean([_endpoints(grouped[k]) for k in keys], axis=0)
        b = np.mean([_endpoints(precise[k]) for k in keys], axis=0)
        means = np.mean([np.asarray(grouped[k])[:, :2].mean(axis=0) for k in keys], axis=0)
        extended = np.mean([np.asarray(precise[k])[:, :2].mean(axis=0) for k in keys], axis=0)
        family_means.append(means)
        precise_means.append(extended)
        strata[stratum] = {
            "independent_groups": len(keys),
            "mean_kl_google_x": float(means[0]), "mean_kl_google_llama": float(means[1]),
            "endpoints": {name: _difference(a[j], b[j])
                          for j, name in enumerate(("mean_kl", "max_kl", "max_tv"))}}
    a, b = np.mean(family_means, axis=0), np.mean(precise_means, axis=0)
    overall = {"families": len(strata), "kl_google_x": float(a[0]),
               "kl_google_llama": float(a[1]), **_difference(a[0]-a[1], b[0]-b[1])}
    return strata, overall


def evaluate(root, capture_dir):
    manifest, corpus, inputs = verify(root)
    capture_dir, root = Path(capture_dir), Path(root)
    captured = verify_capture(capture_dir, corpus, inputs)
    grouped, precise, cases = defaultdict(list), defaultdict(list), []
    position_regressions, uncertain_positions = [], []
    for case in corpus["cases"]:
        key = case["id"]
        n = len(inputs[key]["positions"])
        g = array(checked_file(root, manifest["arrays"]["google"][key]), n, "<f8")
        l = array(checked_file(root, manifest["arrays"]["llama"][key]), n, "<f4")
        x = array(checked_file(capture_dir, captured["files"][key]), n, "<f4")
        values, extended = _metrics(g, x, l), _metrics(g, x, l, True)
        for row, position in enumerate(inputs[key]["positions"]):
            for metric, col in (("kl", 0), ("tv", 2)):
                delta = _difference(values[row, col]-values[row, col+1],
                                    extended[row, col]-extended[row, col+1])
                item = {"case": key, "family": case["stratum"], "position": position,
                        "metric": metric, "google_x": float(values[row, col]),
                        "google_llama": float(values[row, col+1]), **delta}
                lo, hi = delta["resolution_interval"]
                if lo > 0:
                    position_regressions.append(item)
                elif lo <= 0 < hi:
                    uncertain_positions.append(item)
        group = (case["stratum"], case["group"])
        grouped[group].extend(values)
        precise[group].extend(extended)
        cases.append({"id": key, "positions": inputs[key]["positions"],
                      "kl_google_x": values[:, 0].tolist(), "kl_google_llama": values[:, 1].tolist(),
                      "tv_google_x": values[:, 2].tolist(), "tv_google_llama": values[:, 3].tolist()})
    strata, overall = _summarize(grouped, precise)
    family_regressions = [
        {"family": family, "metric": name, **endpoint}
        for family, data in strata.items() for name, endpoint in data["endpoints"].items()
        if endpoint["resolution_interval"][0] > 0]
    legacy = manifest["acceptance"] == LEGACY_ACCEPTANCE
    gates = ([e for s in strata.values() for e in s["endpoints"].values()]
             if legacy else [overall])
    failed = any(e["resolution_interval"][0] > 0 for e in gates)
    unstable = any(e["resolution_interval"][0] <= 0 < e["resolution_interval"][1] for e in gates)
    # Repetitions establish observed stability, not an independent document sample.
    enough = captured["repetitions"] >= 3 and captured["repeatable"]
    outcome = "FAIL" if failed else ("INCONCLUSIVE" if unstable else "PASS")
    if not enough:
        outcome = "INCONCLUSIVE"
    return {"schema": "xe-quality-fixture-report-v1", "outcome": outcome,
            "scope": "fixed-public-corpus", "population_inference": "NOT_ESTABLISHED",
            "reason": ("Legacy per-family endpoint gate." if legacy else
                       "Equal-family mean KL gate; local regressions are diagnostic, not vetoes.")
                      + " Zero margin; three identical native runs required.",
            "acceptance": manifest["acceptance"], "overall": overall,
            "diagnostics": {"blocking": legacy,
                "worse_mean_kl_families": [e["family"] for e in family_regressions if e["metric"] == "mean_kl"],
                "family_endpoint_regressions": family_regressions,
                "position_regressions": position_regressions,
                "uncertain_position_differences": uncertain_positions},
            "fixture_sha256": manifest["content_sha256"], "strata": strata, "cases": cases,
            "comparator": {"python": platform.python_version(), "numpy": np.__version__,
                           "source_sha256": file_hash(__file__),
                           "metrics_sha256": file_hash(Path(__file__).with_name("metrics.py"))},
            "repetitions": captured["repetitions"], "repeatable": captured["repeatable"],
            "resolution_check": "FP64 vs longdouble; empirical sensitivity, not a rigorous error bound",
            "longdouble_mantissa_bits": int(np.finfo(np.longdouble).nmant)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--fixtures", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--backend-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--timeout", type=int, default=1800)
    p = sub.add_parser("seal")
    for arg in ("corpus", "google", "baseline", "output"):
        p.add_argument("--"+arg, required=True)
    p = sub.add_parser("compare")
    for arg in ("fixtures", "model", "backend-config", "output"):
        p.add_argument("--"+arg, required=True)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            m, c, _ = verify(args.fixtures)
            result = {"status": "VERIFIED", "cases": len(c["cases"]), "content_sha256": m["content_sha256"]}
        elif args.command == "capture":
            result = capture(args.corpus, args.model, args.backend_config, args.output, args.repetitions, args.timeout)
            result = {k: result[k] for k in ("repetitions", "repeatable", "seconds")}
        elif args.command == "seal":
            result = seal(args.corpus, args.google, args.baseline, args.output)
            result = {"status": "SEALED", "content_sha256": result["content_sha256"]}
        else:
            m, _, _ = verify(args.fixtures)
            corpus_path = checked_file(args.fixtures, m["corpus"])
            capture(corpus_path, args.model, args.backend_config, args.output, args.repetitions, args.timeout)
            result = evaluate(args.fixtures, args.output)
            write_json(Path(args.output) / "report.json", result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}.get(result.get("outcome"), 0)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"outcome": "ERROR", "scientific_decision": "INCONCLUSIVE", "reason": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
