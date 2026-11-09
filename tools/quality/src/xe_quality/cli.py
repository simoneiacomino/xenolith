#!/usr/bin/env python3
"""Exact-GGUF reference archive and fixed-history full-vocabulary comparison."""

from pathlib import Path
import argparse, json, hashlib, sys, os, tempfile, subprocess, time, fcntl, shutil
import stat
from pathlib import PurePosixPath

try:
    import numpy as np
except ImportError:
    print(
        json.dumps(
            {
                "outcome": "ERROR",
                "reason": "NumPy is required. Install tools/quality with pip or run python3.12 tools/quality/setup_environment.py --profile compare --output NEW_VENV.",
            }
        ),
        file=sys.stderr,
    )
    raise SystemExit(2)
from .artifacts import (
    canonical_json_bytes,
    manifest_hash,
    write_artifact,
    verify_resume,
    read_artifact,
    copy_verified_archive,
)
from .metrics import compare_logits, group_summary
from .corpus import prepare, validate


def read(path):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError("Duplicate JSON key " + k)
            d[k] = v
        return d

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON " + x)
        ),
    )


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


QUALIFICATION_PACKET_PATH = "qualification/QUALIFICATION.json"


def _packet_relative_path(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError(f"unsafe qualification evidence path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name:
        raise ValueError(f"unsafe qualification evidence path: {name!r}")
    return Path(*relative.parts)


def _reject_symlink_ancestors(path, label):
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"{label} contains a symlink: {ancestor}")


def _pinned_regular_file(root, relative, pin):
    root = Path(root)
    _reject_symlink_ancestors(root, "qualification packet root")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"qualification evidence contains a symlink: {relative}")
    try:
        mode = current.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"qualification evidence missing: {relative}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"qualification evidence is not a regular file: {relative}")
    if current.stat().st_size != pin["bytes"] or sha(current) != pin["sha256"]:
        raise ValueError(f"qualification evidence pin mismatch: {relative}")
    return current


def validate_qualification_packet(path, *, expected=None):
    """Validate the original qualification JSON and every pinned raw evidence file."""
    path = Path(path)
    if path.name != "QUALIFICATION.json":
        raise ValueError("qualification packet must contain QUALIFICATION.json")
    _reject_symlink_ancestors(path, "qualification packet path")
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(
            f"qualification JSON is missing or is not a regular file: {path}"
        )
    qualification = read(path)
    if not isinstance(qualification, dict):
        raise ValueError("qualification JSON must be an object")
    if expected is not None and qualification != expected:
        raise ValueError("qualification JSON differs from the archive contract")
    evidence = qualification.get("evidence_files", {})
    if not isinstance(evidence, dict):
        raise ValueError("qualification evidence_files must be an object")
    if qualification.get("status") == "QUALIFIED_SCOPED" and not evidence:
        raise ValueError("QUALIFIED_SCOPED requires pinned raw qualification evidence")
    if qualification.get("status") not in ("QUALIFIED_SCOPED", "UNQUALIFIED"):
        raise ValueError("qualification packet has an unsupported status")
    expected_files = {"QUALIFICATION.json"}
    files = []
    root = path.parent
    for name, pin in evidence.items():
        relative = _packet_relative_path(name)
        if relative.as_posix() == "QUALIFICATION.json":
            raise ValueError(
                "qualification evidence_files cannot include QUALIFICATION.json"
            )
        if (
            not isinstance(pin, dict)
            or set(pin) != {"bytes", "sha256"}
            or type(pin.get("bytes")) is not int
            or pin["bytes"] < 0
            or not isinstance(pin.get("sha256"), str)
            or len(pin["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in pin["sha256"])
        ):
            raise ValueError(f"invalid qualification evidence pin: {name!r}")
        source = _pinned_regular_file(root, relative, pin)
        expected_files.add(relative.as_posix())
        files.append((relative, source, pin))

    # The producer pins every regular file except QUALIFICATION.json. Reject
    # unpinned files and symlink directories so the copied packet is complete.
    observed_files = {"QUALIFICATION.json"}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            if child.is_symlink() or not child.is_dir():
                raise ValueError(
                    f"qualification packet contains a non-directory path: {child}"
                )
        for filename in filenames:
            child = directory_path / filename
            if child.is_symlink() or not stat.S_ISREG(child.lstat().st_mode):
                raise ValueError(
                    f"qualification packet contains a non-regular file: {child}"
                )
            relative = child.relative_to(root).as_posix()
            observed_files.add(relative)
    if observed_files != expected_files:
        raise ValueError(
            "qualification packet file inventory differs from evidence_files: "
            f"missing={sorted(expected_files - observed_files)}, extra={sorted(observed_files - expected_files)}"
        )
    # Hash the JSON too, rejecting a concurrent change during inventory checks.
    if path.is_symlink() or not path.is_file():
        raise ValueError("qualification JSON changed during validation")
    return qualification, files


def _files_byte_identical(left, right):
    with Path(left).open("rb") as a, Path(right).open("rb") as b:
        while True:
            block_a = a.read(1 << 20)
            block_b = b.read(1 << 20)
            if block_a != block_b:
                return False
            if not block_a:
                return True


def _copy_packet_file(source, destination, pin):
    destination = Path(destination)
    _reject_symlink_ancestors(destination.parent, "qualification destination")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(
                f"refusing to replace qualification packet path: {destination}"
            )
        if (
            destination.stat().st_size != pin["bytes"]
            or sha(destination) != pin["sha256"]
            or not _files_byte_identical(source, destination)
        ):
            raise ValueError(
                f"refusing to overwrite different qualification bytes: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(destination, flags, 0o600)
    except FileExistsError:
        return _copy_packet_file(source, destination, pin)
    try:
        with os.fdopen(fd, "wb") as output, Path(source).open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1 << 20)
            output.flush()
            os.fsync(output.fileno())
        if (
            destination.stat().st_size != pin["bytes"]
            or sha(destination) != pin["sha256"]
        ):
            raise ValueError(
                f"copied qualification bytes do not match pin: {destination}"
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def copy_qualification_packet(
    source_path, qualification, archive_root, *, expected_sha256
):
    """Copy a validated packet byte-for-byte without replacing existing files."""
    source_path = Path(source_path)
    actual, evidence = validate_qualification_packet(
        source_path, expected=qualification
    )
    if sha(source_path) != expected_sha256:
        raise ValueError("qualification JSON SHA-256 differs from archive contract")
    destination_root = Path(archive_root) / "qualification"
    destination_root.mkdir(parents=True, exist_ok=True)
    json_pin = {"bytes": source_path.stat().st_size, "sha256": expected_sha256}
    _copy_packet_file(source_path, destination_root / "QUALIFICATION.json", json_pin)
    for relative, source, pin in evidence:
        _copy_packet_file(source, destination_root / relative, pin)
    validate_qualification_packet(
        destination_root / "QUALIFICATION.json", expected=actual
    )


def verify_distribution(archive, manifest_path):
    archive = Path(archive).resolve()
    manifest = read(manifest_path)
    if manifest.get("schema") != "xenolith-reference-distribution-v1" or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("unsupported reference distribution manifest")
    index_path = archive / "index.json"
    if sha(index_path) != manifest.get("index_sha256"):
        raise ValueError("distribution index SHA-256 mismatch")
    index = load_index(index_path)
    if (
        index["reference_version"] != manifest.get("reference_version")
        or index["contract_hash"] != manifest.get("contract_hash")
        or index["schema"] != manifest.get("archive_schema")
        or index["contract"]["config"].get("weights_sha256")
        != manifest.get("weights_sha256")
    ):
        raise ValueError("distribution manifest and archive identity mismatch")
    for name, spec in manifest["files"].items():
        rel = Path(name)
        if (
            not name
            or rel.is_absolute()
            or any(part in ("..", "") for part in rel.parts)
        ):
            raise ValueError(f"unsafe distribution path: {name!r}")
        path = (archive / rel).resolve()
        if archive not in path.parents:
            raise ValueError(f"distribution path escapes archive: {name!r}")
        if (
            not isinstance(spec, dict)
            or type(spec.get("bytes")) is not int
            or not isinstance(spec.get("sha256"), str)
            or len(spec["sha256"]) != 64
        ):
            raise ValueError(f"invalid file pin for {name!r}")
        if (
            not path.is_file()
            or path.stat().st_size != spec["bytes"]
            or sha(path) != spec["sha256"]
        ):
            raise ValueError(f"distribution file mismatch: {name}")
    return manifest


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp-")
    with os.fdopen(fd, "wb") as f:
        f.write(canonical_json_bytes(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def code_identity(model, source, precision):
    """Fingerprint the selected adapter without constructing a model instance."""
    from .reference.stream import code_identity as fingerprint

    return fingerprint(model, source, precision)


def software_identity():
    """Fingerprint the maintained software that defines inputs and comparisons."""
    root = Path(__file__).resolve().parent
    names = (
        "cli.py",
        "metrics.py",
        "artifacts.py",
        "corpus.py",
        "acquisition.py",
        "tokenize_panel.py",
    )
    return {f"xe_quality/{name}": sha(root / name) for name in names}


def selected(record, config):
    n = len(record["token_ids"])
    explicit = config.get("positions", {}).get(
        record["source"] + ":" + record["record_id"]
    )
    pos = explicit if explicit is not None else sorted(set([max(0, n // 2 - 1), n - 1]))
    if (
        not isinstance(pos, list)
        or not pos
        or any(type(p) is not int or p < 0 or p >= n for p in pos)
        or pos != sorted(set(pos))
    ):
        raise ValueError("Invalid selected logit positions")
    return pos


def protocol_chunks(case, mode):
    n = len(case["record"]["token_ids"])
    state = case["record"]["initial_state"]
    if mode == "decode":
        return [1] * n
    if state is not None:
        chunks = state["prefill_chunks"]
    else:
        endpoints = sorted(set([p + 1 for p in case["positions"]] + [n]))
        chunks = np.diff([0] + endpoints).tolist()
    endpoints = np.cumsum(chunks).tolist()
    if sum(chunks) != n or any(p + 1 not in endpoints for p in case["positions"]):
        raise ValueError(
            "Selected logits must coincide with declared chunk endpoints for the session API"
        )
    return chunks


def archive_dtype(index):
    precision = index["contract"]["config"].get("reference_precision", "fp32")
    if precision not in ["fp32", "fp64"]:
        raise ValueError("Archive reference precision invalid")
    return "<f8" if precision == "fp64" else "<f4"


def case_key(record):
    return hashlib.sha256(
        (record["source"] + ":" + record["record_id"]).encode()
    ).hexdigest()[:24]


def group(record):
    return (
        record.get("leakage_group_id")
        or record.get("passage_id")
        or record.get("session_id")
        or record["group_id"]
    )


def input_state_protocol(record):
    state = record.get("initial_state")
    if state is None:
        return "reset_full_history"
    if not isinstance(state, dict):
        raise ValueError("initial_state must be a mapping or null")
    decode_steps = state.get("decode_steps", 0)
    return "reset_chunked_prefill_decode" if decode_steps else "reset_chunked_prefill"


def input_spec_hash(tokens, positions, state_protocol):
    return manifest_hash(
        {
            "tokens": list(tokens),
            "positions": list(positions),
            "input_state_protocol": state_protocol,
        }
    )


def signed_index(path, index):
    payload = dict(index)
    payload["index_sha256"] = manifest_hash(index)
    save(path, payload)


POLICY_SCHEMA = "xe-quality-policy-v1"
POLICY_METRICS = {
    "kl.mean_group_mean": ("kl", "mean_group_mean"),
    "kl.max_observed_position_value": ("kl", "max_observed_position_value"),
    "tv.mean_group_mean": ("tv", "mean_group_mean"),
    "tv.max_observed_position_value": ("tv", "max_observed_position_value"),
    "top1_disagreement.mean_group_mean": ("top1_disagreement", "mean_group_mean"),
    "top1_disagreement.max_observed_position_value": (
        "top1_disagreement",
        "max_observed_position_value",
    ),
    "nll.delta.mean_group_mean": ("nll", "delta", "mean_group_mean"),
    "nll.delta.max_observed_position_value": (
        "nll",
        "delta",
        "max_observed_position_value",
    ),
}


def evaluate_outcome(report, policy):
    """Apply only caller-specified bounds within an explicitly qualified scope."""
    if policy is None:
        return "INCONCLUSIVE", "diagnostic mode: no acceptance policy was supplied"
    if report.get("candidate_execution_mode", "prefill") != "prefill":
        return (
            "INCONCLUSIVE",
            "reference qualification does not cover the forced one-token candidate state variant",
        )
    if not isinstance(policy, dict) or policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("policy schema must be xe-quality-policy-v1")
    domains = set(report["summary"]["kl"]["domains"])
    qualification = report.get("reference_qualification") or {}
    if qualification.get("status") != "QUALIFIED_SCOPED":
        return "INCONCLUSIVE", "reference is not qualified for a scoped domain"
    scope = qualification.get("scope")
    if (
        not isinstance(scope, dict)
        or scope.get("schema") != "xe-quality-reference-scope-v1"
    ):
        return (
            "INCONCLUSIVE",
            "reference qualification has no machine-readable input/state scope",
        )
    if scope.get("domain_class") != "fixed-token-control":
        return (
            "INCONCLUSIVE",
            "reference qualification scope is not an exact fixed-input control",
        )
    if scope.get("supported_domains") != []:
        return (
            "INCONCLUSIVE",
            "reference qualification domain claims are unsupported by this fixed-input protocol",
        )
    input_hash = scope.get("input_hash")
    if (
        not isinstance(input_hash, str)
        or len(input_hash) != 64
        or any(c not in "0123456789abcdef" for c in input_hash.lower())
    ):
        return "INCONCLUSIVE", "reference qualification input hash is malformed"
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        return (
            "INCONCLUSIVE",
            "qualification covers exactly one fixed input; measured input count differs",
        )
    case = cases[0]
    if (
        case.get("input_hash") != scope.get("input_hash")
        or case.get("positions") != scope.get("positions")
        or case.get("input_length") != scope.get("input_length")
        or case.get("input_state_protocol") != scope.get("input_state_protocol")
        or case.get("input_state_protocol") != scope.get("state_protocol")
    ):
        return (
            "INCONCLUSIVE",
            "measured input, positions, length, or state protocol differ from reference qualification scope",
        )
    max_length = scope.get("maximum_supported_length")
    qualification_max = qualification.get("maximum_supported_length")
    if (
        type(max_length) is not int
        or type(case.get("input_length")) is not int
        or type(qualification_max) is not int
        or case["input_length"] > max_length
        or case["input_length"] > qualification_max
    ):
        return "INCONCLUSIVE", "history is outside the qualified length domain"
    if scope.get("qualification_state_controls") != [
        "full_prefill_cache_restore_reset_repeat"
    ]:
        return (
            "INCONCLUSIVE",
            "qualification state/cache control protocol is unsupported",
        )
    if report.get("candidate_execution_mode") != "prefill":
        return (
            "INCONCLUSIVE",
            "reference qualification covers declared prefill chunk protocols, not decode execution",
        )
    allowed_chunks = scope.get("allowed_prefill_chunks")
    if not isinstance(allowed_chunks, list) or not allowed_chunks:
        return (
            "INCONCLUSIVE",
            "reference qualification has no tested prefill chunk protocols",
        )
    input_length = case["input_length"]
    for chunks in allowed_chunks:
        if (
            not isinstance(chunks, list)
            or not chunks
            or any(type(chunk) is not int or chunk < 1 for chunk in chunks)
            or sum(chunks) != input_length
        ):
            return "INCONCLUSIVE", "reference qualification chunk protocol is malformed"
    if case.get("candidate_chunks") not in allowed_chunks:
        return (
            "INCONCLUSIVE",
            "candidate prefill chunks were not exercised by the reference qualification",
        )
    configured = policy.get("domains")
    if not isinstance(configured, dict) or set(configured) != domains:
        return (
            "INCONCLUSIVE",
            "policy must explicitly cover exactly the measured domains",
        )
    failures = []
    for domain in sorted(domains):
        rules = configured[domain]
        if (
            not isinstance(rules, dict)
            or type(rules.get("min_groups")) is not int
            or rules["min_groups"] < 1
        ):
            raise ValueError(
                f"policy domain {domain!r} requires integer min_groups >= 1"
            )
        summary = report["summary"]["kl"]["domains"][domain]
        if summary["group_count"] < rules["min_groups"]:
            return (
                "INCONCLUSIVE",
                f"domain {domain!r} has {summary['group_count']} groups; policy requires {rules['min_groups']}",
            )
        metrics = rules.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(
                f"policy domain {domain!r} requires explicit metric bounds"
            )
        for name, bounds in metrics.items():
            if (
                name not in POLICY_METRICS
                or not isinstance(bounds, dict)
                or len(bounds) != 1
                or next(iter(bounds)) not in ("max", "min")
            ):
                raise ValueError(f"invalid metric bound for {domain!r}: {name!r}")
            operator, limit = next(iter(bounds.items()))
            if (
                isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or not np.isfinite(limit)
            ):
                raise ValueError(f"policy bound for {name!r} must be a finite number")
            section, path_metric = POLICY_METRICS[name][0], POLICY_METRICS[name][-1]
            node = report["summary"].get(section)
            if section == "nll" and len(POLICY_METRICS[name]) == 3:
                node = (
                    node.get(POLICY_METRICS[name][1])
                    if isinstance(node, dict)
                    else None
                )
            value = (
                node.get("domains", {}).get(domain) if isinstance(node, dict) else None
            )
            if not isinstance(value, dict):
                return (
                    "INCONCLUSIVE",
                    f"metric {name!r} is unavailable for domain {domain!r}",
                )
            metric = value.get(path_metric)
            if not isinstance(metric, (int, float)) or not np.isfinite(metric):
                return (
                    "INCONCLUSIVE",
                    f"metric {name!r} is unavailable for domain {domain!r}",
                )
            if (
                type(value.get("group_count")) is not int
                or value["group_count"] < rules["min_groups"]
            ):
                return (
                    "INCONCLUSIVE",
                    f"metric {name!r} has insufficient independent groups for domain {domain!r}",
                )
            if (operator == "max" and metric > limit) or (
                operator == "min" and metric < limit
            ):
                failures.append(f"{domain}:{name}={metric} violates {operator}={limit}")
    return (
        ("FAIL", "; ".join(failures))
        if failures
        else ("PASS", "all explicit bounds met in the qualified measured domain")
    )


def load_index(path):
    index = read(path)
    digest = index.pop("index_sha256", None)
    if digest != manifest_hash(index):
        raise ValueError("Archive index hash mismatch")
    if index.get("schema") != "gemma4-reference-archive-v1" or not index.get(
        "complete"
    ):
        raise ValueError("Archive index incomplete or unsupported")
    contract = index["contract"]
    ch = manifest_hash(contract)
    if (
        ch != index["contract_hash"]
        or index["reference_version"] != "google-gemma4-gguf-" + ch[:20]
    ):
        raise ValueError("Archive reference contract mismatch")
    if "qualification_packet" in index:
        qualification_packet = index["qualification_packet"]
        if qualification_packet != QUALIFICATION_PACKET_PATH:
            raise ValueError("unsupported qualification packet path")
        qualification_path = (
            Path(path).resolve().parent / "qualification" / "QUALIFICATION.json"
        )
        qualification, _ = validate_qualification_packet(
            qualification_path, expected=contract.get("qualification")
        )
        expected_qualification_sha = contract.get("qualification_file_sha256")
        if (
            not isinstance(expected_qualification_sha, str)
            or len(expected_qualification_sha) != 64
            or sha(qualification_path) != expected_qualification_sha
        ):
            raise ValueError("qualification JSON SHA-256 differs from archive contract")
    corpus = dict(index["corpus"])
    mh = corpus.pop("manifest_sha256", None)
    if manifest_hash(corpus) != mh or contract["corpus_sha256"] != mh:
        raise ValueError("Archive corpus identity mismatch")
    config = contract["config"]
    lookup = {case_key(r): r for r in corpus["records"]}
    keys = []
    for case in index["cases"]:
        record = case["record"]
        key = case_key(record)
        pos = selected(record, config)
        if (
            case["key"] != key
            or lookup.get(key) != record
            or case["positions"] != pos
            or case["group_id"] != group(record)
        ):
            raise ValueError("Archive input/state/selection/group mismatch")
        spec = {
            "record": record,
            "logit_positions": pos,
            "observation_semantics": "input token index j is predicted by logit j-1; no target for absent successor",
        }
        identity = {
            "reference_version": index["reference_version"],
            "config_hash": manifest_hash(config),
            "input_hash": manifest_hash(spec),
            "contract_hash": ch,
        }
        if case["identity"] != identity or case["artifact"] != key + ".npy":
            raise ValueError("Archive artifact input identity mismatch")
        keys.append(key)
    wanted = config.get("record_keys")
    expected = [
        case_key(r)
        for r in corpus["records"]
        if wanted is None or r["source"] + ":" + r["record_id"] in wanted
    ]
    if sorted(keys) != sorted(expected) or len(set(keys)) != len(keys):
        raise ValueError("Archive panel incomplete/duplicated")
    return index


def build(args):
    start = time.monotonic()
    config = read(args.config)
    manifest = validate(args.corpus)
    model = Path(args.model)
    qualification_path = Path(args.qualification)
    qualification, _ = validate_qualification_packet(qualification_path)
    qualification_file_sha256 = sha(qualification_path)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lock_file = (out / ".build.lock").open("a")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if config.get("weights_sha256") != sha(model):
        raise ValueError("Exact GGUF hash mismatch")
    try:
        import jax
        import flax as _flax
    except ImportError as exc:
        raise RuntimeError(
            "Reference generation requires the explicit reference environment: run tools/quality/setup_environment.py --profile reference --output NEW_VENV"
        ) from exc
    precision = config.get("reference_precision")
    if precision != "fp64":
        raise ValueError(
            'Public reference archives can only be built with reference_precision="fp64"'
        )
    storage_dtype = "<f8"
    jax.config.update("jax_enable_x64", precision == "fp64")
    jax.config.update("jax_default_matmul_precision", "highest")
    implementation = code_identity(model, args.google_source, precision)
    runtime = implementation["runtime"]
    if jax.default_backend() != "cpu":
        raise ValueError("This qualification covers CPU reference only")
    if (
        qualification.get("implementation") != implementation
        or qualification.get("runtime") != runtime
    ):
        raise ValueError("Qualification implementation/runtime mismatch")
    if qualification.get("weights_sha256") != config["weights_sha256"]:
        raise ValueError("Qualification weights mismatch")
    if (
        type(qualification.get("maximum_supported_length")) is not int
        or not 1 <= qualification["maximum_supported_length"] <= 262144
    ):
        raise ValueError("Invalid explicit qualification length domain")
    if qualification.get("status") not in ["QUALIFIED_SCOPED", "UNQUALIFIED"]:
        raise ValueError("Explicit scoped qualification status required")
    config_hash = manifest_hash(config)
    contract = {
        "qualification": qualification,
        "qualification_file_sha256": qualification_file_sha256,
        "config": config,
        "implementation": implementation,
        "measurement_software": software_identity(),
        "runtime": runtime,
        "corpus_sha256": manifest["manifest_sha256"],
        "mode": "fixed-history; model transforms before sampler; temperature1",
    }
    contract_hash = manifest_hash(contract)
    version = "google-gemma4-gguf-" + contract_hash[:20]
    records = manifest["records"]
    if config.get("record_keys") is not None:
        wanted = config["record_keys"]
        records = [r for r in records if r["source"] + ":" + r["record_id"] in wanted]
        if len(records) != len(wanted):
            raise ValueError("Requested panel record missing/duplicate")
    if not records:
        raise ValueError("Empty reference panel")
    if max(len(r["token_ids"]) for r in records) > qualification.get(
        "maximum_supported_length", 0
    ):
        raise ValueError(
            "Corpus exceeds declared qualification support; explicit new qualification needed"
        )
    index = {
        "schema": "gemma4-reference-archive-v1",
        "complete": False,
        "reference_version": version,
        "contract": contract,
        "contract_hash": contract_hash,
        "vocab_size": 262144,
        "corpus": manifest,
        "cases": [],
        "qualification_packet": QUALIFICATION_PACKET_PATH,
    }
    old = None
    if (out / "index.json").exists():
        old = load_index(out / "index.json")
        if old.get("qualification_packet") != QUALIFICATION_PACKET_PATH:
            raise ValueError(
                "legacy archive has no qualification packet; keep it unchanged and use a new archive directory"
            )
        if old["contract_hash"] != contract_hash:
            raise ValueError(
                "Reference config/input/runtime changed: use a NEW archive directory"
            )
    else:
        copy_qualification_packet(
            qualification_path,
            qualification,
            out,
            expected_sha256=qualification_file_sha256,
        )
    reference = None
    computed = 0
    resumed = 0
    reference_forwards = 0
    max_forwards = config.get("max_reference_forwards")
    if type(max_forwards) is not int or max_forwards < 1:
        raise ValueError("Explicit finite max_reference_forwards required")

    def reference_forward(*a, **kw):
        nonlocal reference_forwards
        reference_forwards += 1
        if reference_forwards > max_forwards:
            raise ValueError("Reference forward budget exceeded")
        print(
            json.dumps(
                {
                    "event": "reference_forward",
                    "request": reference_forwards,
                    "tokens": len(a[0]),
                    "cached": kw.get("cache", False),
                }
            ),
            flush=True,
        )
        return reference.forward(*a, **kw)

    for record in records:
        pos = selected(record, config)
        spec = {
            "record": record,
            "logit_positions": pos,
            "observation_semantics": "input token index j is predicted by logit j-1; no target for absent successor",
        }
        identity = {
            "reference_version": version,
            "config_hash": config_hash,
            "input_hash": manifest_hash(spec),
            "contract_hash": contract_hash,
        }
        key = case_key(record)
        path = out / (key + ".npy")
        case = {
            "key": key,
            "record": record,
            "positions": pos,
            "identity": identity,
            "artifact": path.name,
            "group_id": group(record),
        }
        if old is not None and (
            not path.is_file()
            or not Path(str(path) + ".json").is_file()
            or Path(str(path) + ".incomplete").exists()
        ):
            raise ValueError(
                "Complete archive record missing/incomplete; explicit new archive required, no silent repair"
            )
        if (
            path.exists()
            or Path(str(path) + ".json").exists()
            or Path(str(path) + ".incomplete").exists()
        ):
            verify_resume(
                path,
                identity,
                expected_vocab_size=262144,
                expected_shape=(len(pos), 262144),
                expected_dtype=storage_dtype,
            )
            resumed += 1
        else:
            if reference is None:
                from .reference.stream import Reference

                reference = Reference(model, args.google_source, precision=precision)
            print(
                json.dumps(
                    {
                        "event": "reference_case_begin",
                        "key": key,
                        "source": record["source"],
                        "record_id": record["record_id"],
                        "tokens": len(record["token_ids"]),
                        "positions": pos,
                    }
                ),
                flush=True,
            )
            reference.reset()
            state = record["initial_state"]
            tokens = record["token_ids"]
            if state is None:
                z = reference_forward(tokens, pos)
            else:
                allowed = {"prefill_chunks", "decode_steps", "cache_action"}
                if set(state) - allowed or state.get("cache_action") != "reset":
                    raise ValueError(
                        "Unsupported state protocol; refusing silent fallback"
                    )
                chunks = state.get("prefill_chunks")
                if (
                    not isinstance(chunks, list)
                    or any(type(c) is not int or c < 1 for c in chunks)
                    or sum(chunks) != len(tokens)
                ):
                    raise ValueError("State chunks must partition exact input")
                decode_steps = state.get("decode_steps", 0)
                if (
                    type(decode_steps) is not int
                    or decode_steps < 0
                    or decode_steps > len(chunks)
                    or (decode_steps and chunks[-decode_steps:] != [1] * decode_steps)
                ):
                    raise ValueError(
                        "State decode_steps must match trailing unit chunks"
                    )
                protocol_chunks(case, "prefill")
                pieces = []
                offset = 0
                for count in chunks:
                    local = [p - offset for p in pos if offset <= p < offset + count]
                    zz = reference_forward(
                        tokens[offset : offset + count],
                        local,
                        cache=True,
                        capacity=len(tokens),
                    )
                    pieces.append(zz)
                    offset += count
                z = np.concatenate(pieces, axis=0)
            write_artifact(
                path,
                z,
                identity,
                expected_vocab_size=262144,
                storage_dtype=storage_dtype,
            )
            computed += 1
        index["cases"].append(case)
        if old is None:
            signed_index(out / "index.incomplete.json", index)
    if old is not None:
        if reference is not None:
            reference.close()
        invocation = {
            "wall_seconds": time.monotonic() - start,
            "computed_cases": computed,
            "resumed_cases": resumed,
            "requests": reference_forwards,
            "reference_version": version,
            "index_sha256": sha(out / "index.json"),
            "unchanged_archive": True,
        }
        save(
            out / "invocations" / ("resume-" + str(time.time_ns()) + ".json"),
            invocation,
        )
        print(json.dumps(invocation, indent=2))
        return
    index["complete"] = True
    index["build"] = {
        "wall_seconds": time.monotonic() - start,
        "computed_cases": computed,
        "resumed_cases": resumed,
        "requests": reference_forwards,
        "maximum_reserved_requests": max_forwards,
        "runtime": runtime,
    }
    signed_index(out / "index.json", index)
    if (out / "index.incomplete.json").exists():
        (out / "index.incomplete.json").unlink()
    if reference is not None:
        reference.close()
    print(
        json.dumps(
            {"archive": str(out), "reference_version": version, **index["build"]},
            indent=2,
        )
    )


def _compare(args):
    total_start = time.monotonic()
    archive = Path(args.archive)
    index = load_index(archive / "index.json")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    backend = read(args.backend_config)
    driver = Path(args.driver or backend.get("driver", ""))
    if not driver.is_file():
        raise ValueError(
            "Backend receipt must name an existing driver or --driver must be supplied"
        )
    if (
        backend.get("weights_sha256") != index["contract"]["config"]["weights_sha256"]
        or sha(args.model) != backend["weights_sha256"]
    ):
        raise ValueError("Backend weights mismatch")
    if sha(driver) != backend.get("driver_sha256"):
        raise ValueError("Backend driver changed")
    for p, digest in backend.get("input_pins", {}).items():
        if sha(p) != digest:
            raise ValueError("Backend dependency/source changed " + p)
    if not backend.get("input_pins") or not backend.get("backend"):
        raise ValueError("Backend provenance/pins required")
    # Verify every reference before starting the candidate; compare never imports JAX/model.
    refs = {}
    for case in index["cases"]:
        refs[case["key"]] = read_artifact(
            archive / case["artifact"],
            expected_identity=case["identity"],
            expected_vocab_size=262144,
            expected_shape=(len(case["positions"]), 262144),
            expected_dtype=archive_dtype(index),
        )[0]
    tsv = out / "inputs.tsv"
    tsv.write_text(
        "".join(
            c["key"]
            + "\t"
            + ",".join(map(str, c["record"]["token_ids"]))
            + "\t"
            + ",".join(map(str, c["positions"]))
            + "\t"
            + ",".join(map(str, protocol_chunks(c, args.mode)))
            + "\n"
            for c in index["cases"]
        )
    )
    context = max(64, max(len(c["record"]["token_ids"]) for c in index["cases"]))
    raw = out / "raw"
    raw.mkdir()
    start = time.monotonic()
    argv = [
        str(driver.resolve()),
        str(Path(args.model).resolve()),
        str(tsv.resolve()),
        str(raw.resolve()),
        str(context),
        args.mode,
    ]
    if getattr(args, "layers", None):
        layer_dir = Path(args.layers).resolve()
        layer_dir.mkdir(parents=True, exist_ok=False)
        argv.append(str(layer_dir))
    with (
        (out / "driver.stdout").open("wb") as stdout,
        (out / "driver.stderr").open("wb") as stderr,
    ):
        proc = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=args.timeout)
    driver_end = time.monotonic()
    if proc.returncode:
        raise ValueError("Candidate driver failed; preserve " + str(out))
    reports = []
    allkl = []
    alltv = []
    all_top1 = []
    groups = []
    domains = []
    nll_ref = []
    nll_can = []
    nll_delta = []
    nll_groups = []
    nll_domains = []
    for case in index["cases"]:
        b = refs[case["key"]]
        path = raw / (case["key"] + ".bin")
        want = len(case["positions"]) * 262144 * 4
        if path.stat().st_size != want:
            raise ValueError("Partial/full vocabulary output size mismatch")
        c = np.fromfile(path, dtype="<f4").reshape(len(case["positions"]), 262144)
        m = compare_logits(b, c, expected_vocab_size=262144)
        language = case["record"].get("language")
        stratum = case["record"]["domain"] + (":" + language if language else "")
        obs = {o["position"]: o["token_id"] for o in case["record"]["observed_tokens"]}
        nll = []
        for j, p in enumerate(case["positions"]):
            if p + 1 in obs:
                nm = compare_logits(
                    b[j : j + 1],
                    c[j : j + 1],
                    expected_vocab_size=262144,
                    observed_tokens=np.array([obs[p + 1]], np.int64),
                )
                nll.append(
                    {
                        "logit_position": p,
                        "observed_token_position": p + 1,
                        "observed_token_id": obs[p + 1],
                        **{k: float(v[0]) for k, v in nm.items() if "nll" in k},
                    }
                )
                nll_ref.append(float(nm["reference_nll_nats"][0]))
                nll_can.append(float(nm["candidate_nll_nats"][0]))
                nll_delta.append(float(nm["delta_nll_nats"][0]))
                nll_groups.append(case["group_id"])
                nll_domains.append(stratum)
        input_protocol = input_state_protocol(case["record"])
        reports.append(
            {
                "key": case["key"],
                "source": case["record"]["source"],
                "record_id": case["record"]["record_id"],
                "domain": case["record"]["domain"],
                "language": language,
                "stratum": stratum,
                "split": case["record"]["split"],
                "group_id": case["group_id"],
                "positions": case["positions"],
                "input_length": len(case["record"]["token_ids"]),
                "input_hash": input_spec_hash(
                    case["record"]["token_ids"], case["positions"], input_protocol
                ),
                "archive_input_hash": case["identity"]["input_hash"],
                "input_state_protocol": input_protocol,
                "candidate_chunks": protocol_chunks(case, args.mode),
                "metrics": {k: v.tolist() for k, v in m.items()},
                "nll": nll,
                "raw_sha256": sha(path),
            }
        )
        allkl.extend(m["kl_b_to_c_nats"].tolist())
        alltv.extend(m["tv"].tolist())
        groups.extend([case["group_id"]] * len(case["positions"]))
        domains.extend([stratum] * len(case["positions"]))
        all_top1.extend(
            (m["reference_top1"] != m["candidate_top1"]).astype(float).tolist()
        )
    report = {
        "schema": "gemma4-comparison-v1",
        "outcome": "INCONCLUSIVE",
        "reason": "Diagnostic fidelity measurements; no approved candidate acceptance thresholds",
        "mode": "fixed_history",
        "candidate_execution_mode": args.mode,
        "candidate_state_protocol": "declared chunks plus own cache"
        if args.mode == "prefill"
        else "forced one-token chunks; diagnostic state variant",
        "task_evaluation": None,
        "sampler": "none; complete post-softcap pre-sampler logits, temperature1",
        "backend": backend,
        "reference_version": index["reference_version"],
        "contract_hash": index["contract_hash"],
        "driver_argv": argv,
        "requested_histories": len(index["cases"]),
        "requested_engine_forward_calls": sum(
            len(protocol_chunks(c, args.mode)) for c in index["cases"]
        ),
        "requested_input_tokens": sum(
            len(c["record"]["token_ids"]) for c in index["cases"]
        ),
        "candidate_wall_seconds": time.monotonic() - total_start,
        "phase_seconds": {
            "input_and_archive_verification": start - total_start,
            "driver": driver_end - start,
            "metrics": time.monotonic() - driver_end,
        },
        "timer_scope": "compare function entry through metric/report construction; outer runner invocation includes Python startup and report writes",
        "reference_loaded": False,
        "reference_qualification": index["contract"].get(
            "qualification",
            {
                "status": "UNQUALIFIED",
                "limits": ["No qualification evidence in archive"],
            },
        ),
        "cases": reports,
        "summary": {
            "kl": group_summary(allkl, groups, domains=domains),
            "tv": group_summary(alltv, groups, domains=domains),
            "top1_disagreement": group_summary(all_top1, groups, domains=domains),
            "unique_group_count_across_strata": len(set(groups)),
            "nll": {
                "reference": group_summary(nll_ref, nll_groups, domains=nll_domains),
                "candidate": group_summary(nll_can, nll_groups, domains=nll_domains),
                "delta": group_summary(nll_delta, nll_groups, domains=nll_domains),
            }
            if nll_delta
            else {"status": "No observed successors at selected positions"},
        },
        "limits": [
            "Small fixed panel does not establish general fidelity or task quality",
            "No free-response task scores are inferred from fixed-history KL",
            "No global average across domains with unknown usage weights",
            "Groups use shared passage identity for Belebele EN/IT",
        ],
    }
    grouped_events = {}
    for row in reports:
        event = any(
            a != b
            for a, b in zip(
                row["metrics"]["reference_top1"], row["metrics"]["candidate_top1"]
            )
        )
        cell = grouped_events.setdefault(row["stratum"], {})
        cell[row["group_id"]] = cell.get(row["group_id"], False) or event
    report["summary"]["top1_group_events"] = {
        dom: {
            "group_count": len(events),
            "groups_with_any_disagreement": sum(events.values()),
            "zero_event_one_sided95_upper": 1 - 0.05 ** (1 / len(events))
            if not any(events.values())
            else None,
            "assumptions": "Binomial illustrative bound assumes independent representative groups; this opportunistic fixed panel does not establish that assumption or universal risk.",
        }
        for dom, events in grouped_events.items()
    }
    policy = read(args.policy) if getattr(args, "policy", None) else None
    report["quality_policy"] = policy
    report["quality_policy_sha256"] = (
        manifest_hash(policy) if policy is not None else None
    )
    report["outcome"], report["reason"] = evaluate_outcome(report, policy)
    save(out / "report.json", report)
    text = (
        "INCONCLUSIVE — diagnostic comparison; no approved acceptance thresholds.\n"
        + f"Reference status {report['reference_qualification']['status']}. Backend {backend['backend']}; {len(reports)} histories, {len(allkl)} full-vocabulary positions; {report['candidate_wall_seconds']:.3f}s including verification, driver load and metrics.\n"
    )
    if report["outcome"] == "PASS":
        text = (
            "PASS — all explicitly configured metric bounds met in the qualified declared domain.\n"
            + text.split("\n", 1)[1]
        )
    elif report["outcome"] == "FAIL":
        text = (
            "FAIL — one or more explicitly configured metric bounds exceeded.\n"
            + text.split("\n", 1)[1]
        )
    for dom, d in report["summary"]["kl"]["domains"].items():
        text += f"{dom}: KL mean of {d['group_count']} group means {d['mean_group_mean']:.8g}; max observed position {d['max_observed_position_value']:.8g}.\n"
    (out / "summary.txt").write_text(text)
    print(text, end="")
    return report


def compare(args):
    """Run a comparison and retain a machine-readable ERROR result on failure."""
    output = Path(args.output)
    existed = output.exists()
    try:
        return _compare(args)
    except Exception as exc:
        if not existed and output.is_dir():
            report = {
                "schema": "gemma4-comparison-v1",
                "outcome": "ERROR",
                "reason": f"{type(exc).__name__}: {exc}",
                "mode": "fixed_history",
                "reference_loaded": False,
            }
            policy_path = Path(args.policy) if getattr(args, "policy", None) else None
            if policy_path and policy_path.is_file():
                report["quality_policy_file_sha256"] = sha(policy_path)
                try:
                    policy = read(policy_path)
                    report["quality_policy"] = policy
                    report["quality_policy_sha256"] = manifest_hash(policy)
                except Exception:
                    report["quality_policy"] = None
            save(output / "report.json", report)
            (output / "summary.txt").write_text(
                "ERROR — " + report["reason"] + "\n", encoding="utf-8"
            )
            return report
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ins = sub.add_parser("inspect-weights")
    ins.add_argument("--model", required=True)
    ins.add_argument("--output", required=True)
    cor = sub.add_parser("prepare-corpus")
    cor.add_argument("--input", required=True)
    cor.add_argument("--output", required=True)
    cor.add_argument("--seed", default="gemma4-corpus-v1")
    ref = sub.add_parser("build-reference")
    for arg in [
        "model",
        "google-source",
        "corpus",
        "config",
        "qualification",
        "output",
    ]:
        ref.add_argument("--" + arg, required=True)
    cmp = sub.add_parser(
        "compare",
        help="compare and apply explicit policy bounds; without --policy this is diagnostic",
    )
    for arg in ["archive", "model", "backend-config", "output"]:
        cmp.add_argument("--" + arg, required=True)
    cmp.add_argument(
        "--driver", help="override the driver recorded in the backend receipt"
    )
    cmp.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    cmp.add_argument("--timeout", type=int, required=True)
    cmp.add_argument(
        "--policy", help="explicit xe-quality-policy-v1 JSON; caller owns all bounds"
    )
    cmp.add_argument(
        "--layers", help="write per-layer hidden-state diagnostics to a new directory"
    )
    diag = sub.add_parser(
        "diagnose", help="diagnostic comparison; outcome is always INCONCLUSIVE"
    )
    for arg in ["archive", "model", "backend-config", "output"]:
        diag.add_argument("--" + arg, required=True)
    diag.add_argument(
        "--driver", help="override the driver recorded in the backend receipt"
    )
    diag.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    diag.add_argument("--timeout", type=int, required=True)
    diag.add_argument(
        "--layers", help="write per-layer hidden-state diagnostics to a new directory"
    )
    prep_backend = sub.add_parser(
        "prepare-backend",
        help="prepare a pinned driver for the current public checkout",
    )
    prep_backend.add_argument("--source", required=True)
    prep_backend.add_argument("--model", required=True)
    prep_backend.add_argument("--output", required=True)
    prep_backend.add_argument("--make", default="make")
    prep_backend.add_argument("--jobs", type=int, default=2)
    sub.add_parser("self-check")
    verify = sub.add_parser("verify-archive")
    verify.add_argument("--archive", required=True)
    verify.add_argument(
        "--manifest", help="optional distribution file/checksum manifest"
    )
    export = sub.add_parser("export-archive")
    export.add_argument("--archive", required=True)
    export.add_argument("--output", required=True)
    export.add_argument(
        "--manifest", help="optional distribution file/checksum manifest"
    )
    imp = sub.add_parser(
        "import-archive",
        help="copy and verify an existing archive without changing its identity",
    )
    imp.add_argument("--archive", required=True)
    imp.add_argument("--output", required=True)
    imp.add_argument("--manifest", help="optional distribution file/checksum manifest")
    args = p.parse_args(argv)
    if args.command == "inspect-weights":
        from .gguf_import import GGUF
        from .reference.stream import audit

        with GGUF(args.model) as r:
            save(args.output, {"audit": audit(r), "inventory": r.inventory()})
    elif args.command == "prepare-corpus":
        prepare(args.input, args.output, seed=args.seed)
    elif args.command == "build-reference":
        build(args)
    elif args.command in ("compare", "diagnose"):
        if args.command == "diagnose":
            args.policy = None
        report = compare(args)
        if args.command == "diagnose" and report["outcome"] == "INCONCLUSIVE":
            return 0
        return {"PASS": 0, "FAIL": 1, "ERROR": 2, "INCONCLUSIVE": 3}[report["outcome"]]
    elif args.command == "prepare-backend":
        from .backend import prepare as prepare_backend

        prepare_backend(
            args.source, args.model, args.output, make=args.make, jobs=args.jobs
        )
    elif args.command == "self-check":
        metric = compare_logits(
            np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]]), expected_vocab_size=2
        )
        if float(metric["tv"][0]) != 0.0:
            raise RuntimeError("metric self-check failed")
        print("Software self-check passed; this does not qualify any model/reference.")
    else:
        if args.manifest:
            verify_distribution(args.archive, args.manifest)
        index = load_index(Path(args.archive) / "index.json")
        for c in index["cases"]:
            verify_resume(
                Path(args.archive) / c["artifact"],
                c["identity"],
                expected_vocab_size=262144,
                expected_shape=(len(c["positions"]), 262144),
                expected_dtype=archive_dtype(index),
            )
        if args.command in ("export-archive", "import-archive"):
            dest = Path(args.output)
            dest.mkdir(parents=True, exist_ok=False)
            if args.manifest:
                manifest = read(args.manifest)
                for name, spec in manifest["files"].items():
                    rel = Path(name)
                    source = (Path(args.archive).resolve() / rel).resolve()
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            else:
                for c in index["cases"]:
                    copy_verified_archive(
                        Path(args.archive) / c["artifact"],
                        dest / c["artifact"],
                        c["identity"],
                    )
                shutil.copyfile(Path(args.archive) / "index.json", dest / "index.json")
                if index.get("qualification_packet"):
                    copy_qualification_packet(
                        Path(args.archive) / "qualification" / "QUALIFICATION.json",
                        index["contract"]["qualification"],
                        dest,
                        expected_sha256=index["contract"]["qualification_file_sha256"],
                    )
            if sha(dest / "index.json") != sha(Path(args.archive) / "index.json"):
                raise ValueError("archive index changed during copy")
            load_index(dest / "index.json")
            if args.manifest:
                verify_distribution(dest, args.manifest)
        print("Verified complete archive:", len(index["cases"]), "histories")
    return 0


def entrypoint():
    try:
        return main()
    except Exception as e:
        print(
            json.dumps({"outcome": "ERROR", "error": f"{type(e).__name__}: {e}"}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(entrypoint())
