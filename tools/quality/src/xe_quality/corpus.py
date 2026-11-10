#!/usr/bin/env python3
"""Prepare and validate pinned, token-exact corpus manifests (stdlib only).

Input is explicit JSONL. Each line is one case and includes the provenance and
tokenization metadata; this module does not download datasets or tokenize text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "gemma4-quality-corpus-v1"
SUPPORTED_SOURCES = {
    "wikitext2_raw": "WikiText-2 raw; canonical validation/test split recommended",
    "ifeval": "IFEval original prompt population",
    "humaneval_plus": "EvalPlus HumanEval+",
    "belebele_en": "Belebele eng_Latn; group by shared passage across languages/questions",
    "belebele_it": "Belebele ita_Latn; group by shared passage across languages/questions",
    "ruler": "RULER synthetic configuration; pin generator/configuration/seed",
    "local_state": "local Xenolith state/cache/session cases",
}


class CorpusError(ValueError):
    """Invalid or incomplete corpus provenance/data."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require_text(
    row: dict[str, Any], key: str, line: int, *, allow_empty: bool = False
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise CorpusError(
            f"line {line}: {key} must be {'a string' if allow_empty else 'a non-empty string'}"
        )
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise CorpusError(f"line {line_no}: each JSONL item must be an object")
            rows.append(item)
    if not rows:
        raise CorpusError("source JSONL has no records")
    return rows


def _validate_source_row(row: dict[str, Any], line: int) -> dict[str, Any]:
    source = _require_text(row, "source", line)
    if source not in SUPPORTED_SOURCES:
        raise CorpusError(
            f"line {line}: unsupported source {source!r}; supported: {', '.join(SUPPORTED_SOURCES)}"
        )
    revision = _require_text(row, "dataset_revision", line)
    if revision.lower() in {
        "latest",
        "main",
        "master",
        "head",
        "unknown",
        "unspecified",
    }:
        raise CorpusError(
            f"line {line}: dataset_revision must be an immutable, explicit revision"
        )
    dataset_hash = _require_text(row, "dataset_content_sha256", line)
    if len(dataset_hash) != 64 or any(
        c not in "0123456789abcdef" for c in dataset_hash.lower()
    ):
        raise CorpusError(
            f"line {line}: dataset_content_sha256 must be a 64-digit SHA-256"
        )
    tokenizer = row.get("tokenizer")
    if (
        not isinstance(tokenizer, dict)
        or not _require_text(tokenizer, "identity", line)
        or not _require_text(tokenizer, "revision", line)
    ):
        raise CorpusError(
            f"line {line}: tokenizer.identity and tokenizer.revision are required"
        )
    template = row.get("template")
    if (
        not isinstance(template, dict)
        or not _require_text(template, "identity", line)
        or not _require_text(template, "revision", line)
    ):
        raise CorpusError(
            f"line {line}: template.identity and template.revision are required (use an explicit no-template identity where applicable)"
        )
    for key in ("record_id", "group_id", "document_id", "domain"):
        _require_text(row, key, line)
    for key in ("session_id", "leakage_group_id"):
        if (
            key in row
            and row[key] is not None
            and (not isinstance(row[key], str) or not row[key].strip())
        ):
            raise CorpusError(f"line {line}: {key} must be a non-empty string or null")
    tokens = row.get("token_ids")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise CorpusError(
            f"line {line}: token_ids must be a non-empty list of non-negative integers"
        )
    observations = row.get("observed_tokens", [])
    if not isinstance(observations, list):
        raise CorpusError(f"line {line}: observed_tokens must be a list")
    seen_positions: set[int] = set()
    for observation in observations:
        if (
            not isinstance(observation, dict)
            or type(observation.get("position")) is not int
            or type(observation.get("token_id")) is not int
        ):
            raise CorpusError(
                f"line {line}: observed_tokens entries need integer position and token_id"
            )
        position = observation["position"]
        if (
            position < 0
            or position >= len(tokens)
            or tokens[position] != observation["token_id"]
        ):
            raise CorpusError(
                f"line {line}: observed token at position {position} does not match token_ids"
            )
        if position in seen_positions:
            raise CorpusError(
                f"line {line}: duplicate observed-token position {position}"
            )
        seen_positions.add(position)
    if "initial_state" not in row:
        raise CorpusError(
            f"line {line}: initial_state must be present (null is allowed)"
        )
    if "split" in row and row["split"] not in ("development", "confirmation"):
        raise CorpusError(f"line {line}: split must be development or confirmation")
    return row


def _assigned_split(row: dict[str, Any], seed: str, fraction: float) -> str:
    if "split" in row:
        return row["split"]
    group = row.get("leakage_group_id") or row["group_id"]
    # Belebele language variants share one grouping namespace by design.
    namespace = (
        "belebele" if row["source"] in {"belebele_en", "belebele_it"} else row["source"]
    )
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{group}".encode("utf-8")).digest()
    return (
        "confirmation"
        if int.from_bytes(digest[:8], "big") / 2**64 < fraction
        else "development"
    )


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def prepare(
    source_jsonl: str | Path,
    output_manifest: str | Path,
    *,
    seed: str = "gemma4-corpus-v1",
    confirmation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Build a deterministic manifest from explicitly pinned source JSONL.

    Records with an explicit dataset split retain it. Otherwise split assignment
    is deterministic by source and leakage_group_id (or group_id). Group,
    document, session, and passage-linked records are checked for split leakage.
    """
    if not isinstance(seed, str) or not seed:
        raise CorpusError("seed must be a non-empty string")
    if not (0.0 < confirmation_fraction < 1.0):
        raise CorpusError("confirmation_fraction must be between 0 and 1")
    source_path = Path(source_jsonl)
    rows = [
        _validate_source_row(row, i)
        for i, row in enumerate(_read_jsonl(source_path), 1)
    ]

    source_identity: dict[str, tuple[Any, ...]] = {}
    ids: set[tuple[str, str]] = set()
    prepared: list[dict[str, Any]] = []
    for row in rows:
        source = row["source"]
        identity = (
            row["dataset_revision"],
            row["dataset_content_sha256"].lower(),
            row["tokenizer"],
            row["template"],
        )
        if source in source_identity and source_identity[source] != identity:
            raise CorpusError(
                f"source {source!r} has inconsistent dataset/tokenizer/template identity"
            )
        source_identity[source] = identity
        record_key = (source, row["record_id"])
        if record_key in ids:
            raise CorpusError(f"duplicate record_id {row['record_id']!r} in {source}")
        ids.add(record_key)
        split = _assigned_split(row, seed, confirmation_fraction)
        record = {
            "source": source,
            "dataset_revision": row["dataset_revision"],
            "dataset_content_sha256": row["dataset_content_sha256"].lower(),
            "tokenizer": row["tokenizer"],
            "template": row["template"],
            "record_id": row["record_id"],
            "record_content_sha256": _sha256(_canonical(row)),
            "group_id": row["group_id"],
            "document_id": row["document_id"],
            "session_id": row.get("session_id"),
            "leakage_group_id": row.get("leakage_group_id"),
            "domain": row["domain"],
            "split": split,
            "token_ids": row["token_ids"],
            "observed_tokens": sorted(
                row.get("observed_tokens", []), key=lambda item: item["position"]
            ),
            "initial_state": row["initial_state"],
        }
        if "text" in row:
            record["text"] = row["text"]
        if "passage_id" in row:
            record["passage_id"] = row["passage_id"]
        if "language" in row:
            record["language"] = row["language"]
        prepared.append(record)

    _validate_leakage(prepared)
    prepared.sort(key=lambda r: (r["source"], r["record_id"]))
    datasets = {}
    for source in sorted(source_identity):
        rev, digest, tokenizer, template = source_identity[source]
        datasets[source] = {
            "dataset_revision": rev,
            "dataset_content_sha256": digest,
            "tokenizer": tokenizer,
            "template": template,
            "population_status": "source records explicitly supplied; population completeness not asserted",
        }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "source_jsonl_sha256": _file_sha256(source_path),
        "split": {
            "seed": seed,
            "confirmation_fraction": confirmation_fraction,
            "algorithm": "sha256(seed\\0namespace\\0leakage_group_id) uint64 / 2^64; Belebele EN/IT share namespace; explicit source splits preserved",
        },
        "datasets": datasets,
        "record_count": len(prepared),
        "records": prepared,
    }
    manifest["manifest_sha256"] = _sha256(_canonical(manifest))
    _write_atomic(Path(output_manifest), _canonical(manifest) + b"\n")
    return manifest


def _validate_leakage(records: Iterable[dict[str, Any]]) -> None:
    group_splits: dict[tuple[str, str], str] = {}
    doc_splits: dict[tuple[str, str], str] = {}
    session_splits: dict[tuple[str, str], str] = {}
    passage_splits: dict[str, str] = {}
    declared_leakage_splits: dict[tuple[str, str], str] = {}
    for record in records:
        split = record["split"]
        group_key = (record["source"], record["group_id"])
        doc_key = (record["source"], record["document_id"])
        if group_key in group_splits and group_splits[group_key] != split:
            raise CorpusError(f"group split leakage: {group_key} occurs in both splits")
        if doc_key in doc_splits and doc_splits[doc_key] != split:
            raise CorpusError(
                f"document split leakage: {doc_key} occurs in both splits"
            )
        group_splits[group_key] = split
        doc_splits[doc_key] = split
        if record.get("session_id") is not None:
            key = (record["source"], record["session_id"])
            if key in session_splits and session_splits[key] != split:
                raise CorpusError(f"session split leakage: {key} occurs in both splits")
            session_splits[key] = split
        if record.get("leakage_group_id") is not None and record["source"] not in {
            "belebele_en",
            "belebele_it",
        }:
            namespace = (
                "belebele"
                if record["source"] in {"belebele_en", "belebele_it"}
                else record["source"]
            )
            key = (namespace, record["leakage_group_id"])
            if key in declared_leakage_splits and declared_leakage_splits[key] != split:
                raise CorpusError(
                    f"leakage-group split leakage: {key} occurs in both splits"
                )
            declared_leakage_splits[key] = split
        if record["source"] in {"belebele_en", "belebele_it"}:
            passage = (
                record.get("passage_id")
                or record.get("leakage_group_id")
                or record["group_id"]
            )
            if passage in passage_splits and passage_splits[passage] != split:
                raise CorpusError(
                    f"Belebele passage split leakage across languages/questions: {passage!r}"
                )
            passage_splits[passage] = split


def validate(manifest_path: str | Path) -> dict[str, Any]:
    """Validate schema, content integrity, token observations, and split isolation."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"cannot read manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise CorpusError(f"manifest schema must be {SCHEMA}")
    claimed_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if claimed_hash != _sha256(_canonical(unhashed)):
        raise CorpusError("manifest_sha256 mismatch")
    records = manifest.get("records")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise CorpusError("datasets must be an object")
    if not isinstance(records, list) or manifest.get("record_count") != len(records):
        raise CorpusError("record_count does not match records")
    seen: set[tuple[str, str]] = set()
    for i, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise CorpusError(f"manifest record {i} is not an object")
        for key in (
            "source",
            "dataset_revision",
            "dataset_content_sha256",
            "record_id",
            "group_id",
            "document_id",
            "domain",
            "split",
        ):
            if not isinstance(record.get(key), str) or not record[key]:
                raise CorpusError(f"manifest record {i}: missing {key}")
        if record["source"] not in SUPPORTED_SOURCES:
            raise CorpusError(
                f"manifest record {i}: unsupported source {record['source']!r}"
            )
        source_meta = datasets.get(record["source"])
        if not isinstance(source_meta, dict):
            raise CorpusError(
                f"manifest record {i}: missing dataset metadata for {record['source']}"
            )
        if any(
            record.get(key) != source_meta.get(key)
            for key in (
                "dataset_revision",
                "dataset_content_sha256",
                "tokenizer",
                "template",
            )
        ):
            raise CorpusError(f"manifest record {i}: dataset identity mismatch")
        for hash_key in ("dataset_content_sha256", "record_content_sha256"):
            value = record.get(hash_key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value.lower())
            ):
                raise CorpusError(f"manifest record {i}: invalid {hash_key}")
        if not isinstance(record.get("tokenizer"), dict) or not isinstance(
            record.get("template"), dict
        ):
            raise CorpusError(
                f"manifest record {i}: missing tokenizer/template identity"
            )
        if record["split"] not in {"development", "confirmation"}:
            raise CorpusError(f"manifest record {i}: invalid split")
        key = (record["source"], record["record_id"])
        if key in seen:
            raise CorpusError(f"manifest duplicate record {key}")
        seen.add(key)
        tokens = record.get("token_ids")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise CorpusError(f"manifest record {i}: invalid token_ids")
        observations = record.get("observed_tokens")
        if not isinstance(observations, list):
            raise CorpusError(f"manifest record {i}: invalid observed_tokens")
        seen_positions: set[int] = set()
        for observation in observations:
            if not isinstance(observation, dict):
                raise CorpusError(f"manifest record {i}: invalid observation")
            pos = observation.get("position")
            tok = observation.get("token_id")
            if (
                type(pos) is not int
                or type(tok) is not int
                or not 0 <= pos < len(tokens)
                or tokens[pos] != tok
            ):
                raise CorpusError(
                    f"manifest record {i}: observed token/position mismatch"
                )
            if pos in seen_positions:
                raise CorpusError(
                    f"manifest record {i}: duplicate observed-token position {pos}"
                )
            seen_positions.add(pos)
    _validate_leakage(records)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare", help="prepare a deterministic manifest from explicit JSONL"
    )
    prep.add_argument("source_jsonl", type=Path)
    prep.add_argument("output_manifest", type=Path)
    prep.add_argument("--seed", default="gemma4-corpus-v1")
    prep.add_argument("--confirmation-fraction", type=float, default=0.2)
    check = commands.add_parser("validate", help="validate an existing manifest")
    check.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = (
            prepare(
                args.source_jsonl,
                args.output_manifest,
                seed=args.seed,
                confirmation_fraction=args.confirmation_fraction,
            )
            if args.command == "prepare"
            else validate(args.manifest)
        )
    except CorpusError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "valid": True,
                "record_count": manifest["record_count"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
