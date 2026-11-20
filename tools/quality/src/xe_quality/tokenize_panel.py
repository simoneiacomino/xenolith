#!/usr/bin/env python3
"""Build an immutable token-exact panel from the pinned un-tokenized subset.

The XE and llama.cpp tokenizers are invoked independently on the exact UTF-8
text bytes. Their complete ID outputs must match before any limit is applied.
The model is only read for GGUF metadata/vocabulary tokenization; this script
does not run inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


BOS_ID = 2
VOCAB_SIZE = 262144
RULER_REVISION = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
TEMPLATE_TEXT = "raw-no-chat+explicit-bos2-v1; source text bytes are unchanged; no chat template; prepend BOS token id 2 after exact tokenizer agreement"


class TokenizeError(ValueError):
    """Input provenance, tokenizers, or output panel is invalid."""


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TokenizeError(f"value is not canonical JSON: {exc}") from exc


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TokenizeError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda x: (_ for _ in ()).throw(
                TokenizeError(f"invalid JSON number {x}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenizeError(f"cannot read {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TokenizeError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenizeError(f"cannot read JSONL {path}: {exc}") from exc
    if not rows:
        raise TokenizeError(f"no JSONL records in {path}")
    return rows


def verify_source_manifest(
    source_records: Path, source_manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(source_manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "gemma4-pinned-source-manifest-v1"
    ):
        raise TokenizeError("unsupported pinned source manifest schema")
    unhashed = dict(manifest)
    claimed_manifest = unhashed.pop("manifest_content_sha256", None)
    if claimed_manifest != digest(canonical(unhashed)):
        raise TokenizeError("pinned source manifest content hash mismatch")
    selected = manifest.get("selected_records")
    if not isinstance(selected, dict):
        raise TokenizeError("source manifest has no selected_records metadata")
    selected_path = (source_manifest_path.parent / selected.get("path", "")).resolve()
    if source_records.resolve() != selected_path:
        raise TokenizeError(
            "source-record path differs from pinned selected_records path"
        )
    if file_digest(source_records) != selected.get("sha256"):
        raise TokenizeError("un-tokenized source-record SHA-256 mismatch")
    rows = read_jsonl(source_records)
    if len(rows) != selected.get("count"):
        raise TokenizeError("source record count differs from pinned manifest")

    for source_meta in manifest.get("sources", {}).values():
        artifacts = source_meta.get("artifacts", [])
        artifacts = (
            list(artifacts.values()) if isinstance(artifacts, dict) else artifacts
        )
        for artifact in artifacts:
            rel = artifact.get("path")
            if not rel:
                raise TokenizeError("pinned artifact missing local path")
            path = source_manifest_path.parent / rel
            if (
                not path.is_file()
                or path.stat().st_size != artifact.get("size_bytes")
                or file_digest(path) != artifact.get("sha256")
            ):
                raise TokenizeError(f"source artifact hash mismatch: {path}")

    ids: set[tuple[str, str]] = set()
    for row in rows:
        source, rid = row.get("source"), row.get("record_id")
        if not isinstance(source, str) or not isinstance(rid, str) or not rid:
            raise TokenizeError("source row needs source and record_id")
        key = (source, rid)
        if key in ids:
            raise TokenizeError(f"duplicate source record {key}")
        ids.add(key)
        source_key = "belebele" if source in {"belebele_en", "belebele_it"} else source
        source_meta = manifest["sources"].get(source_key)
        if not isinstance(source_meta, dict):
            raise TokenizeError(
                f"source {source!r} is absent from pinned source manifest"
            )
        if row.get("dataset_revision") != source_meta.get("revision"):
            raise TokenizeError(f"dataset revision mismatch for {source}:{rid}")
        if not isinstance(row.get("text"), str):
            raise TokenizeError(f"text field missing for {source}:{rid}")
        source_artifact_hash = row.get("source_artifact_sha256")
        artifacts = source_meta.get("artifacts", [])
        artifacts = (
            list(artifacts.values()) if isinstance(artifacts, dict) else artifacts
        )
        pinned_hashes = {artifact.get("sha256") for artifact in artifacts}
        # Official RULER artifacts are supplied separately via --official-ruler-dir.
        if source != "ruler" and source_artifact_hash not in pinned_hashes:
            raise TokenizeError(f"record artifact is not pinned for {source}:{rid}")
        if source == "ruler" and source_artifact_hash not in pinned_hashes:
            raise TokenizeError(
                f"local RULER toy record has an unpinned source artifact: {rid}"
            )
    return manifest, rows


def model_metadata(model: Path) -> tuple[dict[str, Any], str]:
    from . import gguf_import as module

    try:
        gguf = module.GGUF(model)
    except Exception as exc:
        raise TokenizeError(f"cannot inspect GGUF metadata: {exc}") from exc
    try:
        metadata = dict(gguf.metadata)
    finally:
        gguf.close()
    if metadata.get("tokenizer.ggml.add_bos_token") is not False:
        raise TokenizeError(
            "GGUF tokenizer.ggml.add_bos_token must be false for both tokenizer commands"
        )
    bos_id = metadata.get("tokenizer.ggml.bos_token_id")
    if type(bos_id) is not int or bos_id != BOS_ID:
        raise TokenizeError(f"GGUF BOS token must be exactly {BOS_ID}, got {bos_id!r}")
    raw = {
        key: metadata[key]
        for key in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.bos_token_id")
    }
    return raw, digest(canonical(metadata))


def run_tokenizer(
    executable: Path, model: Path, text: str, *, timeout: int = 60
) -> list[int]:
    fd, name = tempfile.mkstemp(prefix="gemma4-tokenizer-input-", suffix=".txt")
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(text.encode("utf-8"))
        try:
            result = subprocess.run(
                [str(executable.resolve()), str(model.resolve()), str(path)],
                check=False,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.TimeoutExpired as exc:
            raise TokenizeError(
                f"tokenizer timed out after {timeout}s: {executable}"
            ) from exc
        if result.returncode:
            stderr = result.stderr.decode("utf-8", "replace")[-2000:]
            raise TokenizeError(
                f"tokenizer failed ({executable}, exit {result.returncode}): {stderr}"
            )
        try:
            output = result.stdout.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise TokenizeError(
                f"tokenizer emitted non-ASCII token IDs: {executable}"
            ) from exc
        if not output or "\n" in output or "\r" in output:
            raise TokenizeError(
                f"tokenizer must emit exactly one comma-separated ID line: {executable}"
            )
        try:
            ids = [int(part) for part in output.split(",")]
        except ValueError as exc:
            raise TokenizeError(
                f"tokenizer emitted malformed comma-separated IDs: {executable}"
            ) from exc
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise TokenizeError(
                f"tokenizer emitted empty or negative token IDs: {executable}"
            )
        if ids[0] == BOS_ID:
            raise TokenizeError(
                f"tokenizer unexpectedly emitted BOS {BOS_ID}; add_bos=false is required"
            )
        return ids
    finally:
        path.unlink(missing_ok=True)


def _official_ruler(
    directory: Path, source_meta: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source_meta.get("revision") != RULER_REVISION:
        raise TokenizeError(
            "official RULER input must match the pinned RULER source revision"
        )
    receipt_path = directory / "GENERATION.json"
    if not receipt_path.is_file():
        raise TokenizeError(
            "official RULER requires a GENERATION.json provenance receipt"
        )
    receipt_bytes = receipt_path.read_bytes()
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise TokenizeError("GENERATION.json must be an object")
    if receipt.get("upstream_commit") != RULER_REVISION:
        raise TokenizeError(
            "RULER generation receipt upstream_commit does not match pinned source revision"
        )
    if receipt.get("task") != "niah_single_1":
        raise TokenizeError("RULER generation receipt must identify task niah_single_1")
    if type(receipt.get("num_samples")) is not int or receipt["num_samples"] < 2:
        raise TokenizeError(
            "RULER generation receipt num_samples must be an integer >= 2"
        )
    source_pins = receipt.get("source_pins")
    if not isinstance(source_pins, dict):
        raise TokenizeError("RULER generation receipt source_pins must be an object")
    niah_pin = None
    for key, value in source_pins.items():
        normalized = str(key).replace("\\", "/")
        if normalized.endswith("scripts/data/synthetic/niah.py"):
            niah_pin = value
            break
    if isinstance(niah_pin, dict):
        niah_pin = niah_pin.get("sha256")
    if (
        not isinstance(niah_pin, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", niah_pin) is None
    ):
        raise TokenizeError(
            "RULER source_pins must pin scripts/data/synthetic/niah.py by SHA-256"
        )

    output_pins = receipt.get("output_pins")
    if not isinstance(output_pins, dict) or not output_pins:
        raise TokenizeError(
            "RULER generation receipt output_pins must map JSONL paths to SHA-256"
        )
    directory = directory.resolve()
    pinned_files: list[tuple[Path, str, int]] = []
    for relative, value in output_pins.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise TokenizeError(
                f"invalid relative path in RULER output_pins: {relative!r}"
            )
        if Path(relative).suffix.lower() != ".jsonl":
            raise TokenizeError(
                f"RULER output_pins may list only generated JSONL files: {relative}"
            )
        expected_sha = value.get("sha256") if isinstance(value, dict) else value
        if (
            not isinstance(expected_sha, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha) is None
        ):
            raise TokenizeError(f"RULER output pin must be a SHA-256: {relative}")
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise TokenizeError(
                f"RULER pinned generated JSONL is missing or escapes its directory: {relative}"
            )
        data = path.read_bytes()
        actual_sha = digest(data)
        if actual_sha != expected_sha.lower():
            raise TokenizeError(f"RULER generated-file SHA-256 mismatch: {relative}")
        pinned_files.append((path, actual_sha, len(data)))

    files = sorted(
        pinned_files, key=lambda item: item[0].relative_to(directory).as_posix()
    )
    if not files:
        raise TokenizeError(
            "--official-ruler-dir contains no receipt-pinned generated JSONL files"
        )
    file_meta = []
    candidates = []
    for path, file_sha, file_size in files:
        data = path.read_bytes()
        file_meta.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": file_sha,
                "size_bytes": file_size,
            }
        )
        try:
            source_rows = [
                json.loads(line)
                for line in data.decode("utf-8").splitlines()
                if line.strip()
            ]
            if not isinstance(source_rows, list) or any(
                not isinstance(row, dict) for row in source_rows
            ):
                raise TokenizeError(f"unsupported RULER JSON structure: {path}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TokenizeError(f"invalid RULER generated file {path}: {exc}") from exc
        for line_no, row in enumerate(source_rows, 1):
            text_key = next(
                (
                    key
                    for key in ("input", "prompt", "text")
                    if isinstance(row.get(key), str)
                ),
                None,
            )
            if text_key is None:
                continue
            candidates.append(
                {
                    "path": path,
                    "file_sha256": file_sha,
                    "line_no": line_no,
                    "text_key": text_key,
                    "source_row": row,
                    "text": row[text_key],
                }
            )
    if len(candidates) < 2:
        raise TokenizeError(
            "official RULER directory must contain at least two records with a full input/prompt/text field"
        )
    if len(candidates) != receipt["num_samples"]:
        raise TokenizeError(
            "RULER generated row count differs from num_samples declared in GENERATION.json"
        )
    receipt_sha256 = digest(receipt_bytes)
    bundle_hash = digest(
        canonical(
            {
                "source_revision": RULER_REVISION,
                "generation_receipt_sha256": receipt_sha256,
                "files": file_meta,
            }
        )
    )
    selected = candidates[:2]
    output = []
    for index, item in enumerate(selected):
        row = item["source_row"]
        raw_id = row.get(
            "sample_id", row.get("id", row.get("task_id", item["line_no"]))
        )
        output.append(
            {
                "source": "ruler",
                "dataset_revision": RULER_REVISION,
                "dataset_content_sha256": bundle_hash,
                "source_artifact_sha256": item["file_sha256"],
                "source_file": item["path"].relative_to(directory).as_posix(),
                "record_id": f"official:{item['path'].relative_to(directory).as_posix()}:{raw_id}",
                "group_id": f"ruler-official:{item['path'].relative_to(directory).as_posix()}:{raw_id}",
                "document_id": f"ruler-official:{item['path'].relative_to(directory).as_posix()}:{raw_id}",
                "domain": str(
                    row.get("task", row.get("task_name", "ruler-official-synthetic"))
                ),
                "source_split": "development" if index == 0 else "confirmation",
                "text": item["text"],
                "source_fields": {
                    key: value for key, value in row.items() if key != item["text_key"]
                },
            }
        )
    return output, {
        "revision": RULER_REVISION,
        "generation_identity": receipt,
        "generation_receipt_sha256": receipt_sha256,
        "files": file_meta,
        "bundle_sha256": bundle_hash,
        "selected_record_ids": [row["record_id"] for row in output],
        "selection": "first two eligible records in sorted path/file order; first development, second confirmation; full input retained without token truncation",
    }


def split_for(row: dict[str, Any], belebele_links: list[str]) -> str:
    source = row["source"]
    source_split = row.get("source_split")
    if source == "wikitext2_raw":
        mapping = {"validation": "development", "test": "confirmation"}
        if source_split not in mapping:
            raise TokenizeError(f"unexpected WikiText source split {source_split!r}")
        return mapping[source_split]
    if source in {"ifeval", "humaneval_plus"}:
        if source_split not in {"development", "confirmation"}:
            raise TokenizeError(
                f"{source} selection needs fixed development/confirmation source split"
            )
        return source_split
    if source in {"belebele_en", "belebele_it"}:
        link = row.get("source_fields", {}).get("link")
        if link not in belebele_links or len(belebele_links) != 2:
            raise TokenizeError(
                "Belebele source link is absent from the two pinned passage groups"
            )
        return "development" if belebele_links.index(link) == 0 else "confirmation"
    if source == "local_state":
        if source_split != "development":
            raise TokenizeError("local state fixtures must remain in development")
        return "development"
    raise TokenizeError(f"unrecognized source split mapping for {source}")


def chunk_state(record_id: str, token_count: int) -> dict[str, Any] | None:
    if record_id.startswith("state-prefill-full-"):
        return None
    if not record_id.startswith("state-prefill-chunked-decode-"):
        return None
    if token_count < 2:
        raise TokenizeError("chunked/decode local-state case needs at least two tokens")
    # The first prefill chunk ends immediately after the default midpoint logit
    # position. The final selected logit is in a one-token decode chunk. Up to
    # eight trailing IDs are fed one at a time; any earlier remainder stays a
    # prefill chunk.
    midpoint = max(0, token_count // 2 - 1)
    decode_steps = min(8, token_count - midpoint - 1)
    first_prefill = midpoint + 1
    middle_prefill = token_count - first_prefill - decode_steps
    chunks = [first_prefill]
    if middle_prefill:
        chunks.append(middle_prefill)
    chunks.extend([1] * decode_steps)
    if sum(chunks) != token_count:
        raise TokenizeError("internal local-state chunk partition error")
    return {
        "prefill_chunks": chunks,
        "decode_steps": decode_steps,
        "cache_action": "reset",
    }


def tokenize_panel(
    source_records: str | Path,
    source_manifest: str | Path,
    model: str | Path,
    expected_model_sha256: str,
    xe_tokenizer: str | Path,
    llama_tokenizer: str | Path,
    output: str | Path,
    max_regular_tokens: int,
    official_ruler_dir: str | Path | None = None,
) -> dict[str, Any]:
    source_records, source_manifest = Path(source_records), Path(source_manifest)
    model, xe_tokenizer, llama_tokenizer, output = map(
        Path, (model, xe_tokenizer, llama_tokenizer, output)
    )
    if type(max_regular_tokens) is not int or max_regular_tokens < 1:
        raise TokenizeError("max_regular_tokens must be a positive integer")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_model_sha256):
        raise TokenizeError("expected_model_sha256 must be 64 hexadecimal characters")
    expected_model_sha256 = expected_model_sha256.lower()
    for path in (source_records, source_manifest, model, xe_tokenizer, llama_tokenizer):
        if not path.is_file():
            raise TokenizeError(f"required file does not exist: {path}")
    actual_model_sha256 = file_digest(model)
    if actual_model_sha256 != expected_model_sha256:
        raise TokenizeError("exact model SHA-256 mismatch")
    for exe in (xe_tokenizer, llama_tokenizer):
        if not os.access(exe, os.X_OK):
            raise TokenizeError(f"tokenizer is not executable: {exe}")
    output_audit = Path(str(output) + ".audit.json")
    if output.exists() or output_audit.exists():
        raise TokenizeError(
            "output JSONL and audit paths must both be new (immutable output)"
        )
    if output.resolve() in {
        source_records.resolve(),
        source_manifest.resolve(),
        model.resolve(),
        xe_tokenizer.resolve(),
        llama_tokenizer.resolve(),
    }:
        raise TokenizeError(
            "output path must not replace an input, model, or tokenizer executable"
        )
    metadata, metadata_sha256 = model_metadata(model)
    manifest, source_rows = verify_source_manifest(source_records, source_manifest)
    source_section_hashes = {
        key: digest(canonical(value)) for key, value in manifest["sources"].items()
    }
    source_by_id = {(row["source"], row["record_id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise TokenizeError("duplicate source records")

    ruler_toys = [row for row in source_rows if row["source"] == "ruler"]
    excluded = [
        {
            "source": row["source"],
            "record_id": row["record_id"],
            "reason": "local NIAH toy case excluded from official panel",
        }
        for row in ruler_toys
    ]
    panel_rows = [row for row in source_rows if row["source"] != "ruler"]
    if official_ruler_dir is not None:
        official, official_meta = _official_ruler(
            Path(official_ruler_dir), manifest["sources"].get("ruler", {})
        )
        panel_rows.extend(official)
    else:
        official, official_meta = [], None
    belebele_meta = manifest["sources"].get("belebele", {})
    belebele_links = belebele_meta.get("selected_passage_links", [])
    if not isinstance(belebele_links, list) or len(belebele_links) != 2:
        raise TokenizeError(
            "pinned manifest must define exactly two Belebele passage links"
        )

    template_revision = digest(TEMPLATE_TEXT.encode("utf-8"))
    xe_sha, llama_sha = file_digest(xe_tokenizer), file_digest(llama_tokenizer)
    tokenizer_identity = "Gemma4 GGUF vocabulary; XE/llama.cpp full-ID agreement"
    tokenizer_revision = digest(
        canonical(
            {
                "model_sha256": actual_model_sha256,
                "xe_tokenizer_sha256": xe_sha,
                "llama_tokenizer_sha256": llama_sha,
                "bos": metadata,
            }
        )
    )
    output_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    for source_row in panel_rows:
        source, rid = source_row["source"], source_row["record_id"]
        is_official_ruler = source == "ruler" and "source_file" in source_row
        text = source_row["text"]
        xe_ids = run_tokenizer(xe_tokenizer, model, text)
        llama_ids = run_tokenizer(llama_tokenizer, model, text)
        xe_hash, llama_hash = digest(canonical(xe_ids)), digest(canonical(llama_ids))
        if xe_ids != llama_ids:
            raise TokenizeError(
                f"tokenizer IDs differ before BOS/cutoff for {source}:{rid}; XE={xe_hash} llama.cpp={llama_hash}"
            )
        if any(token >= VOCAB_SIZE for token in xe_ids):
            raise TokenizeError(
                f"token ID outside GGUF vocabulary [0,{VOCAB_SIZE}) for {source}:{rid}"
            )
        full_regular_count = len(xe_ids)
        if is_official_ruler:
            retained_regular = xe_ids
        else:
            retained_regular = xe_ids[:max_regular_tokens]
        truncated = len(retained_regular) != full_regular_count
        token_ids = [BOS_ID, *retained_regular]
        if len(token_ids) < 2:
            raise TokenizeError(f"empty token sequence after BOS for {source}:{rid}")

        if is_official_ruler:
            split = source_row["source_split"]
            section_hash = official_meta["bundle_sha256"]
            dataset_hash = section_hash
            passage_id = None
            session_id = None
            leakage_group_id = None
            group_id = source_row["group_id"]
            doc_id = source_row["document_id"]
            revision = RULER_REVISION
            artifact_sha = source_row["source_artifact_sha256"]
            source_fields = source_row["source_fields"]
            domain = source_row["domain"]
            source_split = split
        else:
            source_key = (
                "belebele" if source in {"belebele_en", "belebele_it"} else source
            )
            section_hash = source_section_hashes[source_key]
            dataset_hash = section_hash
            split = split_for(source_row, belebele_links)
            revision = source_row["dataset_revision"]
            artifact_sha = source_row["source_artifact_sha256"]
            source_fields = source_row.get("source_fields", {})
            domain = source_row["domain"]
            source_split = source_row.get("source_split")
            doc_id = source_row["document_id"]
            session_id = source_row.get("session_id")
            passage_id = None
            leakage_group_id = source_row.get("leakage_group_id")
            group_id = source_row["group_id"]
            if source in {"belebele_en", "belebele_it"}:
                link = source_fields.get("link")
                passage_id = f"belebele-passage:{link}"
                leakage_group_id = passage_id
            if source == "local_state":
                group_id = "local-state-json17-family"
                leakage_group_id = group_id
        state = chunk_state(rid, len(token_ids)) if source == "local_state" else None
        observed = [
            {"position": pos, "token_id": token_ids[pos]}
            for pos in range(1, len(token_ids))
        ]
        record = {
            "source": source,
            "dataset_revision": revision,
            "dataset_content_sha256": dataset_hash,
            "source_manifest_section_sha256": section_hash,
            "source_artifact_sha256": artifact_sha,
            "tokenizer": {
                "identity": tokenizer_identity,
                "revision": tokenizer_revision,
            },
            "template": {
                "identity": "raw-no-chat+explicit-bos2",
                "revision": template_revision,
            },
            "record_id": rid,
            "group_id": group_id,
            "document_id": doc_id,
            "session_id": session_id,
            "leakage_group_id": leakage_group_id,
            "passage_id": passage_id,
            "language": source_row.get("language")
            or source_row.get("source_fields", {}).get("dialect"),
            "domain": domain,
            "split": split,
            "source_split": source_split,
            "text": text,
            "token_ids": token_ids,
            "observed_tokens": observed,
            "initial_state": state,
            "source_fields": source_fields,
        }
        output_records.append(record)
        audit_records.append(
            {
                "source": source,
                "record_id": rid,
                "text_sha256": digest(text.encode("utf-8")),
                "xe_full_token_ids_sha256": xe_hash,
                "llama_full_token_ids_sha256": llama_hash,
                "full_regular_token_count": full_regular_count,
                "tokenizer_agreement": True,
                "preexisting_bos": False,
                "bos_id_prepended": BOS_ID,
                "regular_token_limit": None
                if is_official_ruler
                else max_regular_tokens,
                "truncation_applied": truncated,
                "truncated_regular_token_count": full_regular_count
                - len(retained_regular),
                "retained_regular_token_count": len(retained_regular),
                "final_input_token_count": len(token_ids),
                "final_input_token_ids_sha256": digest(canonical(token_ids)),
                "selected_logit_positions_default": sorted(
                    set([max(0, len(token_ids) // 2 - 1), len(token_ids) - 1])
                ),
                "observed_token_rule": "one record per input token j in [1,n): observed position j has token_ids[j]; logit j-1 predicts it",
                "state_protocol": state,
                "official_ruler_full_input": bool(is_official_ruler),
                "claim_scope": "sampled-prefix-fidelity-only; not an official task score"
                if source == "local_state"
                else None,
            }
        )

    output_records.sort(key=lambda row: (row["source"], row["record_id"]))
    audit_records.sort(key=lambda row: (row["source"], row["record_id"]))
    if not output_records:
        raise TokenizeError("empty output panel")
    output_bytes = b"".join(canonical(row) + b"\n" for row in output_records)
    audit = {
        "schema": "gemma4-tokenized-panel-audit-v1",
        "source_records_sha256": file_digest(source_records),
        "source_manifest_sha256": manifest["manifest_content_sha256"],
        "output_records_sha256": digest(output_bytes),
        "output_record_count": len(output_records),
        "excluded_records": excluded,
        "official_ruler": official_meta,
        "model": {
            "path_basename": model.name,
            "sha256": actual_model_sha256,
            "metadata_sha256": metadata_sha256,
            "raw_metadata": metadata,
        },
        "tokenizers": {
            "xe": {"path_basename": xe_tokenizer.name, "sha256": xe_sha},
            "llama_cpp": {"path_basename": llama_tokenizer.name, "sha256": llama_sha},
            "timeout_seconds": 60,
            "required_invocation": "[executable, model, exact UTF-8 text file]",
        },
        "bos_policy": {
            "gguf_add_bos_token": metadata["tokenizer.ggml.add_bos_token"],
            "gguf_bos_token_id": metadata["tokenizer.ggml.bos_token_id"],
            "tokenizer_output_must_not_start_with_bos": True,
            "explicitly_prepended_bos_id": BOS_ID,
        },
        "template": {
            "identity": "raw-no-chat+explicit-bos2",
            "revision": template_revision,
            "specification": TEMPLATE_TEXT,
        },
        "preprocessing": {
            "text_encoding": "UTF-8",
            "normalization": "none",
            "chat_template": None,
            "tokenizer_bos": False,
            "explicit_bos_token_id": BOS_ID,
        },
        "regular_token_limit": max_regular_tokens,
        "official_ruler_records_are_not_truncated": True,
        "records": audit_records,
    }
    audit["audit_sha256"] = digest(canonical(audit))
    audit_bytes = canonical(audit) + b"\n"
    audit_path = output_audit
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_pair(output, output_bytes, audit_path, audit_bytes)
    return audit


def _atomic_pair(
    output: Path, output_bytes: bytes, audit: Path, audit_bytes: bytes
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temps = []
    try:
        for path, payload in ((output, output_bytes), (audit, audit_bytes)):
            fd, name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
            temps.append(name)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temps[0], output)
        os.replace(temps[1], audit)
    except BaseException:
        for name in temps:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-records", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--xe-tokenizer", required=True, type=Path)
    parser.add_argument("--llama-tokenizer", required=True, type=Path)
    parser.add_argument("--official-ruler-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-regular-tokens", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        result = tokenize_panel(
            args.source_records,
            args.source_manifest,
            args.model,
            args.expected_model_sha256,
            args.xe_tokenizer,
            args.llama_tokenizer,
            args.output,
            args.max_regular_tokens,
            args.official_ruler_dir,
        )
    except TokenizeError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit": str(args.output) + ".audit.json",
                "record_count": result["output_record_count"],
                "output_records_sha256": result["output_records_sha256"],
                "excluded_records": len(result["excluded_records"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
