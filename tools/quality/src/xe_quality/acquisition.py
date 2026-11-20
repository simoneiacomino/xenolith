#!/usr/bin/env python3
"""Acquire tiny, pinned public source subsets; emit un-tokenized source records.

Network access is needed only when this acquisition command is run. All URLs
are public and unauthenticated. The script never downloads a model or executes
an inference backend.
"""

from __future__ import annotations

import gzip
import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources"
RAW = SOURCES / "raw"
PINNED = {
    "wikitext2_raw": {
        "repository": "https://huggingface.co/datasets/Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "config": "wikitext-2-raw-v1",
        "splits": ["validation", "test"],
    },
    "ifeval": {
        "repository": "https://github.com/google-research/google-research",
        "revision": "26d8ccdab6fec61b5c83ad6327ea8bda9e580288",
        "path": "instruction_following_eval/data/input_data.jsonl",
    },
    "humaneval_plus": {
        "repository": "https://github.com/evalplus/humanevalplus_release",
        "revision": "68cd26d53a0dec69f85eafe1f82a2a74155a2bd6",
        "tag": "v0.1.10",
        "path": "HumanEvalPlus.jsonl.gz",
    },
    "belebele": {
        "repository": "https://huggingface.co/datasets/facebook/belebele",
        "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
        "languages": {"belebele_en": "eng_Latn", "belebele_it": "ita_Latn"},
        "split": "test",
    },
    "ruler": {
        "repository": "https://github.com/NVIDIA/RULER",
        "revision": "db43f92ff634bbce1c33d1fb15858aec4bba509a",
        "path": "scripts/synthetic.yaml",
    },
    "local_state": {
        "revision": "local-state-cases-v1",
        "description": "authored deterministic state-transition fixtures; not a published benchmark",
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def get(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "gemma4-corpus-acquisition/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def save(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(
            f"refusing to replace different existing source file: {path}"
        )
    path.write_bytes(data)
    try:
        display_path = str(path.resolve().relative_to(HERE.resolve()))
    except ValueError:
        # Explicit output directories may live outside the package; avoid
        # embedding a machine-specific absolute path in the receipt.
        display_path = path.name
    return {"path": display_path, "size_bytes": len(data), "sha256": sha256(data)}


def rows_api(
    dataset: str, config: str, split: str, offset: int, length: int, revision: str
) -> tuple[bytes, list[dict[str, Any]]]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
            "revision": revision,
        }
    )
    data = get("https://datasets-server.huggingface.co/rows?" + query)
    parsed = json.loads(data)
    return data, [item["row"] for item in parsed.get("rows", [])]


def add_record(
    records: list[dict[str, Any]],
    *,
    source: str,
    revision: str,
    source_sha256: str,
    record_id: str,
    group_id: str,
    document_id: str,
    domain: str,
    text: str,
    split: str,
    fields: dict[str, Any] | None = None,
    session_id: str | None = None,
    leakage_group_id: str | None = None,
) -> None:
    row: dict[str, Any] = {
        "source": source,
        "dataset_revision": revision,
        "source_artifact_sha256": source_sha256,
        "record_id": record_id,
        "group_id": group_id,
        "leakage_group_id": leakage_group_id,
        "document_id": document_id,
        "session_id": session_id,
        "domain": domain,
        "source_split": split,
        "text": text,
        "tokenization_status": "not-tokenized; exact independent GGUF tokenizer/template metadata is required before freezing the corpus",
    }
    if fields:
        row["source_fields"] = fields
    records.append(row)


def acquire_wikitext(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    meta = PINNED["wikitext2_raw"]
    result: dict[str, Any] = {
        **meta,
        "artifacts": [],
        "selection": "first two complete article documents in each canonical validation/test split, retaining source order",
    }
    for split in ("validation", "test"):
        all_rows: list[dict[str, Any]] = []
        page_offset = 0
        while True:
            data, batch = rows_api(
                "Salesforce/wikitext",
                meta["config"],
                split,
                page_offset,
                100,
                meta["revision"],
            )
            if not batch:
                raise RuntimeError(
                    f"WikiText viewer returned no rows at offset {page_offset}"
                )
            result["artifacts"].append(
                save(RAW / f"wikitext2-{split}-rows-{page_offset:05d}.json", data)
            )
            all_rows.extend(batch)
            starts = [
                i
                for i, r in enumerate(all_rows)
                if _is_wikitext_article_title(r.get("text", ""))
            ]
            if len(starts) >= 3:
                break
            page_offset += len(batch)
        starts = [
            i
            for i, r in enumerate(all_rows)
            if _is_wikitext_article_title(r.get("text", ""))
        ]
        for doc_no in range(2):
            start = starts[doc_no]
            end = starts[doc_no + 1] if doc_no + 1 < len(starts) else len(all_rows)
            title = re.sub(r"^\s*=\s+|\s+=\s*$", "", all_rows[start]["text"]).strip()
            doc_rows = all_rows[start:end]
            selected_raw = save(
                RAW / f"wikitext2-{split}-doc-{doc_no + 1}.json",
                canonical(doc_rows) + b"\n",
            )
            result["artifacts"].append(selected_raw)
            text = "".join(row.get("text", "") for row in doc_rows).rstrip("\n")
            group = f"wikitext2:{split}:{title}"
            add_record(
                records,
                source="wikitext2_raw",
                revision=meta["revision"],
                source_sha256=selected_raw["sha256"],
                record_id=group,
                group_id=group,
                document_id=group,
                domain="english-encyclopedic-text",
                text=text,
                split=split,
            )
    manifest["sources"]["wikitext2_raw"] = result


def _is_wikitext_article_title(text: str) -> bool:
    # Article titles have one surrounding '='; section headings have doubled
    # delimiters and must not terminate a document.
    return bool(re.match(r"^\s*=\s+(?![=]).+?(?<![=])\s+=\s*$", text))


def acquire_ifeval(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    meta = PINNED["ifeval"]
    url = f"https://raw.githubusercontent.com/google-research/google-research/{meta['revision']}/{meta['path']}"
    data = get(url)
    artifact = save(RAW / "ifeval-input_data.jsonl", data)
    parsed = [
        json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()
    ]
    selected = sorted(parsed, key=lambda x: int(x["key"]))[:4]
    for index, item in enumerate(selected):
        key = str(item["key"])
        add_record(
            records,
            source="ifeval",
            revision=meta["revision"],
            source_sha256=artifact["sha256"],
            record_id=key,
            group_id=f"ifeval:{key}",
            document_id=f"ifeval:{key}",
            domain="instruction-following",
            text=item["prompt"],
            split="development" if index < 2 else "confirmation",
            fields={
                "key": item["key"],
                "instruction_id_list": item["instruction_id_list"],
                "kwargs": item["kwargs"],
            },
        )
    manifest["sources"]["ifeval"] = {
        **meta,
        "artifacts": [artifact],
        "raw_record_count": len(parsed),
        "selected_record_ids": [str(x["key"]) for x in selected],
        "selection": "four lowest integer keys; first two development and last two confirmation (preselected subset, not official score)",
    }


def acquire_humaneval(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    meta = PINNED["humaneval_plus"]
    url = f"https://raw.githubusercontent.com/evalplus/humanevalplus_release/{meta['revision']}/{meta['path']}"
    data = get(url)
    artifact = save(RAW / "humanevalplus-v0.1.10.jsonl.gz", data)
    parsed = [
        json.loads(line)
        for line in gzip.decompress(data).decode("utf-8").splitlines()
        if line.strip()
    ]
    selected = sorted(parsed, key=lambda x: int(x["task_id"].rsplit("/", 1)[-1]))[:4]
    for index, item in enumerate(selected):
        task = item["task_id"]
        add_record(
            records,
            source="humaneval_plus",
            revision=meta["revision"],
            source_sha256=artifact["sha256"],
            record_id=task,
            group_id=f"humanevalplus:{task}",
            document_id=f"humanevalplus:{task}",
            domain="python-code-generation",
            text=item["prompt"],
            split="development" if index < 2 else "confirmation",
            fields={k: v for k, v in item.items() if k != "prompt"},
        )
    manifest["sources"]["humaneval_plus"] = {
        **meta,
        "artifacts": [artifact],
        "raw_record_count": len(parsed),
        "selected_record_ids": [x["task_id"] for x in selected],
        "selection": "four lowest numeric task IDs; first two development and last two confirmation (preselected subset, not official score)",
    }


def acquire_belebele(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    meta = PINNED["belebele"]
    result: dict[str, Any] = {
        **meta,
        "artifacts": {},
        "selection": "two lowest source-order passage links in eng_Latn test, with all questions for those passages in both language variants",
    }
    english_raw, english_rows = rows_api(
        "facebook/belebele", "eng_Latn", "test", 0, 20, meta["revision"]
    )
    english_artifact = save(RAW / "belebele-eng_Latn-test-first20.json", english_raw)
    links = []
    for row in english_rows:
        if row["link"] not in links:
            links.append(row["link"])
        if len(links) == 2:
            break
    result["artifacts"]["belebele_en_initial"] = english_artifact
    for source, language in meta["languages"].items():
        raw, rows = english_raw, english_rows
        artifact = english_artifact
        if language != "eng_Latn":
            raw, rows = rows_api(
                "facebook/belebele", language, "test", 0, 20, meta["revision"]
            )
            artifact = save(RAW / f"belebele-{language}-test-first20.json", raw)
        result["artifacts"][source] = artifact
        selected = [r for r in rows if r["link"] in links]
        selected.sort(key=lambda r: (links.index(r["link"]), int(r["question_number"])))
        if not selected or {r["link"] for r in selected} != set(links):
            raise RuntimeError(
                f"could not find same two Belebele passages in {language}"
            )
        for item in selected:
            link, qno = item["link"], str(item["question_number"])
            passage = f"belebele-passage:{link}"
            prompt = (
                f"Passage:\n{item['flores_passage']}\n\nQuestion:\n{item['question']}\n\n"
                f"A. {item['mc_answer1']}\nB. {item['mc_answer2']}\n"
                f"C. {item['mc_answer3']}\nD. {item['mc_answer4']}\nAnswer:"
            )
            add_record(
                records,
                source=source,
                revision=meta["revision"],
                source_sha256=artifact["sha256"],
                record_id=f"{language}:{link}:q{qno}",
                group_id=f"{source}:{link}",
                leakage_group_id=passage,
                document_id=f"{source}:{link}",
                domain="reading-comprehension",
                text=prompt,
                split="test",
                fields={
                    "link": link,
                    "question_number": item["question_number"],
                    "correct_answer_num": item["correct_answer_num"],
                    "dialect": item["dialect"],
                    "source_question": item["question"],
                    "source_answers": [item[f"mc_answer{i}"] for i in range(1, 5)],
                },
            )
    result["selected_passage_links"] = links
    result["selected_question_count"] = sum(
        1 for r in records if r["source"] in meta["languages"]
    )
    manifest["sources"]["belebele"] = result


def acquire_ruler(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    meta = PINNED["ruler"]
    url = f"https://raw.githubusercontent.com/NVIDIA/RULER/{meta['revision']}/{meta['path']}"
    config = get(url)
    artifact = save(RAW / "ruler-synthetic.yaml", config)
    # This small stdlib-only generator is explicitly a local approximation of
    # RULER's niah_single_1 configuration, not output from the official runner.
    for count, value in ((128, "314159"), (512, "271828")):
        key = f"needle-key-{count}"
        noise = [f"filler{(i * 17 + 11) % 97}" for i in range(count)]
        insert_at = count // 2
        noise[insert_at:insert_at] = [f"The secret code for {key} is {value}."]
        text = (
            "Context:\n"
            + " ".join(noise)
            + f"\n\nQuestion: What is the secret code for {key}?\nAnswer:"
        )
        record_id = f"local-niah-words-{count}"
        add_record(
            records,
            source="ruler",
            revision=meta["revision"],
            source_sha256=artifact["sha256"],
            record_id=record_id,
            group_id=record_id,
            document_id=record_id,
            domain="synthetic-long-context-toy",
            text=text,
            split="development" if count == 128 else "confirmation",
            fields={
                "official_config": "niah_single_1",
                "key": key,
                "expected_answer": value,
                "word_count_target": count,
                "generator": "local-stdlib-toy-v1",
                "seed": 20261104,
            },
        )
    manifest["sources"]["ruler"] = {
        **meta,
        "artifacts": [artifact],
        "selection": "two local deterministic NIAH toy cases (128/512 filler words), following config niah_single_1; NOT official RULER generated records or benchmark results",
        "generator": "stdlib fixed formula filler values; seed 20261104; deterministic, no external corpus/tokenizer",
    }


def acquire_local(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    meta = PINNED["local_state"]
    cases = [
        {
            "record_id": "state-prefill-full-01",
            "session": "session-prefill-full-01",
            "text": "User: Return the value 17 as JSON.\nAssistant:",
            "state": {"initial": "reset", "prefill_chunks": [9], "decode_steps": 0},
        },
        {
            "record_id": "state-prefill-chunked-decode-01",
            "session": "session-prefill-chunked-decode-01",
            "text": 'User: Return the value 17 as JSON.\nAssistant: {"value": 17}',
            "state": {"initial": "reset", "prefill_chunks": [4, 5], "decode_steps": 1},
        },
    ]
    local_bytes = b"".join(canonical(item) + b"\n" for item in cases)
    artifact = save(RAW / "local-state-cases.jsonl", local_bytes)
    for item in cases:
        add_record(
            records,
            source="local_state",
            revision=meta["revision"],
            source_sha256=artifact["sha256"],
            record_id=item["record_id"],
            group_id=item["session"],
            document_id=item["session"],
            session_id=item["session"],
            domain="cache-state-integration",
            text=item["text"],
            split="development",
            fields={"initial_state": item["state"]},
        )
    manifest["sources"]["local_state"] = {
        **meta,
        "artifacts": [artifact],
        "selected_record_ids": [x["record_id"] for x in cases],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory for acquired files and the pinned source manifest",
    )
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"output directory already exists; choose a new path: {output}")
    global HERE, SOURCES, RAW
    HERE = output
    SOURCES = HERE / "sources"
    RAW = SOURCES / "raw"
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema": "gemma4-pinned-source-manifest-v1",
        "sources": {},
    }
    acquire_wikitext(records, manifest)
    acquire_ifeval(records, manifest)
    acquire_humaneval(records, manifest)
    acquire_belebele(records, manifest)
    acquire_ruler(records, manifest)
    acquire_local(records, manifest)
    records.sort(key=lambda r: (r["source"], r["record_id"]))
    selected_bytes = b"".join(canonical(row) + b"\n" for row in records)
    record_file = SOURCES / "records-un-tokenized.jsonl"
    record_artifact = save(record_file, selected_bytes)
    manifest["selected_records"] = {
        **record_artifact,
        "count": len(records),
        "format": "JSONL text/prompt staging records; no token IDs or tokenizer claims yet",
    }
    manifest["manifest_content_sha256"] = sha256(canonical(manifest))
    manifest_bytes = canonical(manifest) + b"\n"
    path = HERE / "pinned-source-manifest.json"
    if path.exists() and path.read_bytes() != manifest_bytes:
        raise RuntimeError(
            f"refusing to replace different existing pin manifest: {path}"
        )
    path.write_bytes(manifest_bytes)
    print(
        json.dumps(
            {
                "record_count": len(records),
                "source_count": len(manifest["sources"]),
                "selected_records_sha256": record_artifact["sha256"],
                "raw_bytes": sum(
                    item["size_bytes"]
                    for source in manifest["sources"].values()
                    for item in _artifacts(source)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _artifacts(source: dict[str, Any]):
    artifacts = source.get("artifacts", [])
    if isinstance(artifacts, dict):
        return artifacts.values()
    return artifacts


if __name__ == "__main__":
    raise SystemExit(main())
