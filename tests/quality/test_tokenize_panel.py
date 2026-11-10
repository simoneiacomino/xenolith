"""Pure tests for exact dual-tokenizer agreement and frozen panel preparation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality" / "src"))
from xe_quality import tokenize_panel


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def gguf_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def minimal_gguf(*, add_bos=False, bos_id=2):
    # Minimal GGUF v3 with exactly the two tokenizer metadata keys used by the
    # BOS guard. It carries no weights and is never loaded for model execution.
    value = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
    value += (
        gguf_string("tokenizer.ggml.add_bos_token")
        + struct.pack("<I", 7)
        + bytes([add_bos])
    )
    value += (
        gguf_string("tokenizer.ggml.bos_token_id")
        + struct.pack("<I", 4)
        + struct.pack("<I", bos_id)
    )
    return value


def source_rows():
    rows = []

    def add(
        source,
        rid,
        *,
        revision,
        artifact_hash,
        text,
        source_split,
        group,
        fields=None,
        language=None,
        session=None,
        domain="fixture-domain",
    ):
        row = {
            "source": source,
            "dataset_revision": revision,
            "source_artifact_sha256": artifact_hash,
            "record_id": rid,
            "group_id": group,
            "document_id": group,
            "domain": domain,
            "source_split": source_split,
            "text": text,
            "source_fields": fields or {},
        }
        if language:
            row["language"] = language
        if session:
            row["session_id"] = session
        rows.append(row)

    # Belebele has two passages x two languages; each passage stays wholly in
    # one split and its language variants share a canonical leakage group.
    for link, suffix in (("passage-A", "A"), ("passage-B", "B")):
        for lang, source in (("eng_Latn", "belebele_en"), ("ita_Latn", "belebele_it")):
            add(
                source,
                f"{lang}-q-{suffix}",
                revision="hf-belebele-r1",
                artifact_hash="a" * 64 if source.endswith("en") else "b" * 64,
                text=f"Visible question {link} {lang}",
                source_split="test",
                group=f"{source}-{link}",
                fields={"link": link, "correct_answer_num": "1"},
                language=lang,
                domain="reading-comprehension",
            )
    add(
        "wikitext2_raw",
        "wiki-dev",
        revision="wiki-r1",
        artifact_hash="c" * 64,
        text="A validation document",
        source_split="validation",
        group="wiki-dev",
        domain="english-text",
    )
    add(
        "wikitext2_raw",
        "wiki-confirm",
        revision="wiki-r1",
        artifact_hash="c" * 64,
        text="A test document",
        source_split="test",
        group="wiki-test",
        domain="english-text",
    )
    add(
        "ifeval",
        "ifeval-dev",
        revision="ifeval-r1",
        artifact_hash="d" * 64,
        text="Follow this instruction",
        source_split="development",
        group="ifeval-dev",
        fields={"gold": "hidden"},
    )
    add(
        "humaneval_plus",
        "HumanEval/0",
        revision="evalplus-r1",
        artifact_hash="e" * 64,
        text="def f():\n    pass",
        source_split="confirmation",
        group="HumanEval/0",
        fields={"tests": "hidden"},
        domain="code",
    )
    for rid, family in (
        ("state-prefill-full-01", "full"),
        ("state-prefill-chunked-decode-01", "chunked"),
    ):
        add(
            "local_state",
            rid,
            revision="local-v1",
            artifact_hash="f" * 64,
            text=f"Visible local state {family}",
            source_split="development",
            group=f"local-{family}",
            fields={"gold": "never tokenize this"},
            session=rid,
            domain="cache-state",
        )
    for rid in ("local-niah-words-128", "local-niah-words-512"):
        add(
            "ruler",
            rid,
            revision=tokenize_panel.RULER_REVISION,
            artifact_hash="1" * 64,
            text=f"Local RULER toy {rid}",
            source_split="development",
            group=rid,
            fields={"expected_answer": "hidden"},
            domain="synthetic-long-context-toy",
        )
    return rows


class TokenizePanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus_root = self.root / "corpus"
        (self.corpus_root / "sources" / "raw").mkdir(parents=True)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(minimal_gguf())
        self.model_sha = digest(self.model.read_bytes())
        self.xe = self.root / "xe-tokenizer"
        self.llama = self.root / "llama-tokenizer"
        self.write_tokenizer(self.xe)
        self.write_tokenizer(self.llama)
        self.source_records = (
            self.corpus_root / "sources" / "records-un-tokenized.jsonl"
        )
        self.source_manifest_path = self.corpus_root / "pinned-source-manifest.json"
        self.make_source_bundle()

    def write_tokenizer(self, path: Path, *, mismatch=False, prepend_bos=False):
        values = [10 + i for i in range(20)]
        if mismatch:
            values[-1] += 1
        if prepend_bos:
            values.insert(0, 2)
        script = (
            "#!/usr/bin/env python3\n"
            "import pathlib, sys\n"
            "model, textfile = sys.argv[1:]\n"
            "data = pathlib.Path(textfile).read_bytes()\n"
            "if b'never tokenize this' in data or b'gold' in data or b'hidden' in data: raise SystemExit(7)\n"
            "if not data: raise SystemExit(8)\n"
            f"print({','.join(map(str, values))!r})\n"
        )
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)

    def make_source_bundle(self, *, wrong_records_hash=False, include_ruler=False):
        records = source_rows()
        records_bytes = b"".join(canonical(row) + b"\n" for row in records)
        self.source_records.write_bytes(records_bytes)
        source_meta = {}
        for source in sorted({row["source"] for row in records}):
            rows = [row for row in records if row["source"] == source]
            hashes = sorted({row["source_artifact_sha256"] for row in rows})
            artifacts = []
            for index, source_hash in enumerate(hashes):
                # Local bounded artifacts are just provenance fixtures. Their
                # on-disk byte hashes remain real and are checked by the code.
                name = f"{source}-{index}.raw"
                payload = (f"raw fixture artifact {source} {index}\n").encode()
                path = self.corpus_root / "sources" / "raw" / name
                path.write_bytes(payload)
                actual = digest(payload)
                # The test source rows refer to this actual artifact hash.
                for row in rows:
                    if row["source_artifact_sha256"] == source_hash:
                        row["source_artifact_sha256"] = actual
                artifacts.append(
                    {
                        "path": f"sources/raw/{name}",
                        "size_bytes": len(payload),
                        "sha256": actual,
                    }
                )
            revision = rows[0]["dataset_revision"]
            source_meta[source] = {"revision": revision, "artifacts": artifacts}

        # Rewrite after replacing fixture provenance digests with the actual
        # local raw artifact digests.
        records_bytes = b"".join(canonical(row) + b"\n" for row in records)
        self.source_records.write_bytes(records_bytes)
        source_meta["belebele"] = {
            "revision": "hf-belebele-r1",
            "languages": {"belebele_en": "eng_Latn", "belebele_it": "ita_Latn"},
            "selected_passage_links": ["passage-A", "passage-B"],
            "artifacts": source_meta.pop("belebele_en")["artifacts"]
            + source_meta.pop("belebele_it")["artifacts"],
        }
        raw = b"official config fixture\n"
        ruler_path = self.corpus_root / "sources" / "raw" / "ruler.yaml"
        ruler_path.write_bytes(raw)
        source_meta["ruler"] = {
            "revision": tokenize_panel.RULER_REVISION,
            "artifacts": [
                {
                    "path": "sources/raw/ruler.yaml",
                    "size_bytes": len(raw),
                    "sha256": digest(raw),
                }
            ]
            + source_meta["ruler"]["artifacts"],
        }
        selected_hash = digest(records_bytes)
        if wrong_records_hash:
            selected_hash = "0" * 64
        manifest = {
            "schema": "gemma4-pinned-source-manifest-v1",
            "selected_records": {
                "path": "sources/records-un-tokenized.jsonl",
                "sha256": selected_hash,
                "size_bytes": len(records_bytes),
                "count": len(records),
            },
            "sources": source_meta,
        }
        manifest["manifest_content_sha256"] = digest(canonical(manifest))
        self.source_manifest_path.write_bytes(canonical(manifest) + b"\n")

    def run_panel(
        self,
        *,
        output=None,
        max_tokens=10,
        xe=None,
        llama=None,
        official=None,
        expected_sha=None,
    ):
        output = output or self.root / "panel.jsonl"
        return tokenize_panel.tokenize_panel(
            self.source_records,
            self.source_manifest_path,
            self.model,
            expected_sha or self.model_sha,
            xe or self.xe,
            llama or self.llama,
            output,
            max_tokens,
            official,
        )

    def write_official_ruler(
        self, directory: Path, *, upstream=None, task="niah_single_1", output_sha=None
    ):
        directory.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                {
                    "input": "Full official synthetic input one",
                    "outputs": ["gold-a"],
                    "task": "niah",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "input": "Full official synthetic input two",
                    "outputs": ["gold-b"],
                    "task": "niah",
                }
            )
            + "\n"
        ).encode("utf-8")
        generated = directory / "task.jsonl"
        generated.write_bytes(payload)
        receipt = {
            "upstream_commit": upstream or tokenize_panel.RULER_REVISION,
            "task": task,
            "num_samples": 2,
            "source_pins": {"scripts/data/synthetic/niah.py": "a" * 64},
            "output_pins": {"task.jsonl": output_sha or digest(payload)},
        }
        (directory / "GENERATION.json").write_bytes(canonical(receipt) + b"\n")
        return generated

    def test_dual_tokenizer_panel_mapping_cutoff_bos_and_state(self):
        output = self.root / "panel.jsonl"
        audit = self.run_panel(output=output)
        rows = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
        ]
        by_key = {(row["source"], row["record_id"]): row for row in rows}
        self.assertEqual(10, len(rows))
        self.assertEqual(
            [2, *range(10, 20)], by_key[("wikitext2_raw", "wiki-dev")]["token_ids"]
        )
        self.assertEqual("development", by_key[("wikitext2_raw", "wiki-dev")]["split"])
        self.assertEqual(
            "confirmation", by_key[("wikitext2_raw", "wiki-confirm")]["split"]
        )
        self.assertEqual("development", by_key[("ifeval", "ifeval-dev")]["split"])
        self.assertEqual(
            "confirmation", by_key[("humaneval_plus", "HumanEval/0")]["split"]
        )
        en_a = by_key[("belebele_en", "eng_Latn-q-A")]
        it_a = by_key[("belebele_it", "ita_Latn-q-A")]
        en_b = by_key[("belebele_en", "eng_Latn-q-B")]
        it_b = by_key[("belebele_it", "ita_Latn-q-B")]
        self.assertEqual("eng_Latn", en_a["language"])
        self.assertEqual("ita_Latn", it_a["language"])
        self.assertEqual(en_a["leakage_group_id"], it_a["leakage_group_id"])
        self.assertEqual("development", en_a["split"])
        self.assertEqual("development", it_a["split"])
        self.assertEqual("confirmation", en_b["split"])
        self.assertEqual("confirmation", it_b["split"])
        for row in rows:
            self.assertEqual(11, len(row["token_ids"]))
            self.assertEqual(
                [
                    {"position": i, "token_id": row["token_ids"][i]}
                    for i in range(1, 11)
                ],
                row["observed_tokens"],
            )
            self.assertEqual({"identity", "revision"}, set(row["tokenizer"]))
            self.assertEqual("raw-no-chat+explicit-bos2", row["template"]["identity"])
            self.assertFalse(any(token == 2 for token in row["token_ids"][1:]))
        full_state = by_key[("local_state", "state-prefill-full-01")]
        chunked = by_key[("local_state", "state-prefill-chunked-decode-01")]
        self.assertIsNone(full_state["initial_state"])
        self.assertEqual("reset", chunked["initial_state"]["cache_action"])
        self.assertEqual(6, chunked["initial_state"]["decode_steps"])
        self.assertEqual(
            [5, 1, 1, 1, 1, 1, 1], chunked["initial_state"]["prefill_chunks"]
        )
        self.assertEqual("local-state-json17-family", full_state["group_id"])
        self.assertEqual(full_state["group_id"], chunked["group_id"])
        self.assertNotEqual(full_state["session_id"], chunked["session_id"])
        self.assertTrue(all("source_fields" in row for row in rows))
        self.assertFalse(any("never tokenize this" in row["text"] for row in rows))
        self.assertEqual(2, len(audit["excluded_records"]))
        audit_record = next(
            item for item in audit["records"] if item["record_id"] == "wiki-dev"
        )
        self.assertEqual(20, audit_record["full_regular_token_count"])
        self.assertTrue(audit_record["truncation_applied"])
        self.assertEqual(2, audit["bos_policy"]["explicitly_prepended_bos_id"])
        local_audit = next(
            item for item in audit["records"] if item["source"] == "local_state"
        )
        self.assertEqual(
            "sampled-prefix-fidelity-only; not an official task score",
            local_audit["claim_scope"],
        )
        self.assertEqual(digest(output.read_bytes()), audit["output_records_sha256"])
        self.assertTrue(Path(str(output) + ".audit.json").is_file())

    def test_tokenizer_mismatch_or_preexisting_bos_fails_before_writing(self):
        mismatch = self.root / "mismatch-tokenizer"
        self.write_tokenizer(mismatch, mismatch=True)
        out = self.root / "mismatch.jsonl"
        with self.assertRaisesRegex(
            tokenize_panel.TokenizeError, "IDs differ before BOS/cutoff"
        ):
            self.run_panel(output=out, llama=mismatch)
        self.assertFalse(out.exists())

        bos = self.root / "bos-tokenizer"
        self.write_tokenizer(bos, prepend_bos=True)
        out2 = self.root / "bos.jsonl"
        with self.assertRaisesRegex(
            tokenize_panel.TokenizeError, "unexpectedly emitted BOS"
        ):
            self.run_panel(output=out2, xe=bos, llama=bos)
        self.assertFalse(out2.exists())

    def test_source_hash_and_model_pin_are_checked(self):
        bad_model_out = self.root / "bad-model.jsonl"
        with self.assertRaisesRegex(
            tokenize_panel.TokenizeError, "model SHA-256 mismatch"
        ):
            self.run_panel(output=bad_model_out, expected_sha="0" * 64)
        self.assertFalse(bad_model_out.exists())

        altered = self.source_records.read_bytes() + b"\n"
        self.source_records.write_bytes(altered)
        with self.assertRaisesRegex(
            tokenize_panel.TokenizeError, "source-record SHA-256 mismatch"
        ):
            self.run_panel(output=self.root / "bad-source.jsonl")

    def test_official_ruler_records_are_pinned_and_never_truncated(self):
        self.make_source_bundle(include_ruler=True)
        ruler_dir = self.root / "official-ruler"
        self.write_official_ruler(ruler_dir)
        output = self.root / "with-ruler.jsonl"
        audit = self.run_panel(output=output, max_tokens=3, official=ruler_dir)
        rows = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
        ]
        official = [row for row in rows if row["source"] == "ruler"]
        self.assertEqual(2, len(official))
        self.assertEqual(
            ["development", "confirmation"], [row["split"] for row in official]
        )
        self.assertEqual([2, *range(10, 30)], official[0]["token_ids"])
        self.assertEqual("gold-a", official[0]["source_fields"]["outputs"][0])
        per_record = {row["record_id"]: row for row in audit["records"]}
        self.assertFalse(per_record[official[0]["record_id"]]["truncation_applied"])
        self.assertIsNone(per_record[official[0]["record_id"]]["regular_token_limit"])
        self.assertTrue(audit["official_ruler"]["bundle_sha256"])
        self.assertEqual(
            tokenize_panel.RULER_REVISION,
            audit["official_ruler"]["generation_identity"]["upstream_commit"],
        )
        self.assertEqual(
            digest((ruler_dir / "GENERATION.json").read_bytes()),
            audit["official_ruler"]["generation_receipt_sha256"],
        )

    def test_official_ruler_requires_matching_generation_receipt_and_output_hash(self):
        self.make_source_bundle(include_ruler=True)
        missing = self.root / "missing-receipt"
        missing.mkdir()
        (missing / "task.jsonl").write_text(
            '{"input":"arbitrary JSONL"}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(
            tokenize_panel.TokenizeError, "requires a GENERATION.json"
        ):
            self.run_panel(
                output=self.root / "missing-receipt-panel.jsonl", official=missing
            )

        wrong_revision = self.root / "wrong-revision"
        self.write_official_ruler(wrong_revision, upstream="0" * 40)
        with self.assertRaisesRegex(
            tokenize_panel.TokenizeError, "upstream_commit does not match"
        ):
            self.run_panel(
                output=self.root / "wrong-revision-panel.jsonl", official=wrong_revision
            )

        bad_output_pin = self.root / "bad-output-pin"
        self.write_official_ruler(bad_output_pin, output_sha="0" * 64)
        with self.assertRaisesRegex(
            tokenize_panel.TokenizeError, "generated-file SHA-256 mismatch"
        ):
            self.run_panel(
                output=self.root / "bad-output-pin-panel.jsonl", official=bad_output_pin
            )

    def test_output_is_immutable_and_requires_new_paths(self):
        out = self.root / "panel.jsonl"
        self.run_panel(output=out)
        with self.assertRaisesRegex(tokenize_panel.TokenizeError, "must both be new"):
            self.run_panel(output=out)


if __name__ == "__main__":
    unittest.main()
