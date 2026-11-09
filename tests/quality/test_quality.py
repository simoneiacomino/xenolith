"""Pure software tests for the archive index and candidate comparison path."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality" / "src"))
from xe_quality import cli as quality
from xe_quality.artifacts import manifest_hash, write_artifact
from xe_quality.corpus import prepare


VOCAB = 262144


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tokenized_rows():
    common = {
        "dataset_revision": "pinned-fixture-v1",
        "tokenizer": {
            "identity": "fake-gguf-tokenizer",
            "revision": "fixture-tokenizer-1",
        },
        "template": {
            "identity": "fixture-no-template",
            "revision": "fixture-template-1",
        },
        "token_ids": [11, 12, 42, 14, 15],
        "observed_tokens": [{"position": 2, "token_id": 42}],
        "initial_state": None,
    }
    records = []
    for source, rid, group_id, language in (
        ("belebele_en", "en-q1", "en-passage", "eng_Latn"),
        ("belebele_it", "it-q1", "it-passage", "ita_Latn"),
    ):
        records.append(
            {
                **common,
                "source": source,
                "dataset_content_sha256": "a" * 64
                if source == "belebele_en"
                else "b" * 64,
                "record_id": rid,
                "group_id": group_id,
                "leakage_group_id": "shared-passage-17",
                "passage_id": "shared-passage-17",
                "document_id": group_id,
                "language": language,
                "domain": "reading-comprehension",
                "text": f"fixture {language}",
            }
        )
    records.append(
        {
            **common,
            "source": "local_state",
            "dataset_content_sha256": "c" * 64,
            "record_id": "state-no-successor",
            "group_id": "state-session",
            "document_id": "state-session",
            "session_id": "state-session",
            "domain": "cache-state",
            # Position 4 is observed, but no selected logit predicts token index 5.
            "observed_tokens": [{"position": 4, "token_id": 15}],
            "text": "fixture state case",
        }
    )
    return records


def create_corpus(root: Path, records=None):
    input_path = root / "source.jsonl"
    rows = tokenized_rows() if records is None else records
    input_path.write_text(
        "".join(canonical(row) + "\n" for row in rows), encoding="utf-8"
    )
    corpus_path = root / "corpus.json"
    manifest = prepare(
        input_path, corpus_path, seed="integration-fixture", confirmation_fraction=0.5
    )
    return corpus_path, manifest


def make_index(
    archive: Path,
    corpus: dict,
    *,
    config=None,
    allowed_prefill_chunks=None,
    qualification_packet=False,
):
    config = config or {"weights_sha256": "d" * 64, "positions": {}}
    wanted = config.get("record_keys")
    records = [
        r
        for r in corpus["records"]
        if wanted is None or f"{r['source']}:{r['record_id']}" in wanted
    ]
    first_positions = quality.selected(records[0], config)
    first_state_protocol = quality.input_state_protocol(records[0])
    first_input_hash = quality.input_spec_hash(
        records[0]["token_ids"], first_positions, first_state_protocol
    )
    input_length = len(records[0]["token_ids"])
    midpoint = input_length // 2
    if allowed_prefill_chunks is None:
        allowed_prefill_chunks = [[input_length], [midpoint, input_length - midpoint]]
    scope = {
        "schema": "xe-quality-reference-scope-v1",
        "domain_class": "fixed-token-control",
        "input_hash": first_input_hash,
        "input_length": len(records[0]["token_ids"]),
        "positions": first_positions,
        "state_protocol": first_state_protocol,
        "input_state_protocol": first_state_protocol,
        "maximum_supported_length": 512,
        "allowed_prefill_chunks": allowed_prefill_chunks,
        "qualification_state_controls": ["full_prefill_cache_restore_reset_repeat"],
        "supported_domains": [],
    }
    qualification = {
        "status": "QUALIFIED_SCOPED",
        "maximum_supported_length": 512,
        "scope": scope,
    }
    qualification_file_sha256 = None
    if qualification_packet:
        raw_path = archive / "qualification" / "evidence" / "control.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = b'{"fixture":"raw independent control"}\n'
        raw_path.write_bytes(raw_bytes)
        qualification["evidence_files"] = {
            "evidence/control.json": {
                "bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }
        }
        qualification_path = archive / "qualification" / "QUALIFICATION.json"
        qualification_bytes = (
            json.dumps(qualification, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        qualification_path.write_bytes(qualification_bytes)
        qualification_file_sha256 = hashlib.sha256(qualification_bytes).hexdigest()
    config_hash = manifest_hash(config)
    contract = {
        "config": config,
        "implementation": {"fixture": "pure-test"},
        "runtime": {"backend": "cpu", "python": "fixture"},
        "corpus_sha256": corpus["manifest_sha256"],
        "mode": "fixed-history; model transforms before sampler; temperature1",
        "qualification": qualification,
    }
    if qualification_file_sha256 is not None:
        contract["qualification_file_sha256"] = qualification_file_sha256
    contract_hash = manifest_hash(contract)
    version = "google-gemma4-gguf-" + contract_hash[:20]
    index = {
        "schema": "gemma4-reference-archive-v1",
        "complete": True,
        "reference_version": version,
        "contract": contract,
        "contract_hash": contract_hash,
        "vocab_size": VOCAB,
        "corpus": corpus,
        "cases": [],
    }
    if qualification_packet:
        index["qualification_packet"] = quality.QUALIFICATION_PACKET_PATH
    for record in records:
        positions = quality.selected(record, config)
        spec = {
            "record": record,
            "logit_positions": positions,
            "observation_semantics": "input token index j is predicted by logit j-1; no target for absent successor",
        }
        identity = {
            "reference_version": version,
            "config_hash": config_hash,
            "input_hash": manifest_hash(spec),
            "contract_hash": contract_hash,
        }
        key = quality.case_key(record)
        arr = np.zeros((len(positions), VOCAB), dtype="<f4")
        observations = {
            entry["position"]: entry["token_id"] for entry in record["observed_tokens"]
        }
        for row_index, logit_position in enumerate(positions):
            target_position = logit_position + 1
            if target_position in observations:
                arr[row_index, observations[target_position]] = 2.0
        artifact_name = key + ".npy"
        write_artifact(
            archive / artifact_name, arr, identity, expected_vocab_size=VOCAB
        )
        index["cases"].append(
            {
                "key": key,
                "record": record,
                "positions": positions,
                "identity": identity,
                "artifact": artifact_name,
                "group_id": quality.group(record),
            }
        )
    quality.signed_index(archive / "index.json", index)
    return index


def write_fake_driver(root: Path, *, truncate=False) -> Path:
    driver = root / "fake-driver.py"
    truncate_line = "data = data[:-1]" if truncate else ""
    driver.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "import numpy as np\n"
        "model, tsv, raw, context, mode = sys.argv[1:]\n"
        "out = pathlib.Path(raw)\n"
        "for line in pathlib.Path(tsv).read_text().splitlines():\n"
        "    fields = line.split('\\t')\n"
        "    key, tokens, positions = fields[:3]\n"
        "    token_ids = [int(x) for x in tokens.split(',')]\n"
        "    pos = [int(x) for x in positions.split(',')]\n"
        f"    logits = np.zeros((len(pos), {VOCAB}), dtype='<f4')\n"
        "    for i, p in enumerate(pos):\n"
        "        if p + 1 < len(token_ids): logits[i, token_ids[p + 1]] = 2.0\n"
        "    data = logits.tobytes(order='C')\n"
        f"    {truncate_line}\n"
        "    (out / (key + '.bin')).write_bytes(data)\n",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    return driver


def compare_args(
    root: Path,
    *,
    truncate=False,
    record_keys=None,
    allowed_prefill_chunks=None,
    input_length=None,
    explicit_positions=None,
):
    records = tokenized_rows()
    if input_length is not None:
        records[0]["token_ids"] = [11, 12, 42] + list(range(43, 43 + input_length - 3))
        if input_length >= 64:
            records[0]["observed_tokens"] = [
                {"position": position, "token_id": token_id}
                for position, token_id in enumerate(records[0]["token_ids"])
            ]
    corpus_path, corpus = create_corpus(root, records=records)
    model = root / "fake.gguf"
    model.write_bytes(b"not a real GGUF; subprocess fixture only")
    archive = root / "archive"
    archive.mkdir()
    config = {"weights_sha256": sha(model), "positions": {}}
    if explicit_positions is not None:
        config["positions"]["belebele_en:en-q1"] = explicit_positions
    if record_keys is not None:
        config["record_keys"] = record_keys
    make_index(
        archive, corpus, config=config, allowed_prefill_chunks=allowed_prefill_chunks
    )
    driver = write_fake_driver(root, truncate=truncate)
    pin = {str(model): sha(model)}
    backend_config = root / "backend.json"
    backend_config.write_text(
        canonical(
            {
                "backend": "fake-stdlib-test-driver",
                "weights_sha256": sha(model),
                "driver_sha256": sha(driver),
                "input_pins": pin,
            }
        ),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "archive": str(archive),
            "model": str(model),
            "driver": str(driver),
            "backend_config": str(backend_config),
            "output": str(root / "comparison"),
            "mode": "prefill",
            "timeout": 30,
        },
    )()
    return args, corpus, archive


class QualityIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_compare_uses_full_vocab_and_aligns_nll_to_next_input_token(self):
        args, corpus, archive = compare_args(self.root)
        # Compare must only read verified FP32 artifacts and invoke the driver.
        original_import = __import__

        def import_without_reference(name, *a, **kwargs):
            if name in {"jax", "flax", "stream"}:
                raise AssertionError(f"compare unexpectedly imported {name}")
            return original_import(name, *a, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_without_reference):
            quality.compare(args)
        report = json.loads((Path(args.output) / "report.json").read_text())
        self.assertFalse(report["reference_loaded"])
        by_source = {case["source"]: case for case in report["cases"]}
        en = by_source["belebele_en"]
        self.assertEqual("shared-passage-17", en["group_id"])
        self.assertEqual("shared-passage-17", by_source["belebele_it"]["group_id"])
        self.assertEqual([1, 4], en["positions"])
        self.assertEqual(1, len(en["nll"]))
        self.assertEqual(
            {
                "logit_position": 1,
                "observed_token_position": 2,
                "observed_token_id": 42,
            },
            {
                k: en["nll"][0][k]
                for k in (
                    "logit_position",
                    "observed_token_position",
                    "observed_token_id",
                )
            },
        )
        expected_nll = math.log((VOCAB - 1) + math.exp(2.0)) - 2.0
        self.assertAlmostEqual(
            expected_nll, en["nll"][0]["reference_nll_nats"], places=5
        )
        self.assertAlmostEqual(0.0, en["nll"][0]["delta_nll_nats"], places=8)
        self.assertEqual(
            [],
            by_source["local_state"]["nll"],
            "no observed token follows selected final position",
        )
        raw_size = (Path(args.output) / "raw" / (en["key"] + ".bin")).stat().st_size
        self.assertEqual(len(en["positions"]) * VOCAB * 4, raw_size)
        self.assertEqual(
            1_048_576, VOCAB * 4, "one complete FP32 vocabulary row is 1 MiB"
        )

    def test_index_rejects_hash_case_state_and_configuration_mismatches(self):
        _, corpus = create_corpus(self.root)
        archive = self.root / "archive"
        archive.mkdir()
        index = make_index(archive, corpus)
        self.assertEqual(
            index["cases"], quality.load_index(archive / "index.json")["cases"]
        )

        # Stale index digest is rejected before any artifact use.
        payload = json.loads((archive / "index.json").read_text())
        payload["index_sha256"] = "0" * 64
        (archive / "bad-hash.json").write_text(canonical(payload))
        with self.assertRaisesRegex(ValueError, "index hash mismatch"):
            quality.load_index(archive / "bad-hash.json")

        altered_corpus = json.loads(json.dumps(index))
        altered_corpus["corpus"]["records"][0]["token_ids"][2] = 99
        quality.signed_index(archive / "bad-corpus.json", altered_corpus)
        with self.assertRaisesRegex(ValueError, "corpus identity mismatch"):
            quality.load_index(archive / "bad-corpus.json")

        # An otherwise correctly signed record with a changed token or initial
        # state cannot be mistaken for the corpus record that produced logits.
        for field, replacement in (
            ("token_ids", [11, 12, 99, 14, 15]),
            (
                "initial_state",
                {"cache_action": "reset", "prefill_chunks": [2, 3], "decode_steps": 0},
            ),
        ):
            altered = json.loads(json.dumps(index))
            altered["cases"][0]["record"][field] = replacement
            quality.signed_index(archive / f"bad-{field}.json", altered)
            with self.assertRaisesRegex(
                ValueError, "input/state/selection/group mismatch"
            ):
                quality.load_index(archive / f"bad-{field}.json")

        altered_config = json.loads(json.dumps(index))
        altered_config["contract"]["config"]["positions"] = {"belebele_en:en-q1": [0]}
        quality.signed_index(archive / "bad-config.json", altered_config)
        with self.assertRaisesRegex(ValueError, "reference contract mismatch"):
            quality.load_index(archive / "bad-config.json")

        altered_identity = json.loads(json.dumps(index))
        altered_identity["cases"][0]["identity"]["config_hash"] = "e" * 64
        quality.signed_index(archive / "bad-case-config.json", altered_identity)
        with self.assertRaisesRegex(ValueError, "artifact input identity mismatch"):
            quality.load_index(archive / "bad-case-config.json")

    def test_compare_rejects_truncated_full_vocabulary_driver_output(self):
        args, _, _ = compare_args(self.root, truncate=True)
        report = quality.compare(args)
        self.assertEqual("ERROR", report["outcome"])
        self.assertIn("Partial/full vocabulary output size mismatch", report["reason"])
        self.assertEqual(
            "ERROR",
            json.loads((Path(args.output) / "report.json").read_text())["outcome"],
        )
        self.assertTrue((Path(args.output) / "raw").is_dir())

    def test_group_prefers_cross_language_passage_identity(self):
        records = tokenized_rows()[:2]
        self.assertNotEqual(records[0]["group_id"], records[1]["group_id"])
        self.assertEqual("shared-passage-17", quality.group(records[0]))
        self.assertEqual(quality.group(records[0]), quality.group(records[1]))

    def test_policy_outcomes_are_explicit_and_scope_checked(self):
        domain = "reading-comprehension:eng_Latn"
        row = {
            "group_count": 12,
            "mean_group_mean": 0.01,
            "max_observed_position_value": 0.03,
        }
        report = {
            "summary": {
                "kl": {"domains": {domain: row}},
                "tv": {"domains": {domain: row}},
                "top1_disagreement": {
                    "domains": {domain: {**row, "mean_group_mean": 0.0}}
                },
                "nll": {"delta": {"domains": {domain: row}}},
            },
            "reference_qualification": {
                "status": "QUALIFIED_SCOPED",
                "maximum_supported_length": 512,
                "scope": {
                    "schema": "xe-quality-reference-scope-v1",
                    "domain_class": "fixed-token-control",
                    "input_hash": "a" * 64,
                    "input_length": 3,
                    "positions": [1],
                    "state_protocol": "reset_full_history",
                    "input_state_protocol": "reset_full_history",
                    "maximum_supported_length": 512,
                    "allowed_prefill_chunks": [[3], [1, 2]],
                    "qualification_state_controls": [
                        "full_prefill_cache_restore_reset_repeat"
                    ],
                    "supported_domains": [],
                },
            },
            "cases": [
                {
                    "input_hash": "a" * 64,
                    "input_length": 3,
                    "positions": [1],
                    "input_state_protocol": "reset_full_history",
                    "candidate_chunks": [1, 2],
                }
            ],
            "candidate_execution_mode": "prefill",
        }
        policy = {
            "schema": "xe-quality-policy-v1",
            "domains": {
                domain: {
                    "min_groups": 10,
                    "metrics": {
                        "kl.mean_group_mean": {"max": 0.02},
                        "top1_disagreement.mean_group_mean": {"max": 0.0},
                    },
                }
            },
        }
        self.assertEqual(
            ("PASS", "all explicit bounds met in the qualified measured domain"),
            quality.evaluate_outcome(report, policy),
        )
        policy["domains"][domain]["metrics"]["kl.mean_group_mean"] = {"max": 0.001}
        outcome, reason = quality.evaluate_outcome(report, policy)
        self.assertEqual("FAIL", outcome)
        self.assertIn("violates max", reason)
        self.assertEqual("INCONCLUSIVE", quality.evaluate_outcome(report, None)[0])
        report["reference_qualification"]["status"] = "UNQUALIFIED"
        self.assertEqual("INCONCLUSIVE", quality.evaluate_outcome(report, policy)[0])
        report["reference_qualification"]["status"] = "QUALIFIED_SCOPED"
        report["reference_qualification"]["scope"]["input_hash"] = "b" * 64
        self.assertEqual("INCONCLUSIVE", quality.evaluate_outcome(report, policy)[0])
        report["reference_qualification"]["scope"]["input_hash"] = "a" * 64
        report["cases"][0]["input_length"] = 513
        self.assertEqual("INCONCLUSIVE", quality.evaluate_outcome(report, policy)[0])

    def test_real_compare_policy_pass_and_out_of_scope_inconclusive(self):
        args, _, _ = compare_args(self.root, record_keys=["belebele_en:en-q1"])
        domain = "reading-comprehension:eng_Latn"
        policy = {
            "schema": "xe-quality-policy-v1",
            "domains": {
                domain: {
                    "min_groups": 1,
                    "metrics": {
                        "kl.mean_group_mean": {"max": 0.01},
                        "tv.max_observed_position_value": {"max": 0.01},
                        "top1_disagreement.mean_group_mean": {"max": 0.0},
                    },
                }
            },
        }
        policy_path = self.root / "policy.json"
        policy_path.write_text(canonical(policy), encoding="utf-8")
        args.policy = str(policy_path)
        report = quality.compare(args)
        self.assertEqual("PASS", report["outcome"])
        self.assertEqual(policy, report["quality_policy"])
        self.assertEqual(manifest_hash(policy), report["quality_policy_sha256"])
        case = report["cases"][0]
        self.assertEqual(5, case["input_length"])
        self.assertEqual([1, 4], case["positions"])
        self.assertEqual("reset_full_history", case["input_state_protocol"])

        fail_policy = json.loads(json.dumps(policy))
        fail_policy["domains"][domain]["metrics"][
            "top1_disagreement.mean_group_mean"
        ] = {"max": -1.0}
        fail_policy_path = self.root / "fail-policy.json"
        fail_policy_path.write_text(canonical(fail_policy), encoding="utf-8")
        fail_root = self.root / "numeric-fail"
        fail_root.mkdir()
        args, _, _ = compare_args(fail_root, record_keys=["belebele_en:en-q1"])
        args.policy = str(fail_policy_path)
        report = quality.compare(args)
        self.assertEqual("FAIL", report["outcome"])
        self.assertIn("violates max", report["reason"])

        other_root = self.root / "other-scope"
        other_root.mkdir()
        args, _, _ = compare_args(other_root, record_keys=["belebele_it:it-q1"])
        args.policy = str(policy_path)
        report = quality.compare(args)
        self.assertEqual("INCONCLUSIVE", report["outcome"])

        midpoint_root = self.root / "midpoint-prefill"
        midpoint_root.mkdir()
        args, _, _ = compare_args(
            midpoint_root, record_keys=["belebele_en:en-q1"], input_length=64
        )
        args.policy = str(policy_path)
        report = quality.compare(args)
        self.assertEqual("PASS", report["outcome"])
        self.assertEqual([31, 63], report["cases"][0]["positions"])
        self.assertEqual([32, 32], report["cases"][0]["candidate_chunks"])

        boundary_root = self.root / "boundary-prefill"
        boundary_root.mkdir()
        args, _, _ = compare_args(
            boundary_root,
            record_keys=["belebele_en:en-q1"],
            input_length=64,
            explicit_positions=[0, 63],
        )
        args.policy = str(policy_path)
        report = quality.compare(args)
        self.assertEqual("INCONCLUSIVE", report["outcome"])
        self.assertEqual([1, 63], report["cases"][0]["candidate_chunks"])
        self.assertIn("not exercised", report["reason"])

        decode_root = self.root / "decode-mode"
        decode_root.mkdir()
        args, _, _ = compare_args(
            decode_root, record_keys=["belebele_en:en-q1"], input_length=64
        )
        args.mode = "decode"
        args.policy = str(policy_path)
        report = quality.compare(args)
        self.assertEqual("INCONCLUSIVE", report["outcome"])

    def test_distribution_manifest_checks_identity_hashes_and_safe_paths(self):
        _, corpus = create_corpus(self.root)
        archive = self.root / "archive"
        archive.mkdir()
        index = make_index(archive, corpus)
        (archive / ".build.lock").write_bytes(b"")
        manifest = {
            "schema": "xenolith-reference-distribution-v1",
            "reference_version": index["reference_version"],
            "archive_schema": index["schema"],
            "contract_hash": index["contract_hash"],
            "index_sha256": sha(archive / "index.json"),
            "weights_sha256": index["contract"]["config"]["weights_sha256"],
            "files": {},
        }
        for path in archive.iterdir():
            manifest["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
        manifest_path = self.root / "distribution.json"
        manifest_path.write_text(canonical(manifest), encoding="utf-8")
        self.assertEqual(manifest, quality.verify_distribution(archive, manifest_path))
        imported = self.root / "imported"
        self.assertEqual(
            0,
            quality.main(
                [
                    "import-archive",
                    "--archive",
                    str(archive),
                    "--output",
                    str(imported),
                    "--manifest",
                    str(manifest_path),
                ]
            ),
        )
        self.assertEqual(
            (archive / "index.json").read_bytes(),
            (imported / "index.json").read_bytes(),
        )
        for name in manifest["files"]:
            self.assertEqual(sha(archive / name), sha(imported / name))
        manifest["files"]["../outside"] = {"bytes": 0, "sha256": "0" * 64}
        manifest_path.write_text(canonical(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe distribution path"):
            quality.verify_distribution(archive, manifest_path)

    def _packet_archive(self, name):
        root = self.root / name
        root.mkdir()
        _, corpus = create_corpus(root)
        archive = root / "archive"
        archive.mkdir()
        index = make_index(archive, corpus, qualification_packet=True)
        return archive, index

    def test_qualification_packet_load_validates_q_json_and_every_raw_pin(self):
        archive, index = self._packet_archive("valid-packet")
        loaded = quality.load_index(archive / "index.json")
        self.assertEqual(index["reference_version"], loaded["reference_version"])
        self.assertEqual(
            (archive / "qualification" / "QUALIFICATION.json").read_bytes(),
            (archive / loaded["qualification_packet"]).read_bytes(),
        )

        missing, _ = self._packet_archive("missing-evidence")
        (missing / "qualification" / "evidence" / "control.json").unlink()
        with self.assertRaisesRegex(ValueError, "qualification evidence missing"):
            quality.load_index(missing / "index.json")

        truncated, _ = self._packet_archive("truncated-evidence")
        raw = truncated / "qualification" / "evidence" / "control.json"
        raw.write_bytes(raw.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "evidence pin mismatch"):
            quality.load_index(truncated / "index.json")

        changed_q, _ = self._packet_archive("changed-q-json")
        q_path = changed_q / "qualification" / "QUALIFICATION.json"
        q_path.write_bytes(q_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "qualification JSON differs|SHA-256"):
            quality.load_index(changed_q / "index.json")

    def test_qualification_packet_rejects_traversal_symlinks_and_unpinned_files(self):
        archive, _ = self._packet_archive("traversal-packet")
        q_path = archive / "qualification" / "QUALIFICATION.json"
        qualification = json.loads(q_path.read_text(encoding="utf-8"))
        pin = qualification["evidence_files"].pop("evidence/control.json")
        qualification["evidence_files"]["../outside.json"] = pin
        q_path.write_text(canonical(qualification), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe qualification evidence path"):
            quality.validate_qualification_packet(q_path)

        symlink_archive, _ = self._packet_archive("symlink-packet")
        evidence = symlink_archive / "qualification" / "evidence" / "control.json"
        external = self.root / "external-control.json"
        external.write_bytes(evidence.read_bytes())
        evidence.unlink()
        evidence.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            quality.load_index(symlink_archive / "index.json")

        extra_archive, _ = self._packet_archive("extra-packet-file")
        (extra_archive / "qualification" / "unlisted.txt").write_text("extra")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            quality.load_index(extra_archive / "index.json")

    def test_unqualified_diagnostic_packet_may_contain_only_qualification_json(self):
        root = self.root / "unqualified-packet"
        root.mkdir()
        qualification_path = root / "QUALIFICATION.json"
        qualification_path.write_text(
            canonical({"status": "UNQUALIFIED", "evidence_files": {}}),
            encoding="utf-8",
        )
        qualification, evidence = quality.validate_qualification_packet(
            qualification_path
        )
        self.assertEqual("UNQUALIFIED", qualification["status"])
        self.assertEqual([], evidence)

    def test_corrupt_qualification_packet_fails_before_model_or_output_setup(self):
        archive, _ = self._packet_archive("preflight-packet")
        q_path = archive / "qualification" / "QUALIFICATION.json"
        (archive / "qualification" / "evidence" / "control.json").unlink()
        root = archive.parent
        config_path = root / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        args = type(
            "Args",
            (),
            {
                "config": str(config_path),
                "corpus": str(root / "corpus.json"),
                "model": str(root / "model-that-does-not-exist.gguf"),
                "qualification": str(q_path),
                "output": str(root / "output-not-created"),
            },
        )()
        with self.assertRaisesRegex(ValueError, "qualification evidence missing"):
            quality.build(args)
        self.assertFalse(Path(args.output).exists())

    def test_unmanifested_export_import_retains_packet_and_legacy_v1_stays_compatible(
        self,
    ):
        archive, index = self._packet_archive("packet-export")
        source_packet = archive / "qualification"
        before = {
            path.relative_to(archive).as_posix(): sha(path)
            for path in source_packet.rglob("*")
            if path.is_file()
        }
        imported = self.root / "packet-imported"
        self.assertEqual(
            0,
            quality.main(
                ["import-archive", "--archive", str(archive), "--output", str(imported)]
            ),
        )
        exported = self.root / "packet-exported"
        self.assertEqual(
            0,
            quality.main(
                [
                    "export-archive",
                    "--archive",
                    str(imported),
                    "--output",
                    str(exported),
                ]
            ),
        )
        self.assertEqual(
            (archive / "index.json").read_bytes(),
            (exported / "index.json").read_bytes(),
        )
        after = {
            path.relative_to(exported).as_posix(): sha(path)
            for path in (exported / "qualification").rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertEqual(
            index["reference_version"],
            quality.load_index(exported / "index.json")["reference_version"],
        )

        # A historical v1 index without the optional packet pointer remains valid.
        old_root = self.root / "old-v1"
        old_root.mkdir()
        _, old_corpus = create_corpus(old_root)
        old_archive = old_root / "archive"
        old_archive.mkdir()
        old_index = make_index(old_archive, old_corpus)
        self.assertNotIn("qualification_packet", old_index)
        self.assertEqual(
            old_index["reference_version"],
            quality.load_index(old_archive / "index.json")["reference_version"],
        )

    def test_diagnostic_command_succeeds_while_gate_reports_inconclusive(self):
        args = [
            "--archive",
            "a",
            "--model",
            "m",
            "--backend-config",
            "b",
            "--output",
            "o",
            "--timeout",
            "10",
        ]
        with mock.patch.object(
            quality, "compare", return_value={"outcome": "INCONCLUSIVE"}
        ):
            self.assertEqual(0, quality.main(["diagnose", *args]))
            self.assertEqual(3, quality.main(["compare", *args]))
        with mock.patch.object(quality, "compare", return_value={"outcome": "ERROR"}):
            self.assertEqual(2, quality.main(["diagnose", *args]))

    def test_installed_package_runs_outside_checkout_and_cli_errors_are_json(self):
        env = dict(os.environ)
        package_root = str(ROOT / "tools" / "quality" / "src")
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        outside = self.root / "outside-checkout"
        outside.mkdir()
        good = subprocess.run(
            [sys.executable, "-m", "xe_quality", "self-check"],
            cwd=outside,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, good.returncode, good.stderr)
        self.assertIn("does not qualify", good.stdout)
        identity = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json; from xe_quality.cli import software_identity; print(json.dumps(software_identity()))",
            ],
            cwd=outside,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, identity.returncode, identity.stderr)
        self.assertIn("xe_quality/cli.py", json.loads(identity.stdout))
        bad = subprocess.run(
            [
                sys.executable,
                "-m",
                "xe_quality",
                "verify-archive",
                "--archive",
                str(outside / "missing"),
            ],
            cwd=outside,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(2, bad.returncode)
        self.assertEqual("ERROR", json.loads(bad.stderr)["outcome"])


if __name__ == "__main__":
    unittest.main()
