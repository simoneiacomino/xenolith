import json
import tempfile
import unittest
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality" / "src"))

from xe_quality.corpus import CorpusError, prepare, validate


FIXTURE = ROOT / "tests" / "quality" / "fixtures" / "minimal.jsonl"


class CorpusTests(unittest.TestCase):
    def test_public_prepare_corpus_command(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "xe_quality",
                    "prepare-corpus",
                    "--input",
                    str(FIXTURE),
                    "--output",
                    str(output),
                    "--seed",
                    "cli",
                ],
                check=True,
                timeout=30,
            )
            self.assertEqual(validate(output)["record_count"], 5)

    def test_prepare_is_reproducible_and_preserves_exact_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            built = prepare(
                FIXTURE, first, seed="fixed-seed", confirmation_fraction=0.5
            )
            prepare(FIXTURE, second, seed="fixed-seed", confirmation_fraction=0.5)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(built, validate(first))
            self.assertEqual(5, built["record_count"])
            by_id = {record["record_id"]: record for record in built["records"]}
            self.assertEqual([101, 205, 42, 9], by_id["en-q1"]["token_ids"])
            self.assertEqual(
                {1: 205, 3: 9},
                {
                    x["position"]: x["token_id"]
                    for x in by_id["en-q1"]["observed_tokens"]
                },
            )
            self.assertEqual(
                [2, 1],
                by_id["local-prefill-decode-01"]["initial_state"]["prefill_chunks"],
            )

    def test_belebele_translations_and_questions_share_confirmation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare(
                FIXTURE,
                Path(tmp) / "manifest.json",
                seed="fixed-seed",
                confirmation_fraction=0.5,
            )
            records = manifest["records"]
            passage_records = [
                r for r in records if r.get("passage_id") == "belebele-passage-01"
            ]
            self.assertEqual(4, len(passage_records))
            self.assertEqual(1, len({r["split"] for r in passage_records}))

    def test_rejects_belebele_cross_language_split_leakage(self):
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
        for row in rows:
            if row["source"] == "belebele_en":
                row["split"] = "development"
            elif row["source"] == "belebele_it":
                row["split"] = "confirmation"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "leaking.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(CorpusError, "Belebele passage split leakage"):
                prepare(source, Path(tmp) / "out.json")

    def test_rejects_wrong_observed_token(self):
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
        rows[0]["observed_tokens"][0]["token_id"] = 999
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bad.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(CorpusError, "does not match token_ids"):
                prepare(source, Path(tmp) / "out.json")

    def test_rejects_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "manifest.json"
            prepare(FIXTURE, output)
            doc = json.loads(output.read_text())
            doc["records"][0]["token_ids"][0] += 1
            output.write_text(json.dumps(doc))
            with self.assertRaisesRegex(CorpusError, "manifest_sha256 mismatch"):
                validate(output)


if __name__ == "__main__":
    unittest.main()
