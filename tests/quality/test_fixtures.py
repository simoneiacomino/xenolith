"""Finite-suite acceptance, provenance and no-HF producer contract controls."""
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import numpy as np

from xe_quality.artifacts import manifest_hash
from xe_quality.fixture_reference import file_hash, load_corpus, write_json
from xe_quality.fixtures import LEGACY_ACCEPTANCE, _summarize, evaluate, seal, verify


class FixturesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus_dir = self.root / "corpus"
        self.corpus_dir.mkdir()
        self.google = self.root / "google"
        self.google.mkdir()
        self.baseline = self.root / "llama"
        self.baseline.mkdir()
        self.spec = {"tokens": [2, 3, 4, 5], "positions": [1, 3], "chunks": [2, 2],
                     "context": 64, "initial_state": "empty", "mode": "prefill"}
        write_json(self.corpus_dir / "a.json", self.spec)
        self.corpus = {"schema": "xe-quality-fixture-corpus-v1", "weights_sha256": "a"*64,
                       "vocab_size": 262144, "documents": [], "cases": [
                           {"id": "a", "group": "document-a", "stratum": "text",
                            "input": {"path": "a.json", "sha256": file_hash(self.corpus_dir/"a.json")}}]}
        write_json(self.corpus_dir / "manifest.json", self.corpus)
        self.g = np.full((2, 262144), -12.0, dtype="<f8")
        self.g[:, :2] = [1.0, 0.0]
        self.l = self.g.astype("<f4")
        self.l[:, 0] += 0.5
        np.save(self.google / "a.npy", self.g)
        np.save(self.baseline / "a.npy", self.l)
        write_json(self.google / "google.json", {"corpus_sha256": manifest_hash(self.corpus),
            "producer": {"kind": "trusted-google-fp64"}, "files": {
                "a": {"path": "a.npy", "sha256": file_hash(self.google/"a.npy")}}})
        self.capture = {"schema": "xe-quality-fixture-capture-v1", "corpus_sha256": manifest_hash(self.corpus),
            "backend": {"backend": "llama.cpp CPU", "source_commit": "test-pinned-source",
                        "driver_sha256": "c"*64, "public_profile": {"configuration": {"threads": 8}}},
            "files": {"a": {"path": "a.npy", "sha256": file_hash(self.baseline/"a.npy")}},
            "repetitions": 3, "repeatable": True}
        self.capture["runs"] = self.runs(self.baseline, self.capture["files"], 3)
        write_json(self.baseline / "capture.json", self.capture)
        self.suite = self.root / "suite"
        seal(self.corpus_dir/"manifest.json", self.google, self.baseline, self.suite)

    def tearDown(self):
        self.temp.cleanup()

    def runs(self, folder, files, count):
        result = [files]
        for i in range(1, count):
            path = folder/f"repeat-{i}.npy"
            shutil.copyfile(folder/"a.npy", path)
            result.append({"a": {"path": path.name, "sha256": file_hash(path)}})
        return result

    def candidate(self, logits, repetitions=3, repeatable=True):
        out = self.root / "candidate"
        out.mkdir()
        np.save(out/"a.npy", np.asarray(logits, dtype="<f4"))
        capture = dict(self.capture, repetitions=repetitions, repeatable=repeatable)
        capture["files"] = {"a": {"path": "a.npy", "sha256": file_hash(out/"a.npy")}}
        capture["runs"] = self.runs(out, capture["files"], repetitions)
        write_json(out/"capture.json", capture)
        return out

    def test_finite_equality_is_not_population_equivalence(self):
        report = evaluate(self.suite, self.candidate(self.l))
        self.assertEqual(report["outcome"], "PASS")
        self.assertEqual(report["population_inference"], "NOT_ESTABLISHED")
        self.assertEqual(report["strata"]["text"]["endpoints"]["mean_kl"]["resolution_interval"], [0, 0])

    def test_google_exact_candidate_is_better(self):
        self.assertEqual(evaluate(self.suite, self.candidate(self.g))["outcome"], "PASS")

    def test_worse_candidate_fails(self):
        x = self.l.copy()
        x[:, 0] += 2
        self.assertEqual(evaluate(self.suite, self.candidate(x))["outcome"], "FAIL")

    def test_better_mean_passes_but_worse_tail_remains_visible(self):
        x = self.g.astype("<f4")
        x[1, 0] += 0.65
        report = evaluate(self.suite, self.candidate(x))
        endpoints = report["strata"]["text"]["endpoints"]
        self.assertLess(endpoints["mean_kl"]["delta_x_minus_llama"], 0)
        self.assertGreater(endpoints["max_kl"]["delta_x_minus_llama"], 0)
        self.assertEqual(report["outcome"], "PASS")
        self.assertFalse(report["diagnostics"]["blocking"])
        self.assertTrue(any(p["position"] == 3 and p["metric"] == "kl"
                            for p in report["diagnostics"]["position_regressions"]))
        self.assertTrue(any(p["metric"] == "max_kl"
                            for p in report["diagnostics"]["family_endpoint_regressions"]))

    def test_equal_family_and_document_weights_despite_unequal_position_counts(self):
        grouped = {("a", "one"): [[10., 0., 0., 0.]],
                   ("a", "two"): [[0., 0., 0., 0.]] * 9,
                   ("b", "three"): [[2., 0., 0., 0.]]}
        precise = {k: np.asarray(v, dtype=np.longdouble) for k, v in grouped.items()}
        strata, total = _summarize(grouped, precise)
        self.assertEqual(strata["a"]["mean_kl_google_x"], 5.)
        self.assertEqual(total["kl_google_x"], 3.5)
        self.assertEqual(total["resolution_interval"], [3.5, 3.5])

    def test_old_manifest_keeps_its_stricter_rule(self):
        path = self.suite/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("content_sha256")
        manifest["acceptance"] = LEGACY_ACCEPTANCE
        manifest["content_sha256"] = manifest_hash(manifest)
        path.write_text(json.dumps(manifest))
        x = self.g.astype("<f4")
        x[1, 0] += 0.65
        report = evaluate(self.suite, self.candidate(x))
        self.assertEqual(report["outcome"], "FAIL")
        self.assertTrue(report["diagnostics"]["blocking"])

    def test_positive_margin_is_not_silently_accepted(self):
        path = self.suite/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("content_sha256")
        manifest["acceptance"]["margin"] = 0.001
        manifest["content_sha256"] = manifest_hash(manifest)
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            verify(self.suite)

    def test_variability_cannot_be_averaged_away(self):
        self.assertEqual(evaluate(self.suite, self.candidate(self.g, repeatable=False))["outcome"], "INCONCLUSIVE")

    def test_single_run_is_diagnostic(self):
        self.assertEqual(evaluate(self.suite, self.candidate(self.g, repetitions=1))["outcome"], "INCONCLUSIVE")

    def test_modified_logits_rejected_before_comparison(self):
        path = self.suite/"google/a.npy"
        with path.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaises(ValueError):
            verify(self.suite)

    def test_wrong_candidate_pairing_rejected(self):
        out = self.candidate(self.g)
        capture = json.loads((out/"capture.json").read_text())
        capture["corpus_sha256"] = "0"*64
        (out/"capture.json").write_text(json.dumps(capture))
        with self.assertRaises(ValueError):
            evaluate(self.suite, out)

    def test_nonfinite_candidate_rejected(self):
        x = self.l.copy()
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            evaluate(self.suite, self.candidate(x))

    def test_duplicate_cases_do_not_create_groups(self):
        self.corpus["cases"] *= 2
        (self.corpus_dir/"manifest.json").write_text(json.dumps(self.corpus))
        with self.assertRaises(ValueError):
            load_corpus(self.corpus_dir/"manifest.json")

    def test_path_escape_rejected(self):
        self.corpus["cases"][0]["input"]["path"] = "../corpus/a.json"
        (self.corpus_dir/"manifest.json").write_text(json.dumps(self.corpus))
        with self.assertRaises(ValueError):
            load_corpus(self.corpus_dir/"manifest.json")

    def test_unobserved_chunk_positions_rejected(self):
        self.spec["chunks"] = [4]
        (self.corpus_dir/"a.json").write_text(json.dumps(self.spec))
        self.corpus["cases"][0]["input"]["sha256"] = file_hash(self.corpus_dir/"a.json")
        (self.corpus_dir/"manifest.json").write_text(json.dumps(self.corpus))
        with self.assertRaises(ValueError):
            load_corpus(self.corpus_dir/"manifest.json")

    def test_repeated_artifact_is_not_another_execution(self):
        out = self.candidate(self.l)
        receipt = json.loads((out/"capture.json").read_text())
        receipt["runs"] = [receipt["files"]]*3
        (out/"capture.json").write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            evaluate(self.suite, out)

    def test_resolved_auto_dispatch_is_checked_across_runs(self):
        out = self.candidate(self.l)
        receipt = json.loads((out/"capture.json").read_text())
        for i, run in enumerate(receipt["runs"]):
            folder = out/f"execution-{i}"
            folder.mkdir()
            shutil.copyfile(out/run["a"]["path"], folder/"a.npy")
            (folder/"stderr.txt").write_text(
                "resolve_fused_ops: Flash Attention " + ("disabled" if i == 2 else "enabled") + "\n")
            receipt["runs"][i] = {"a": {"path": f"execution-{i}/a.npy",
                                      "sha256": file_hash(folder/"a.npy")}}
        receipt["files"] = receipt["runs"][0]
        (out/"capture.json").write_text(json.dumps(receipt))
        self.assertEqual(evaluate(self.suite, out)["outcome"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
