"""Finite-suite acceptance, provenance and no-HF producer contract controls."""
import json
from pathlib import Path
import tempfile
import unittest
import shutil

import numpy as np

from xe_quality.artifacts import manifest_hash
from xe_quality.fixture_reference import file_hash, load_corpus, write_json
from xe_quality.fixtures import LEGACY_ACCEPTANCE, _summarize, evaluate, seal, seal_vulkan, verify, verify_capture, main, capture, check_runtime_environment


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

    def vulkan_capture(self, name, offset=0.5):
        folder = self.root / name
        folder.mkdir()
        identity = ("XE_QUALITY_VULKAN device=Vulkan0 description=Test GPU "
                    "vendor_id=1 device_id=2 driver_version=3 api_version=4 physical_type=2")
        log = (identity + "\nllama_model_load: offloaded 36/36 layers to GPU\n"
               "llama_context: flash_attn = 0\n"
               "XE_QUALITY_VULKAN dispatch=full-layer-offload flash_attn=OFF offload_kqv=1 op_offload=1\n")
        receipt = dict(self.capture, schema="xe-quality-fixture-capture-v2")
        receipt["backend"] = dict(self.capture["backend"], backend="llama.cpp Vulkan",
            public_profile={"profile": "vulkan", "configuration": {"devices": ["Vulkan0"],
                "n_gpu_layers": 999, "flash_attn_type": "OFF", "offload_kqv": True,
                "op_offload": True}, "vulkan_runtime": {"identity": [identity]}})
        receipt["runs"] = []
        for i in range(3):
            dest = folder / str(i)
            dest.mkdir()
            logits = self.g.astype("<f4")
            logits[:, 0] += offset
            np.save(dest / "a.npy", logits)
            (dest / "stderr.txt").write_text(log)
            receipt["runs"].append({"a": {"path": f"{i}/a.npy", "sha256": file_hash(dest/"a.npy"),
                "stderr": {"path": f"{i}/stderr.txt", "sha256": file_hash(dest/"stderr.txt")}}})
        receipt["files"] = receipt["runs"][0]
        write_json(folder / "capture.json", receipt)
        return folder

    def named_suite(self):
        output = self.root / "vulkan-suite"
        seal_vulkan(self.suite, [("panther-vulkan", self.vulkan_capture("panther", 0.25)),
                                ("raptor-vulkan", self.vulkan_capture("raptor", 1.0))], output)
        return output

    def test_named_selection_changes_verdict_and_reports_identity(self):
        suite = self.named_suite()
        out = self.candidate(self.l)
        self.assertEqual(evaluate(suite, out, "panther-vulkan")["outcome"], "FAIL")
        report = evaluate(suite, out, "raptor-vulkan")
        self.assertEqual(report["outcome"], "PASS")
        self.assertEqual(report["baseline"]["id"], "raptor-vulkan")
        self.assertEqual(report["baseline"]["profile_sha256"], manifest_hash(report["baseline"]["profile"]))
        manifest, _, _ = verify(suite)
        original, _, _ = verify(self.suite)
        self.assertEqual(manifest["reference"], original["reference"])
        self.assertEqual((suite/"google/a.npy").read_bytes(), (self.suite/"google/a.npy").read_bytes())
        self.assertEqual((suite/"corpus/manifest.json").read_bytes(), (self.suite/"corpus/manifest.json").read_bytes())
        self.assertEqual((suite/"llama/panther-vulkan/a.npy").read_bytes(), (self.root/"panther/0/a.npy").read_bytes())

    def test_v2_selector_required_and_unknown_rejected(self):
        suite = self.named_suite()
        out = self.candidate(self.l)
        for selector in (None, "unknown"):
            with self.assertRaisesRegex(ValueError, "baseline"):
                evaluate(suite, out, selector)
        from unittest.mock import patch
        with patch("xe_quality.fixtures.capture") as native:
            code = main(["compare", "--fixtures", str(suite), "--model", "unused",
                         "--backend-config", "unused", "--output", "unused"])
            self.assertEqual(code, 2)
            native.assert_not_called()

    def test_unselected_baseline_tamper_is_rejected(self):
        suite = self.named_suite()
        with (suite/"llama/raptor-vulkan/a.npy").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaises(ValueError):
            evaluate(suite, self.candidate(self.l), "panther-vulkan")

    def test_seal_rejects_duplicate_or_unsafe_baseline_names(self):
        folder = self.vulkan_capture("vk")
        for names in (("same", "same"), ("../escape",), ("Bad Name",)):
            with self.assertRaises(ValueError):
                seal_vulkan(self.suite, [(n, folder) for n in names], self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())

    def test_vulkan_stderr_hash_is_checked(self):
        folder = self.vulkan_capture("vk")
        (folder/"2/stderr.txt").write_text("tampered")
        _, inputs = load_corpus(self.corpus_dir/"manifest.json")
        with self.assertRaises(ValueError):
            verify_capture(folder, self.corpus, inputs)

    def test_vulkan_fallback_and_identity_and_flash_rejected(self):
        for name, before, after in (("fallback", "36/36", "0/36"),
                                    ("device", "description=Test GPU", "description=Other GPU"),
                                    ("flash", "flash_attn = 0", "flash_attn = 1"),
                                    ("missing", "offloaded 36/36 layers to GPU", "no offload evidence")):
            folder = self.vulkan_capture(name)
            receipt = json.loads((folder/"capture.json").read_text())
            for run in receipt["runs"]:
                stderr = folder/run["a"]["stderr"]["path"]
                stderr.write_text(stderr.read_text().replace(before, after))
                run["a"]["stderr"]["sha256"] = file_hash(stderr)
            receipt["files"] = receipt["runs"][0]
            (folder/"capture.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "dispatch mismatch"):
                seal_vulkan(self.suite, [("test-vulkan", folder)], self.root/(name+"-suite"))

    def test_vulkan_incomplete_nonrepeatable_capture_rejected(self):
        for name, mutate in (("short", lambda r: r.update(repetitions=2, runs=r["runs"][:2])),
                             ("vary", lambda r: r.update(repeatable=False)),
                             ("incomplete", lambda r: r["runs"][2].clear())):
            folder = self.vulkan_capture(name)
            receipt = json.loads((folder/"capture.json").read_text())
            mutate(receipt)
            (folder/"capture.json").write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                seal_vulkan(self.suite, [("test-vulkan", folder)], self.root/(name+"-suite"))

    def test_public_vulkan_profile_has_portable_runtime_provenance(self):
        folder = self.vulkan_capture("portable")
        receipt_path = folder/"capture.json"
        receipt = json.loads(receipt_path.read_text())
        backend = receipt["backend"]
        paths = ["/private/home/gpu/libvulkan.so.1", "/tmp/private/icd.json"]
        backend["input_pins"] = {path: "d"*64 for path in paths}
        backend["public_profile"]["vulkan_runtime"].update(mapped_files=paths[:1], icd_manifests=paths[1:])
        backend["public_profile"]["build_environment"] = {"PATH": "/private/home/bin:/tmp/build/bin"}
        backend["public_profile"]["reused_static_build"] = {"original_backend_sha256": "e"*64}
        backend["public_profile"]["builder_sha256"] = "f"*64
        backend["public_profile"]["cmake_cache_sha256"] = "c"*64
        backend["public_profile"]["build_affinity"] = [12]
        backend["public_profile"]["affinity"] = [12, 13]
        backend["public_profile"]["environment"] = {"VK_DRIVER_FILES": paths[1],
            "LD_LIBRARY_PATH": "/private/home/gpu", "GGML_VK_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "8", "OMP_DYNAMIC": "FALSE", "OPENBLAS_NUM_THREADS": "8",
            "LD_BIND_NOW": "1", "VK_LOADER_DEBUG": "error", "HOME": "/private/home",
            "XDG_SESSION_ID": "2", "XDG_CONFIG_HOME": "/tmp/private/config"}
        receipt_path.write_text(json.dumps(receipt))
        before = receipt_path.read_bytes()
        output = self.root/"portable-suite"
        manifest = seal_vulkan(self.suite, [("portable-vulkan", folder)], output)
        self.assertEqual(receipt_path.read_bytes(), before)
        profile = manifest["baselines"]["portable-vulkan"]["profile"]
        self.assertEqual(profile["vulkan_runtime"]["mapped_files"],
                         [{"basename": "libvulkan.so.1", "sha256": "d"*64}])
        self.assertNotIn("icd_manifests", profile["vulkan_runtime"])
        self.assertEqual(profile["environment"], {"GGML_VK_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "8", "OMP_DYNAMIC": "FALSE", "OPENBLAS_NUM_THREADS": "8"})
        self.assertEqual(profile["affinity"], [12, 13])
        for field in ("build_environment", "build_affinity", "reused_static_build",
                      "builder_sha256", "cmake_cache_sha256", "stderr_sha256"):
            self.assertNotIn(field, profile)
        text = json.dumps(manifest)
        for private in ("/tmp/", "/home/", "/private/", "value_sha256"):
            self.assertNotIn(private, text)
        verify(output)

    def test_public_profile_rejects_private_paths_in_exported_settings(self):
        folder = self.vulkan_capture("private-toolchain")
        receipt_path = folder/"capture.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["backend"]["public_profile"]["toolchain"] = {"gcc": "/home/builder/bin/gcc"}
        receipt_path.write_text(json.dumps(receipt))
        output = self.root/"invalid-public-settings"
        with self.assertRaisesRegex(ValueError, "private path"):
            seal_vulkan(self.suite, [("test-vulkan", folder)], output)
        self.assertFalse(output.exists())

    def test_runtime_environment_added_removed_changed_rejected_before_inference(self):
        from unittest.mock import patch
        backend = {"backend": "llama.cpp Vulkan", "public_profile": {"environment": {
            "VK_DRIVER_FILES": "/private/icd.json", "GGML_VK_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": "/private/lib", "XDG_CONFIG_HOME": "/private/config"}}}
        expected = backend["public_profile"]["environment"]
        with patch.dict("os.environ", expected, clear=True):
            check_runtime_environment(backend)
        for changed in ({**expected, "VK_LOADER_DEBUG": "all"},
                        {k: v for k, v in expected.items() if k != "VK_DRIVER_FILES"},
                        {**expected, "GGML_VK_VISIBLE_DEVICES": "1"},
                        {**expected, "LD_PRELOAD": "/private/injected.so"},
                        {**expected, "HOME": "/other/home"},
                        {**expected, "XDG_CONFIG_HOME": "/other/config"}):
            with patch.dict("os.environ", changed, clear=True), \
                    patch("xe_quality.fixtures.backend_receipt", return_value=backend), \
                    patch("xe_quality.fixtures.subprocess.run") as native:
                with self.assertRaisesRegex(ValueError, "runtime environment differs"):
                    capture(self.corpus_dir/"manifest.json", "unused", "unused", self.root/"never-created")
                native.assert_not_called()
                self.assertFalse((self.root/"never-created").exists())

    def test_vulkan_capture_records_execution_affinity_and_checks_changes(self):
        from unittest.mock import patch
        folder = self.vulkan_capture("affinity-source")
        backend = json.loads((folder/"capture.json").read_text())["backend"]
        backend["driver"] = "/unused/driver"
        backend["public_profile"]["affinity"] = [12]
        original = json.dumps(backend, sort_keys=True)
        log = (folder/"0/stderr.txt").read_bytes()
        def native(argv, stdout, stderr, **kwargs):
            dest = Path(argv[3])
            self.l.tofile(dest/"a.bin")
            stderr.write(log)
        out = self.root/"affinity-capture"
        with patch("xe_quality.fixtures.backend_receipt", return_value=backend), \
                patch("xe_quality.fixtures.check_runtime_environment"), \
                patch("xe_quality.fixtures.check_backend"), \
                patch("xe_quality.fixtures.os.sched_getaffinity", return_value={0, 1, 2, 3}), \
                patch("xe_quality.fixtures.subprocess.run", side_effect=native):
            captured = capture(self.corpus_dir/"manifest.json", "unused", "unused", out, repetitions=1)
        profile = captured["backend"]["public_profile"]
        self.assertEqual(profile["build_affinity"], [12])
        self.assertEqual(profile["affinity"], [0, 1, 2, 3])
        self.assertEqual(json.dumps(backend, sort_keys=True), original)
        with patch("xe_quality.fixtures.backend_receipt", return_value=backend), \
                patch("xe_quality.fixtures.check_runtime_environment"), \
                patch("xe_quality.fixtures.os.sched_getaffinity", side_effect=[{0, 1}, {0}]), \
                patch("xe_quality.fixtures.subprocess.run") as process:
            with self.assertRaisesRegex(ValueError, "affinity changed"):
                capture(self.corpus_dir/"manifest.json", "unused", "unused", self.root/"affinity-changed")
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
