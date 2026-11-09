from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import subprocess
import sys
import venv
import os
from unittest.mock import patch
from pathlib import Path

from xe_quality.qualification.driver import (
    _compare_routes,
    _evidence_files,
    _interpreter_path,
    _package_manifest,
    _probe_interpreter,
    main,
    _run_worker,
    _sha256,
)
from xe_quality.qualification import worker as qualification_worker


class ReferenceQualificationTests(unittest.TestCase):
    def test_evidence_manifest_is_deterministic_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "qualification"
            nested = root / "google" / "layers"
            nested.mkdir(parents=True)
            text_file = root / "PREFLIGHT.json"
            text_file.write_text('{"status": "pass"}\n', encoding="utf-8")
            binary_file = nested / "layer-00.npy"
            binary_file.write_bytes(b"\x00\x01\xff")
            (root / "QUALIFICATION.json").write_text(
                "self is not included", encoding="utf-8"
            )

            first = _evidence_files(root)
            second = _evidence_files(root)
            self.assertEqual(first, second)
            self.assertEqual(
                set(first), {"PREFLIGHT.json", "google/layers/layer-00.npy"}
            )
            self.assertEqual(first["PREFLIGHT.json"]["bytes"], text_file.stat().st_size)
            self.assertEqual(first["google/layers/layer-00.npy"]["bytes"], 3)

            binary_file.write_bytes(b"changed bytes")
            changed = _evidence_files(root)
            self.assertNotEqual(
                first["google/layers/layer-00.npy"]["sha256"],
                changed["google/layers/layer-00.npy"]["sha256"],
            )
            (root / "linked.json").symlink_to(text_file)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _evidence_files(root)

    def test_selected_environment_does_not_inherit_callers_site_packages(self):
        from xe_quality.qualification import driver

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            observed = {}

            def fake_run(argv, **kwargs):
                observed.update(kwargs["env"])
                observed["argv"] = argv
                target = root / "hf"
                target.mkdir()
                (target / "WORKER.json").write_text(
                    json.dumps(
                        {
                            "worker_sha256": _sha256(
                                Path(driver.__file__).with_name("worker.py")
                            )
                        }
                    )
                )

            with patch.dict(
                os.environ,
                {
                    "PYTHONPATH": "/wrong/reference/site-packages",
                    "PYTHONHOME": "/wrong/python/home",
                },
            ):
                with patch.object(driver.subprocess, "run", side_effect=fake_run):
                    _run_worker(
                        Path(sys.executable),
                        "hf",
                        root / "model",
                        root / "input",
                        root,
                        root / "source",
                        root / "config",
                        10,
                        {
                            "selected_executable": str(
                                _interpreter_path(Path(sys.executable))
                            ),
                            "package_sha256": "verified",
                            "caller_package_sha256": "verified",
                        },
                    )
            self.assertNotIn("PYTHONPATH", observed)
            self.assertNotIn("PYTHONHOME", observed)
            self.assertEqual(
                observed["argv"][1:4], ["-I", "-m", "xe_quality.qualification.worker"]
            )

    def test_suffix_free_state_qualification_is_rejected_before_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, request = root / "model.gguf", root / "input.json"
            model.write_bytes(b"not loaded: input must fail before execution")
            request.write_text(
                json.dumps({"tokens": [2, 100, 101, 102], "positions": [0]})
            )
            with self.assertRaises(SystemExit) as error:
                main(
                    [
                        "--model",
                        str(model),
                        "--input",
                        str(request),
                        "--google-source",
                        str(root),
                        "--google-python",
                        sys.executable,
                        "--hf-python",
                        sys.executable,
                        "--output",
                        str(root / "output"),
                    ]
                )
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root / "output").exists())

    def test_two_installed_copies_match_from_different_venvs_and_ignore_poisoned_env(
        self,
    ):
        from xe_quality.qualification import driver

        package_root = Path(driver.__file__).resolve().parents[1]
        manifest = _package_manifest(package_root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probes = []
            for name in ("google", "hf"):
                env_path = root / f"{name}-venv"
                venv.EnvBuilder(with_pip=False, symlinks=True).create(env_path)
                site = next((env_path / "lib").glob("python*/site-packages"))
                shutil.copytree(package_root, site / "xe_quality")
                selected = _interpreter_path(env_path / "bin/python")
                with patch.dict(
                    os.environ,
                    {
                        "PYTHONPATH": "/poison/site-packages",
                        "PYTHONHOME": "/poison/home",
                    },
                ):
                    probes.append(_probe_interpreter(selected, manifest))
                    self.assertEqual(
                        os.environ.get("PYTHONPATH"), "/poison/site-packages"
                    )
                    self.assertEqual(os.environ.get("PYTHONHOME"), "/poison/home")
            self.assertEqual(probes[0]["package_sha256"], manifest["package_sha256"])
            self.assertEqual(probes[1]["package_sha256"], manifest["package_sha256"])
            self.assertNotEqual(probes[0]["package_root"], probes[1]["package_root"])
            self.assertNotEqual(probes[0]["prefix"], probes[1]["prefix"])

    def test_missing_required_module_is_reported_by_isolated_probe(self):
        from xe_quality.qualification import driver

        package_root = Path(driver.__file__).resolve().parents[1]
        manifest = _package_manifest(package_root)
        with tempfile.TemporaryDirectory() as temp:
            env_path = Path(temp) / "selected-venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(env_path)
            site = next((env_path / "lib").glob("python*/site-packages"))
            shutil.copytree(package_root, site / "xe_quality")
            selected = _interpreter_path(env_path / "bin/python")
            with self.assertRaisesRegex(
                RuntimeError, "missing required modules: xe_missing_probe_module"
            ):
                _probe_interpreter(
                    selected, manifest, required_modules=("xe_missing_probe_module",)
                )

    def test_symlinked_venv_prefix_is_recorded_without_resolving(self):
        from xe_quality.qualification import driver

        package_root = Path(driver.__file__).resolve().parents[1]
        manifest = _package_manifest(package_root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            actual = root / "actual-venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(actual)
            site = next((actual / "lib").glob("python*/site-packages"))
            shutil.copytree(package_root, site / "xe_quality")
            alias = root / "selected-venv"
            alias.symlink_to(actual, target_is_directory=True)
            selected = _interpreter_path(alias / "bin/python")
            probe = _probe_interpreter(selected, manifest)
            self.assertEqual(probe["prefix"], str(alias))
            self.assertEqual(probe["selected_executable"], str(selected))
            self.assertNotEqual(Path(probe["prefix"]), Path(probe["prefix"]).resolve())

    def test_missing_or_tampered_hf_package_fails_before_any_worker(self):
        from xe_quality.qualification import driver

        package_root = Path(driver.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, request = root / "model.gguf", root / "input.json"
            model.write_bytes(b"preflight must fail before model access")
            request.write_text(
                json.dumps({"tokens": [2, 100, 101, 102], "positions": [3]})
            )
            for tamper in ("missing", "changed"):
                with self.subTest(tamper=tamper):
                    google_env = root / f"google-{tamper}"
                    hf_env = root / f"hf-{tamper}"
                    for env_path in (google_env, hf_env):
                        venv.EnvBuilder(with_pip=False, symlinks=True).create(env_path)
                        site = next((env_path / "lib").glob("python*/site-packages"))
                        shutil.copytree(package_root, site / "xe_quality")
                    hf_site = next((hf_env / "lib").glob("python*/site-packages"))
                    copied = hf_site / "xe_quality"
                    if tamper == "missing":
                        shutil.rmtree(copied)
                    else:
                        with (copied / "qualification/worker.py").open("a") as stream:
                            stream.write("\n# modified child package\n")
                    output = root / f"output-{tamper}"
                    original_probe = driver._probe_interpreter

                    def probe_without_dependencies(
                        python, expected, timeout=30, required_modules=()
                    ):
                        if "google-" in str(python):
                            required_modules = ()
                        return original_probe(
                            python, expected, timeout, required_modules=required_modules
                        )

                    with (
                        patch.object(driver, "verify_google_source"),
                        patch.object(
                            driver,
                            "_run_worker",
                            side_effect=AssertionError("worker must not launch"),
                        ),
                        patch.object(
                            driver,
                            "_probe_interpreter",
                            side_effect=probe_without_dependencies,
                        ),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                            main(
                                [
                                    "--model",
                                    str(model),
                                    "--input",
                                    str(request),
                                    "--google-source",
                                    str(root),
                                    "--google-python",
                                    str(google_env / "bin/python"),
                                    "--hf-python",
                                    str(hf_env / "bin/python"),
                                    "--output",
                                    str(output),
                                ]
                            )
                    receipt = json.loads((output / "PREFLIGHT.json").read_text())
                    self.assertEqual(receipt["status"], "fail")
                    self.assertEqual(receipt["children"]["google"]["status"], "pass")
                    self.assertEqual(receipt["children"]["hf"]["status"], "fail")

    def test_framework_preflight_checks_both_children_before_workers(self):
        from xe_quality.qualification import driver

        package_root = Path(driver.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, request = root / "model.gguf", root / "input.json"
            model.write_bytes(b"preflight must fail before model access")
            request.write_text(
                json.dumps({"tokens": [2, 100, 101, 102], "positions": [3]})
            )
            envs = {}
            for role in ("google", "hf"):
                env_path = root / f"{role}-venv"
                venv.EnvBuilder(with_pip=False, symlinks=True).create(env_path)
                site = next((env_path / "lib").glob("python*/site-packages"))
                shutil.copytree(package_root, site / "xe_quality")
                envs[role] = env_path
            output = root / "output"
            with (
                patch.object(driver, "verify_google_source"),
                patch.object(
                    driver,
                    "_run_worker",
                    side_effect=AssertionError("worker must not launch"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                    main(
                        [
                            "--model",
                            str(model),
                            "--input",
                            str(request),
                            "--google-source",
                            str(root),
                            "--google-python",
                            str(envs["google"] / "bin/python"),
                            "--hf-python",
                            str(envs["hf"] / "bin/python"),
                            "--output",
                            str(output),
                        ]
                    )
            receipt = json.loads((output / "PREFLIGHT.json").read_text())
            self.assertEqual(receipt["status"], "fail")
            self.assertIn("numpy", receipt["children"]["google"]["error"])
            self.assertIn(
                "tools/quality/setup_environment.py --profile reference",
                receipt["children"]["google"]["error"],
            )
            self.assertIn("numpy", receipt["children"]["hf"]["error"])
            self.assertIn(
                "tools/quality/setup_environment.py --profile qualification",
                receipt["children"]["hf"]["error"],
            )

    def test_hf_runtime_inventory_is_path_independent_and_tracks_source_and_native_bytes(
        self,
    ):
        class Distribution:
            version = "1.2.3"

            def __init__(self, root, relative_files):
                self.root = root
                self.files = relative_files

            def locate_file(self, relative):
                return self.root / Path(str(relative))

        relative_files = [
            Path("package/module.py"),
            Path("package/native/libbackend.so.1"),
            Path("package/README.txt"),
            Path("../../../bin/torchfrtrace"),
            Path("../../../bin/torchrun"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            roots = [root / "install-a", root / "install-b"]
            for install in roots:
                for name in qualification_worker._HF_RUNTIME_DISTRIBUTIONS:
                    dist_root = install / name
                    for relative in relative_files:
                        path = dist_root / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"unchanged source or library")

            def inventory(install):
                with patch.object(
                    qualification_worker.importlib.metadata,
                    "distribution",
                    side_effect=lambda name: Distribution(
                        install / name, relative_files
                    ),
                ):
                    return qualification_worker._hf_runtime_inventory()

            first = inventory(roots[0])
            second = inventory(roots[1])
            self.assertEqual(first["fingerprint_sha256"], second["fingerprint_sha256"])
            self.assertNotIn(str(roots[0]), json.dumps(first))
            self.assertNotIn(str(roots[1]), json.dumps(second))
            self.assertEqual(
                set(first["distributions"]["torch"]["files"]),
                {"package/module.py", "package/native/libbackend.so.1"},
            )

            source = roots[1] / "torch/package/module.py"
            source.write_bytes(b"changed Python source")
            self.assertNotEqual(
                first["fingerprint_sha256"], inventory(roots[1])["fingerprint_sha256"]
            )
            source.write_bytes(b"unchanged source or library")
            native = roots[1] / "torch/package/native/libbackend.so.1"
            native.write_bytes(b"changed native library")
            self.assertNotEqual(
                first["fingerprint_sha256"], inventory(roots[1])["fingerprint_sha256"]
            )

    def test_route_comparison_normalizes_google_batch_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            choices = [[0, 1], [2, 3]]
            google = [{"layer": i, "choices": [choices]} for i in range(30)]
            hf = [
                {"layer": i, "choices": [row[:] for row in choices]} for i in range(30)
            ]
            gp, hp = root / "google.json", root / "hf.json"
            gp.write_text(json.dumps(google), encoding="utf-8")
            hp.write_text(json.dumps(hf), encoding="utf-8")
            self.assertTrue(_compare_routes(gp, hp)["passed"])
            hf[17]["choices"][0][0] = 99
            hp.write_text(json.dumps(hf), encoding="utf-8")
            result = _compare_routes(gp, hp)
            self.assertFalse(result["passed"])
            self.assertEqual(result["mismatches"][0]["layer"], 17)
            self.assertEqual(result["mismatches"][0]["slot_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
