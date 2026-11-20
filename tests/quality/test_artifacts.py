"""Artifact identity, atomic completion, endian, and archive-copy checks."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality" / "src"))
from xe_quality import artifacts


def identity(**changes):
    result = {
        "reference_version": "ref-1",
        "config_hash": "a" * 64,
        "input_hash": "b" * 64,
        "contract_hash": "c" * 64,
    }
    result.update(changes)
    return result


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "sample.npy"
        self.logits = np.array([[1.25, -2.5, 7.0], [0.0, 4.0, -1.0]], dtype=np.float64)

    def test_roundtrip_little_endian_hash_and_manifest_canonicalization(self):
        rec = artifacts.write_artifact(
            self.path, self.logits, identity(), expected_vocab_size=3
        )
        verified = artifacts.verify_resume(self.path, identity(), expected_vocab_size=3)
        self.assertEqual(rec, verified)
        arr, read_record = artifacts.read_artifact(
            self.path,
            expected_identity=identity(),
            expected_vocab_size=3,
            expected_shape=(2, 3),
        )
        self.assertEqual(read_record, rec)
        self.assertEqual(arr.dtype.str, "<f4")
        np.testing.assert_array_equal(arr, self.logits.astype("<f4"))
        raw = self.path.read_bytes()
        self.assertIn(np.array([1.25], dtype="<f4").tobytes(), raw)
        self.assertEqual(
            artifacts.manifest_hash({"b": 2, "a": 1}),
            artifacts.manifest_hash({"a": 1, "b": 2}),
        )

    def test_identity_changes_invalidate_reference_config_and_input(self):
        artifacts.write_artifact(
            self.path, self.logits, identity(), expected_vocab_size=3
        )
        for changed in (
            identity(reference_version="ref-2"),
            identity(config_hash="d" * 64),
            identity(input_hash="e" * 64),
            identity(contract_hash="f" * 64),
        ):
            with self.assertRaises(artifacts.ArtifactError):
                artifacts.verify_resume(self.path, changed)

    def test_incomplete_marker_corrupt_bytes_and_vocab_truncation_fail(self):
        artifacts.write_artifact(
            self.path, self.logits, identity(), expected_vocab_size=3
        )
        marker = Path(str(self.path) + ".incomplete")
        marker.write_text("interrupted\n")
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.verify_artifact(self.path)
        marker.unlink()
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.verify_artifact(self.path, expected_vocab_size=4)
        content = bytearray(self.path.read_bytes())
        content[-1] ^= 1
        self.path.write_bytes(content)
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.verify_artifact(self.path)

    def test_nonfinite_and_fp32_overflow_are_rejected(self):
        bad = self.root / "bad.npy"
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.write_artifact(
                bad, np.array([[0.0, np.inf]]), identity(), expected_vocab_size=2
            )
        self.assertTrue(Path(str(bad) + ".incomplete").exists())
        huge = self.root / "huge.npy"
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.write_artifact(
                huge, np.array([[0.0, 1e300]]), identity(), expected_vocab_size=2
            )

    def test_writer_rejects_partial_vocabulary_before_creating_artifact(self):
        truncated = self.root / "truncated.npy"
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.write_artifact(
                truncated, self.logits[:, :2], identity(), expected_vocab_size=3
            )
        self.assertFalse(truncated.exists())
        self.assertFalse(Path(str(truncated) + ".incomplete").exists())

    def test_fp64_preserves_small_differences_and_big_endian_inputs(self):
        value = 1.0 + 2.0**-40
        precise = np.array([[1.0, value, -3.25]], dtype=np.float64)
        f64_path = self.root / "precise-f64.npy"
        rec = artifacts.write_artifact(
            f64_path, precise, identity(), expected_vocab_size=3, storage_dtype="<f8"
        )
        self.assertEqual(rec["dtype"], "<f8")
        f64, _ = artifacts.read_artifact(
            f64_path,
            expected_identity=identity(),
            expected_vocab_size=3,
            expected_dtype="<f8",
        )
        self.assertEqual(f64.dtype.str, "<f8")
        self.assertEqual(float(f64[0, 1]), value)
        self.assertNotEqual(float(f64[0, 1]), 1.0)

        f32_path = self.root / "rounded-f32.npy"
        artifacts.write_artifact(f32_path, precise, identity(), expected_vocab_size=3)
        f32, _ = artifacts.read_artifact(
            f32_path, expected_vocab_size=3, expected_dtype="<f4"
        )
        self.assertEqual(float(f32[0, 1]), 1.0)

        big_endian = np.array([[2.0**-40, value, 123.5]], dtype=">f8")
        be_path = self.root / "from-big-endian.npy"
        artifacts.write_artifact(
            be_path, big_endian, identity(), expected_vocab_size=3, storage_dtype="<f8"
        )
        canonical, _ = artifacts.read_artifact(
            be_path, expected_vocab_size=3, expected_dtype="<f8"
        )
        self.assertEqual(canonical.dtype.str, "<f8")
        np.testing.assert_array_equal(canonical, big_endian.astype("<f8"))

    def test_fp64_record_and_expected_dtype_mismatch_rejected(self):
        path = self.root / "record-f64.npy"
        artifacts.write_artifact(
            path, self.logits, identity(), expected_vocab_size=3, storage_dtype="<f8"
        )
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.verify_artifact(path, expected_dtype="<f4")
        record_path = Path(str(path) + ".json")
        import json

        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["dtype"] = "<f4"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.verify_artifact(path)

    def test_fp64_archive_copy_and_resume_preserve_dtype(self):
        source = self.root / "source-f64.npy"
        artifacts.write_artifact(
            source, self.logits, identity(), expected_vocab_size=3, storage_dtype="<f8"
        )
        archive = self.root / "f64-archive" / "logits.npy"
        copied = artifacts.copy_verified_archive(source, archive, identity())
        self.assertEqual(copied["dtype"], "<f8")
        resumed = artifacts.verify_resume(
            archive, identity(), expected_vocab_size=3, expected_dtype="<f8"
        )
        self.assertEqual(resumed, copied)
        self.assertEqual(source.read_bytes(), archive.read_bytes())
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.verify_resume(archive, identity(), expected_dtype="<f4")

    def test_invalid_identity_fields_and_hashes_are_rejected(self):
        for bad in (
            identity(reference_version=None),
            identity(reference_version="  "),
            identity(config_hash=None),
            identity(input_hash=""),
            identity(contract_hash="g" * 64),
            identity(config_hash="a" * 63),
        ):
            path = self.root / f"bad-{len(list(self.root.iterdir()))}.npy"
            with self.assertRaises(artifacts.ArtifactError):
                artifacts.write_artifact(path, self.logits, bad, expected_vocab_size=3)
            self.assertFalse(path.exists())
            self.assertFalse(Path(str(path) + ".json").exists())
            self.assertFalse(Path(str(path) + ".incomplete").exists())

    def test_write_never_overwrites_complete_corrupt_or_incomplete_state(self):
        artifacts.write_artifact(
            self.path, self.logits, identity(), expected_vocab_size=3
        )
        before_data = self.path.read_bytes()
        before_record = Path(str(self.path) + ".json").read_bytes()
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.write_artifact(
                self.path,
                self.logits + 1,
                identity(reference_version="ref-2"),
                expected_vocab_size=3,
            )
        self.assertEqual(self.path.read_bytes(), before_data)
        self.assertEqual(Path(str(self.path) + ".json").read_bytes(), before_record)
        corrupted = bytearray(before_data)
        corrupted[-1] ^= 1
        self.path.write_bytes(corrupted)
        corrupt_before = self.path.read_bytes()
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.write_artifact(
                self.path,
                self.logits + 1,
                identity(reference_version="ref-2"),
                expected_vocab_size=3,
            )
        self.assertEqual(self.path.read_bytes(), corrupt_before)
        self.assertEqual(Path(str(self.path) + ".json").read_bytes(), before_record)
        marker = Path(str(self.path) + ".incomplete")
        marker.write_bytes(b"prior incomplete marker\n")
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.write_artifact(
                self.path, self.logits + 2, identity(), expected_vocab_size=3
            )
        self.assertEqual(self.path.read_bytes(), corrupt_before)
        self.assertEqual(Path(str(self.path) + ".json").read_bytes(), before_record)
        self.assertEqual(marker.read_bytes(), b"prior incomplete marker\n")

    def test_export_archive_copy_is_reverified(self):
        artifacts.write_artifact(
            self.path, self.logits, identity(), expected_vocab_size=3
        )
        archive = self.root / "archive" / "sample.npy"
        copied = artifacts.copy_verified_archive(self.path, archive, identity())
        self.assertEqual(
            copied, artifacts.verify_artifact(archive, expected_identity=identity())
        )
        self.assertEqual(self.path.read_bytes(), archive.read_bytes())
        self.assertEqual(
            Path(str(self.path) + ".json").read_bytes(),
            Path(str(archive) + ".json").read_bytes(),
        )
        archived_data = archive.read_bytes()
        archived_record = Path(str(archive) + ".json").read_bytes()
        again = artifacts.copy_verified_archive(self.path, archive, identity())
        self.assertEqual(again, copied)
        self.assertEqual(archive.read_bytes(), archived_data)
        self.assertEqual(Path(str(archive) + ".json").read_bytes(), archived_record)
        other = self.root / "other.npy"
        artifacts.write_artifact(
            other,
            self.logits + 1,
            identity(reference_version="ref-2"),
            expected_vocab_size=3,
        )
        other_data_before = other.read_bytes()
        other_record_before = Path(str(other) + ".json").read_bytes()
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.copy_verified_archive(self.path, other, identity())
        self.assertEqual(other.read_bytes(), other_data_before)
        self.assertEqual(Path(str(other) + ".json").read_bytes(), other_record_before)


if __name__ == "__main__":
    unittest.main()
