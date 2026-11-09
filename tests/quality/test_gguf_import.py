from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xe_quality.gguf_import import GGUF, GGUFError


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _fixture(tensors, metadata=(), *, version=3):
    """Build tiny valid GGUF headers directly, independently of the reader."""
    header = bytearray(
        b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata))
    )
    for key, typ, value in metadata:
        header += _string(key) + struct.pack("<I", typ)
        if typ == 6:
            header += struct.pack("<f", value)
        elif typ == 4:
            header += struct.pack("<I", value)
        elif typ == 8:
            header += _string(value)
        else:
            raise AssertionError("test fixture metadata type unsupported")
    blobs, relative = [], 0
    for name, dims, kind, raw in tensors:
        relative = (relative + 31) // 32 * 32
        header += _string(name) + struct.pack("<I", len(dims))
        header += struct.pack("<" + "Q" * len(dims), *dims)
        header += struct.pack("<IQ", kind, relative)
        blobs.append((relative, raw))
        relative += len(raw)
    data_start = (len(header) + 31) // 32 * 32
    output = bytearray(data_start + relative)
    output[: len(header)] = header
    for offset, raw in blobs:
        output[data_start + offset : data_start + offset + len(raw)] = raw
    return bytes(output)


def _gguf(path, *, kind=0, shape=(4,), payload=None, version=3):
    if payload is None:
        payload = np.arange(np.prod(shape), dtype="<f4").tobytes()
    dims = tuple(reversed(shape))
    header = bytearray(b"GGUF" + struct.pack("<IQQ", version, 1, 0))
    header += _string("sample") + struct.pack("<I", len(dims))
    header += struct.pack("<" + "Q" * len(dims), *dims)
    header += struct.pack("<IQ", kind, 0)
    header += b"\0" * ((-len(header)) % 32)
    path.write_bytes(header + payload)


class GGUFImportTests(unittest.TestCase):
    def test_bounded_f32_rows_and_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tiny.gguf"
            _gguf(path, shape=(3, 4))
            with GGUF(path) as reader:
                self.assertEqual(reader.tensors["sample"].shape, (3, 4))
                np.testing.assert_array_equal(reader.read("sample", 1), [[4, 5, 6, 7]])
                np.testing.assert_array_equal(
                    reader.read("sample", slice(1, 3)), [[4, 5, 6, 7], [8, 9, 10, 11]]
                )
                self.assertEqual(
                    len(reader.inventory(hash_tensors=False)["file_sha256"]), 64
                )
                with self.assertRaisesRegex(ValueError, "step 1"):
                    reader.read("sample", slice(None, None, 2))

    def test_f16_and_q4_0_bounded_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "f16.gguf"
            _gguf(path, kind=1, shape=(4,), payload=np.arange(4, dtype="<f2").tobytes())
            with GGUF(path) as reader:
                np.testing.assert_array_equal(reader.read("sample"), [0, 1, 2, 3])
            block = np.zeros(18, dtype=np.uint8)
            block[:2] = np.frombuffer(np.float16(0.5).tobytes(), dtype=np.uint8)
            block[2:] = 0x91
            _gguf(path, kind=2, shape=(32,), payload=block.tobytes())
            with GGUF(path) as reader:
                decoded = reader.read("sample")
                np.testing.assert_array_equal(
                    decoded[:16], np.full(16, -3.5, np.float32)
                )
                np.testing.assert_array_equal(
                    decoded[16:], np.full(16, 0.5, np.float32)
                )
                with self.assertRaisesRegex(GGUFError, "block"):
                    reader.read("sample", slice(1, 2))

    def test_q4_all_nibbles_and_independent_expected_order(self):
        low, high = list(range(16)), list(reversed(range(16)))
        raw = struct.pack("<e", 0.5) + bytes(
            lo | (hi << 4) for lo, hi in zip(low, high)
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "q4.gguf"
            path.write_bytes(_fixture([("q", (32, 1), 2, raw)]))
            with GGUF(path) as reader:
                self.assertEqual(reader.tensors["q"].shape, (1, 32))
                expected = (
                    np.array(
                        [n - 8 for n in low] + [n - 8 for n in high], dtype=np.float32
                    )
                    * 0.5
                )
                np.testing.assert_array_equal(reader.read("q")[0], expected)
                np.testing.assert_array_equal(
                    reader.read("q", rows=0), expected[None, :]
                )

    def test_q_and_expert_axis_transforms_are_explicit(self):
        from xe_quality import gguf_import as module

        q = np.arange(12).reshape(4, 3)
        np.testing.assert_array_equal(
            module.q_to_gemma(q, 2, 2), q.reshape(2, 2, 3).transpose(0, 2, 1)
        )
        np.testing.assert_array_equal(
            module.kv_to_gemma(q, q + 20), np.stack((q, q + 20))
        )
        exp = np.arange(2 * 4 * 3).reshape(2, 4, 3)
        np.testing.assert_array_equal(
            module.expert_gate_up_to_gemma(exp, 2, 2), exp.reshape(2, 2, 2, 3)
        )
        down = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        np.testing.assert_array_equal(
            module.expert_down_to_gemma(down), down.transpose(0, 2, 1)
        )

    def test_inventory_hashes_exact_file_and_tensor_bytes(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.gguf"
            raw = struct.pack("<2f", 1.25, -3.0)
            path.write_bytes(_fixture([("x", (2,), 0, raw)]))
            with GGUF(path) as reader:
                inventory = reader.inventory(chunk_bytes=7)
                self.assertEqual(
                    inventory["file_sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    inventory["tensors"]["x"]["sha256"], hashlib.sha256(raw).hexdigest()
                )

    def test_unknown_type_bad_q4_shape_nonfinite_metadata_and_truncation_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.gguf"
            cases = [
                _fixture([("x", (1,), 99, b"\0" * 4)]),
                _fixture([("bad_q4_rows", (16, 2), 2, bytes(18))]),
                b"NOPE" + struct.pack("<IQQ", 3, 0, 0),
                _fixture([], [("test.bad", 6, float("nan"))]),
            ]
            for index, payload in enumerate(cases):
                candidate = path.with_name(f"bad-{index}.gguf")
                candidate.write_bytes(payload)
                with self.assertRaises(GGUFError):
                    GGUF(candidate)
            valid = _fixture([("x", (2,), 0, struct.pack("<2f", 1.0, 2.0))])
            path.write_bytes(valid[:-1])
            with self.assertRaises(GGUFError):
                GGUF(path)

    def test_rejects_nonfinite_bad_magic_and_unsupported_version(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.gguf"
            path.write_bytes(b"NOPE")
            with self.assertRaisesRegex(GGUFError, "magic"):
                GGUF(path)
            _gguf(path, version=1)
            with self.assertRaisesRegex(GGUFError, "version"):
                GGUF(path)
            _gguf(path, payload=np.array([np.nan] * 4, dtype="<f4").tobytes())
            with GGUF(path) as reader, self.assertRaisesRegex(GGUFError, "non-finite"):
                reader.read("sample")


if __name__ == "__main__":
    unittest.main()
