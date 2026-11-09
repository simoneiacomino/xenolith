"""Bounded, independent GGUF reader for Gemma 4 numerical validation.

The module intentionally supports only GGUF v2/v3 and tensor encodings F32, F16 and Q4_0. Tensor ``shape`` is logical NumPy order:
GGUF's ``ne[]`` dimensions are reversed. Construction reads only the header;
``read`` and ``inventory`` access tensor data on demand.

Mapping functions convert GGUF projection layouts to Gemma 4 parameter axes.
They validate source shapes; callers must also validate the complete
architecture-specific tensor inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import mmap
import os
import struct
from typing import Any, Iterable, Mapping

import numpy as np


class GGUFError(ValueError):
    """Malformed or unsupported GGUF input."""


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]  # logical row-major shape (reverse of GGUF ne[])
    ggml_type: int
    type_name: str
    offset: int  # absolute file offset
    nbytes: int


_TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0"}
_BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18)}
_META_TYPES = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}


class _Cursor:
    def __init__(self, data: mmap.mmap):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise GGUFError(f"truncated GGUF at byte {self.pos} (need {n})")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.take(size))

    def string(self) -> str:
        (n,) = self.unpack("Q")
        if n > len(self.data) - self.pos:
            raise GGUFError("GGUF string length exceeds remaining file")
        try:
            return self.take(n).decode("utf-8")
        except UnicodeDecodeError as e:
            raise GGUFError("invalid UTF-8 in GGUF string") from e


def _finite_metadata(value: Any, key: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GGUFError(f"non-finite metadata value: {key}")
    if isinstance(value, list):
        for item in value:
            _finite_metadata(item, key)


def _metadata_value(c: _Cursor, typ: int, key: str) -> Any:
    if typ == 8:
        return c.string()
    if typ == 9:
        elem_type, count = c.unpack("IQ")
        if count > (len(c.data) - c.pos) + 1:
            raise GGUFError(f"implausible array length in metadata {key}")
        vals = [_metadata_value(c, elem_type, key) for _ in range(count)]
        return vals
    if typ not in _META_TYPES:
        raise GGUFError(f"unknown GGUF metadata type {typ} for {key}")
    fmt, _ = _META_TYPES[typ]
    (v,) = c.unpack(fmt)
    _finite_metadata(v, key)
    return v


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class GGUF:
    """Read a GGUF header and expose bounded row access to supported tensors."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.fspath(path)
        self._file = open(self.path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._closed = False
        c = _Cursor(self._map)
        if c.take(4) != b"GGUF":
            self.close()
            raise GGUFError("bad GGUF magic")
        (version,) = c.unpack("I")
        if version not in (2, 3):
            self.close()
            raise GGUFError(f"unsupported GGUF version {version}; expected 2 or 3")
        tensor_count, metadata_count = c.unpack("QQ")
        if tensor_count > len(self._map) // 24 or metadata_count > len(self._map) // 8:
            self.close()
            raise GGUFError("implausible GGUF header counts")
        self.version = version
        self.metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = c.string()
            (typ,) = c.unpack("I")
            if key in self.metadata:
                raise GGUFError(f"duplicate metadata key: {key}")
            self.metadata[key] = _metadata_value(c, typ, key)
        descs: dict[str, tuple[tuple[int, ...], int, int]] = {}
        for _ in range(tensor_count):
            name = c.string()
            (ndim,) = c.unpack("I")
            if not 1 <= ndim <= 4:
                raise GGUFError(f"unsupported dimension count {ndim} for {name}")
            dims = c.unpack("Q" * ndim)
            if any(d == 0 for d in dims):
                raise GGUFError(f"zero dimension in tensor {name}")
            typ, rel = c.unpack("IQ")
            if name in descs:
                raise GGUFError(f"duplicate tensor name: {name}")
            if typ not in _TYPE_NAMES:
                raise GGUFError(f"unknown tensor type {typ} for {name}")
            descs[name] = (tuple(reversed(dims)), typ, rel)
        alignment = self.metadata.get("general.alignment", 32)
        if (
            not isinstance(alignment, int)
            or alignment <= 0
            or alignment & (alignment - 1)
        ):
            raise GGUFError("general.alignment must be a positive power of two")
        data_start = _align(c.pos, alignment)
        self.tensors: dict[str, Tensor] = {}
        ranges = []
        for name, (shape, typ, rel) in descs.items():
            count = math.prod(shape)
            block, block_bytes = _BLOCK[typ]
            if count % block or shape[-1] % block:
                raise GGUFError(
                    f"tensor {name} last logical dimension must be divisible by block {block}"
                )
            nbytes = count // block * block_bytes
            start, end = data_start + rel, data_start + rel + nbytes
            if rel % alignment:
                raise GGUFError(f"tensor {name} offset is not aligned to {alignment}")
            if end > len(self._map):
                raise GGUFError(f"tensor {name} extends beyond end of file")
            ranges.append((start, end, name))
            self.tensors[name] = Tensor(
                name, shape, typ, _TYPE_NAMES[typ], start, nbytes
            )
        ranges.sort()
        for left, right in zip(ranges, ranges[1:]):
            if left[1] > right[0]:
                raise GGUFError(f"overlapping tensor data: {left[2]} and {right[2]}")
        self.data_start = data_start

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            self._closed = True
            self._map.close()
            self._file.close()

    def __del__(self):
        # Also release descriptors if header parsing raises midway through __init__.
        try:
            if not getattr(self, "_closed", True):
                self.close()
        except Exception:
            pass

    def __enter__(self) -> "GGUF":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read(self, name: str, rows: int | slice | None = None) -> np.ndarray:
        """Decode all rows or a first-axis integer/slice; Q4_0 rows must align
        to its 32-element blocks. An integer returns a one-row array.
        """
        if self._closed:
            raise GGUFError("reader is closed")
        try:
            t = self.tensors[name]
        except KeyError as e:
            raise KeyError(f"unknown tensor {name!r}") from e
        if len(t.shape) < 1:
            raise GGUFError("scalar tensor row access is unsupported")
        nrows = t.shape[0]
        width = math.prod(t.shape[1:]) if len(t.shape) > 1 else 1
        if rows is None:
            start, stop = 0, nrows
        elif isinstance(rows, int):
            idx = rows if rows >= 0 else nrows + rows
            if not 0 <= idx < nrows:
                raise IndexError("row index out of range")
            start, stop = idx, idx + 1
        elif isinstance(rows, slice):
            start, stop, step = rows.indices(nrows)
            if step != 1:
                raise ValueError("row slices must have step 1")
        else:
            raise TypeError("rows must be None, an integer, or a slice")
        block, bbytes = _BLOCK[t.ggml_type]
        flat_q4 = t.ggml_type == 2 and len(t.shape) == 1
        if width % block and not flat_q4:
            raise GGUFError(
                f"row width {width} of {name} is not divisible by block {block}"
            )
        elem0, elem1 = start * width, stop * width
        if flat_q4 and (elem0 % block or elem1 % block):
            raise GGUFError("1D Q4_0 row slice must start and end on a 32-value block")
        if t.ggml_type == 2 and (start * width) % block:
            raise GGUFError("Q4_0 row slice does not begin at a block boundary")
        byte0, byte1 = elem0 // block * bbytes, elem1 // block * bbytes
        raw = memoryview(self._map)[t.offset + byte0 : t.offset + byte1]
        if t.ggml_type == 0:
            out = (
                np.frombuffer(raw, dtype="<f4")
                .copy()
                .reshape((stop - start,) + t.shape[1:])
            )
        elif t.ggml_type == 1:
            out = (
                np.frombuffer(raw, dtype="<f2")
                .astype(np.float32)
                .reshape((stop - start,) + t.shape[1:])
            )
        else:
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
            scales = blocks[:, :2].copy().view("<f2").astype(np.float32).reshape(-1)
            if not np.isfinite(scales).all():
                raise GGUFError(f"non-finite Q4_0 scale in {name}")
            packed = blocks[:, 2:]
            low = (packed & 0x0F).astype(np.int16) - 8
            high = (packed >> 4).astype(np.int16) - 8
            vals = np.empty((len(blocks), 32), dtype=np.float32)
            vals[:, :16] = low * scales[:, None]
            vals[:, 16:] = high * scales[:, None]
            out = vals.reshape(-1)[
                elem0 % block : (elem0 % block) + (stop - start) * width
            ]
            out = out.reshape((stop - start,) + t.shape[1:])
        if not np.isfinite(out).all():
            raise GGUFError(f"non-finite decoded value in {name}")
        return out

    def inventory(
        self, hash_tensors: bool = True, chunk_bytes: int = 8 << 20
    ) -> dict[str, Any]:
        """Return file SHA-256 and tensor descriptors; hash tensor bytes by default.

        Hashing is streamed in bounded chunks. The full file digest is still a
        deliberate full-file read and should be requested only when needed.
        """
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        fh = hashlib.sha256()
        for off in range(0, len(self._map), chunk_bytes):
            fh.update(self._map[off : off + chunk_bytes])
        entries = {}
        for name, t in self.tensors.items():
            item: dict[str, Any] = {
                "shape": list(t.shape),
                "type": t.type_name,
                "offset": t.offset,
                "nbytes": t.nbytes,
            }
            if hash_tensors:
                digest = hashlib.sha256()
                for off in range(t.offset, t.offset + t.nbytes, chunk_bytes):
                    digest.update(
                        self._map[off : min(off + chunk_bytes, t.offset + t.nbytes)]
                    )
                item["sha256"] = digest.hexdigest()
            entries[name] = item
        return {
            "path": self.path,
            "file_size": len(self._map),
            "file_sha256": fh.hexdigest(),
            "version": self.version,
            "metadata": self.metadata,
            "tensors": entries,
        }


# Architecture transforms validate source shapes before reordering axes.
# Inputs are ordinary arrays already read within the GGUF bounds.
def q_to_gemma(x: np.ndarray, heads: int, head_dim: int) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] != heads * head_dim:
        raise ValueError("Q source must have shape [heads*head_dim, features]")
    return x.reshape(heads, head_dim, x.shape[1]).transpose(0, 2, 1)


def global_k_to_gemma(k: np.ndarray, heads: int, head_dim: int) -> np.ndarray:
    """Map shared global-attention K; Gemma4 has no corresponding global V."""
    return q_to_gemma(k, heads, head_dim)


def kv_to_gemma(k: np.ndarray, v: np.ndarray) -> np.ndarray:
    if k.shape != v.shape or k.ndim != 2:
        raise ValueError("K and V must have matching [heads*head_dim, features] shapes")
    return np.stack((k, v), axis=0)


def attention_output_to_gemma(x: np.ndarray, heads: int, head_dim: int) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] != heads * head_dim:
        raise ValueError(
            "attention output source must have shape [features, heads*head_dim]"
        )
    return x.reshape(x.shape[0], heads, head_dim).transpose(1, 2, 0)


def dense_gate_up_to_gemma(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    if gate.shape != up.shape or gate.ndim != 2:
        raise ValueError(
            "dense gate and up must have matching [hidden, features] shapes"
        )
    return np.stack((gate, up), axis=0)


def expert_gate_up_to_gemma(x: np.ndarray, experts: int, hidden: int) -> np.ndarray:
    if x.ndim != 3 or x.shape[0] != experts or x.shape[1] != 2 * hidden:
        raise ValueError(
            "expert gate/up source must have shape [experts, 2*hidden, features]"
        )
    return x.reshape(experts, 2, hidden, x.shape[2])


def dense_down_to_gemma(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("dense down source must be rank 2")
    return x.T


def expert_down_to_gemma(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError("expert down source must be rank 3")
    return x.transpose(0, 2, 1)


def router_to_gemma(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("router source must be rank 2 [experts, features]")
    return x.T


def audit_mapping(
    source_names: Iterable[str], mapping: Mapping[str, str]
) -> dict[str, list[str]]:
    """Check one-to-one source assignment; return missing/unexpected/collisions.

    ``mapping`` maps GGUF tensor names to destination parameter names. Multiple
    sources may intentionally alias a destination only if the caller records
    those destinations separately instead of using this strict function.
    """
    sources = list(source_names)
    seen, duplicate_set = set(), set()
    for name in sources:
        (duplicate_set if name in seen else seen).add(name)
    duplicates = sorted(duplicate_set)
    missing = sorted(set(sources) - set(mapping))
    unexpected = sorted(set(mapping) - set(sources))
    destinations: dict[str, list[str]] = {}
    for source, dest in mapping.items():
        destinations.setdefault(dest, []).append(source)
    collisions = sorted(dest for dest, ss in destinations.items() if len(ss) > 1)
    return {
        "missing": missing,
        "unexpected": unexpected,
        "duplicate_sources": duplicates,
        "destination_collisions": collisions,
    }
