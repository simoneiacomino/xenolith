"""Atomic full-vocabulary FP32/FP64 logit artifacts with identity checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np


class ArtifactError(ValueError):
    """Artifact incomplete, corrupt, incompatible, or malformed."""


REQUIRED_IDENTITY = ("reference_version", "config_hash", "input_hash", "contract_hash")
SCHEMA_VERSION = 1
SUPPORTED_STORAGE_DTYPES = ("<f4", "<f8")


def _storage_dtype(value: str) -> str:
    if value not in SUPPORTED_STORAGE_DTYPES:
        raise ArtifactError(
            f"storage dtype must be one of {SUPPORTED_STORAGE_DTYPES}, got {value!r}"
        )
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise ArtifactError(f"value is not strict canonical JSON: {e}") from e


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """SHA-256 of a strict canonical JSON manifest (key order independent)."""
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def _identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ArtifactError("identity must be a mapping")
    out = dict(identity)
    missing = [key for key in REQUIRED_IDENTITY if key not in out]
    if missing:
        raise ArtifactError(f"identity missing required fields: {missing}")
    version = out["reference_version"]
    if not isinstance(version, str) or not version.strip():
        raise ArtifactError("reference_version must be a nonempty stable string")
    for key in ("config_hash", "input_hash", "contract_hash"):
        value = out[key]
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
        ):
            raise ArtifactError(f"{key} must be a 64-character hexadecimal SHA-256")
    canonical_json_bytes(out)
    return out


def _assert_destination_absent(data: Path, record: Path, marker: Path) -> None:
    existing = [str(path) for path in (data, record, marker) if path.exists()]
    if existing:
        raise ArtifactError(
            f"refusing to overwrite existing artifact state: {existing}"
        )


def _create_marker_exclusive(marker: Path, payload: bytes) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as e:
        raise ArtifactError(f"artifact marker already exists: {marker}") from e
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def _paths(path: str | os.PathLike[str]) -> tuple[Path, Path, Path]:
    data = Path(path)
    return data, Path(str(data) + ".json"), Path(str(data) + ".incomplete")


def _sha_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _write_npy_little_endian(
    path: Path, logits: Any, storage_dtype: str
) -> tuple[tuple[int, ...], int]:
    storage_dtype = _storage_dtype(storage_dtype)
    arr = np.asarray(logits)
    if arr.ndim < 2 or arr.shape[-1] < 2:
        raise ArtifactError(
            "logits must have position axes and a full vocabulary axis >= 2"
        )
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise ArtifactError("logits must be real numeric values")
    shape, vocab = tuple(int(x) for x in arr.shape), int(arr.shape[-1])
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.lib.format.write_array_header_1_0(
                f, {"descr": storage_dtype, "fortran_order": False, "shape": shape}
            )
            for i in range(shape[0]):
                with np.errstate(over="ignore", invalid="ignore"):
                    chunk = np.asarray(arr[i]).astype(storage_dtype, copy=False)
                if not np.isfinite(chunk).all():
                    raise ArtifactError(
                        f"logits contain NaN, infinity, or overflow in {storage_dtype}"
                    )
                f.write(chunk.tobytes(order="C"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return shape, vocab


def write_artifact(
    path: str | os.PathLike[str],
    logits: Any,
    identity: Mapping[str, Any],
    *,
    expected_vocab_size: int,
    storage_dtype: str = "<f4",
) -> dict[str, Any]:
    """Write little-endian FP32 (default) or FP64 logits atomically."""
    data_path, record_path, marker_path = _paths(path)
    ident = _identity(identity)
    storage_dtype = _storage_dtype(storage_dtype)
    if type(expected_vocab_size) is not int or expected_vocab_size < 2:
        raise ArtifactError("expected_vocab_size must be an integer >= 2")
    array_shape = np.shape(logits)
    if len(array_shape) < 2 or array_shape[-1] != expected_vocab_size:
        raise ArtifactError("logits do not contain the expected full vocabulary")
    _assert_destination_absent(data_path, record_path, marker_path)
    _create_marker_exclusive(marker_path, b"artifact write in progress\n")
    try:
        shape, vocab = _write_npy_little_endian(data_path, logits, storage_dtype)
        record = {
            "schema_version": SCHEMA_VERSION,
            "shape": list(shape),
            "vocab_size": vocab,
            "dtype": storage_dtype,
            "data_sha256": _sha_file(data_path),
            "identity": ident,
            "identity_sha256": manifest_hash(ident),
        }
        _atomic_bytes(record_path, canonical_json_bytes(record) + b"\n")
        verify_artifact(data_path, expected_identity=ident, _allow_incomplete=True)
        marker_path.unlink()
        return record
    except BaseException:
        # The marker remains, preventing an old/new partial pair from resuming.
        raise


def _strict_json(path: Path) -> Any:
    def no_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ArtifactError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(
                f,
                object_pairs_hook=no_duplicates,
                parse_constant=lambda x: (_ for _ in ()).throw(
                    ArtifactError(f"invalid JSON number {x}")
                ),
            )
    except (OSError, json.JSONDecodeError) as e:
        raise ArtifactError(f"cannot read artifact record {path}: {e}") from e


def verify_artifact(
    path: str | os.PathLike[str],
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_vocab_size: int | None = None,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: str | None = None,
    chunk_rows: int = 32,
    _allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate completion marker, canonical identity, byte hash, dtype, shape and finiteness."""
    data_path, record_path, marker_path = _paths(path)
    if marker_path.exists() and not _allow_incomplete:
        raise ArtifactError(f"artifact marked incomplete: {data_path}")
    record = _strict_json(record_path)
    required = {
        "schema_version",
        "shape",
        "vocab_size",
        "dtype",
        "data_sha256",
        "identity",
        "identity_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ArtifactError("artifact record has missing or unexpected fields")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ArtifactError("unsupported artifact schema or dtype")
    try:
        record_dtype = _storage_dtype(record["dtype"])
    except ArtifactError as e:
        raise ArtifactError("unsupported artifact schema or dtype") from e
    if expected_dtype is not None and record_dtype != _storage_dtype(expected_dtype):
        raise ArtifactError(
            f"artifact storage dtype mismatch: expected {expected_dtype}, got {record_dtype}"
        )
    ident = _identity(record["identity"])
    if manifest_hash(ident) != record["identity_sha256"]:
        raise ArtifactError("identity manifest hash mismatch")
    if expected_identity is not None and ident != _identity(expected_identity):
        raise ArtifactError(
            "artifact identity mismatch (reference/config/input/contract invalidation)"
        )
    if _sha_file(data_path) != record["data_sha256"]:
        raise ArtifactError("logit data SHA-256 mismatch")
    try:
        arr = np.load(data_path, mmap_mode="r", allow_pickle=False)
    except Exception as e:
        raise ArtifactError(f"invalid NumPy logits artifact: {e}") from e
    if (
        list(arr.shape) != record["shape"]
        or arr.dtype.str != record_dtype
        or arr.ndim < 2
    ):
        raise ArtifactError("logit array shape or little-endian storage dtype mismatch")
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ArtifactError("logit shape does not match expected shape")
    if arr.shape[-1] != record["vocab_size"] or (
        expected_vocab_size is not None and arr.shape[-1] != expected_vocab_size
    ):
        raise ArtifactError("logit vocabulary size mismatch")
    if chunk_rows < 1:
        raise ArtifactError("chunk_rows must be positive")
    for start in range(0, arr.shape[0], chunk_rows):
        if not np.isfinite(arr[start : start + chunk_rows]).all():
            raise ArtifactError("logits contain NaN or infinity")
    return record


def verify_resume(
    path: str | os.PathLike[str],
    expected_identity: Mapping[str, Any],
    *,
    expected_vocab_size: int | None = None,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: str | None = None,
) -> dict[str, Any]:
    """Resume only when a complete artifact exactly matches all frozen identities."""
    return verify_artifact(
        path,
        expected_identity=expected_identity,
        expected_vocab_size=expected_vocab_size,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
    )


def read_artifact(
    path: str | os.PathLike[str],
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_vocab_size: int | None = None,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: str | None = None,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read only after complete hash, identity, shape, dtype, vocab and finite checks."""
    record = verify_artifact(
        path,
        expected_identity=expected_identity,
        expected_vocab_size=expected_vocab_size,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
    )
    data_path, _, _ = _paths(path)
    return np.load(data_path, mmap_mode=mmap_mode, allow_pickle=False), record


def copy_verified_archive(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy a verified pair; existing destinations must already be identical."""
    src, src_record, _ = _paths(source)
    source_record = verify_artifact(src, expected_identity=expected_identity)
    dst, dst_record, dst_marker = _paths(destination)
    existing = [path.exists() for path in (dst, dst_record, dst_marker)]
    if any(existing):
        if existing[2] or not existing[0] or not existing[1]:
            raise ArtifactError(
                "refusing to replace incomplete or partial archive destination"
            )
        destination_record = verify_artifact(dst, expected_identity=expected_identity)
        if (
            destination_record != source_record
            or destination_record["data_sha256"] != source_record["data_sha256"]
            or dst.stat().st_size != src.stat().st_size
        ):
            raise ArtifactError(
                "existing archive destination is not byte-identical to source"
            )
        return destination_record
    _create_marker_exclusive(dst_marker, b"archive copy in progress\n")
    try:
        for origin, target in ((src, dst), (src_record, dst_record)):
            fd, temp_name = tempfile.mkstemp(
                prefix=target.name + ".tmp-", dir=target.parent
            )
            try:
                with os.fdopen(fd, "wb") as out, origin.open("rb") as inp:
                    shutil.copyfileobj(inp, out, length=8 << 20)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temp_name, target)
            except BaseException:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
                raise
        result = verify_artifact(
            dst, expected_identity=expected_identity, _allow_incomplete=True
        )
        dst_marker.unlink()
        return result
    except BaseException:
        raise
