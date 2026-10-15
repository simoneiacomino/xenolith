#!/usr/bin/env python3
"""Cross-build snapshot checks using the real engine reader and writer.

By default host session buffers replace GPU allocation. Full mode
extracts historical writers from Git (or source directories), rather than making
the new writer imitate the old header. --model also validates real inference in
separate writer/reader processes on the GPU. All artifacts live in a temporary tree.
"""

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile

BASELINE = "68b8fbf55787eda392c8cbfa32bab5ad14622e8a"
METADATA_V2 = "4521185b381bf9f48e5a12ff9abb05eb5a3a164b"
ROOT = Path(__file__).resolve().parent.parent


def run(args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def source_hash(root):
    entries = "".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in ("xenolith.c", "xenolith.cl")
    )
    return hashlib.sha256(entries.encode()).hexdigest()


def crc64(data):
    value = 0
    for byte in data:
        value ^= byte << 56
        for _ in range(8):
            value = ((value << 1) ^ (0x42F0E1EBA9EA3693 if value >> 63 else 0))
            value &= (1 << 64) - 1
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-only", action="store_true")
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--metadata-source", type=Path)
    parser.add_argument("--model", type=Path, help="also check GPU inference continuation")
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--base-cflags", default="-O3 -march=native -std=c11 -Wall -Wextra -pthread")
    args = parser.parse_args()
    if args.model and (args.current_only or not args.model.is_file()):
        parser.error("--model needs an existing model file and full compatibility checks")
    flags_hash = hashlib.sha256(args.base_cflags.encode()).hexdigest()
    gpu_object = ROOT / "xenolith_gpu_spv.o"
    if not gpu_object.is_file():
        parser.error("build xenolith_gpu_spv.o first (or use the Makefile targets)")

    with tempfile.TemporaryDirectory(prefix="xenolith-snapshot-compat-") as tmp:
        tmp = Path(tmp)

        def compile_fixture(name, source, commit, numeric_hash, cflags_hash, compiler=None,
                            fixture="snapshot_compat_fixture.c"):
            output = tmp / name
            defines = {
                "SNAPSHOT_ENGINE_SOURCE": str(source / "xenolith.c"),
                "XE_BUILD_COMMIT": commit,
                "XE_NUMERIC_SOURCE_HASH": numeric_hash,
                "XE_NUMERIC_CFLAGS_HASH": cflags_hash,
            }
            command = shlex.split(args.cc) + shlex.split(args.base_cflags)
            command += ["-ffunction-sections", "-fdata-sections"]
            command += [f'-D{key}="{value}"' for key, value in defines.items()]
            if compiler:
                command += ["-U__VERSION__", f'-D__VERSION__="{compiler}"']
            command += [ROOT / "tests" / fixture, source / "format.c",
                        gpu_object, "-Wl,--gc-sections", "-lm", "-lze_loader", "-o", output]
            run(command)
            return output

        model = tmp / "model.bin"
        model.write_bytes(bytes(range(129)))
        original_model = model.read_bytes()
        env = dict(os.environ, SNAPSHOT_MODEL_FILE=str(model))

        def exercise(binary, mode, path, want="ok", capacity=129, tokens=17, state="unchanged"):
            run([binary, mode, path, want, capacity, tokens, state], env=env)

        a = compile_fixture("current-a", ROOT, "build-a", source_hash(ROOT), flags_hash)
        # Different diagnostic/compiler/flag identities with unchanged numerical
        # implementation. This is not a test of a second physical compiler.
        b = compile_fixture("current-b", ROOT, "build-b", "different-source",
                            "different-flags", "compat-test-compiler")
        snapshot = tmp / "current.xkv"
        exercise(a, "write", snapshot)
        original = snapshot.read_bytes()
        assert struct.unpack_from("<H", original, 8)[0] == 2
        assert original[328:344].split(b"\0")[0] == b"build-a"
        exercise(b, "read", snapshot)
        exercise(b, "write", tmp / "other.xkv")
        exercise(a, "read", tmp / "other.xkv")
        other = (tmp / "other.xkv").read_bytes()
        assert other[344:376].split(b"\0")[0] == b"compat-test-compiler"
        assert original[256:288] == other[256:288]
        assert original[96:128] == hashlib.sha256(original_model).digest()

        # Same bytes at a new path/inode/timestamp retain their identity.
        copied = tmp / "copy.bin"
        copied.write_bytes(original_model)
        os.utime(copied, ns=(123456789, 987654321))
        env["SNAPSHOT_MODEL_FILE"] = str(copied)
        exercise(b, "read", snapshot)
        env["SNAPSHOT_MODEL_FILE"] = str(model)

        # Same header and length, different last payload byte: old identity
        # cannot distinguish these weights, but both new directions must reject.
        model.write_bytes(original_model[:-1] + bytes([original_model[-1] ^ 1]))
        exercise(b, "read", snapshot, "model-mismatch")
        variant = tmp / "variant.xkv"
        exercise(b, "write", variant)
        assert variant.read_bytes()[96:128] == hashlib.sha256(model.read_bytes()).digest()
        exercise(a, "read", variant)
        model.write_bytes(original_model)
        exercise(a, "read", variant, "model-mismatch")
        exercise(a, "read", snapshot)
        print("model identity: complete SHA-256, payload changes, copies and cached reads PASS", flush=True)

        def changed(data, offset, replacement, want, state="unchanged", fix_crc=True):
            copy = bytearray(data)
            copy[offset:offset + len(replacement)] = replacement
            if fix_crc:
                struct.pack_into("<Q", copy, 376, crc64(copy[:376]))
            path = tmp / "changed.xkv"
            path.write_bytes(copy)
            exercise(a, "read", path, want, state=state)

        for i, kind in enumerate(("model", "tokenizer", "template", "layout", "context", "numeric")):
            offset = 96 + i * 32
            changed(original, offset, bytes([original[offset] ^ 1]), f"{kind}-mismatch")
        abi = struct.unpack_from("<I", original, 320)[0]
        changed(original, 320, struct.pack("<I", abi + 1), "numeric-mismatch")
        changed(original, 8, struct.pack("<H", 99), "version")
        changed(original, 10, struct.pack("<H", 1), "version")
        changed(original, 324, b"\1", "format")
        changed(original, 343, b"x", "format")
        changed(original, 375, b"x", "format")
        changed(original, 328, b"new-origin", "ok")
        changed(original, 328, b"new-origin", "format", fix_crc=False)
        exercise(a, "read", snapshot, "context-mismatch", capacity=130)
        # Payload corruption must leave no partially restored session usable.
        global_offset = struct.unpack_from("<Q", original, 384 + 4 * 40 + 8)[0]
        changed(original, global_offset, bytes([original[global_offset] ^ 1]),
                "format", state="empty")
        print("snapshot compatibility: cross-build v2, rejection checks and corruption PASS", flush=True)

        # Global rows and SWA wraparound at a non-power-of-two context capacity.
        large = tmp / "wrapped.xkv"
        exercise(a, "write", large, capacity=2049, tokens=1057)
        exercise(b, "read", large, capacity=2049, tokens=1057)
        large.unlink()
        print("snapshot compatibility: exact wrapped SWA/global state PASS", flush=True)

        if not args.current_only:
            writers = []
            for version, commit, supplied in ((1, BASELINE, args.legacy_source),
                                               (2, METADATA_V2, args.metadata_source)):
                legacy = tmp / f"metadata-v{version}"
                legacy.mkdir()
                for name in ("xenolith.c", "xenolith.cl", "xenolith.h", "xenolith_internal.h",
                             "format.c", "format.h"):
                    if supplied:
                        data = (supplied / name).read_bytes()
                    else:
                        data = run(["git", "show", f"{commit}:{name}"], cwd=ROOT,
                                   stdout=subprocess.PIPE).stdout
                    (legacy / name).write_bytes(data)
                if version == 1 and source_hash(legacy) != "528b6db377397368158a28dc155047628ac30a5a06c38aa6c32a5680ccd56d84":
                    parser.error("legacy source does not match the supported baseline")
                if (legacy / "xenolith.cl").read_bytes() != (ROOT / "xenolith.cl").read_bytes():
                    parser.error("kernel changed: build a separate historical kernel for inference checks")
                old = compile_fixture(f"metadata-v{version}-writer", legacy, commit[:9],
                                      source_hash(legacy), flags_hash)
                saved = tmp / f"metadata-v{version}.xkv"
                exercise(old, "write", saved)
                legacy_bytes = saved.read_bytes()
                assert struct.unpack_from("<H", legacy_bytes, 8)[0] == version
                exercise(old, "read", saved)
                # Even the audited v1 numeric bridge cannot establish weights
                # from a header-only identity. Reject before changing state.
                exercise(a, "read", saved, "model-mismatch")
                exercise(b, "read", saved, "model-mismatch")
                assert saved.read_bytes() == legacy_bytes
                exercise(old, "read", snapshot, "version" if version == 1 else "model-mismatch")
                model.write_bytes(original_model[:-1] + bytes([original_model[-1] ^ 1]))
                exercise(old, "read", saved)  # Demonstrates the old blind spot.
                exercise(a, "read", saved, "model-mismatch")
                model.write_bytes(original_model)
                exercise(old, "write", large, capacity=2049, tokens=1057)
                exercise(a, "read", large, "model-mismatch", capacity=2049, tokens=1057)
                large.unlink()
                writers.append((version, commit, legacy))
                print(f"snapshot compatibility: metadata v{version} rejected without state changes PASS", flush=True)
            if args.model:
                new_model = compile_fixture("current-model", ROOT, "build-a", source_hash(ROOT),
                                            flags_hash, fixture="snapshot_compat_model.c")
                other_model = compile_fixture("other-model", ROOT, "build-b", "different-source",
                                              "different-flags", "compat-test-compiler",
                                              fixture="snapshot_compat_model.c")
                pairs = [("v2 content cross-build", new_model, other_model, [])]
                for version, commit, legacy in writers:
                    old_model = compile_fixture(f"metadata-v{version}-model", legacy, commit[:9],
                                                source_hash(legacy), flags_hash,
                                                fixture="snapshot_compat_model.c")
                    pairs.append((f"metadata v{version} -> replay", old_model, new_model,
                                  ["model-mismatch"]))
                for label, writer, reader, extra in pairs:
                    artifacts = tmp / "model-snapshots"
                    artifacts.mkdir()
                    run([writer, "write", args.model.resolve(), artifacts])
                    run([reader, "read", args.model.resolve(), artifacts, *extra])
                    for artifact in artifacts.iterdir():
                        artifact.unlink()
                    artifacts.rmdir()
                    print(f"snapshot model compatibility: {label}, exact continuation PASS", flush=True)
    if not args.model:
        print("Host buffers only; model continuation still requires GPU/model validation.")


if __name__ == "__main__":
    main()
