#!/usr/bin/env python3
"""Run the actual OpenCL K-load blocks on the host under AddressSanitizer.

This checks their memory footprint, not GPU execution or attention arithmetic.
Extraction fails closed if K pointers move outside the supported load blocks.
Three deliberately broken copies must be rejected to validate test sensitivity.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
KERNELS = {
    "regular": ("xe_prefill_attn_online_b8_global_shared", ("k",)),
    "cow": ("xe_prefill_attn_online_b8_global_cow", ("prefix_k", "tail_k")),
}


def masked(source):
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',
                  lambda m: " " * len(m[0]), source, flags=re.S)


def load_block(source, name, pointers):
    scan = masked(source)
    match = re.search(r"\b__kernel\s+void\s+" + re.escape(name) + r"\s*\(", scan)
    if not match:
        raise ValueError(f"missing kernel {name}; review the bounds harness")
    begin = scan.index("{", match.end()) + 1
    depth, end = 1, begin
    while depth and end < len(scan):
        depth += (scan[end] == "{") - (scan[end] == "}")
        end += 1
    if depth:
        raise ValueError(f"unclosed kernel {name}")
    body = source[begin:end - 1]
    code = scan[begin:end - 1]
    loops = list(re.finditer(r"for\s*\(\s*int\s+key0\b", code))
    if len(loops) != 1:
        raise ValueError(f"unexpected K loop in {name}; review the bounds harness")
    start = loops[0].start()
    barrier = re.search(r"\bbarrier\s*\(\s*CLK_LOCAL_MEM_FENCE\s*\)\s*;", code[start:])
    if not barrier:
        raise ValueError(f"missing load barrier in {name}")
    stop = start + barrier.start()
    outside = code[:start] + code[stop:]
    for pointer in pointers:
        if re.search(r"\b" + pointer + r"\b", outside):
            raise ValueError(f"{pointer} used outside checked block in {name}")
        if not re.search(r"\b" + pointer + r"\b", code[start:stop]):
            raise ValueError(f"missing {pointer} load in {name}")
    block = body[start:stop] + "}\n"
    # OpenCL scalar-to-vector zero syntax -> C vector initializer.
    return re.sub(r"\(uint4\)\s*\(0\)", "(uint4){0}", block)


def host_source(source):
    regular = load_block(source, *KERNELS["regular"])
    cow = load_block(source, *KERNELS["cow"])
    return r"""
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef _Float16 half;
typedef unsigned int uint4 __attribute__((vector_size(16)));
#define __global
/* Observe every loaded vector, including rows later masked by attention.
 * Volatile destinations prevent dead-load elimination in this host test. */
#define __local volatile
static void regular_load(half *k, int n_count, int position) {
    half lk[8 * 512] __attribute__((aligned(64)));
    int maximum_position = position < n_count ? position : n_count - 1;
    for (int kv_head = 0; kv_head < 2; kv_head++)
        for (int lid = 0; lid < 128; lid++) {
""" + regular + r"""
        }
}
static void cow_load(half *prefix_k, half *tail_k, int prefix_capacity,
                     int tail_capacity, int split, int position) {
    half lk[8 * 512] __attribute__((aligned(64)));
    for (int kv_head = 0; kv_head < 2; kv_head++)
        for (int lid = 0; lid < 128; lid++) {
""" + cow + r"""
        }
}
static half *cache(int capacity) {
    size_t bytes = (size_t)2 * capacity * 512 * sizeof(half);
    half *p = aligned_alloc(64, bytes);
    assert(p);
    memset(p, 0, bytes);
    return p;
}
static void check(int capacity, int split) {
    int tail_capacity = capacity - split;
    half *prefix = cache(capacity), *tail = cache(tail_capacity);
    for (int position = capacity - 3; position < capacity; position++) {
        regular_load(prefix, capacity, position);
        if (position >= split)
            cow_load(prefix, tail, capacity, tail_capacity, split, position);
    }
    free(tail);
    free(prefix);
}
int main(void) {
    int cases = 0;
    for (int capacity = 64; capacity < 72; capacity++)
        for (int split = 8; split < 16; split++, cases++) check(capacity, split);
    for (int split = 8; split < 16; split++)
        for (int tail = 511; tail <= 513; tail++, cases++) check(split + tail, split);
    for (int capacity = 64; capacity < 72; capacity++)
        for (int tail = 1; tail <= 8; tail++, cases++) check(capacity, capacity - tail);
    printf("K load bounds: %d host cases PASS\n", cases);
    return 0;
}
"""


def mutate(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"mutation anchor changed: {old!r}; review the sensitivity test")
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, default=ROOT / "xenolith.cl",
                        help="candidate kernel source; allows a separate trusted test harness")
    args = parser.parse_args()
    source_bytes = args.kernel.read_bytes()
    source = source_bytes.decode()
    print(f"K load bounds: source SHA256 {hashlib.sha256(source_bytes).hexdigest()}",
          flush=True)
    variants = [("current", source)]
    variants += [
        ("unbounded-regular", mutate(source, "if (key0 + 8 <= n_count)", "if (1)")),
        ("unbounded-cow-tile", mutate(source, "source_key + 8 <= capacity", "1")),
        ("unbounded-cow-rows", mutate(source, "if (key <= position)", "if (1)")),
    ]
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=0")
    with tempfile.TemporaryDirectory(prefix="xe-kv-bounds-") as directory:
        directory = Path(directory)
        for name, code in variants:
            src, binary = directory / f"{name}.c", directory / name
            src.write_text(host_source(code))
            cmd = shlex.split(os.environ.get("CC", "gcc")) + [
                "-O1", "-g", "-std=c11", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address", "-fno-omit-frame-pointer", "-fno-pie", "-no-pie",
                str(src), "-o", str(binary),
            ]
            subprocess.run(cmd, check=True, timeout=60)
            result = subprocess.run([str(binary)], env=env, text=True,
                                    capture_output=True, timeout=60)
            if name == "current":
                if result.returncode != 0:
                    raise RuntimeError(result.stdout + result.stderr)
                print(result.stdout.strip(), flush=True)
            elif (result.returncode == 0 or
                  "AddressSanitizer: heap-buffer-overflow" not in result.stderr or
                  not re.search(r"\bREAD of size \d+", result.stderr)):
                raise RuntimeError(f"mutation {name} was not detected as an out-of-bounds read:\n"
                                   + result.stdout + result.stderr)
            else:
                print(f"K load bounds: mutation {name} rejected PASS", flush=True)


if __name__ == "__main__":
    main()
