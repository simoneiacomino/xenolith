"""Build the pinned, ordinary CPU llama.cpp fixture backend from source."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile

from .fixture_reference import file_hash, write_json

COMMIT = "f0458bd05e698306ce423b1850164042112f59c4"
ARCHIVE_SHA256 = "8e13888498d8e2e9723b7662dd40985cdb79d94dbfc4841212ba521d5cfb924a"
OPTIONS = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_CPU=ON",
           "-DGGML_SYCL=OFF", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_BLAS=OFF",
           "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_BUILD_COMMON=OFF", "-DLLAMA_CURL=OFF",
           "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=OFF",
           "-DBUILD_SHARED_LIBS=OFF", "-DGGML_BACKEND_DL=OFF"]


def prepare(source, archive, driver_source, model, output, jobs=8):
    output = Path(output).absolute()
    if output.exists() or not 1 <= jobs <= 32:
        raise ValueError("new output directory and bounded build parallelism required")
    if bool(source) == bool(archive):
        raise ValueError("provide exactly one pinned checkout or archive")
    output.mkdir(parents=True)
    if archive:
        if file_hash(archive) != ARCHIVE_SHA256:
            raise ValueError("llama source archive identity mismatch")
        source = output / "source"
        source.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(source, filter="data")
    else:
        source = Path(source).resolve()
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != COMMIT:
            raise ValueError("wrong llama source commit")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
            raise ValueError("llama source checkout must be clean")
    driver_source = Path(driver_source).resolve()
    pins = {str(p): file_hash(p) for p in source.rglob("*")
            if p.is_file() and ".git" not in p.parts}
    pins[str(driver_source)] = file_hash(driver_source)
    tools = {n: str(Path(shutil.which(n) or n).resolve()) for n in ("gcc", "g++", "cmake")}
    versions = {n: subprocess.check_output([p, "--version"], text=True).splitlines()[0] for n, p in tools.items()}
    if any("15.2.0" not in versions[n] for n in ("gcc", "g++")):
        raise ValueError("fixture CPU profile requires GCC/G++15.2.0")
    pins.update({p: file_hash(p) for p in tools.values()})
    build = output / "build"
    commands = [[tools["cmake"], "-S", str(source), "-B", str(build),
                 "-DCMAKE_C_COMPILER="+tools["gcc"], "-DCMAKE_CXX_COMPILER="+tools["g++"], *OPTIONS],
                [tools["cmake"], "--build", str(build), "-j", str(jobs)]]
    with (output / "build.log").open("wb") as log:
        for argv in commands:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        libraries = []
        for name in ("libllama.a", "libggml.a", "libggml-cpu.a", "libggml-base.a"):
            matches = list(build.rglob(name))
            if len(matches) != 1:
                raise ValueError("ambiguous static library: "+name)
            libraries.append(str(matches[0]))
        driver = output / "llama-driver"
        argv = [tools["g++"], "-O2", "-I"+str(source/"include"), "-I"+str(source/"ggml/include"),
                str(driver_source), "-Wl,--start-group", *libraries, "-Wl,--end-group",
                "-fopenmp", "-pthread", "-lm", "-ldl", "-o", str(driver)]
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        commands.append(argv)
    for p, digest in pins.items():
        if file_hash(p) != digest:
            raise ValueError("source/compiler changed during build")
    pins.update({p: file_hash(p) for p in [str(driver), *libraries, str(build/"CMakeCache.txt")]})
    linked = subprocess.check_output(["ldd", str(driver)], text=True)
    for line in linked.splitlines():
        for word in line.split():
            if word.startswith("/") and Path(word).is_file():
                pins[word] = file_hash(word)
    config = {"n_gpu_layers": 0, "use_extra_bufts": True, "n_ctx": "per-case context",
              "n_batch": 1024, "n_ubatch": 1024, "n_seq_max": 1,
              "n_threads": 8, "n_threads_batch": 8, "type_k": "F16", "type_v": "F16",
              "offload_kqv": False, "op_offload": False, "swa_full": True,
              "kv_unified": True, "no_perf": True, "flash_attn_type": "AUTO",
              "other_parameters": "Unmodified defaults at the pinned upstream commit; no sampler"}
    cpu = {}
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().split("\n\n")[0].splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in ("vendor_id", "model name", "cpu family", "model", "stepping", "flags", "microcode"):
                cpu[key.strip()] = value.strip()
    result = {"schema": "xe-quality-llama-backend-v1", "backend": "llama.cpp CPU",
              "source_commit": COMMIT, "driver": str(driver), "driver_sha256": file_hash(driver),
              "weights_sha256": file_hash(model), "input_pins": pins, "build_argv": commands,
              "public_profile": {"configuration": config, "toolchain": versions,
                  "platform": platform.platform(), "cpu": cpu,
                  "affinity": sorted(os.sched_getaffinity(0)),
                  "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS")},
                  "cmake_options": OPTIONS, "wrapper_sha256": file_hash(driver_source),
                  "source_archive_sha256": ARCHIVE_SHA256 if archive else None,
                  "cmake_cache_sha256": file_hash(build/"CMakeCache.txt"),
                  "builder_sha256": file_hash(__file__)}}
    write_json(output / "backend.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    p = parser.add_mutually_exclusive_group(required=True)
    p.add_argument("--source")
    p.add_argument("--source-archive")
    for name in ("driver-source", "model", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    prepare(args.source, args.source_archive, args.driver_source, args.model, args.output, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
