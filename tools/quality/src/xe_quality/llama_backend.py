"""Build a pinned Vulkan (default) or historical CPU llama.cpp fixture backend."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import re
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


def profile_options(profile):
    if profile not in ("vulkan", "cpu-historical"):
        raise ValueError("unknown llama profile")
    return ["-DGGML_VULKAN=ON" if profile == "vulkan" and x == "-DGGML_VULKAN=OFF" else x
            for x in OPTIONS]


def runtime_probe(driver, output):
    """Probe device creation without loading a model; retain dynamic ICD libraries."""
    probe = subprocess.run([str(driver), "--probe"], capture_output=True, text=True,
                           check=True, timeout=120)
    (output / "probe.stdout.txt").write_text(probe.stdout)
    (output / "probe.stderr.txt").write_text(probe.stderr)
    files = {line.partition("\t")[2] for line in probe.stdout.splitlines()
             if line.startswith("runtime_file\t")}
    identities = [line for line in probe.stderr.splitlines() if line.startswith("XE_QUALITY_VULKAN ")]
    if not identities or not any("libvulkan.so" in p for p in files):
        raise ValueError("Vulkan device/loader probe is incomplete")
    pins = {p: file_hash(p) for p in sorted(files)}
    # The loader opens ICD manifests dynamically, so ldd alone is insufficient.
    manifests = set()
    for key in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES"):
        manifests.update(Path(p).absolute() for p in os.environ.get(key, "").split(":") if p)
    for folder in ("/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d",
                   str(Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home()/".config"))) / "vulkan/icd.d"),
                   str(Path(os.environ.get("XDG_DATA_HOME", str(Path.home()/".local/share"))) / "vulkan/icd.d")):
        manifests.update(Path(folder).glob("*.json"))
    pins.update({str(p): file_hash(p) for p in sorted(manifests) if p.is_file()})
    return {"identity": identities, "mapped_files": sorted(files),
            "icd_manifests": sorted(str(p) for p in manifests if p.is_file())}, pins


def prepare(source, archive, driver_source, model, output, jobs=8,
            profile="vulkan", device="Vulkan0", cc="gcc", cxx="g++"):
    options = profile_options(profile)
    vulkan = profile == "vulkan"
    if not re.fullmatch(r"Vulkan[0-9]+", device):
        raise ValueError("device must be an explicit Vulkan device name (e.g. Vulkan0)")
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
    tools = {n: str(Path(shutil.which(value) or value).resolve())
             for n, value in (("gcc", cc), ("g++", cxx), ("cmake", "cmake"))}
    versions = {n: subprocess.check_output([p, "--version"], text=True).splitlines()[0] for n, p in tools.items()}
    if not vulkan and any("15.2.0" not in versions[n] for n in ("gcc", "g++")):
        raise ValueError("fixture CPU profile requires GCC/G++15.2.0")
    pins.update({p: file_hash(p) for p in tools.values()})
    build = output / "build"
    commands = [[tools["cmake"], "-S", str(source), "-B", str(build),
                 "-DCMAKE_C_COMPILER="+tools["gcc"], "-DCMAKE_CXX_COMPILER="+tools["g++"], *options],
                [tools["cmake"], "--build", str(build), "-j", str(jobs)]]
    with (output / "build.log").open("wb") as log:
        for argv in commands:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        libraries = []
        for name in ("libllama.a", "libggml.a", "libggml-cpu.a", "libggml-base.a") + (("libggml-vulkan.a",) if vulkan else ()):
            matches = list(build.rglob(name))
            if len(matches) != 1:
                raise ValueError("ambiguous static library: "+name)
            libraries.append(str(matches[0]))
        if vulkan:
            cache = (build / "CMakeCache.txt").read_text()
            match = re.search(r"^Vulkan_GLSLC_EXECUTABLE:[^=]+=([^\n]+)$", cache, re.MULTILINE)
            if not match or not Path(match[1]).is_file():
                raise ValueError("Vulkan shader compiler identity missing from CMake cache")
            include_match = re.search(r"^Vulkan_INCLUDE_DIR:[^=]+=([^\n]+)$", cache, re.MULTILINE)
            if not include_match or not (Path(include_match[1]) / "vulkan/vulkan.h").is_file():
                raise ValueError("Vulkan header identity missing from CMake cache")
            library_match = re.search(r"^Vulkan_LIBRARY:[^=]+=([^\n]+)$", cache, re.MULTILINE)
            if not library_match or not Path(library_match[1]).is_file():
                raise ValueError("Vulkan library identity missing from CMake cache")
            vulkan_library = str(Path(library_match[1]).absolute())
            pins[vulkan_library] = file_hash(vulkan_library)
            vulkan_include = str(Path(include_match[1]).absolute())
            for folder in ("vulkan", "vk_video"):
                for header in (Path(include_match[1]) / folder).rglob("*"):
                    if header.is_file():
                        pins[str(header)] = file_hash(header)
            shader_compiler = str(Path(match[1]).resolve())
            pins[shader_compiler] = file_hash(shader_compiler)
            versions["glslc"] = subprocess.check_output([shader_compiler, "--version"], text=True).splitlines()[0]
        driver = output / "llama-driver"
        argv = [tools["g++"], "-O2", "-I"+str(source/"include"), "-I"+str(source/"ggml/include"),
                *(["-I"+vulkan_include, "-DXE_QUALITY_VULKAN", '-DXE_QUALITY_DEVICE="'+device+'"'] if vulkan else []),
                str(driver_source), "-Wl,--start-group", *libraries, "-Wl,--end-group",
                *([vulkan_library] if vulkan else []), "-fopenmp", "-pthread", "-lm", "-ldl", "-o", str(driver)]
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
    runtime = None
    if vulkan:
        runtime, runtime_pins = runtime_probe(driver, output)
        pins.update(runtime_pins)
    config = {"n_gpu_layers": 0, "use_extra_bufts": True, "n_ctx": "per-case context",
              "n_batch": 1024, "n_ubatch": 1024, "n_seq_max": 1,
              "n_threads": 8, "n_threads_batch": 8, "type_k": "F16", "type_v": "F16",
              "offload_kqv": False, "op_offload": False, "swa_full": True,
              "kv_unified": True, "no_perf": True, "flash_attn_type": "AUTO",
              "other_parameters": "Unmodified defaults at the pinned upstream commit; no sampler"}
    if vulkan:
        config.update(n_gpu_layers=999, devices=[device], split_mode="NONE", main_gpu=0,
                      offload_kqv=True, op_offload=True, flash_attn_type="OFF")
    cpu = {}
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().split("\n\n")[0].splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in ("vendor_id", "model name", "cpu family", "model", "stepping", "flags", "microcode"):
                cpu[key.strip()] = value.strip()
    environment_keys = {"OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS",
                        "VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES",
                        "GGML_VK_VISIBLE_DEVICES", "VULKAN_SDK", "HOME", "LD_LIBRARY_PATH",
                        "LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_ORIGIN_PATH", "LD_HWCAP_MASK",
                        "LD_BIND_NOW", "LD_BIND_NOT", "LD_ASSUME_KERNEL",
                        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CONFIG_DIRS", "XDG_DATA_DIRS", "XDG_RUNTIME_DIR"}
    environment_keys.update(k for k in os.environ if k.startswith(("GGML_VK_", "VK_", "LD_")))
    result = {"schema": "xe-quality-llama-backend-v1", "backend": "llama.cpp Vulkan" if vulkan else "llama.cpp CPU",
              "source_commit": COMMIT, "driver": str(driver), "driver_sha256": file_hash(driver),
              "weights_sha256": file_hash(model), "input_pins": pins, "build_argv": commands,
              "public_profile": {"profile": profile, "configuration": config, "vulkan_runtime": runtime, "toolchain": versions,
                  "platform": platform.platform(), "cpu": cpu,
                  "affinity": sorted(os.sched_getaffinity(0)),
                  "environment": {k: os.environ.get(k) for k in sorted(environment_keys)},
                  "build_environment": {k: os.environ.get(k) for k in
                      ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "PATH", "VULKAN_SDK", "CMAKE_LIBRARY_PATH", "CMAKE_PREFIX_PATH")},
                  "cmake_options": options, "wrapper_sha256": file_hash(driver_source),
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
    parser.add_argument("--profile", choices=("vulkan", "cpu-historical"), default="vulkan")
    parser.add_argument("--device", default="Vulkan0")
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args(argv)
    prepare(args.source, args.source_archive, args.driver_source, args.model, args.output, args.jobs,
            args.profile, args.device, args.cc, args.cxx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
