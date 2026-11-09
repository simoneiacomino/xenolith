"""Discover and fingerprint the actual CPU reference runtime, without inference."""

import hashlib, json, pathlib, platform, sys, os, importlib.metadata


def digest(path):
    with pathlib.Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def package_sources(name):
    dist = importlib.metadata.distribution(name)
    pins = {}
    for rel in dist.files or []:
        if str(rel).endswith(".py"):
            path = pathlib.Path(dist.locate_file(rel))
            if path.is_file():
                pins[str(rel)] = digest(path)
    return pins


def runtime_info(pin_output=None):
    import jax, jaxlib, flax, numpy

    cpu = {}
    try:
        first = pathlib.Path("/proc/cpuinfo").read_text().split("\n\n")[0]
        for line in first.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                if k.strip() in [
                    "vendor_id",
                    "cpu family",
                    "model",
                    "stepping",
                    "model name",
                    "microcode",
                    "flags",
                ]:
                    cpu[k.strip()] = v.strip()
    except OSError:
        cpu = {"unavailable": True}
    pins = {name: package_sources(name) for name in ["jax", "jaxlib", "flax", "numpy"]}
    binaries = {}
    for package in [jaxlib, numpy]:
        root = pathlib.Path(package.__file__).parent
        for path in root.rglob("*.so"):
            binaries[package.__name__ + "/" + str(path.relative_to(root))] = digest(
                path
            )
        libs = root.parent / (package.__name__ + ".libs")
        if libs.is_dir():
            for path in libs.iterdir():
                if path.is_file():
                    binaries[libs.name + "/" + path.name] = digest(path)
    pin_sha = hashlib.sha256(
        json.dumps(
            {"python_sources": pins, "binaries": binaries},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    result = {
        "python": platform.python_version(),
        "python_executable_sha256": digest(pathlib.Path(sys.executable).resolve()),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "flax": flax.__version__,
        "numpy": numpy.__version__,
        "backend": str(jax.default_backend()),
        "hardware": {
            "machine": platform.machine(),
            "cpu": cpu,
            "kernel": platform.release(),
        },
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "JAX_PLATFORMS",
                "XLA_FLAGS",
            ]
        },
        "precision": {
            "matmul": str(jax.config.jax_default_matmul_precision),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "runtime_source_and_binary_sha256": pin_sha,
    }
    if pin_output:
        pathlib.Path(pin_output).write_text(
            json.dumps(
                {"identity": result, "python_sources": pins, "binaries": binaries},
                indent=2,
                sort_keys=True,
            )
        )
    return result


if __name__ == "__main__":
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    print(
        json.dumps(runtime_info(sys.argv[1] if len(sys.argv) > 1 else None), indent=2)
    )
