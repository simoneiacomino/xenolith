"""Load audited Google operators with import-only observer/type dependencies removed.

No arithmetic changes here. Exact stable top-k is installed separately by runner.
"""

from pathlib import Path
import sys, types, importlib.util, hashlib, difflib, json


def load(source):
    from .source_pin import verify_google_source

    source = Path(source)
    verify_google_source(source)
    packages = {
        "gemma": source / "gemma",
        "gemma.gm": source / "gemma/gm",
        "gemma.gm.nn": source / "gemma/gm/nn",
        "gemma.gm.nn.gemma4": source / "gemma/gm/nn/gemma4",
        "gemma.gm.math": source / "gemma/gm/math",
    }
    for name, path in packages.items():
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m
    import flax.linen as nn

    class Identity(nn.Module):
        def __call__(self, x):
            return x

    patches = {}
    for name, relative in [
        ("gemma.gm.nn.gemma4._layers", "gemma/gm/nn/gemma4/_layers.py"),
        (
            "gemma.gm.math._positional_embeddings",
            "gemma/gm/math/_positional_embeddings.py",
        ),
        ("gemma.gm.nn.gemma4._modules", "gemma/gm/nn/gemma4/_modules.py"),
        ("gemma.gm.nn.gemma4._moe", "gemma/gm/nn/gemma4/_moe.py"),
    ]:
        raw = (source / relative).read_text()
        code = raw
        if name.endswith("_modules"):
            code = "from __future__ import annotations\n" + code
            code = code.replace(
                "from kauldron import kd",
                "import types\nkd = types.SimpleNamespace(nn=types.SimpleNamespace(Identity=__identity__))",
            )
            code = code.replace(
                "from kauldron.ktyping import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member",
                "def typechecked(fn): return fn",
            )
        patches[relative] = {
            "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "patch": "".join(
                difflib.unified_diff(
                    raw.splitlines(True),
                    code.splitlines(True),
                    fromfile=relative,
                    tofile=relative,
                )
            ),
        }
        mod = types.ModuleType(name)
        mod.__file__ = str(source / relative)
        mod.__dict__["__identity__"] = Identity
        sys.modules[name] = mod
        exec(compile(code, str(source / relative), "exec"), mod.__dict__)
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, mod)
    return (
        sys.modules["gemma.gm.nn.gemma4._modules"],
        sys.modules["gemma.gm.nn.gemma4._moe"],
        sys.modules["gemma.gm.nn.gemma4._layers"],
        patches,
    )
