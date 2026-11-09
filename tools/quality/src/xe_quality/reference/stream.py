"""Layer-streamed official Google Gemma4 text reference over exact GGUF values.

Attention, norms, residuals, dense MLP, RoPE and routing probabilities execute
Google's pinned operators. MoE expert products are dispatched in bounded pieces
instead of CPU ragged_dot; the numerical qualification checks this adapter.
"""

from pathlib import Path
import importlib.util, sys, json, time, gc, hashlib, resource
import numpy as np
from .upstream import load
from .source_pin import verify_google_source


def code_identity(model, source, precision="fp64", import_module=None):
    """Fingerprint the selected weights, pinned source and runtime without loading weights."""
    from .source_pin import verify_google_source

    source = Path(source).expanduser().resolve()
    verify_google_source(source)
    if precision not in ("fp32", "fp64"):
        raise ValueError("Explicit precision must be fp32 or fp64")
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", precision == "fp64")
    from .runtime_info import runtime_info

    model_path = Path(model).expanduser().resolve()
    with model_path.open("rb") as stream:
        weight_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    importer_path = Path(
        import_module or Path(__file__).resolve().parents[1] / "gguf_import.py"
    ).resolve()
    reference_path = Path(__file__).resolve()

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    pin = verify_google_source(source)
    return {
        "weights_sha256": weight_sha,
        "reference_code_sha256": digest(reference_path),
        "importer_sha256": digest(importer_path),
        "upstream_adapter_sha256": digest(reference_path.with_name("upstream.py")),
        "runtime_info_code_sha256": digest(reference_path.with_name("runtime_info.py")),
        "numerics_sha256": digest(reference_path.with_name("NUMERICS.json")),
        "google_pin_sha256": digest(reference_path.with_name("GOOGLE-PIN.json")),
        "effective_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "precise_rope_sha256": digest(reference_path.with_name("precise_rope.py"))
        if precision == "fp64"
        else None,
        "google_sources": pin,
        "precision": precision
        + " parameters/activations/KV; exact top-k"
        + (
            "; FP64variant lifts RoPE/router implementation casts"
            if precision == "fp64"
            else ""
        ),
        "runtime": runtime_info(),
    }


def importer(path):
    spec = importlib.util.spec_from_file_location("quality_gguf", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


NORMS = {
    "attn_norm": "pre_attention_norm",
    "post_attention_norm": "post_attention_norm",
    "ffn_norm": "pre_ffw2_norm",
    "post_ffw_norm": "post_ffw_norm",
    "post_ffw_norm_1": "post_ffw2_norm",
    "post_ffw_norm_2": "post_ffw1_norm",
    "pre_ffw_norm_2": "pre_ffw_norm",
}


def expected_tensors():
    expected = {
        "token_embd.weight": ((262144, 2816), 2),
        "output_norm.weight": ((2816,), 0),
        "rope_freqs.weight": ((256,), 0),
    }
    for i in range(30):
        glob = i % 6 == 5
        d = 512 if glob else 256
        k = 2 if glob else 8
        shapes = {n + ".weight": ((2816,), 0) for n in NORMS}
        shapes.update(
            {
                "layer_output_scale.weight": ((1,), 0),
                "attn_q.weight": ((16 * d, 2816), 2),
                "attn_k.weight": ((k * d, 2816), 2),
                "attn_output.weight": ((2816, 16 * d), 2),
                "attn_q_norm.weight": ((d,), 0),
                "attn_k_norm.weight": ((d,), 0),
                "ffn_gate.weight": ((2112, 2816), 2),
                "ffn_up.weight": ((2112, 2816), 2),
                "ffn_down.weight": ((2816, 2112), 2),
                "ffn_gate_inp.weight": ((128, 2816), 0),
                "ffn_gate_inp.scale": ((2816,), 0),
                "ffn_gate_up_exps.weight": ((128, 1408, 2816), 2),
                "ffn_down_exps.weight": ((128, 2816, 704), 2),
                "ffn_down_exps.scale": ((128,), 0),
            }
        )
        if not glob:
            shapes["attn_v.weight"] = ((k * d, 2816), 2)
        expected.update({"blk." + str(i) + "." + n: v for n, v in shapes.items()})
    return expected


def audit(reader):
    expected = expected_tensors()
    actual = reader.tensors
    required = {
        "general.architecture": "gemma4",
        "gemma4.block_count": 30,
        "gemma4.embedding_length": 2816,
        "gemma4.feed_forward_length": 2112,
        "gemma4.attention.head_count": 16,
        "gemma4.expert_count": 128,
        "gemma4.expert_used_count": 8,
        "gemma4.expert_feed_forward_length": 704,
        "gemma4.attention.sliding_window": 1024,
        "gemma4.attention.shared_kv_layers": 0,
        "gemma4.embedding_length_per_layer_input": 0,
        "gemma4.final_logit_softcapping": 30.0,
        "gemma4.rope.freq_base": 1000000.0,
        "gemma4.rope.freq_base_swa": 10000.0,
    }
    for key, value in required.items():
        if reader.metadata.get(key) != value:
            raise ValueError("Architecture metadata mismatch " + key)
    epsilon = reader.metadata.get("gemma4.attention.layer_norm_rms_epsilon")
    if epsilon != float(np.float32(1e-6)):
        raise ValueError("RMS epsilon mismatch")
    if set(actual) != set(expected):
        raise ValueError(
            "Unmapped/missing tensors: "
            + str(
                {
                    "missing": sorted(set(expected) - set(actual)),
                    "unexpected": sorted(set(actual) - set(expected)),
                }
            )
        )
    for name, (shape, typ) in expected.items():
        t = actual[name]
        if t.shape != shape or t.ggml_type != typ:
            raise ValueError("Tensor contract mismatch " + name + ": " + str(t))
    rope = reader.read("rope_freqs.weight")
    # Conversion encodes unrotated global half-pairs with 1e30 factors.
    if not (np.all(rope[:64] == 1) and np.all(rope[64:] > 1e29)):
        raise ValueError("Unrecognized global RoPE factors")
    return {
        "tensor_count": len(expected),
        "expanded_parameter_bytes": sum(
            int(np.prod(t.shape)) * 4 for t in actual.values()
        ),
        "alias": "embedding=head; global K/V same raw projection then distinct key/value normalization",
        "rope": "Google infinite timescale; GGUF nonrotated factor >1e29; finite perturbation bounded separately",
    }


class Reference:
    def __init__(self, model, source, import_module=None, precision="fp64"):
        verify_google_source(source)
        import jax, jax.numpy as jnp

        jax.config.update("jax_default_matmul_precision", "highest")
        if precision not in ["fp32", "fp64"]:
            raise ValueError("Explicit precision must be fp32 or fp64")
        if precision == "fp64":
            jax.config.update("jax_enable_x64", True)
        self.source = Path(source).expanduser().resolve()
        self.precision = precision
        self.dtype = jnp.float64 if precision == "fp64" else jnp.float32
        self.np_dtype = np.float64 if precision == "fp64" else np.float32
        self.jax, self.jnp = jax, jnp
        self.import_module = Path(
            import_module or Path(__file__).resolve().parents[1] / "gguf_import.py"
        )
        self.reader = importer(self.import_module).GGUF(model)
        self.audit = audit(self.reader)
        self.modules, self.moe, self.layers, self.patches = load(source)
        if precision == "fp64":
            from .precise_rope import apply_rope
            from gemma.gm.math import _positional_embeddings

            _positional_embeddings.apply_rope = apply_rope

        # top_k documents exact ties: lower source index first.
        def exact_router(module, logits):
            logits = logits.astype(self.dtype)
            probs = jax.nn.softmax(logits, axis=-1)
            _, choices = jax.lax.top_k(logits, module.num_experts_per_datapoint)
            return probs / self.moe._renormalization_factor(probs, choices), choices

        self.moe.MoERagged._router = exact_router
        self.active_layer = None
        self.route_trace = []
        outer = self

        def streamed_experts(module, x, choices, weights):
            counts, sorted_x, reverse, combine = outer.moe._expert_dispatch(
                x, choices, weights
            )
            counts = np.asarray(counts)
            out = []
            offset = 0
            trace = {
                "layer": outer.active_layer,
                "choices": np.asarray(choices).tolist(),
            }
            outer.route_trace.append(trace)
            prefix = f"blk.{outer.active_layer}."
            for e, count in enumerate(counts):
                n = int(count)
                if not n:
                    continue
                xx = sorted_x[offset : offset + n]
                offset += n
                gate = outer.reader.read(prefix + "ffn_gate_up_exps.weight", e)[
                    0
                ].reshape(2, 704, 2816)
                down = outer.reader.read(prefix + "ffn_down_exps.weight", e)[0].T
                g = jnp.einsum(
                    "bf,khf->bkh",
                    xx,
                    jnp.asarray(gate, dtype=outer.dtype),
                    precision=jax.lax.Precision.HIGHEST,
                )
                a = jax.nn.gelu(g[:, 0, :], approximate=True) * g[:, 1, :]
                y = jnp.einsum(
                    "bh,hf->bf",
                    a,
                    jnp.asarray(down, dtype=outer.dtype),
                    precision=jax.lax.Precision.HIGHEST,
                )
                y = y * module.per_expert_scale[e]
                out.append(np.asarray(y))
                del gate, down, g, a, y
            expert_outputs = jnp.asarray(np.concatenate(out, axis=0))
            gathered = outer.moe._expert_collect(expert_outputs, reverse, combine)
            return jnp.einsum(
                "blkd,blk->bld",
                gathered,
                combine,
                preferred_element_type=outer.dtype,
                precision=jax.lax.Precision.HIGHEST,
            )

        self.moe.MoERagged._run_ffw_and_routing = streamed_experts
        self.caches = {}
        self.position = 0
        self.history = []
        self.timings = []
        self._identity_cache = code_identity(
            self.reader.path, self.source, self.precision, self.import_module
        )

    def params(self, i):
        jnp = self.jnp
        r = self.reader
        pre = f"blk.{i}."
        glob = i % 6 == 5
        d = 512 if glob else 256
        k = 2 if glob else 8
        get = lambda n: r.read(pre + n).astype(self.np_dtype, copy=False)
        p = {
            dest: {"scale": jnp.asarray(get(src + ".weight"))}
            for src, dest in NORMS.items()
        }
        p["skip_scale"] = jnp.asarray(get("layer_output_scale.weight"))
        at = {
            "q_einsum": {
                "w": jnp.asarray(
                    get("attn_q.weight").reshape(16, d, 2816).transpose(0, 2, 1)
                )
            },
            "attn_vec_einsum": {
                "w": jnp.asarray(
                    get("attn_output.weight").reshape(2816, 16, d).transpose(1, 2, 0)
                )
            },
            "query_norm": {"scale": jnp.asarray(get("attn_q_norm.weight"))},
            "key_norm": {"scale": jnp.asarray(get("attn_k_norm.weight"))},
        }
        key = get("attn_k.weight").reshape(k, d, 2816).transpose(0, 2, 1)
        if glob:
            at["k_einsum"] = {"w": jnp.asarray(key)}
        else:
            at["kv_einsum"] = {
                "w": jnp.asarray(
                    np.stack(
                        [
                            key,
                            get("attn_v.weight").reshape(k, d, 2816).transpose(0, 2, 1),
                        ]
                    )
                )
            }
        p["attn"] = at
        p["mlp2"] = {
            "gating_einsum": jnp.asarray(
                np.stack([get("ffn_gate.weight"), get("ffn_up.weight")])
            ),
            "linear": jnp.asarray(get("ffn_down.weight").T),
        }
        p["mlp"] = {
            "router_logits": {"w": jnp.asarray(get("ffn_gate_inp.weight").T)},
            "router_scale": jnp.asarray(get("ffn_gate_inp.scale")),
            "per_expert_scale": jnp.asarray(get("ffn_down_exps.scale")),
        }
        return p

    def reset(self):
        self.caches = {}
        self.position = 0
        self.history = []

    def close(self):
        self.reader.close()

    def forward(
        self,
        tokens,
        positions=None,
        cache=False,
        capacity=None,
        max_layers=30,
        trace_dir=None,
    ):
        jnp = self.jnp
        jax = self.jax
        original = np.asarray(tokens)
        if (
            original.ndim != 1
            or not np.issubdtype(original.dtype, np.integer)
            or np.issubdtype(original.dtype, np.bool_)
            or (original < 0).any()
            or (original >= 262144).any()
        ):
            raise ValueError("Exact integer token IDs required")
        tokens = original.astype(np.int32)
        if (
            tokens.ndim != 1
            or not len(tokens)
            or (tokens < 0).any()
            or (tokens >= 262144).any()
        ):
            raise ValueError("Invalid exact token history")
        if positions is None:
            positions = list(range(len(tokens)))
        if any(p < 0 or p >= len(tokens) for p in positions):
            raise ValueError("Invalid selected positions")
        if cache:
            if max_layers != 30:
                raise ValueError("Cached partial-layer forwards are forbidden")
            self._validate_cache(capacity)
        begin = self.position if cache else 0
        pos = jnp.arange(begin, begin + len(tokens), dtype=jnp.int32)[None, :]
        x = jnp.asarray(
            np.stack(
                [self.reader.read("token_embd.weight", int(t))[0] for t in tokens]
            ),
            dtype=self.dtype,
        )[None, :, :] * jnp.sqrt(jnp.asarray(2816, dtype=self.dtype))
        for i in range(max_layers):
            t = time.monotonic()
            glob = i % 6 == 5
            d = 512 if glob else 256
            k = 2 if glob else 8
            module = self.modules.Block(
                num_heads=16,
                num_kv_heads=8,
                embed_dim=2816,
                head_dim=256,
                hidden_dim=2112,
                use_post_attn_norm=True,
                use_post_ffw_norm=True,
                attn_type=self.modules.AttentionType.GLOBAL
                if glob
                else self.modules.AttentionType.LOCAL_SLIDING,
                rope_base_frequency=1000000 if glob else 10000,
                sliding_window_size=1024,
                qk_norm_with_scale=True,
                num_global_kv_heads=2,
                global_key_size=512,
                k_eq_v_global=True,
                global_rope_proportion=0.25,
                local_rope_proportion=1.0,
                enable_moe=True,
                num_experts=128,
                expert_dim=704,
                top_k_experts=8,
            )
            params = self.params(i)
            load = time.monotonic() - t
            if cache:
                if capacity is None or begin + len(tokens) > capacity:
                    raise ValueError("Explicit cache capacity exceeded")
                kv = self.caches.get(i)
                if kv is None:
                    kv = self.modules.Attention.init_cache(
                        capacity, k, d, 1, self.dtype
                    )
                mask = jnp.arange(capacity)[None, None, :] <= pos[:, :, None]
            else:
                kv = None
                mask = jnp.arange(len(tokens))[None, None, :] <= pos[:, :, None]
            self.active_layer = i
            kv, x = module.apply({"params": params}, x, pos, kv, mask)
            x.block_until_ready()
            if not np.isfinite(np.asarray(x)).all():
                raise ValueError("Nonfinite reference hidden state")
            if cache:
                self.caches[i] = kv
            if trace_dir:
                Path(trace_dir).mkdir(parents=True, exist_ok=True)
                np.save(Path(trace_dir) / f"layer-{i:02d}.npy", np.asarray(x))
            self.timings.append(
                {
                    "layer": i,
                    "tokens": len(tokens),
                    "load_seconds": load,
                    "total_seconds": time.monotonic() - t,
                    "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
            print(json.dumps({"event": "layer", **self.timings[-1]}), flush=True)
            del params, kv
            gc.collect()
        if cache:
            self.position += len(tokens)
            self.history.extend(tokens.tolist())
        if max_layers != 30:
            return np.asarray(x)
        if not positions:
            return np.empty((0, 262144), self.np_dtype)
        x = self.layers.RMSNorm().apply(
            {
                "params": {
                    "scale": jnp.asarray(
                        self.reader.read("output_norm.weight"), dtype=self.dtype
                    )
                }
            },
            x,
        )
        selected = np.asarray(x)[0, positions, :]
        logits = np.empty((len(positions), 262144), self.np_dtype)
        for lo in range(0, 262144, 4096):
            emb = self.reader.read("token_embd.weight", slice(lo, lo + 4096))
            logits[:, lo : lo + len(emb)] = np.asarray(
                jnp.einsum(
                    "pf,vf->pv",
                    jnp.asarray(selected),
                    jnp.asarray(emb, dtype=self.dtype),
                    precision=jax.lax.Precision.HIGHEST,
                )
            )
        logits = np.asarray(
            30 * jnp.tanh(jnp.asarray(logits) / 30), dtype=self.np_dtype
        )
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite reference logits")
        return logits

    def _validate_cache(self, capacity=None):
        if len(self.history) != self.position:
            raise ValueError("Cache history/index mismatch; reset required")
        if not self.position:
            if self.caches:
                raise ValueError("Unexpected cache at reset position")
            return
        if set(self.caches) != set(range(30)):
            raise ValueError("Incomplete per-layer cache; reset required")
        for i, c in self.caches.items():
            glob = i % 6 == 5
            k, d = (2, 512) if glob else (8, 256)
            shape = c["k"].shape
            if (
                len(shape) != 4
                or shape[0] != 1
                or shape[2:] != (k, d)
                or c["v"].shape != shape
                or (capacity is not None and shape[1] != capacity)
            ):
                raise ValueError("Per-layer cache capacity/layout mismatch")
            if (
                str(c["k"].dtype) != str(np.dtype(self.np_dtype))
                or str(c["v"].dtype) != str(np.dtype(self.np_dtype))
                or int(np.asarray(c["end_index"])[0]) != self.position
            ):
                raise ValueError("Per-layer cache precision/index mismatch")
            if c["positions"].shape != (1, shape[1]) or not np.array_equal(
                np.asarray(c["positions"])[0, : self.position], np.arange(self.position)
            ):
                raise ValueError("Per-layer cache positions mismatch")

    def _state_identity(self):
        return self._identity_cache

    def code_identity(self, *, refresh=False):
        if refresh:
            self._identity_cache = code_identity(
                self.reader.path, self.source, self.precision, self.import_module
            )
        return self._identity_cache

    def save_state(self, path):
        import os, tempfile

        path = Path(path)
        meta = Path(str(path) + ".json")
        marker = Path(str(path) + ".incomplete")
        if any(p.exists() for p in [path, meta, marker]):
            raise ValueError("State export refuses overwrite")
        if set(self.caches) != set(range(30)) or len(self.history) != self.position:
            raise ValueError("Cannot export partial model state")
        path.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x") as f:
            f.write("incomplete")
        self._validate_cache()
        arrays = {
            f"layer_{i:02d}_{key}": np.asarray(value)
            for i, cache in self.caches.items()
            for key, value in cache.items()
        }
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Cannot export nonfinite cache")
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp-")
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        document = {
            "schema": "gemma4-reference-state-v1",
            "identity": self._state_identity(),
            "history": self.history,
            "position": self.position,
            "data_sha256": digest,
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=meta.name + ".tmp-")
        with os.fdopen(fd, "w") as f:
            json.dump(document, f, sort_keys=True, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, meta)
        marker.unlink()
        return document

    def load_state(self, path, expected_history):
        path = Path(path)
        if Path(str(path) + ".incomplete").exists():
            raise ValueError("Incomplete state")
        document = json.loads(Path(str(path) + ".json").read_text())
        if (
            document.get("schema") != "gemma4-reference-state-v1"
            or document.get("identity") != self._state_identity()
        ):
            raise ValueError("Reference state identity mismatch")
        expected = np.asarray(expected_history)
        if (
            expected.ndim != 1
            or not np.issubdtype(expected.dtype, np.integer)
            or (expected < 0).any()
            or (expected >= 262144).any()
        ):
            raise ValueError("Invalid expected state history")
        if document.get("history") != expected.tolist() or document.get(
            "position"
        ) != len(expected):
            raise ValueError("Reference state history mismatch")
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != document["data_sha256"]:
                raise ValueError("Reference state data hash mismatch")
        caches = {}
        with np.load(path, allow_pickle=False) as arrays:
            names = {
                f"layer_{i:02d}_{key}"
                for i in range(30)
                for key in ["k", "v", "positions", "end_index"]
            }
            if set(arrays.files) != names:
                raise ValueError("Reference state missing/unexpected arrays")
            capacity = None
            for i in range(30):
                glob = i % 6 == 5
                k, d = (2, 512) if glob else (8, 256)
                cache = {}
                for key in ["k", "v", "positions", "end_index"]:
                    a = arrays[f"layer_{i:02d}_{key}"]
                    if key == "k":
                        if a.ndim != 4 or a.shape[0] != 1 or a.shape[2:] != (k, d):
                            raise ValueError("Reference KV layout mismatch")
                        if capacity is None:
                            capacity = a.shape[1]
                    shape = (
                        (1, capacity, k, d)
                        if key in ["k", "v"]
                        else (1, capacity)
                        if key == "positions"
                        else (1,)
                    )
                    dtype = (
                        np.dtype(self.np_dtype)
                        if key in ["k", "v"]
                        else np.dtype("int32")
                    )
                    if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
                        raise ValueError(
                            "Reference state shape/dtype/nonfinite mismatch"
                        )
                    cache[key] = self.jnp.asarray(a)
                if int(np.asarray(cache["end_index"])[0]) != len(
                    expected
                ) or capacity < len(expected):
                    raise ValueError("Reference state cache index mismatch")
                if not np.array_equal(
                    np.asarray(cache["positions"])[0, : len(expected)],
                    np.arange(len(expected)),
                ):
                    raise ValueError("Reference state cache positions mismatch")
                caches[i] = cache
        self.caches = caches
        self.position = len(expected)
        self.history = expected.tolist()
        return document
