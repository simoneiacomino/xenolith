"""Independent official HF CPU FP32 layer-streamed qualification control.

Complete expert bank for one layer only; no precision reduction. The GGUF
import is shared and separately checked against gguf-py. Architecture operations
and reduction ordering are independent from Google/JAX.
"""

import pathlib, sys, json, time, gc, resource
import numpy as np
import importlib.util


def run(
    model,
    config_path,
    import_path,
    tokens,
    positions,
    out_dir,
    max_layers=30,
    precision="fp32",
):
    from .hf_source_pin import verify_hf_source

    pins_before = verify_hf_source(config_path)
    import torch
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextDecoderLayer,
        Gemma4TextRotaryEmbedding,
        Gemma4RMSNorm,
        Gemma4TextRouter,
    )

    if precision not in ["fp32", "fp64"]:
        raise ValueError("Unknown explicit precision")
    dtype = torch.float64 if precision == "fp64" else torch.float32
    npdtype = np.float64 if precision == "fp64" else np.float32
    ledger = None
    if precision == "fp64":
        from .hf_precise import enable

        ledger = enable()
    route_trace = []
    original_router_forward = Gemma4TextRouter.forward

    def traced_router(self, hidden_states):
        result = original_router_forward(self, hidden_states)
        route_trace.append(
            {
                "layer": getattr(self, "_qualification_layer", None),
                "choices": result[-1].detach().cpu().tolist(),
            }
        )
        return result

    Gemma4TextRouter.forward = traced_router
    spec = importlib.util.spec_from_file_location("hf_gguf", import_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    r = m.GGUF(model)
    config = Gemma4TextConfig(**json.loads(pathlib.Path(config_path).read_text()))
    config._attn_implementation = "eager"
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.set_grad_enabled(False)
    x = (
        torch.from_numpy(
            np.stack([r.read("token_embd.weight", int(t))[0] for t in tokens]).astype(
                npdtype
            )
        )[None, :, :]
        * 2816**0.5
    )
    rotary = Gemma4TextRotaryEmbedding(config)
    pos = torch.arange(len(tokens))[None, :]
    mask = torch.zeros((1, 1, len(tokens), len(tokens)), dtype=dtype).masked_fill(
        ~torch.tril(torch.ones(len(tokens), len(tokens), dtype=torch.bool))[
            None, None, :, :
        ],
        torch.finfo(dtype).min,
    )
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = []
    for i in range(max_layers):
        t = time.monotonic()
        pre = f"blk.{i}."
        glob = i % 6 == 5
        get = lambda n: torch.from_numpy(r.read(pre + n).astype(npdtype, copy=False))
        with torch.device("meta"):
            layer = Gemma4TextDecoderLayer(config, i)
        sd = {}
        norms = {
            "attn_norm": "input_layernorm",
            "post_attention_norm": "post_attention_layernorm",
            "ffn_norm": "pre_feedforward_layernorm",
            "post_ffw_norm": "post_feedforward_layernorm",
            "post_ffw_norm_1": "post_feedforward_layernorm_1",
            "post_ffw_norm_2": "post_feedforward_layernorm_2",
            "pre_ffw_norm_2": "pre_feedforward_layernorm_2",
        }
        for src, dst in norms.items():
            sd[dst + ".weight"] = get(src + ".weight")
        for src, dst in [
            ("attn_q", "q_proj"),
            ("attn_k", "k_proj"),
            ("attn_output", "o_proj"),
            ("attn_q_norm", "q_norm"),
            ("attn_k_norm", "k_norm"),
        ]:
            sd["self_attn." + dst + ".weight"] = get(src + ".weight")
        if not glob:
            sd["self_attn.v_proj.weight"] = get("attn_v.weight")
        for src, dst in [
            ("ffn_gate", "gate_proj"),
            ("ffn_up", "up_proj"),
            ("ffn_down", "down_proj"),
        ]:
            sd["mlp." + dst + ".weight"] = get(src + ".weight")
        sd.update(
            {
                "layer_scalar": get("layer_output_scale.weight"),
                "router.proj.weight": get("ffn_gate_inp.weight"),
                "router.scale": get("ffn_gate_inp.scale"),
                "router.per_expert_scale": get("ffn_down_exps.scale"),
                "experts.gate_up_proj": get("ffn_gate_up_exps.weight"),
                "experts.down_proj": get("ffn_down_exps.weight"),
            }
        )
        layer.load_state_dict(sd, strict=True, assign=True)
        layer.eval()
        layer.router._qualification_layer = i
        typ = config.layer_types[i]
        emb = rotary(x, pos, typ)
        attn_mask = mask
        if not glob:
            local = (pos[:, :, None] - pos[:, None, :]) < 1024
            attn_mask = mask.masked_fill(~local[:, None, :, :], torch.finfo(dtype).min)
        x = layer(
            x,
            shared_kv_states={},
            position_embeddings=emb,
            attention_mask=attn_mask,
            position_ids=pos,
        )
        if x.dtype != dtype:
            raise ValueError("HF layer output precision mismatch")
        np.save(out_dir / f"layer-{i:02d}.npy", x.numpy())
        timings.append(
            {
                "layer": i,
                "seconds": time.monotonic() - t,
                "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        print(json.dumps({"event": "hf-layer", **timings[-1]}), flush=True)
        del layer, sd
        gc.collect()
    if max_layers == 30:
        norm = Gemma4RMSNorm(2816, eps=1e-6)
        norm.weight = torch.nn.Parameter(
            torch.from_numpy(r.read("output_norm.weight").astype(npdtype, copy=False)),
            requires_grad=False,
        )
        xx = norm(x)[0, positions, :]
        z = np.empty((len(positions), 262144), npdtype)
        for lo in range(0, 262144, 4096):
            z[:, lo : lo + 4096] = (
                xx
                @ torch.from_numpy(
                    r.read("token_embd.weight", slice(lo, lo + 4096)).astype(
                        npdtype, copy=False
                    )
                ).T
            ).numpy()
        z = (30 * torch.tanh(torch.from_numpy(z) / 30)).numpy()
        np.save(out_dir / "logits.npy", z)
    if ledger:
        (out_dir / "CAST-LEDGER.json").write_text(json.dumps(ledger, indent=2))
    pins_after = verify_hf_source(config_path)
    if pins_after != pins_before:
        raise ValueError("HF source changed during qualification control")
    (out_dir / "TIMINGS.json").write_text(json.dumps(timings, indent=2))
    (out_dir / "ROUTES.json").write_text(json.dumps(route_trace, indent=2))
    return x.numpy()


if __name__ == "__main__":
    model, config, imp, out = sys.argv[1:5]
    run(
        model,
        config,
        imp,
        [2, 9259, 573, 1707],
        [2, 3],
        out,
        int(sys.argv[5]) if len(sys.argv) > 5 else 30,
    )
