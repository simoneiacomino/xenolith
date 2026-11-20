"""Explicit true-FP64 cast lifting for independent HF qualification only.

Same architectural formulas and learned values. Natural routing with canonical
stable exact score top-k; no fixed expert selections and no GELU substitution.
Use one precision per process. Original installed source bytes remain untouched.
"""

import torch
from transformers.models.gemma4 import modeling_gemma4 as hf


def enable():
    counts = {"norm": 0, "rope": 0, "router": 0, "attention": 0}

    def norm(self, x):
        counts["norm"] += 1
        if x.dtype != torch.float64:
            raise ValueError("Precise HF norm received non-FP64")
        out = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return out * self.weight if self.with_scale else out

    def rope(self, x, position_ids, layer_type):
        counts["rope"] += 1
        if x.dtype != torch.float64:
            raise ValueError("Precise HF RoPE received non-FP64")
        config = self.config.per_layer_config[layer_type]
        d = config.head_dim
        params = self.config.rope_parameters[layer_type]
        base = params["rope_theta"]
        prop = params.get("partial_rotary_factor", 1.0)
        n = int(d * prop // 2)
        exponents = torch.arange(d // 2, dtype=torch.float64, device=x.device) * 2 / d
        inv = (
            torch.as_tensor(base, dtype=torch.float64, device=x.device) ** exponents
        ).reciprocal()
        inv[n:] = 0.0
        angles = position_ids[..., None].to(torch.float64) * inv
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()

    def router(self, x):
        counts["router"] += 1
        if x.dtype != torch.float64:
            raise ValueError("Precise HF router received non-FP64")
        normalized = self.norm(x) * self.scale * self.scalar_root_size
        scores = self.proj(normalized)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float64)
        choices = torch.argsort(scores, dim=-1, descending=True, stable=True)[
            ..., : self.config.top_k_experts
        ]
        weights = torch.gather(probs, -1, choices)
        weights = weights / weights.sum(-1, keepdim=True)
        weights = weights * self.per_expert_scale[choices]
        return probs, weights, choices

    def attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=None,
        softcap=None,
        **kwargs,
    ):
        counts["attention"] += 1
        if query.dtype != torch.float64 or dropout != 0.0:
            raise ValueError("Precise HF attention dtype/dropout mismatch")
        scale = module.head_dim**-0.5 if scaling is None else scaling
        k = hf.repeat_kv(key, module.num_key_value_groups)
        v = hf.repeat_kv(value, module.num_key_value_groups)
        score = torch.matmul(query, k.transpose(2, 3)) * scale
        if softcap is not None:
            score = torch.tanh(score / softcap) * softcap
        if attention_mask is not None:
            score = score + attention_mask
        probabilities = torch.softmax(score, dim=-1, dtype=torch.float64)
        return torch.matmul(probabilities, v).transpose(
            1, 2
        ).contiguous(), probabilities

    hf.Gemma4RMSNorm.forward = norm
    hf.Gemma4TextRotaryEmbedding.forward = rope
    hf.Gemma4TextRouter.forward = router
    hf.eager_attention_forward = attention
    return {
        "executed_calls": counts,
        "parameter_values": "Exact GGUF values decodedFP32 then promotedFP64",
        "RMS": "FP64 pow(-.5), removed x.float and weight.float",
        "RoPE": "Recomputed inverse frequencies/positions/cos/sinFP64, canonical partial dimensions",
        "attention": "FP64 QK/softmax/AV; same causal/GQA formulas",
        "router": "FP64 norm/scale/dot/softmax, natural stable exactscore top-k; no route locking",
        "experts": "FP64 tanh GELU/products/index_add, architectural per-expert scales",
        "head": "FP64 dot and softcap before any serialization cast",
    }
