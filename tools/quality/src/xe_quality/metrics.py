"""Full-vocabulary logit metrics and group-aware summaries.

``compare_logits`` requires the expected vocabulary size: equal but truncated
arrays cannot otherwise be distinguished from a complete vocabulary. Every
position is evaluated in FP64 log space; inputs are expected to be untempered,
pre-sampling logits for identical histories.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable

import numpy as np


class MetricError(ValueError):
    """Invalid or incomparable metric input."""


def _as_logits(value: Any, label: str, expected_vocab_size: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim < 2 or arr.shape[-1] != expected_vocab_size:
        raise MetricError(
            f"{label} must have a full final vocabulary axis of {expected_vocab_size}"
        )
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise MetricError(f"{label} must be real numeric logits")
    arr = arr.astype(np.float64, copy=False)
    if not np.isfinite(arr).all():
        raise MetricError(f"{label} contains NaN or infinity")
    return arr


def _logsoftmax(x: np.ndarray) -> np.ndarray:
    # Subtracting the per-position maximum avoids overflow and preserves the
    # requested full support; never clip, temperature-scale or truncate logits.
    m = np.max(x, axis=-1, keepdims=True)
    shifted = x - m
    lse = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True, dtype=np.float64))
    return shifted - lse


def compare_logits(
    reference: Any,
    candidate: Any,
    *,
    expected_vocab_size: int,
    observed_tokens: Any | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-position KL(B||C), TV and diagnostics over the full vocab.

    ``observed_tokens`` must have the leading shape of the logits; omitted
    observations yield no NLL fields. Positive ``delta_nll`` means candidate C
    assigned less probability to the observed token.
    """
    if type(expected_vocab_size) is not int or expected_vocab_size < 2:
        raise MetricError("expected_vocab_size must be an integer >= 2")
    b = _as_logits(reference, "reference", expected_vocab_size)
    c = _as_logits(candidate, "candidate", expected_vocab_size)
    if b.shape != c.shape:
        raise MetricError(f"logit shape mismatch: {b.shape} != {c.shape}")
    lb, lc = _logsoftmax(b), _logsoftmax(c)
    p, q = np.exp(lb), np.exp(lc)
    diff = lb - lc
    kl_terms = np.zeros_like(p)
    np.multiply(p, diff, out=kl_terms, where=p != 0)
    kl = np.sum(kl_terms, axis=-1, dtype=np.float64)
    tv = 0.5 * np.sum(np.abs(p - q), axis=-1, dtype=np.float64)
    b_top = np.argmax(b, axis=-1)
    c_top = np.argmax(c, axis=-1)
    b2 = np.sort(np.partition(b, -2, axis=-1)[..., -2:], axis=-1)
    c2 = np.sort(np.partition(c, -2, axis=-1)[..., -2:], axis=-1)
    result = {
        "kl_b_to_c_nats": kl,
        "kl_negative_observed": kl < 0.0,
        "tv": tv,
        "reference_top1": b_top,
        "candidate_top1": c_top,
        "reference_top1_margin_logits": b2[..., 1] - b2[..., 0],
        "candidate_top1_margin_logits": c2[..., 1] - c2[..., 0],
    }
    if observed_tokens is not None:
        y = np.asarray(observed_tokens)
        if y.shape != b.shape[:-1] or not np.issubdtype(y.dtype, np.integer):
            raise MetricError(
                "observed_tokens must be integer IDs with the logits leading shape"
            )
        if np.any(y < 0) or np.any(y >= expected_vocab_size):
            raise MetricError("observed token ID is outside the vocabulary")
        nll_b = -np.take_along_axis(lb, y[..., None], axis=-1)[..., 0]
        nll_c = -np.take_along_axis(lc, y[..., None], axis=-1)[..., 0]
        result["reference_nll_nats"] = nll_b
        result["candidate_nll_nats"] = nll_c
        result["delta_nll_nats"] = nll_c - nll_b
    return result


def group_summary(
    values: Any,
    group_ids: Any,
    *,
    domains: Any | None = None,
    quantile_min_groups: int = 10,
) -> dict[str, Any]:
    """Summarize position metrics within groups, then groups within domains.

    The domain mean gives each document/session group equal weight. Position
    counts are reported separately; tokens are never treated as independent
    bootstrap units. Quantiles are omitted for small group counts.
    """
    x = np.asarray(values, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=object)
    if x.shape != groups.shape or x.ndim != 1:
        raise MetricError("values and group_ids must be matching 1D arrays")
    if x.size == 0:
        raise MetricError("cannot summarize an empty sample")
    if not np.isfinite(x).all():
        raise MetricError("values contain NaN or infinity")
    if domains is None:
        domain_values = np.full(groups.shape, "all", dtype=object)
    else:
        domain_values = np.asarray(domains, dtype=object)
        if domain_values.shape != groups.shape:
            raise MetricError("domains must match values shape")
    grouped: dict[tuple[Hashable, Hashable], list[float]] = defaultdict(list)
    for value, group, domain in zip(x, groups, domain_values):
        grouped[(domain, group)].append(float(value))
    per_domain: dict[Hashable, list[tuple[Hashable, list[float]]]] = defaultdict(list)
    for (domain, group), observations in grouped.items():
        per_domain[domain].append((group, observations))
    result = {}
    for domain, entries in per_domain.items():
        group_means = np.asarray([np.mean(obs, dtype=np.float64) for _, obs in entries])
        item: dict[str, Any] = {
            "group_count": len(entries),
            "position_count": sum(len(obs) for _, obs in entries),
            "mean_group_mean": float(np.mean(group_means, dtype=np.float64)),
            "max_observed_position_value": float(max(max(obs) for _, obs in entries)),
            "groups": {
                str(group): {"position_count": len(obs), "mean": float(np.mean(obs))}
                for group, obs in entries
            },
        }
        if len(entries) >= quantile_min_groups:
            item["group_mean_quantiles"] = {
                "p50": float(np.quantile(group_means, 0.50)),
                "p90": float(np.quantile(group_means, 0.90)),
            }
        result[str(domain)] = item
    return {
        "domains": result,
        "grouping_unit": "document_or_session",
        "independent_unit": "group",
    }


def paired_group_bootstrap(
    reference: Any,
    candidate: Any,
    group_ids: Any,
    *,
    seed: int = 0,
    resamples: int = 10000,
    confidence: float = 0.95,
    min_groups: int = 10,
) -> dict[str, Any]:
    """Percentile CI for mean(candidate-reference), resampling paired groups.

    Each sampled cluster is a whole document/session. The returned interval is
    descriptive for the supplied groups; opportunistic groups are not made
    representative by bootstrapping.
    """
    a, b = (
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    groups = np.asarray(group_ids, dtype=object)
    if a.ndim != 1 or a.shape != b.shape or a.shape != groups.shape:
        raise MetricError(
            "reference, candidate, and group_ids must be matching 1D arrays"
        )
    if a.size == 0:
        raise MetricError("cannot bootstrap an empty sample")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise MetricError("bootstrap values must be finite")
    if resamples < 1 or min_groups < 1 or not 0 < confidence < 1:
        raise MetricError("invalid bootstrap settings")
    group_order = list(dict.fromkeys(groups.tolist()))
    group_delta = np.asarray(
        [np.mean((b - a)[groups == group], dtype=np.float64) for group in group_order]
    )
    n = len(group_delta)
    point = float(np.mean(group_delta, dtype=np.float64)) if n else float("nan")
    result = {
        "estimate_candidate_minus_reference": point,
        "group_count": n,
        "position_count": len(a),
        "resamples": resamples,
        "seed": seed,
        "confidence": confidence,
        "group_deltas": group_delta.tolist(),
    }
    if n < min_groups:
        result.update(
            {
                "status": "insufficient_groups",
                "warning": f"need >= {min_groups} groups for requested bootstrap",
            }
        )
        return result
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(resamples, n))
    estimates = np.mean(group_delta[draws], axis=1, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    result.update(
        {
            "status": "available",
            "interval": [
                float(np.quantile(estimates, alpha)),
                float(np.quantile(estimates, 1 - alpha)),
            ],
        }
    )
    return result
