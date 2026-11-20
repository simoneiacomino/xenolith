"""Independent FP32 and true-FP64 operator controls; no model weights are loaded."""

from pathlib import Path
import numpy as np, json
from .upstream import load


def run(source, out):
    import jax, jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    modules, moe, layers, patches = load(source)
    rng = np.random.default_rng(41290)
    rows = []

    def check(name, actual, expected, tol=3e-5, element_bound=None):
        a = np.asarray(actual, dtype=np.float64)
        b = np.asarray(expected, dtype=np.float64)
        scale = max(1.0, float(np.sqrt(np.mean(b * b))))
        err = float(np.max(np.abs(a - b)))
        rel = float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(b)))
        passed = (
            err <= tol * scale and rel <= tol
            if element_bound is None
            else bool(np.all(np.abs(a - b) <= element_bound))
        )
        rows.append(
            dict(
                name=name,
                max_abs=err,
                relative_l2=rel,
                absolute_gate=tol * scale
                if element_bound is None
                else float(np.max(element_bound)),
                relative_gate=tol if element_bound is None else None,
                passed=passed,
            )
        )
        if not rows[-1]["passed"]:
            raise ValueError("Operator gate " + name + ": " + str(rows[-1]))

    x = rng.normal(size=(2, 5, 32)).astype(np.float32)
    scale = rng.uniform(0.1, 2, size=32).astype(np.float32)
    check(
        "rmsnorm_fp64",
        layers.RMSNorm().apply(
            {"params": {"scale": jnp.asarray(scale)}}, jnp.asarray(x)
        ),
        x.astype(float)
        / np.sqrt(np.mean(x.astype(float) ** 2, axis=-1, keepdims=True) + 1e-6)
        * scale,
    )
    # Tanh GELU is architecture, independent FP64 formula (not erf).
    check(
        "gelu_tanh_fp64",
        jax.nn.gelu(jnp.asarray(x), approximate=True),
        0.5
        * x.astype(float)
        * (
            1
            + np.tanh(
                np.sqrt(2 / np.pi) * (x.astype(float) + 0.044715 * x.astype(float) ** 3)
            )
        ),
    )
    from gemma.gm.math import _positional_embeddings as rope

    for prop, base in [(1.0, 10000), (0.25, 1000000)]:
        xx = rng.normal(size=(1, 4, 2, 32)).astype(np.float32)
        pos = np.array([[0, 1, 1023, 8191]], np.int32)
        r = int(prop * 16)
        freq = np.full(16, np.inf)
        freq[:r] = base ** (np.arange(r, dtype=float) * 2 / 32)
        ang = pos[:, :, None, None] / freq[None, None, None, :]
        a, b = np.split(xx.astype(float), 2, axis=-1)
        expected = np.concatenate(
            [a * np.cos(ang) - b * np.sin(ang), b * np.cos(ang) + a * np.sin(ang)],
            axis=-1,
        )
        # FP32 power/angle errors are magnified by absolute position. The Jacobian
        # of each rotated component is <=|a|+|b|. This pilot envelope reserves
        # 8 eps relative timescale/phase plus 8 eps arithmetic, per element.
        # It is an empirical operator resolution gate, not a global proof for pow.
        half_bound = (np.abs(a) + np.abs(b)) * (
            8 * np.finfo(np.float32).eps * (np.abs(ang) + 1)
        )
        bound = np.concatenate([half_bound, half_bound], axis=-1) + 2e-6
        check(
            "rope_" + str(prop),
            rope.apply_rope(
                jnp.asarray(xx),
                jnp.asarray(pos),
                base_frequency=base,
                rope_proportion=prop,
            ),
            expected,
            element_bound=bound,
        )
        if prop < 1:
            output = np.asarray(
                rope.apply_rope(
                    jnp.asarray(xx),
                    jnp.asarray(pos),
                    base_frequency=base,
                    rope_proportion=prop,
                )
            )
            assert np.array_equal(output[..., r:16], xx[..., r:16]) and np.array_equal(
                output[..., 16 + r :], xx[..., 16 + r :]
            )
    # Small ragged reference is upstream implementation, checked against independent
    # FP64 token/expert loops incl natural routing; same adapter layout as real.
    F, H, E, K = 32, 16, 6, 2
    xx = rng.normal(size=(1, 5, F)).astype(np.float32)
    gate = rng.normal(scale=0.2, size=(E, 2, H, F)).astype(np.float32)
    down = rng.normal(scale=0.2, size=(E, H, F)).astype(np.float32)
    router = rng.normal(scale=0.2, size=(F, E)).astype(np.float32)
    rs = rng.uniform(0.5, 1.5, F).astype(np.float32)
    es = rng.uniform(0.5, 1.5, E).astype(np.float32)
    params = {
        "router_logits": {"w": jnp.asarray(router)},
        "gating_einsum": {"w": jnp.asarray(gate)},
        "linear": {"w": jnp.asarray(down)},
        "router_scale": jnp.asarray(rs),
        "per_expert_scale": jnp.asarray(es),
    }
    actual = moe.MoERagged(F, H, E, K).apply({"params": params}, jnp.asarray(xx))
    rn = xx.astype(float) / np.sqrt(
        np.mean(xx.astype(float) ** 2, axis=-1, keepdims=True) + 1e-6
    )
    logits = (rn / np.sqrt(F) * rs) @ router.astype(float)
    chosen = np.argsort(-logits, axis=-1, kind="stable")[..., :K]
    expected = np.zeros_like(xx, dtype=float)
    for t in range(5):
        scores = logits[0, t, chosen[0, t]]
        p = np.exp(scores - scores.max())
        p /= p.sum()
        for e, w in zip(chosen[0, t], p):
            gu = np.einsum("f,khf->kh", xx[0, t].astype(float), gate[e].astype(float))
            a = (
                0.5
                * gu[0]
                * (1 + np.tanh(np.sqrt(2 / np.pi) * (gu[0] + 0.044715 * gu[0] ** 3)))
                * gu[1]
            )
            expected[0, t] += w * (a @ down[e].astype(float)) * es[e]
    check("upstream_ragged_vs_independent_fp64", actual, expected)
    tie = np.array([[1.0, 1.0, 1.0, 0.0]], np.float32)
    _, choices = jax.lax.top_k(jnp.asarray(tie), 2)
    assert np.array_equal(np.asarray(choices), [[0, 1]])
    mask = np.asarray(
        modules._create_sliding_mask(
            jnp.asarray([[1023, 1024, 2048]]),
            cache_positions=jnp.asarray([[0, 1, 1023, 1024, 2048]]),
            sliding_window_size=1024,
        )
    )
    expected = np.array(
        [
            [
                [True, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, True],
            ]
        ]
    )
    assert np.array_equal(mask, expected)
    # Separate true-FP64 tests mirror the retained longdouble scalar controls.
    # They test the FP64 helpers used by the streamed candidate path, not a relaxed
    # tolerance around the stock FP32 implementation above.
    ld = np.longdouble
    wide = np.finfo(ld).nmant > np.finfo(np.float64).nmant
    precise_rows = []
    if wide:
        eps = np.finfo(np.float64).eps

        def bounded64(name, actual, expected, bound):
            a = np.asarray(actual)
            self_dtype = a.dtype == np.dtype(np.float64)
            error = np.abs(a.astype(ld) - expected)
            passed = (
                self_dtype
                and bool(np.isfinite(a).all())
                and bool(np.all(error <= bound))
            )
            item = {
                "name": name,
                "dtype": str(a.dtype),
                "max_abs": float(np.max(error)),
                "max_error_over_bound": float(np.max(error / bound)),
                "passed": passed,
            }
            precise_rows.append(item)
            if not passed:
                raise ValueError("True-FP64 operator gate " + name + ": " + str(item))

        for width in (256, 512, 2816):
            rg = np.random.default_rng(96170 + width)
            xx = rg.normal(size=(3, width)) * np.array([1e-8, 0.5, 32.0])[:, None]
            scale = rg.uniform(0.125, 3.0, width).astype(np.float32).astype(np.float64)
            xl = xx.astype(ld)
            expected64 = (
                xl
                / np.sqrt(
                    np.sum(xl * xl, axis=-1, keepdims=True, dtype=ld) / ld(width)
                    + ld(1e-6)
                )
                * scale.astype(ld)
            )
            gamma = (width + 4) * eps / (1 - (width + 4) * eps)
            bound = 32 * gamma * np.maximum(ld(1), np.abs(expected64))
            actual64 = layers.RMSNorm().apply(
                {"params": {"scale": jnp.asarray(scale, dtype=jnp.float64)}},
                jnp.asarray(xx, dtype=jnp.float64),
            )
            bounded64("rmsnorm_true_fp64_" + str(width), actual64, expected64, bound)
        from .precise_rope import apply_rope as precise_apply_rope

        pos = np.array([[0, 1, 255, 509, 511, 1023, 8191]], dtype=np.int32)
        for width, prop, base in ((256, 1.0, 10000), (512, 0.25, 1000000)):
            rg = np.random.default_rng(62812 + width)
            xx = rg.normal(size=(1, len(pos[0]), 2, width))
            half = width // 2
            active = int(width * prop) // 2
            expected64 = np.empty_like(xx, dtype=ld)
            bounds = np.empty_like(xx, dtype=ld)
            for j in range(half):
                phase = (
                    pos.astype(ld) / (ld(base) ** (ld(2 * j) / ld(width)))
                    if j < active
                    else np.zeros_like(pos, dtype=ld)
                )
                c = np.cos(phase)[..., None]
                ss = np.sin(phase)[..., None]
                a = xx[..., j].astype(ld)
                b = xx[..., half + j].astype(ld)
                expected64[..., j] = a * c - b * ss
                expected64[..., half + j] = b * c + a * ss
                bound = (
                    128
                    * eps
                    * (1 + np.abs(phase))[..., None]
                    * (1 + np.abs(a) + np.abs(b))
                )
                bounds[..., j] = bound
                bounds[..., half + j] = bound
            actual64 = precise_apply_rope(
                jnp.asarray(xx, dtype=jnp.float64),
                jnp.asarray(pos),
                base_frequency=base,
                rope_proportion=prop,
            )
            bounded64("rope_true_fp64_" + str(prop), actual64, expected64, bounds)
            actual_np = np.asarray(actual64)
            np.testing.assert_array_equal(actual_np[:, 0], xx[:, 0])
        gel = np.array(
            [-12.0, -8.0, -5.0, -3.0, -1.0, -1e-9, 0.0, 1e-9, 1.0, 3.0, 5.0, 8.0, 12.0]
        )
        gl = gel.astype(ld)
        pi = np.arccos(ld(-1))
        expected_g = (
            ld(".5")
            * gl
            * (1 + np.tanh(np.sqrt(ld(2) / pi) * (gl + ld(".044715") * gl**3)))
        )
        gel_bound = 256 * eps * (1 + np.abs(gl) + np.abs(expected_g))
        bounded64(
            "gelu_tanh_true_fp64",
            jax.nn.gelu(jnp.asarray(gel, dtype=jnp.float64), approximate=True),
            expected_g,
            gel_bound,
        )
    report = {
        "passed": bool(wide and all(x["passed"] for x in precise_rows)),
        "precision64_status": "pass" if wide and precise_rows else "unavailable",
        "precision64_controls": precise_rows,
        "controls": rows,
        "router_ties": "lower index first verified",
        "sliding_mask": "boundary exact verified",
        "limitations": "Small independent operator controls; not full26B qualification",
    }
    Path(out).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import sys

    run(sys.argv[1], sys.argv[2])
