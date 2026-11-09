"""Google's canonical split-half partial RoPE, with implementation casts lifted.

Same formula/base/partial dimensions/GELU-independent architecture. This is an
explicit higher-precision reference variant, not bit-identical upstream FP32.
"""

import jax.numpy as jnp


def apply_rope(
    inputs, positions, *, base_frequency, scale_factor=1.0, rope_proportion=1.0
):
    if inputs.dtype != jnp.float64:
        raise ValueError("Precise RoPE requires FP64 operands")
    if scale_factor < 1.0:
        raise ValueError("RoPE scale must be >=1")
    d = inputs.shape[-1]
    n = int(rope_proportion * d // 2)
    exponent = (2.0 / d) * jnp.arange(n, dtype=jnp.float64)
    timescale = jnp.pad(
        jnp.asarray(base_frequency, dtype=jnp.float64) ** exponent,
        (0, d // 2 - n),
        constant_values=(0, jnp.inf),
    )
    angles = positions.astype(jnp.float64)[..., None] / timescale[None, None, :]
    angles = angles[..., None, :] / jnp.asarray(scale_factor, dtype=jnp.float64)
    first, second = jnp.split(inputs, 2, axis=-1)
    return jnp.concatenate(
        [
            first * jnp.cos(angles) - second * jnp.sin(angles),
            second * jnp.cos(angles) + first * jnp.sin(angles),
        ],
        axis=-1,
    )
