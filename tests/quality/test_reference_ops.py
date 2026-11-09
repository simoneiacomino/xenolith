from __future__ import annotations

import unittest


class PreciseRopeTests(unittest.TestCase):
    def test_precise_rope_fp64_rotates_prefix_and_preserves_suffix(self):
        try:
            import jax
            import jax.numpy as jnp
            import numpy as np
        except ImportError as error:
            self.skipTest(f"optional JAX dependency unavailable: {error}")
        jax.config.update("jax_enable_x64", True)
        from xe_quality.reference.precise_rope import apply_rope

        x = jnp.asarray([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=jnp.float64)
        positions = jnp.asarray([[7]], dtype=jnp.int32)
        actual = np.asarray(
            apply_rope(x, positions, base_frequency=10000, rope_proportion=0.5)
        )
        theta = 7.0
        expected_pair = np.asarray(
            [
                1.0 * np.cos(theta) - 3.0 * np.sin(theta),
                3.0 * np.cos(theta) + 1.0 * np.sin(theta),
            ]
        )
        np.testing.assert_allclose(
            actual[0, 0, 0, [0, 2]], expected_pair, rtol=1e-14, atol=1e-14
        )
        np.testing.assert_array_equal(actual[0, 0, 0, [1, 3]], [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
