"""Independent checks for full-vocabulary metrics and grouped uncertainty."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality" / "src"))
from xe_quality import metrics


class MetricsTests(unittest.TestCase):
    def test_two_token_hand_calculation_nll_sign_and_identical(self):
        b = np.array([[0.0, np.log(3.0)]])
        c = np.array([[0.0, 0.0]])
        got = metrics.compare_logits(b, c, expected_vocab_size=2, observed_tokens=[0])
        p = np.array([0.25, 0.75])
        q = np.array([0.5, 0.5])
        expected_kl = np.sum(p * np.log(p / q))
        self.assertAlmostEqual(got["kl_b_to_c_nats"][0], expected_kl, places=14)
        self.assertAlmostEqual(got["tv"][0], 0.25, places=14)
        self.assertAlmostEqual(
            got["delta_nll_nats"][0], np.log(2.0) - np.log(4.0), places=14
        )
        self.assertEqual(got["reference_top1"][0], 1)
        self.assertEqual(got["candidate_top1"][0], 0)
        same = metrics.compare_logits(b, b, expected_vocab_size=2, observed_tokens=[1])
        np.testing.assert_array_equal(same["kl_b_to_c_nats"], [0.0])
        np.testing.assert_array_equal(same["tv"], [0.0])
        np.testing.assert_array_equal(same["delta_nll_nats"], [0.0])
        worse_for_observed_one = metrics.compare_logits(
            b, c, expected_vocab_size=2, observed_tokens=[1]
        )
        self.assertGreater(worse_for_observed_one["delta_nll_nats"][0], 0.0)

    def test_shift_permutation_and_extremes(self):
        b = np.array([[4.0, 0.0, -2.0], [10000.0, 9999.0, -10000.0]])
        c = np.array([[-6.0, -10.0, -12.0], [10000.0, 9999.0, -10000.0]])
        baseline = metrics.compare_logits(b, b, expected_vocab_size=3)
        shifted = metrics.compare_logits(b, c, expected_vocab_size=3)
        np.testing.assert_allclose(
            shifted["kl_b_to_c_nats"], baseline["kl_b_to_c_nats"], atol=1e-14
        )
        perm = [2, 0, 1]
        permuted = metrics.compare_logits(b[:, perm], c[:, perm], expected_vocab_size=3)
        np.testing.assert_allclose(
            permuted["kl_b_to_c_nats"], shifted["kl_b_to_c_nats"], atol=1e-14
        )
        np.testing.assert_allclose(permuted["tv"], shifted["tv"], atol=1e-14)
        self.assertTrue(np.isfinite(shifted["kl_b_to_c_nats"]).all())
        self.assertGreaterEqual(shifted["tv"].min(), 0.0)

    def test_full_vocab_mismatch_truncation_and_nonfinite_rejected(self):
        full = np.zeros((2, 4), dtype=np.float32)
        with self.assertRaises(metrics.MetricError):
            metrics.compare_logits(full[:, :3], full[:, :3], expected_vocab_size=4)
        with self.assertRaises(metrics.MetricError):
            metrics.compare_logits(full, full[:, :3], expected_vocab_size=4)
        with self.assertRaises(metrics.MetricError):
            metrics.compare_logits(
                np.array([[0.0, np.nan]]), np.zeros((1, 2)), expected_vocab_size=2
            )
        with self.assertRaises(metrics.MetricError):
            metrics.compare_logits(
                full, full, expected_vocab_size=4, observed_tokens=[4, 1]
            )

    def test_groups_weight_documents_equally_and_bootstrap_repeats(self):
        summary = metrics.group_summary(
            [0.0, 2.0, 10.0],
            ["doc-a", "doc-a", "doc-b"],
            domains=["code", "code", "code"],
        )
        domain = summary["domains"]["code"]
        self.assertEqual(domain["group_count"], 2)
        self.assertEqual(domain["position_count"], 3)
        self.assertEqual(domain["mean_group_mean"], 5.5)
        self.assertNotIn("group_mean_quantiles", domain)
        a, b, groups = [0.0, 1.0, 5.0, 6.0], [1.0, 2.0, 7.0, 8.0], ["a", "a", "b", "b"]
        first = metrics.paired_group_bootstrap(
            a, b, groups, seed=4, resamples=500, min_groups=2
        )
        second = metrics.paired_group_bootstrap(
            a, b, groups, seed=4, resamples=500, min_groups=2
        )
        self.assertEqual(first, second)
        self.assertEqual(first["estimate_candidate_minus_reference"], 1.5)
        self.assertEqual(first["interval"], [1.0, 2.0])
        short = metrics.paired_group_bootstrap(a, b, groups, min_groups=3)
        self.assertEqual(short["status"], "insufficient_groups")


if __name__ == "__main__":
    unittest.main()
