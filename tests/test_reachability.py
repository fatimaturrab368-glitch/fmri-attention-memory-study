"""Mathematical validation (standard-library unittest).

Run:  python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory import (  # noqa: E402
    cyclic_impl_score_bytes,
    global_score_bytes,
    partition_score_bytes,
    swift_illustration,
)
from src.reachability import (  # noqa: E402
    ARCHITECTURES,
    adjacency_from_partition,
    bool_matmul,
    cyclic_masked_adjacency,
    layer_adjacency,
    reachability,
    regular_partition,
    shifted_partition,
)

NS = [16, 32, 64, 128, 256]
W, S, DEPTH = 4, 2, 16


def block_ids(n, w):
    return np.arange(n) // w


class TestManualExample(unittest.TestCase):
    """Hand-derived example: N = 8, w = 4, s = 2.

    Layer 1 (regular):  windows {0,1,2,3} {4,5,6,7}
    Layer 2 (shifted):  windows {0,1} {2,3,4,5} {6,7}   (no wraparound)

    R_1 = A_1 (two 4x4 blocks).
    R_2 = A_2 R_1:
      rows 0,1 : union of R_1 rows 0,1         -> {0..3}
      rows 2-5 : union of R_1 rows 2,3,4,5     -> {0..7}
      rows 6,7 : union of R_1 rows 6,7         -> {4..7}
    """

    def test_partitions(self):
        self.assertEqual(regular_partition(8, 4), [(0, 4), (4, 8)])
        self.assertEqual(shifted_partition(8, 4, 2), [(0, 2), (2, 6), (6, 8)])

    def test_R2_by_hand(self):
        expected = np.zeros((8, 8), dtype=bool)
        expected[0:2, 0:4] = True
        expected[2:6, :] = True
        expected[6:8, 4:8] = True
        R = reachability("shifted", 8, 2, W, S)
        np.testing.assert_array_equal(R[2], expected)

    def test_multiplication_order_matters(self):
        """A_2 @ A_1 (correct) differs from A_1 @ A_2 (wrong order)."""
        A1 = layer_adjacency("shifted", 1, 8, W, S)
        A2 = layer_adjacency("shifted", 2, 8, W, S)
        correct, wrong = bool_matmul(A2, A1), bool_matmul(A1, A2)
        self.assertFalse(np.array_equal(correct, wrong))
        # wrong order would let token 0 reach token 5 after two layers
        self.assertTrue(wrong[0, 5])
        self.assertFalse(correct[0, 5])


class TestReachability(unittest.TestCase):
    def test_1_R0_identity(self):
        for n in NS:
            for arch in ARCHITECTURES:
                np.testing.assert_array_equal(reachability(arch, n, 0)[0], np.eye(n, dtype=bool))

    def test_2_global_full_at_L1(self):
        for n in NS:
            self.assertTrue(reachability("global", n, 1)[1].all())

    def test_3_fixed_never_crosses_boundaries(self):
        for n in NS:
            b = block_ids(n, W)
            same_block = b[:, None] == b[None, :]
            for R in reachability("fixed", n, DEPTH)[1:]:
                self.assertFalse((R & ~same_block).any())
                np.testing.assert_array_equal(R, same_block)

    def test_4_shifted_nondecreasing(self):
        for n in NS:
            Rs = reachability("shifted", n, DEPTH)
            for prev, cur in zip(Rs[:-1], Rs[1:]):
                self.assertTrue((cur | ~prev).all(), "R_l must contain R_(l-1)")
            # self-connections present in every layer
            for layer in range(1, DEPTH + 1):
                self.assertTrue(np.diag(layer_adjacency("shifted", layer, n, W, S)).all())

    def test_4b_one_step_per_layer(self):
        """Each layer extends the receptive field by at most one window
        (w tokens) of new reach, so the maximal |i - j| grows by <= w."""
        for n in NS:
            prev_d = 0
            for R in reachability("shifted", n, DEPTH)[1:]:
                i, j = np.nonzero(R)
                d = int(np.abs(i - j).max())
                self.assertLessEqual(d - prev_d, W)
                prev_d = d

    def test_4c_depth_laws_stated_in_readme(self):
        """Exact laws for w=4, s=2 at all five N, L = 1..16."""
        for n in NS:
            for L, R in enumerate(reachability("shifted", n, DEPTH)[1:], start=1):
                rf = R.sum(axis=1)
                i, j = np.nonzero(R)
                self.assertEqual(int(np.abs(i - j).max()), min(2 * L + 1, n - 1))
                self.assertEqual(int(rf.max()), min(4 * L, n))
                self.assertEqual(int(rf.min()), min(4 * -(-L // 2), n))
        for n in (16, 32):
            Rs = reachability("shifted", n, DEPTH)
            first = next(L for L in range(1, DEPTH + 1) if Rs[L].all())
            self.assertEqual(first, (n - 2) // 2)

    def test_4d_functional_no_dependency_outside_R(self):
        """Ground truth from an actual masked-softmax attention network
        (random weights, residual + tanh): perturbing input j never changes
        output i when R_L[i, j] is False."""
        rng = np.random.default_rng(0)
        n, d = 16, 3
        for arch in ARCHITECTURES:
            for L in (1, 2, 3, 4):
                Ws = rng.normal(size=(L, 3, d, d))
                masks = [np.where(layer_adjacency(arch, l, n, W, S), 0.0, -np.inf)
                         for l in range(1, L + 1)]

                def net(x):
                    h = x
                    for (Wq, Wk, Wv), m in zip(Ws, masks):
                        sc = (h @ Wq) @ (h @ Wk).T / np.sqrt(d) + m
                        p = np.exp(sc - sc.max(axis=1, keepdims=True))
                        h = h + np.tanh((p / p.sum(axis=1, keepdims=True)) @ (h @ Wv))
                    return h

                x = rng.normal(size=(n, d))
                base = net(x)
                R = reachability(arch, n, L, W, S)[L]
                for j in range(n):
                    xp = x.copy()
                    xp[j] += 1e-3
                    changed = np.abs(net(xp) - base).max(axis=1) > 0
                    self.assertFalse((changed & ~R[:, j]).any(), f"{arch} L={L} j={j}")

    def test_5_masking_matches_swin_cyclic_reference(self):
        """Our non-cyclic partition == Swin/SwiFT roll + region mask."""
        for n in [8] + NS:
            ours = adjacency_from_partition(n, shifted_partition(n, W, S))
            ref = cyclic_masked_adjacency(n, W, S, apply_mask=True)
            np.testing.assert_array_equal(ours, ref)

    def test_5b_no_wraparound(self):
        for n in NS:
            naive = cyclic_masked_adjacency(n, W, S, apply_mask=False)
            self.assertTrue(naive[0, n - 1], "negative control should wrap")
            for R in reachability("shifted", n, 2)[1:]:
                self.assertFalse(R[0, n - 1] or R[n - 1, 0])

    def test_bool_matmul_no_overflow(self):
        """Compare against an explicit OR/AND reference at n = 256 with
        dense inputs (path counts exceed 255)."""
        rng = np.random.default_rng(0)
        A = rng.random((256, 256)) < 0.9
        B = rng.random((256, 256)) < 0.9
        ref = np.array([[np.any(A[i] & B[:, j]) for j in range(256)] for i in range(256)])
        np.testing.assert_array_equal(bool_matmul(A, B), ref)
        # all-ones: exactly 256 paths per entry, which a uint8 product wraps to 0
        ones = np.ones((256, 256), dtype=bool)
        self.assertFalse(((ones.astype(np.uint8) @ ones.astype(np.uint8)) > 0).any())
        self.assertTrue(bool_matmul(ones, ones).all())


class TestMemory(unittest.TestCase):
    def test_6_analytic_matches_explicit_partition(self):
        for n in NS:
            for part in (regular_partition(n, W), shifted_partition(n, W, S)):
                A = adjacency_from_partition(n, part)
                # explicit count of materialized (unmasked) score entries
                self.assertEqual(partition_score_bytes(part), 2 * int(A.sum()))
            self.assertEqual(global_score_bytes(n), 2 * n * n)
            self.assertEqual(partition_score_bytes(regular_partition(n, W)), 8 * n)
            self.assertEqual(partition_score_bytes(shifted_partition(n, W, S)), 8 * n - 16)
            self.assertEqual(cyclic_impl_score_bytes(n, W), 8 * n)

    def test_swift_numbers(self):
        r = swift_illustration()
        self.assertEqual(r["grid"], (16, 16, 16, 20))
        self.assertEqual(r["tokens"], 81_920)
        self.assertEqual(r["n_windows"], 320)
        self.assertEqual(r["global_bytes"], 2 * 81_920 ** 2)
        self.assertEqual(r["window_bytes"], 2 * 320 * 256 ** 2)
        self.assertAlmostEqual(r["reduction_factor"], 320.0)


class TestDeterminism(unittest.TestCase):
    def test_7_deterministic(self):
        for arch in ARCHITECTURES:
            a = reachability(arch, 64, DEPTH)
            b = reachability(arch, 64, DEPTH)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
