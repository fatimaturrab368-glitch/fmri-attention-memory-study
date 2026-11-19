"""Tests for the upgrade: sweep laws, 4D separability, attention ops,
synthetic task, matched models, and learning-pipeline separation.

Run:  python -m unittest discover -s tests -v       (CPU only, < 1 min)
"""

import itertools
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attention_ops import dense_masked, shifted_window, window_attention  # noqa: E402
from src.benchmark import theoretical_score_bytes  # noqa: E402
from src.reachability import (  # noqa: E402
    adjacency_from_partition, bool_matmul, connected_components, cyclic_masked_adjacency,
    layer_adjacency, reachability, regular_partition, shifted_partition)
from src.sweep import layers_to_distance  # noqa: E402
from src.synthetic_task import IGNORE, make_dataset, make_split  # noqa: E402
from src.torch_models import MaskedTransformer, build_masks, n_params  # noqa: E402

NS = [16, 32, 64, 128, 256]


class TestSweepLaws(unittest.TestCase):
    """Exact laws reported in reports/04 (masked = Swin/SwiFT boundary)."""

    def test_distance_and_receptive_field_laws(self):
        for n in NS:
            for w, shifts in ((4, (1, 2)), (8, (1, 2, 4))):
                for s in shifts:
                    Rs = reachability("shifted", n, 32, w, s)
                    for L in range(1, 33):
                        R = Rs[L]
                        i, j = np.nonzero(R)
                        pred = (w // 2) * (L - 1) + (w - 1) + (L % 2 == 0) * (w // 2 - s)
                        self.assertEqual(int(np.abs(i - j).max()), min(pred, n - 1))
                        self.assertEqual(int(R.sum(1).max()), min(w * L, n))

    def test_pair_count_independent_of_shift(self):
        for n in NS:
            for w, shifts in ((4, (1, 2)), (8, (1, 2, 4))):
                counts = [[int(R.sum()) for R in reachability("shifted", n, 32, w, s)]
                          for s in shifts]
                self.assertTrue(all(c == counts[0] for c in counts))

    def test_cyclic_boundary_fraction(self):
        for n in NS:
            for w, s in ((4, 2), (8, 4), (8, 1)):
                for L, R in enumerate(reachability("shifted", n, 32, w, s, "cyclic")):
                    if L:
                        self.assertAlmostEqual(R.mean(), min(w * L / n, 1.0))
                # wraparound present only in the cyclic condition
                self.assertTrue(reachability("shifted", n, 2, w, s, "cyclic")[2][0, n - 1])

    def test_components_vs_reachability(self):
        """Components (closure-like) differ from depth-limited reachability."""
        for n in NS:
            self.assertEqual(connected_components(reachability("fixed", n, 8)[8]), n // 4)
            R2 = reachability("shifted", n, 2)[2]
            self.assertEqual(connected_components(R2), 1)
            if n > 16:
                self.assertLess(R2.mean(), 0.5)  # connected, yet most pairs unreachable

    def test_layers_to_distance(self):
        Rs = reachability("shifted", 64, 32)
        self.assertEqual(layers_to_distance(Rs, 1), (1, 2))   # some at L1, all at L2
        la, ll = layers_to_distance(reachability("fixed", 64, 32), 4)
        self.assertEqual((la, ll), (None, None))


class TestFourDSeparability(unittest.TestCase):
    """SwiFT-style 4D roll + region mask == Kronecker product of 1D masks."""

    @staticmethod
    def swin4d_adjacency(dims, win, shift):
        coords = list(itertools.product(*[range(d) for d in dims]))
        idx = {c: k for k, c in enumerate(coords)}
        n = len(coords)
        labels = []
        for d, w, s in zip(dims, win, shift):
            lab = np.zeros(d, dtype=int)
            lab[d - w: d - s] = 1
            lab[d - s:] = 2
            labels.append(lab)
        A = np.zeros((n, n), dtype=bool)
        for p in coords:          # rolled positions
            for q in coords:
                if any(a // w != b // w for a, b, w in zip(p, q, win)):
                    continue
                if any(lab[a] != lab[b] for a, b, lab in zip(p, q, labels)):
                    continue
                tp = tuple((a + s) % d for a, s, d in zip(p, shift, dims))
                tq = tuple((b + s) % d for b, s, d in zip(q, shift, dims))
                A[idx[tp], idx[tq]] = True
        return A

    def test_kronecker(self):
        dims, win, shift = (4, 4, 2, 8), (2, 2, 2, 4), (1, 1, 1, 2)
        A4 = self.swin4d_adjacency(dims, win, shift)
        A1 = [cyclic_masked_adjacency(d, w, s) for d, w, s in zip(dims, win, shift)]
        K = A1[0]
        for a in A1[1:]:
            K = np.kron(K, a)
        np.testing.assert_array_equal(A4, K)
        # two-layer reachability also factorises (Boolean mixed-product property)
        R1 = [adjacency_from_partition(d, regular_partition(d, w)) for d, w in zip(dims, win)]
        Kr = R1[0]
        for a in R1[1:]:
            Kr = np.kron(Kr, a)
        R2_4d = bool_matmul(A4, Kr)
        R2_axes = [bool_matmul(a, r) for a, r in zip(A1, R1)]
        Kp = R2_axes[0]
        for a in R2_axes[1:]:
            Kp = np.kron(Kp, a)
        np.testing.assert_array_equal(R2_4d, Kp)


class TestAttentionOps(unittest.TestCase):
    def test_window_ops_match_dense_reference(self):
        torch.manual_seed(0)
        for n, w, s in ((16, 4, 2), (64, 4, 2), (64, 8, 4), (64, 8, 1)):
            q, k, v = (torch.randn(2, 3, n, 5, dtype=torch.float64) for _ in range(3))
            A_reg = torch.from_numpy(adjacency_from_partition(n, regular_partition(n, w)))
            A_sh = torch.from_numpy(adjacency_from_partition(n, shifted_partition(n, w, s)))
            torch.testing.assert_close(window_attention(q, k, v, w), dense_masked(q, k, v, A_reg))
            torch.testing.assert_close(shifted_window(q, k, v, w, s), dense_masked(q, k, v, A_sh))

    def test_theoretical_bytes(self):
        self.assertEqual(theoretical_score_bytes("global_math", 256, 0, 4, 2), 4 * 2 * 256 ** 2)
        self.assertEqual(theoretical_score_bytes("fixed_window", 256, 4, 4, 2), 4 * 2 * 64 * 16)


class TestSyntheticTask(unittest.TestCase):
    def test_labels(self):
        x, y = make_split(50, 16, 3, 4, seed=0, split="train")
        np.testing.assert_array_equal(y[:, :13], (x[:, :13] + x[:, 3:]) % 4)
        self.assertTrue((y[:, 13:] == IGNORE).all())

    def test_splits_disjoint_and_deterministic(self):
        cfg = dict(n_train=2000, n_val=500, n_test=500, n_tokens=64, vocab=4)
        a, b = make_dataset(cfg, 8, 0), make_dataset(cfg, 8, 0)
        for k in a:
            np.testing.assert_array_equal(a[k][0], b[k][0])
        rows = {k: {r.tobytes() for r in a[k][0]} for k in a}
        self.assertFalse(rows["train"] & rows["test"])
        self.assertFalse(rows["train"] & rows["val"])
        self.assertFalse(rows["val"] & rows["test"])
        # tuning seed (100) streams differ from final seeds
        self.assertFalse(np.array_equal(make_dataset(cfg, 8, 100)["val"][0], a["val"][0]))

    def test_no_single_token_shortcut(self):
        """P(y_i | x_i) and P(y_i | x_{i+d}) are uniform (up to sampling noise)."""
        x, y = make_split(20000, 32, 5, 4, seed=1, split="train")
        for src in (x[:, :27], x[:, 5:]):
            for val in range(4):
                p = np.bincount(y[:, :27][src == val], minlength=4) / (src == val).sum()
                self.assertLess(np.abs(p - 0.25).max(), 0.01)


class TestMaskedModel(unittest.TestCase):
    def test_matched_params_and_shapes(self):
        counts = set()
        for arch in ("global", "fixed", "shifted", "none"):
            m = MaskedTransformer(4, 32, 16, 2, 4, build_masks(arch, 32, 4, 4, 2), 4)
            counts.add(n_params(m))
            self.assertEqual(tuple(m(torch.zeros(3, 32, dtype=torch.long)).shape), (3, 32, 4))
            for l in range(4):
                np.testing.assert_array_equal(m.masks[l].numpy(),
                                              layer_adjacency(arch, l + 1, 32, 4, 2))
        self.assertEqual(len(counts), 1)

    def test_gradient_dependency_equals_reachability(self):
        """d logits_i / d embedding_j is nonzero iff R_L[i, j]."""
        torch.manual_seed(0)
        n, L = 16, 3
        for arch in ("global", "fixed", "shifted", "none"):
            m = MaskedTransformer(4, n, 16, 2, L, build_masks(arch, n, L, 4, 2), 4).double()
            h0 = (m.tok(torch.randint(0, 4, (1, n))) + m.pos).detach().requires_grad_(True)

            def f(h):
                for l, blk in enumerate(m.blocks):
                    h = blk(h, m.masks[l])
                return m.head(m.ln(h))

            J = torch.autograd.functional.jacobian(f, h0)  # (1,n,4,1,n,16)
            dep = (J.abs().sum(dim=(0, 2, 3, 5)) > 0).numpy()  # (n, n)
            np.testing.assert_array_equal(dep, reachability(arch, n, L)[L])


if __name__ == "__main__":
    unittest.main()
