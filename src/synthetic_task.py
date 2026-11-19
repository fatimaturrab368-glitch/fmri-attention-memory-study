"""Synthetic long-range 'paired modular sum' task.

Input : x in {0..K-1}^N, i.i.d. uniform.
Target: for each position i < N - d,   y_i = (x_i + x_{i+d}) mod K.
        positions i >= N - d are ignored (label -100).

Why there is no shortcut: for uniform i.i.d. tokens, y_i is independent of
x_i alone and independent of x_{i+d} alone (each marginal is uniform), so any
predictor that cannot combine BOTH tokens is at chance 1/K in expectation.
A token-wise model can therefore not exceed chance at any position.
"""

from __future__ import annotations

import numpy as np

IGNORE = -100

# disjoint RNG streams for each split: seed -> (seed * 10 + offset)
SPLIT_OFFSET = {"train": 1, "val": 2, "test": 3}


def make_split(n_seq: int, n_tokens: int, d: int, k: int, seed: int, split: str):
    rng = np.random.default_rng(seed * 10 + SPLIT_OFFSET[split])
    x = rng.integers(0, k, size=(n_seq, n_tokens), dtype=np.int64)
    y = np.full_like(x, IGNORE)
    y[:, : n_tokens - d] = (x[:, : n_tokens - d] + x[:, d:]) % k
    return x, y


def make_dataset(cfg: dict, d: int, seed: int):
    return {s: make_split(cfg[f"n_{s}"], cfg["n_tokens"], d, cfg["vocab"], seed, s)
            for s in ("train", "val", "test")}
