"""Attention cores used by the GPU benchmark (Option B).

All functions take q, k, v of shape (B, H, N, Dh) and return (B, H, N, Dh).

global_math       : softmax(QK^T / sqrt(Dh)) V with the N x N score matrix materialised.
global_sdpa       : torch.nn.functional.scaled_dot_product_attention (backend chosen
                    by PyTorch or forced by the caller); exact global attention.
window_attention  : non-overlapping windows of size w (fixed-window layer).
shifted_window    : Swin-style cyclic roll by -s, window attention with the
                    region-label mask (-inf), roll back by +s. Equivalent to the
                    non-cyclic shifted partition (verified in tests).
dense_masked      : reference implementation with an explicit N x N Boolean mask
                    (used only for correctness tests).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def global_math(q, k, v):
    att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    return att.softmax(dim=-1) @ v


def global_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def _windows(x, w):
    b, h, n, d = x.shape
    return x.reshape(b, h, n // w, w, d)


def window_attention(q, k, v, w, mask=None):
    b, h, n, d = q.shape
    if n % w:
        raise ValueError("benchmark assumes N divisible by w")
    qw, kw, vw = _windows(q, w), _windows(k, w), _windows(v, w)
    att = (qw @ kw.transpose(-2, -1)) * (d ** -0.5)        # (B, H, N/w, w, w)
    if mask is not None:
        att = att.masked_fill(~mask, float("-inf"))
    return (att.softmax(dim=-1) @ vw).reshape(b, h, n, d)


def shifted_region_mask(n, w, s, device):
    """(N/w, w, w) bool mask for the rolled sequence, built like Swin's
    `compute_mask`: label regions slice(0,-w), slice(-w,-s), slice(-s,None)."""
    label = torch.zeros(n, dtype=torch.long, device=device)
    label[n - w: n - s] = 1
    label[n - s:] = 2
    lw = label.reshape(n // w, w)
    return lw[:, :, None] == lw[:, None, :]


def shifted_window(q, k, v, w, s, mask=None):
    n = q.shape[2]
    if mask is None:
        mask = shifted_region_mask(n, w, s, q.device)
    roll = lambda x, sh: torch.roll(x, shifts=sh, dims=2)  # noqa: E731
    out = window_attention(roll(q, -s), roll(k, -s), roll(v, -s), w, mask)
    return roll(out, s)


def dense_masked(q, k, v, allowed):
    """Reference: allowed is an (N, N) bool matrix (query i -> key j)."""
    att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    att = att.masked_fill(~allowed, float("-inf"))
    return att.softmax(dim=-1) @ v
