"""Theoretical attention-score storage.

Assumptions (stated once, applied everywhere):
  * one attention head, one layer, batch size 1;
  * only the materialized QK^T score matrix is counted (FP16 = 2 bytes/entry);
  * activations, Q/K/V, softmax outputs, weights, gradients, optimizer state,
    workspace and allocator overhead are NOT counted.
These are therefore NOT GPU-memory measurements.
"""

from __future__ import annotations

BYTES_FP16 = 2


def score_bytes(window_sizes, bytes_per_entry: int = BYTES_FP16) -> int:
    """Bytes for per-window score matrices: bytes * sum(n_i^2)."""
    return bytes_per_entry * sum(int(k) ** 2 for k in window_sizes)


def global_score_bytes(n: int, bytes_per_entry: int = BYTES_FP16) -> int:
    return bytes_per_entry * n * n


def partition_score_bytes(windows, bytes_per_entry: int = BYTES_FP16) -> int:
    """Score storage of a partition using its ACTUAL window sizes
    (partial boundary windows counted at their true size)."""
    return score_bytes([b - a for a, b in windows], bytes_per_entry)


def cyclic_impl_score_bytes(n: int, w: int, bytes_per_entry: int = BYTES_FP16) -> int:
    """Shifted layer as Swin/SwiFT actually compute it: after the cyclic roll the
    sequence is cut into ceil(n/w) full windows of size w, and the masked
    (cross-region) entries are still materialized. Reported for comparison."""
    n_windows = -(-n // w)
    return bytes_per_entry * n_windows * w * w


# --------------------------------------------------------------------------
# Illustrative scale-up to SwiFT's input (NOT a model of SwiFT's memory)
# --------------------------------------------------------------------------
def swift_illustration(
    volume=(96, 96, 96), frames=20, patch=(6, 6, 6, 1), window=(4, 4, 4, 4)
) -> dict:
    """Stage-1 token grid of SwiFT and per-head, per-layer FP16 score storage
    for (a) hypothetical global attention over all tokens and (b) 4D window
    attention. Defaults follow the SwiFT paper (96^3 x 20, P = 6) and the
    repository's sample script (window 4x4x4x4, patch [6,6,6,1])."""
    grid = tuple(v // p for v, p in zip(volume, patch[:3])) + (frames // patch[3],)
    tokens = 1
    for g in grid:
        tokens *= g
    win_tokens = 1
    for k in window:
        win_tokens *= k
    n_windows = 1
    for g, k in zip(grid, window):
        n_windows *= -(-g // k)
    g_bytes = global_score_bytes(tokens)
    w_bytes = BYTES_FP16 * n_windows * win_tokens ** 2
    return {
        "grid": grid,
        "tokens": tokens,
        "window_tokens": win_tokens,
        "n_windows": n_windows,
        "global_bytes": g_bytes,
        "window_bytes": w_bytes,
        "reduction_factor": g_bytes / w_bytes,
    }
