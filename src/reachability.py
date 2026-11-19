"""Window partitioning and exact depth-limited Boolean reachability.

Convention
----------
A[i, j] is True when OUTPUT token i of a layer can attend to INPUT token j.

    R_0 = I
    R_l = A_l (Boolean product) R_{l-1}

so R_L[i, j] is True exactly when the output of token i after L layers
can depend on input token j through some chain of L attention steps,
one step per layer. Every window contains its own token, so A_l[i, i] is
always True, which also covers the residual path in a Transformer block.

Shifted windows
---------------
Swin/SwiFT use a cyclic roll by -s followed by a region-label mask. After
masking, this is equivalent to the NON-cyclic partition

    [0, s), [s, s + w), [s + w, s + 2w), ..., [last cut, N)

`shifted_partition` builds that partition directly. `cyclic_masked_adjacency`
independently re-implements the roll + region-mask procedure, and the test
suite checks that the two agree, so no wraparound edges are introduced.
"""

from __future__ import annotations

import numpy as np

ARCHITECTURES = ("global", "fixed", "shifted")


# --------------------------------------------------------------------------
# Window partitions (lists of half-open [start, stop) intervals)
# --------------------------------------------------------------------------
def regular_partition(n: int, w: int) -> list[tuple[int, int]]:
    """Non-overlapping windows [0,w), [w,2w), ...; the last one may be partial."""
    return [(a, min(a + w, n)) for a in range(0, n, w)]


def shifted_partition(n: int, w: int, s: int) -> list[tuple[int, int]]:
    """Windows displaced by s tokens, with partial windows at both boundaries
    and NO cyclic wraparound."""
    if not 0 < s < w:
        raise ValueError("shift must satisfy 0 < s < w")
    cuts = [0] + list(range(s, n, w)) + [n]
    cuts = sorted(set(cuts))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def adjacency_from_partition(n: int, windows: list[tuple[int, int]]) -> np.ndarray:
    """Block-diagonal Boolean adjacency: tokens attend within their window."""
    covered = sorted(i for a, b in windows for i in range(a, b))
    if covered != list(range(n)):
        raise ValueError("windows must partition range(n) exactly")
    A = np.zeros((n, n), dtype=bool)
    for a, b in windows:
        A[a:b, a:b] = True
    return A


# --------------------------------------------------------------------------
# Independent reference: Swin/SwiFT cyclic roll + region-label mask
# --------------------------------------------------------------------------
def cyclic_masked_adjacency(n: int, w: int, s: int, apply_mask: bool = True) -> np.ndarray:
    """Re-implements, in 1D, the procedure in Swin `SwinTransformerBlock` and
    SwiFT `SwinTransformerBlock4D.forward_part1` / `compute_mask`:

    1. roll the sequence by -s, so rolled position p holds token (p + s) % n;
    2. label rolled positions with three regions, slice(0, -w) -> 0,
       slice(-w, -s) -> 1, slice(-s, None) -> 2;
    3. partition rolled positions into windows of size w;
    4. allow attention only within a window AND between equal labels
       (the source adds -100 to unequal-label scores; here it is a hard mask).

    With apply_mask=False this returns the naive cyclic (wraparound) version,
    which is used as a negative control in the tests.
    """
    if n % w != 0:
        raise ValueError("reference assumes n divisible by w (SwiFT pads otherwise)")
    token_at = (np.arange(n) + s) % n  # torch.roll(x, -s): out[p] = x[(p + s) % n]
    label = np.zeros(n, dtype=int)
    label[n - w : n - s] = 1
    label[n - s :] = 2
    A = np.zeros((n, n), dtype=bool)
    for start in range(0, n, w):
        for p in range(start, start + w):
            for q in range(start, start + w):
                if apply_mask and label[p] != label[q]:
                    continue
                A[token_at[p], token_at[q]] = True
    return A


# --------------------------------------------------------------------------
# Layer schedules and exact propagation
# --------------------------------------------------------------------------
BOUNDARIES = ("masked", "cyclic")


def layer_adjacency(arch: str, layer: int, n: int, w: int, s: int,
                    boundary: str = "masked") -> np.ndarray:
    """Adjacency of layer `layer` (1-indexed) for a given architecture.

    'shifted' alternates regular (odd layers) and shifted (even layers)
    windows, matching SwiFT's `shift_size = no_shift if i % 2 == 0` with
    0-indexed blocks.

    boundary='masked' : Swin/SwiFT semantics (cyclic roll + region mask, which
                        equals the non-cyclic partition). Default.
    boundary='cyclic' : the shifted layer wraps around (roll WITHOUT the mask),
                        i.e. a ring topology. Used only as a comparison condition.
    'none' is a no-communication control (each token attends only to itself).
    """
    if layer < 1:
        raise ValueError("layers are 1-indexed")
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}")
    if arch == "global":
        return np.ones((n, n), dtype=bool)
    if arch == "none":
        return np.eye(n, dtype=bool)
    if arch == "fixed":
        return adjacency_from_partition(n, regular_partition(n, w))
    if arch == "shifted":
        if layer % 2 == 1:
            return adjacency_from_partition(n, regular_partition(n, w))
        if boundary == "cyclic":
            return cyclic_masked_adjacency(n, w, s, apply_mask=False)
        return adjacency_from_partition(n, shifted_partition(n, w, s))
    raise ValueError(f"unknown architecture {arch!r}")


def layer_partition(arch: str, layer: int, n: int, w: int, s: int) -> list[tuple[int, int]]:
    """Window partition used by a layer ('global' = one window of size n)."""
    if arch == "global":
        return [(0, n)]
    if arch == "fixed" or layer % 2 == 1:
        return regular_partition(n, w)
    return shifted_partition(n, w, s)


def bool_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Boolean product C[i,j] = OR_k (A[i,k] AND B[k,j]).

    Integer counts use int32 so they cannot overflow for any n considered
    here (a uint8 product would wrap at 256 paths and silently drop edges).
    """
    return (A.astype(np.int32) @ B.astype(np.int32)) > 0


def reachability(arch: str, n: int, depth: int, w: int = 4, s: int = 2,
                 boundary: str = "masked") -> list[np.ndarray]:
    """Return [R_0, R_1, ..., R_depth] for an architecture."""
    R = np.eye(n, dtype=bool)
    out = [R]
    for layer in range(1, depth + 1):
        R = bool_matmul(layer_adjacency(arch, layer, n, w, s, boundary), R)
        out.append(R)
    return out


def connected_components(R: np.ndarray) -> int:
    """Number of connected components of the undirected graph with an edge
    i -- j whenever R[i, j] or R[j, i] (union-find, independent of matmul)."""
    n = R.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.nonzero(R | R.T)):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    return len({find(i) for i in range(n)})


def summarize(R: np.ndarray) -> dict:
    """Scalar summaries of one reachability matrix."""
    n = R.shape[0]
    rf = R.sum(axis=1)  # receptive field size of each output token
    i, j = np.nonzero(R)
    return {
        "reachable_pairs": int(R.sum()),
        "reachable_fraction": float(R.sum()) / (n * n),
        "mean_receptive_field": float(rf.mean()),
        "min_receptive_field": int(rf.min()),
        "max_receptive_field": int(rf.max()),
        "max_token_distance": int(np.abs(i - j).max()),
    }
