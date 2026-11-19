"""Option A: exact reachability sweep over N, window w, shift s, boundary and depth."""

from __future__ import annotations

import numpy as np

from .memory import global_score_bytes, partition_score_bytes
from .reachability import (
    connected_components,
    regular_partition,
    reachability,
    summarize,
)


def layers_to_distance(Rs: list[np.ndarray], d: int):
    """For distance d, return (L_any, L_all):
    L_any = first depth at which SOME ordered pair at linear distance d is reachable,
    L_all = first depth at which ALL ordered pairs at distance d are reachable.
    None if not reached within len(Rs) - 1 layers."""
    n = Rs[0].shape[0]
    if d >= n:
        return None, None
    i = np.arange(n - d)
    L_any = L_all = None
    for L, R in enumerate(Rs):
        fwd, bwd = R[i, i + d], R[i + d, i]
        if L_any is None and (fwd.any() or bwd.any()):
            L_any = L
        if L_all is None and fwd.all() and bwd.all():
            L_all = L
            break
    return L_any, L_all


def run_sweep(cfg: dict):
    rows, dist_rows = [], []
    max_depth = max(cfg["depths"])
    for n in cfg["N"]:
        for w in cfg["windows"]:
            if w >= n:
                continue
            shifts = sorted({s for s in cfg["shifts"] if 0 < s < w} | {w // 2})
            # (arch, shift, boundary) conditions; fixed/global do not depend on s/boundary
            conds = [("global", 0, "n/a"), ("fixed", 0, "n/a")]
            conds += [("shifted", s, b) for s in shifts for b in cfg["boundaries"]]
            win_bytes = partition_score_bytes(regular_partition(n, w))
            for arch, s, b in conds:
                Rs = reachability(arch, n, max_depth, w, s or w // 2,
                                  "masked" if b == "n/a" else b)
                for L in cfg["depths"]:
                    R = Rs[L]
                    row = {"N": n, "window": w, "shift": s, "boundary": b,
                           "architecture": arch, "depth": L}
                    row.update(summarize(R))
                    row["components"] = connected_components(R)
                    row["peak_layer_score_bytes"] = (global_score_bytes(n) if arch == "global"
                                                     else win_bytes)
                    rows.append(row)
                for d in cfg["distances"]:
                    if d >= n:
                        continue
                    la, ll = layers_to_distance(Rs, d)
                    dist_rows.append({"N": n, "window": w, "shift": s, "boundary": b,
                                      "architecture": arch, "distance": d,
                                      "layers_any_pair": "" if la is None else la,
                                      "layers_all_pairs": "" if ll is None else ll,
                                      "max_depth_checked": max_depth})
    return rows, dist_rows
