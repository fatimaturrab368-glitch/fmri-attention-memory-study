"""Reproduce every result, table and figure:   python run.py"""

import csv
import os

from src.memory import (
    cyclic_impl_score_bytes,
    global_score_bytes,
    partition_score_bytes,
    swift_illustration,
)
from src.reachability import ARCHITECTURES, layer_partition, reachability, summarize
from src.visualize import fig1_heatmaps, fig2_growth, fig3_tradeoff

NS = [16, 32, 64, 128, 256]
W, S, MAX_DEPTH = 4, 2, 16
ROOT = os.path.dirname(os.path.abspath(__file__))


def compute():
    rows = []
    for n in NS:
        for arch in ARCHITECTURES:
            Rs = reachability(arch, n, MAX_DEPTH, W, S)
            peak = 0
            for L, R in enumerate(Rs):
                row = {"N": n, "architecture": arch, "depth": L, "window": W, "shift": S}
                if L == 0:
                    row.update(layer_type="input", layer_score_bytes=0,
                               layer_score_bytes_cyclic_impl=0)
                else:
                    part = layer_partition(arch, L, n, W, S)
                    if arch == "global":
                        ltype, b, b_cyc = "global", global_score_bytes(n), global_score_bytes(n)
                    elif arch == "shifted" and L % 2 == 0:
                        ltype = "shifted_window"
                        b, b_cyc = partition_score_bytes(part), cyclic_impl_score_bytes(n, W)
                    else:
                        ltype = "regular_window"
                        b = b_cyc = partition_score_bytes(part)
                    peak = max(peak, b)
                    row.update(layer_type=ltype, layer_score_bytes=b,
                               layer_score_bytes_cyclic_impl=b_cyc)
                row["peak_layer_score_bytes"] = peak
                row["global_over_peak_reduction"] = (
                    global_score_bytes(n) / peak if peak else "")
                row.update(summarize(R))
                rows.append(row)
    return rows


def main():
    for d in ("results", "figures"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    rows = compute()
    with open(os.path.join(ROOT, "results", "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)

    sw = swift_illustration()
    with open(os.path.join(ROOT, "results", "swift_scale_illustration.csv"), "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["quantity", "value"])
        for k, v in sw.items():
            wr.writerow([k, "x".join(map(str, v)) if isinstance(v, tuple) else v])

    fig1_heatmaps(os.path.join(ROOT, "figures", "fig1_reachability_heatmaps.png"))
    fig2_growth(os.path.join(ROOT, "figures", "fig2_reachability_growth.png"), rows, NS)
    fig3_tradeoff(os.path.join(ROOT, "figures", "fig3_memory_vs_access.png"), rows, NS)

    # console summary
    print(f"{'N':>4} {'arch':>8} | frac@L1  frac@L4  frac@L16 | first full L | peak bytes/layer | global/peak")
    for n in NS:
        for a in ARCHITECTURES:
            sel = {r["depth"]: r for r in rows if r["N"] == n and r["architecture"] == a}
            full = next((L for L in range(1, MAX_DEPTH + 1) if sel[L]["reachable_fraction"] == 1.0), None)
            print(f"{n:>4} {a:>8} | {sel[1]['reachable_fraction']:7.4f}  {sel[4]['reachable_fraction']:7.4f}"
                  f"  {sel[16]['reachable_fraction']:8.4f} | {str(full or '>16'):>12} |"
                  f" {sel[16]['peak_layer_score_bytes']:>16,} | {sel[16]['global_over_peak_reduction']:.1f}x")
    print("\nSwiFT-scale illustration:", sw)


if __name__ == "__main__":
    main()
