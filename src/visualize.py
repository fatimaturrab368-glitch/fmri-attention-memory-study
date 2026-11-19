"""Figures (Matplotlib, 300 DPI PNG)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .reachability import reachability, summarize  # noqa: E402

# Fixed categorical order (validated default palette, slots 1-3); line style and
# marker give each architecture a second, color-independent encoding.
STYLE = {
    "global": dict(color="#2a78d6", ls="-", marker="o", label="Global attention"),
    "fixed": dict(color="#eb6834", ls="--", marker="s", label="Fixed windows"),
    "shifted": dict(color="#1baf7a", ls="-.", marker="^", label="Shifted windows (alternating)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd7"
REACH_CMAP = ListedColormap(["#f0efec", "#184f95"])  # unreachable / reachable

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def _grid(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig1_heatmaps(path, n=16, depths=(1, 2, 4, 7), w=4, s=2):
    archs = ("global", "fixed", "shifted")
    fig, axes = plt.subplots(3, len(depths), figsize=(1.9 * len(depths) + 1.2, 6.4),
                             constrained_layout=True)
    reach = {a: reachability(a, n, max(depths), w, s) for a in archs}
    for r, a in enumerate(archs):
        for c, L in enumerate(depths):
            ax = axes[r, c]
            R = reach[a][L]
            ax.imshow(R, cmap=REACH_CMAP, vmin=0, vmax=1, interpolation="nearest",
                      origin="upper", extent=(-0.5, n - 0.5, n - 0.5, -0.5))
            for b in range(w, n, w):  # original (layer-1) window boundaries
                ax.axhline(b - 0.5, color="white", lw=0.5, alpha=0.8)
                ax.axvline(b - 0.5, color="white", lw=0.5, alpha=0.8)
            ax.set_xticks(range(0, n, 4))
            ax.set_yticks(range(0, n, 4))
            ax.tick_params(length=2, labelsize=7)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(MUTED)
            ax.set_title(f"L = {L}   ({summarize(R)['reachable_fraction']:.0%} of pairs)",
                         color=INK)
            if c == 0:
                ax.set_ylabel(f"{STYLE[a]['label']}\noutput token i")
            if r == 2:
                ax.set_xlabel("input token j")
    fig.legend(handles=[Patch(color=REACH_CMAP(1.0), label="R_L[i, j] = True (dependency possible)"),
                        Patch(facecolor=REACH_CMAP(0.0), edgecolor=MUTED,
                              label="R_L[i, j] = False")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(f"Figure 1. Depth-limited reachability R_L = A_L ··· A_1 (N = {n}, w = {w}, s = {s}); "
                 "white lines = original window boundaries", fontsize=9.5, color=INK)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_growth(path, rows, ns):
    fig, axes = plt.subplots(1, len(ns), figsize=(13, 3.1), sharey=True, constrained_layout=True)
    for ax, n in zip(axes, ns):
        for a, st in STYLE.items():
            pts = sorted((r["depth"], r["reachable_fraction"]) for r in rows
                         if r["N"] == n and r["architecture"] == a and r["depth"] >= 1)
            x, y = zip(*pts)
            ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5,
                    lw=1.6, label=st["label"])
        ax.set_title(f"N = {n}")
        ax.set_xlabel("depth L (layers)")
        ax.set_xticks([1, 4, 8, 12, 16])
        ax.set_ylim(0, 1.04)
        _grid(ax)
    axes[0].set_ylabel("reachable ordered pairs / N²")
    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.suptitle("Figure 2. Fraction of ordered token pairs (i, j) with a possible computational "
                 "dependency of output i on input j, versus depth (w = 4, s = 2). "
                 "Not a measure of information retention.", fontsize=9.5, color=INK)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_tradeoff(path, rows, ns, depth=4):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
    for a, st in STYLE.items():
        sel = sorted((r for r in rows if r["architecture"] == a and r["depth"] == depth),
                     key=lambda r: r["N"])
        n = np.array([r["N"] for r in sel])
        mem = np.array([r["peak_layer_score_bytes"] for r in sel])
        frac = np.array([r["reachable_fraction"] for r in sel])
        a1.plot(n, mem, color=st["color"], ls=st["ls"], marker=st["marker"], ms=5, lw=1.6,
                label=st["label"])
        a2.plot(mem, frac, color=st["color"], ls=st["ls"], marker=st["marker"], ms=5, lw=1.6,
                label=st["label"])
        for ni, m, f in zip(n, mem, frac):
            if a != "fixed":  # fixed and shifted share memory; label one series only
                a2.annotate(f"N={ni}", (m, f), textcoords="offset points",
                            xytext=(5, 4 if a == "global" else -10), fontsize=6.5, color=MUTED)
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xticks(ns, [str(v) for v in ns])
    a1.set_xlabel("sequence length N (tokens)")
    a1.set_ylabel("peak per-layer FP16 score storage (bytes)")
    a1.set_title("(a) Theoretical score storage, 1 head, 1 layer\n"
                 "(fixed and shifted coincide: peak layer = regular windows, 8N bytes)")
    a1.legend(frameon=False, fontsize=7.5)
    _grid(a1)
    a2.set_xscale("log")
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("peak per-layer FP16 score storage (bytes, log scale)")
    a2.set_ylabel(f"reachable pair fraction at L = {depth}")
    a2.set_title(f"(b) Storage vs. architectural reachability at L = {depth}")
    _grid(a2)
    fig.suptitle("Figure 3. Score storage (theoretical; not measured GPU memory) against "
                 "reachability (possible dependencies; not accuracy).", fontsize=9.5, color=INK)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
