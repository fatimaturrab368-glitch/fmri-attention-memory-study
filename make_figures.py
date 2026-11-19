"""Generate upgrade figures from SAVED outputs only: python make_figures.py
(figures/fig1-3 of the original study are produced by run.py and left unchanged)."""

import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.reachability import layer_adjacency, reachability  # noqa: E402
from src.visualize import GRID, INK, MUTED, REACH_CMAP  # noqa: E402  (shared style)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures", "upgrade")
COL = {"global": "#2a78d6", "fixed": "#eb6834", "shifted": "#1baf7a", "none": "#52514e"}
MARK = {"global": "o", "fixed": "s", "shifted": "^", "none": "x"}
LS = {"global": "-", "fixed": "--", "shifted": "-.", "none": ":"}
LABEL = {"global": "Global", "fixed": "Fixed windows", "shifted": "Shifted windows",
         "none": "No cross-token attention (control)"}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def grid(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig U1
def fig_architecture(n=16, w=4, s=2):
    archs = ("global", "fixed", "shifted")
    cols = [("A_1 (layer 1 mask)", lambda a: layer_adjacency(a, 1, n, w, s)),
            ("A_2 (layer 2 mask)", lambda a: layer_adjacency(a, 2, n, w, s)),
            ("R_2 = A_2 A_1", lambda a: reachability(a, n, 2, w, s)[2]),
            ("R_4", lambda a: reachability(a, n, 4, w, s)[4])]
    fig, axes = plt.subplots(3, 4, figsize=(8.6, 6.6), constrained_layout=True)
    for r, a in enumerate(archs):
        for c, (title, f) in enumerate(cols):
            ax = axes[r, c]
            M = f(a)
            ax.imshow(M, cmap=REACH_CMAP, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([0, 4, 8, 12]); ax.set_yticks([0, 4, 8, 12])
            ax.tick_params(labelsize=7, length=2)
            ax.set_title(f"{title}\n{M.mean():.0%} of pairs", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"{LABEL[a]}\nquery / output i", fontsize=8.5)
            if r == 2:
                ax.set_xlabel("key / input j", fontsize=8.5)
    fig.suptitle(f"Fig. U1  Attention masks per layer and resulting reachability "
                 f"(N={n}, w={w}, s={s}; shifted layer uses the masked, non-cyclic boundary)",
                 fontsize=9.5, color=INK)
    fig.savefig(os.path.join(OUT, "figU1_architecture.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig U2
def fig_sweep():
    rows = read(os.path.join(ROOT, "results", "sweep", "reachability_sweep.csv"))
    dist = read(os.path.join(ROOT, "results", "sweep", "layers_to_distance.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)
    n = 256
    ax = axes[0]
    for w, col in ((4, "#1baf7a"), (8, "#6250d6")):
        for s, mk in sorted({(int(r["shift"]), None) for r in rows
                             if r["architecture"] == "shifted" and int(r["window"]) == w}):
            for b, ls in (("masked", "-"), ("cyclic", ":")):
                sel = sorted((int(r["depth"]), float(r["reachable_fraction"])) for r in rows
                             if int(r["N"]) == n and int(r["window"]) == w and r["shift"] == str(s)
                             and r["boundary"] == b and r["architecture"] == "shifted")
                x, y = zip(*sel)
                lab = f"shifted w={w}, {b}" if s == w // 2 else None
                ax.plot(x, y, color=col, ls=ls, lw=1.4, label=lab, alpha=0.9)
        sel = sorted((int(r["depth"]), float(r["reachable_fraction"])) for r in rows
                     if int(r["N"]) == n and int(r["window"]) == w and r["architecture"] == "fixed")
        x, y = zip(*sel)
        ax.plot(x, y, color=col, ls="--", lw=1.2, label=f"fixed w={w}")
    ax.set_xlabel("depth L (layers)"); ax.set_ylabel("reachable ordered pairs / N²")
    ax.set_title(f"(a) N = {n}: all shifts s overlap exactly", fontsize=9)
    ax.legend(frameon=False, fontsize=7); grid(ax)

    ax = axes[1]
    for w, col in ((4, "#1baf7a"), (8, "#6250d6")):
        shifts = sorted({int(r["shift"]) for r in rows
                         if r["architecture"] == "shifted" and int(r["window"]) == w})
        for s, ls in zip(shifts, ("-", "--", ":")):
            sel = sorted((int(r["depth"]), int(r["max_token_distance"])) for r in rows
                         if int(r["N"]) == n and int(r["window"]) == w and int(r["shift"]) == s
                         and r["boundary"] == "masked" and r["architecture"] == "shifted")
            x, y = zip(*sel)
            ax.plot(x, y, color=col, ls=ls, lw=1.4, label=f"w={w}, s={s}")
    ax.set_xlabel("depth L (layers)"); ax.set_ylabel("max |i − j| reachable (tokens)")
    ax.set_title("(b) Farthest dependency: slope w/2 per layer; s only shifts offset", fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=2); grid(ax)

    ax = axes[2]
    for w, col in ((4, "#1baf7a"), (8, "#6250d6")):
        s = w // 2
        sel = sorted((int(r["distance"]), int(r["layers_all_pairs"])) for r in dist
                     if int(r["N"]) == n and int(r["window"]) == w and int(r["shift"]) == s
                     and r["boundary"] == "masked" and r["architecture"] == "shifted"
                     and r["layers_all_pairs"] != "")
        x, y = zip(*sel)
        ax.plot(x, y, color=col, marker="^", lw=1.4, label=f"shifted w={w}, s={s}")
    ax.axhline(1, color=COL["global"], lw=1.4, label="global (always L = 1)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("token distance d"); ax.set_ylabel("layers until ALL pairs at distance d reachable")
    ax.set_title("(c) Depth needed to connect distance d (N = 256, L ≤ 32)\n"
                 "missing point = not connected within 32 layers; fixed windows never connect d ≥ w",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=7); grid(ax)
    fig.suptitle("Fig. U2  Exact reachability sweep (theoretical; possible dependencies, "
                 "not learned information)", fontsize=9.5, color=INK)
    fig.savefig(os.path.join(OUT, "figU2_reachability_sweep.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig U3
def fig_efficiency():
    path = os.path.join(ROOT, "results", "benchmark", "benchmark.csv")
    if not os.path.exists(path):
        print("benchmark results missing; Fig U3 skipped")
        return
    rows = read(path)
    env = json.load(open(os.path.join(ROOT, "results", "benchmark", "environment.json")))["environment"]
    series = defaultdict(list)
    for r in rows:
        key = r["mechanism"] + (f" (w={r['window']})" if r["window"] else "")
        series[key].append(r)
    style = {"global_math": ("#2a78d6", "o", "-"), "global_sdpa_efficient": ("#6250d6", "D", "-"),
             "global_sdpa_flash": ("#e87ba4", "P", "-"), "global_sdpa_math": ("#52514e", "v", ":"),
             "fixed_window": ("#eb6834", "s", "--"), "shifted_window": ("#1baf7a", "^", "-.")}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for key, rs in series.items():
        mech = rs[0]["mechanism"]
        col, mk, ls = style[mech]
        if "w=256" in key:
            ls = (0, (1, 1)); mk = mk.lower() if mk.isalpha() else mk
        N = np.array([int(r["N"]) for r in rs])
        th = np.array([float(r["theoretical_score_bytes"]) for r in rs]) / 2**20
        axes[0].plot(N, th, color=col, marker=mk, ls=ls, lw=1.3, ms=4, label=key)
        ok = [r for r in rs if r["status"] == "ok"]
        if ok:
            Nk = np.array([int(r["N"]) for r in ok])
            mem = np.array([float(r["peak_extra_mem_fwdbwd_bytes"]) for r in ok]) / 2**20
            med = np.array([float(r["fwdbwd_ms_median"]) for r in ok])
            lo = med - np.array([float(r["fwdbwd_ms_q1"]) for r in ok])
            hi = np.array([float(r["fwdbwd_ms_q3"]) for r in ok]) - med
            axes[1].plot(Nk, mem, color=col, marker=mk, ls=ls, lw=1.3, ms=4, label=key)
            axes[2].errorbar(Nk, med, yerr=[lo, hi], color=col, marker=mk, ls=ls, lw=1.3, ms=4,
                             capsize=2, label=key)
    for ax, ttl, yl in ((axes[0], "(a) THEORETICAL score storage (H·2·Σn²)", "MiB (theoretical)"),
                        (axes[1], "(b) MEASURED peak extra GPU memory, fwd+bwd", "MiB (measured)"),
                        (axes[2], "(c) MEASURED fwd+bwd latency (median, IQR)", "ms (measured)")):
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("sequence length N (tokens)"); ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=9); grid(ax)
    oom = sorted({(r["mechanism"], int(r["N"])) for r in rows if r["status"] == "OOM"})
    unavailable = sorted({r["mechanism"] for r in rows if r["status"].startswith("unavailable")})
    note = ("Not plotted in (b, c): OOM (fwd+bwd, allocator capped at 95% of 16 GiB VRAM): " +
            ", ".join(f"{m} N={n}" for m, n in oom) if oom else "no OOM") + \
           ("; not available in this PyTorch build: " + ", ".join(unavailable) if unavailable else "") + \
           ". Measured memory = peak allocation beyond inputs, after warm-up."
    fig.text(0.5, -0.03, note, ha="center", fontsize=7.5, color=MUTED)
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    fig.suptitle(f"Fig. U3  Single-layer attention core, B=1, H=4, Dh=32, fp16 on {env['gpu']} "
                 f"(torch {env['torch']}). Panel (a) is calculated; (b, c) are measured.",
                 fontsize=9.5, color=INK)
    fig.savefig(os.path.join(OUT, "figU3_efficiency.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig U4
def fig_learning():
    d0 = os.path.join(ROOT, "results", "learning")
    if not os.path.exists(os.path.join(d0, "runs.csv")):
        print("learning results missing; Fig U4 skipped")
        return
    runs = read(os.path.join(d0, "runs.csv"))
    pos = read(os.path.join(d0, "per_position.csv"))
    cfg = json.load(open(os.path.join(d0, "config_used.json")))["config"]
    archs = cfg["architectures"]
    has_fu = os.path.exists(os.path.join(ROOT, "results", "learning_followup", "runs.csv"))
    fig, axes = plt.subplots(1, 4 if has_fu else 3, figsize=(19 if has_fu else 14.5, 4.4),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.2, 1, 1.3, 0.75] if has_fu else [1.2, 1, 1.3]})

    ax = axes[0]
    for a in archs:
        ds = sorted({int(r["distance"]) for r in runs if r["architecture"] == a})
        acc = [[float(r["test_acc_all"]) for r in runs if r["architecture"] == a
                and int(r["distance"]) == d] for d in ds]
        m = np.array([np.mean(v) for v in acc]); sd = np.array([np.std(v, ddof=1) if len(v) > 1 else 0
                                                                 for v in acc])
        ax.errorbar(ds, m, yerr=sd, color=COL[a], marker=MARK[a], ls=LS[a], lw=1.4, ms=5,
                    capsize=3, label=LABEL[a])
        for d, v in zip(ds, acc):
            ax.scatter([d] * len(v), v, color=COL[a], s=7, alpha=0.45, zorder=3)
    ax.axhline(1 / cfg["vocab"], color=MUTED, lw=0.8, ls=":")
    ax.text(ds[-1], 1 / cfg["vocab"] + 0.015, "chance", fontsize=7, color=MUTED, ha="right")
    ax.set_xlabel("dependency distance d (tokens)"); ax.set_ylabel("test accuracy (all scored positions)")
    ax.set_title(f"(a) Test accuracy vs distance (mean ± SD, {len(cfg['seeds'])} seeds; dots = seeds)",
                 fontsize=9)
    ax.set_ylim(0, 1.03); ax.legend(frameon=False, fontsize=7); grid(ax)

    ax = axes[1]
    width = 0.38
    for k, a in enumerate(archs):
        for g, (key, hatch) in enumerate((("test_acc_reachable", ""), ("test_acc_unreachable", "///"))):
            vals = [float(r[key]) for r in runs if r["architecture"] == a and r[key] != "nan"]
            # pool per seed across distances, weighted by number of positions
            nkey = "n_reachable_positions" if g == 0 else "n_unreachable_positions"
            per_seed = []
            for s in cfg["seeds"]:
                rr = [r for r in runs if r["architecture"] == a and int(r["seed"]) == s
                      and int(r[nkey]) > 0]
                if rr:
                    per_seed.append(sum(float(r[key]) * int(r[nkey]) for r in rr) /
                                    sum(int(r[nkey]) for r in rr))
            if not per_seed:
                continue
            x = k + (g - 0.5) * width
            ax.bar(x, np.mean(per_seed), width * 0.92, color=COL[a], alpha=1 if g == 0 else 0.45,
                   hatch=hatch, edgecolor="white",
                   yerr=np.std(per_seed, ddof=1) if len(per_seed) > 1 else None, capsize=3)
            ax.text(x, np.mean(per_seed) + 0.03, f"{np.mean(per_seed):.2f}", ha="center", fontsize=7)
    ax.axhline(1 / cfg["vocab"], color=MUTED, lw=0.8, ls=":")
    short = {"global": "Global", "fixed": "Fixed", "shifted": "Shifted", "none": "None\n(control)"}
    ax.set_xticks(range(len(archs)), [short[a] for a in archs], fontsize=8)
    ax.set_ylabel("test accuracy, pooled over distances")
    ax.set_title("(b) Positions grouped by exact reachability R_L[i, i+d]\n"
                 "solid = reachable, hatched = unreachable (mean ± SD over seeds)", fontsize=9)
    ax.set_ylim(0, 1.1); grid(ax)

    ax = axes[2]
    d_show = cfg.get("figure_distance", 8)
    sel = [p for p in pos if p["architecture"] == "shifted" and int(p["distance"]) == d_show]
    n_ok = 0
    for k, s in enumerate(sorted({p["seed"] for p in sel})):
        ps = [p for p in sel if p["seed"] == s]
        P = np.array([int(p["position"]) for p in ps])
        A = np.array([float(p["test_acc"]) for p in ps])
        Rr = np.array([p["reachable"] == "True" for p in ps])
        n_ok += int(A[Rr].mean() > 0.9)
        off = (k - 1) * 0.25
        ax.scatter(P[Rr] + off, A[Rr], s=10, marker="os^"[k], color=COL["shifted"],
                   label=f"seed {s}: R_L[i,i+d] = True")
        ax.scatter(P[~Rr] + off, A[~Rr], s=10, marker="os^"[k], color="#a5a49d",
                   label="R_L[i,i+d] = False" if k == 0 else None)
    ax.axhline(1 / cfg["vocab"], color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("output position i"); ax.set_ylabel("test accuracy (each seed)")
    ax.set_title(f"(c) Shifted windows, d = {d_show}, 6,000 steps: unreachable positions at chance\n"
                 f"in every seed; reachable positions learned in {n_ok} of 3 seeds", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2)
    grid(ax)

    fu = os.path.join(ROOT, "results", "learning_followup", "runs.csv")
    if os.path.exists(fu):
        ax = axes[3]
        fr = read(fu)
        for k, d in enumerate((8, 9)):
            for j, (src, lab) in enumerate(((runs, "6,000 steps (main)"), (fr, "18,000 steps (post hoc)"))):
                v = [float(r["test_acc_reachable"]) for r in src
                     if r["architecture"] == "shifted" and int(r["distance"]) == d]
                x = k + (j - 0.5) * 0.36
                ax.bar(x, np.mean(v), 0.33, color=COL["shifted"], alpha=1 if j else 0.45,
                       label=lab if k == 0 else None, edgecolor="white")
                ax.scatter([x] * len(v), v, color=INK, s=12, zorder=3)
        ax.axhline(1 / cfg["vocab"], color=MUTED, lw=0.8, ls=":")
        ax.set_xticks([0, 1], ["d = 8", "d = 9"])
        ax.set_ylabel("test accuracy on REACHABLE positions")
        ax.set_title("(d) Post-hoc diagnostic: same runs with 3x budget\n(bars = mean, dots = seeds)",
                     fontsize=9)
        ax.set_ylim(0, 1.08); ax.legend(frameon=False, fontsize=7, loc="lower right"); grid(ax)
    fig.suptitle(f"Fig. U4  Paired modular sum y_i = (x_i + x_(i+d)) mod {cfg['vocab']}, "
                 f"N = {cfg['n_tokens']}, L = {cfg['depth']}, w = {cfg['window']}, s = {cfg['shift']}; "
                 f"identical networks differing only in attention masks", fontsize=9.5, color=INK)
    fig.savefig(os.path.join(OUT, "figU4_learning.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_architecture()
    fig_sweep()
    fig_efficiency()
    fig_learning()
    print("figures written to", OUT)
