"""Aggregate results/learning/runs.csv into summary tables (mean, SD, min over seeds):
python summarize_learning.py"""

import csv
import os
import statistics as st

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "results", "learning")


def agg(v):
    v = [x for x in v if x == x]  # drop NaN
    if not v:
        return {"mean": "", "sd": "", "min": "", "n": 0}
    return {"mean": round(st.mean(v), 4), "sd": round(st.stdev(v), 4) if len(v) > 1 else 0.0,
            "min": round(min(v), 4), "n": len(v)}


def main():
    runs = list(csv.DictReader(open(os.path.join(D, "runs.csv"))))
    archs = list(dict.fromkeys(r["architecture"] for r in runs))
    by_d = []
    for a in archs:
        for d in sorted({int(r["distance"]) for r in runs if r["architecture"] == a}):
            rr = [r for r in runs if r["architecture"] == a and int(r["distance"]) == d]
            row = {"architecture": a, "distance": d,
                   "reachable_positions": rr[0]["n_reachable_positions"],
                   "scored_positions": int(rr[0]["n_reachable_positions"]) + int(rr[0]["n_unreachable_positions"])}
            for k in ("test_acc_all", "test_acc_reachable", "test_acc_unreachable", "val_acc"):
                for s, v in agg([float(r[k]) for r in rr]).items():
                    row[f"{k}_{s}"] = v
            row["best_steps"] = " ".join(r["best_step"] for r in rr)
            by_d.append(row)
    # pooled by reachability, per seed (position-weighted), then mean/SD over seeds
    by_r = []
    for a in archs:
        for key, nkey in (("test_acc_reachable", "n_reachable_positions"),
                          ("test_acc_unreachable", "n_unreachable_positions")):
            per_seed = []
            for s in sorted({r["seed"] for r in runs}):
                rr = [r for r in runs if r["architecture"] == a and r["seed"] == s and int(r[nkey]) > 0]
                if rr:
                    per_seed.append(sum(float(r[key]) * int(r[nkey]) for r in rr) /
                                    sum(int(r[nkey]) for r in rr))
            if per_seed:
                by_r.append({"architecture": a, "group": key.replace("test_acc_", ""),
                             **{f"acc_{k}": v for k, v in agg(per_seed).items()}})
    for name, rows in (("summary_by_distance.csv", by_d), ("summary_by_reachability.csv", by_r)):
        with open(os.path.join(D, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    for r in by_d:
        print(f"{r['architecture']:>7} d={r['distance']:<2} reach {r['reachable_positions']:>2}/{r['scored_positions']:<2} "
              f"all={r['test_acc_all_mean']}±{r['test_acc_all_sd']}  reachable={r['test_acc_reachable_mean']} "
              f"(min {r['test_acc_reachable_min']})  unreachable={r['test_acc_unreachable_mean']} "
              f"(min {r['test_acc_unreachable_min']})  best_steps={r['best_steps']}")
    print()
    for r in by_r:
        print(r)


if __name__ == "__main__":
    main()
