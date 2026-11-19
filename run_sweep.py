"""Option A: python run_sweep.py [--config configs/sweep.json]"""

import argparse
import csv
import json
import os

from src.sweep import run_sweep

ROOT = os.path.dirname(os.path.abspath(__file__))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "sweep.json"))
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = os.path.join(ROOT, "results", "sweep")
    os.makedirs(out, exist_ok=True)
    rows, dist_rows = run_sweep(cfg)
    write_csv(os.path.join(out, "reachability_sweep.csv"), rows)
    write_csv(os.path.join(out, "layers_to_distance.csv"), dist_rows)
    json.dump(cfg, open(os.path.join(out, "config_used.json"), "w"), indent=2)
    print(f"wrote {len(rows)} reachability rows and {len(dist_rows)} distance rows to {out}")


if __name__ == "__main__":
    main()
