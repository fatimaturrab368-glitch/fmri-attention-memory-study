"""Option C: python run_learning.py --config configs/learning.json [--out results/learning]"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA init (determinism)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from src.learning import train_one  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "learning.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "learning"))
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--seeds", type=int, nargs="*", help="override config seeds (parallel shards)")
    ap.add_argument("--archs", nargs="*", help="override config architectures")
    ap.add_argument("--merge", nargs="*", metavar="SHARD_DIR",
                    help="only concatenate CSVs of finished shard directories into --out")
    args = ap.parse_args()
    if args.merge:
        os.makedirs(args.out, exist_ok=True)
        for name in ("runs.csv", "per_position.csv", "curves.csv"):
            rows = []
            for d in args.merge:
                with open(os.path.join(d, name)) as f:
                    rows += list(csv.DictReader(f))
            write_csv(os.path.join(args.out, name), rows)
        cfg0 = json.load(open(os.path.join(args.merge[0], "config_used.json")))
        cfg0["config"]["seeds"] = sorted({int(r["seed"]) for r in rows})
        cfg0["config"]["architectures"] = list(dict.fromkeys(r["architecture"] for r in rows))
        cfg0["merged_from"] = [os.path.relpath(d, ROOT) for d in args.merge]
        json.dump(cfg0, open(os.path.join(args.out, "config_used.json"), "w"), indent=2)
        print("merged", len(args.merge), "shards into", args.out)
        return
    cfg = json.load(open(args.config))
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.archs:
        cfg["architectures"] = args.archs
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    torch.use_deterministic_algorithms(True)
    os.makedirs(args.out, exist_ok=True)

    env = {"python": platform.python_version(), "torch": torch.__version__,
           "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
           "deterministic_algorithms": True, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump({"config": cfg, "environment": env},
              open(os.path.join(args.out, "config_used.json"), "w"), indent=2)

    summaries, positions, curves = [], [], []
    lrs = cfg.get("lr_grid", [None])  # tuning mode: grid of learning rates
    per_arch = cfg.get("distances_per_arch", {})
    runs = [(a, d, s, lr) for s in cfg["seeds"] for a in cfg["architectures"]
            for d in per_arch.get(a, cfg["distances"]) for lr in lrs]
    for k, (a, d, s, lr) in enumerate(runs, 1):
        sm, pos, cur = train_one(cfg, a, d, s, device, lr=lr)
        for c in cur:
            c["lr"] = sm["lr"]
        summaries.append(sm); positions += pos; curves += cur
        print(f"[{k}/{len(runs)}] {a:>7} d={d:<2} seed={s} lr={sm['lr']:g}  val={sm['val_acc']:.3f} "
              f"test reach={sm['test_acc_reachable']:.3f} unreach={sm['test_acc_unreachable']:.3f} "
              f"({sm['train_time_s']}s)", flush=True)
        # write incrementally so partial results survive interruption
        write_csv(os.path.join(args.out, "runs.csv"), summaries)
        write_csv(os.path.join(args.out, "per_position.csv"), positions)
        write_csv(os.path.join(args.out, "curves.csv"), curves)


if __name__ == "__main__":
    main()
