"""Option B: python run_benchmark.py --config configs/benchmark.json
Requires a CUDA GPU; run it alone (no other GPU jobs) for clean timings."""

import argparse
import csv
import json
import os
import platform
import subprocess
import time
import warnings

import torch

# SDPA prints a UserWarning on every call when a forced backend is unavailable;
# availability is recorded explicitly in environment.json instead.
warnings.filterwarnings("ignore", category=UserWarning, module="src.attention_ops")
warnings.filterwarnings("ignore", message=".*AccumulateGrad.*")

from src.benchmark import bench_one, sdpa_backend_support

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "benchmark.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "benchmark"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for the benchmark; nothing was measured.")
    cfg = json.load(open(args.config))
    dev = torch.device("cuda", torch.cuda.current_device())
    # On Windows (WDDM) the driver may silently spill allocations beyond dedicated VRAM
    # into shared system RAM, so "GPU memory" and latency would include host paging.
    # Capping the caching allocator turns that situation into a genuine, recorded OOM.
    frac = cfg.get("memory_fraction", 0.95)
    torch.cuda.set_per_process_memory_fraction(frac, dev)
    vram = torch.cuda.get_device_properties(dev).total_memory
    os.makedirs(args.out, exist_ok=True)
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        drv = ""
    env = {"gpu": torch.cuda.get_device_name(0), "driver": drv, "torch": torch.__version__,
           "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
           "os": platform.platform(),
           "sdpa_backends_available": sdpa_backend_support(dev, getattr(torch, cfg["dtype"])),
           "dedicated_vram_bytes": vram, "allocator_memory_fraction": frac,
           "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump({"config": cfg, "environment": env},
              open(os.path.join(args.out, "environment.json"), "w"), indent=2)
    print(env)

    rows = []
    # windowed mechanisms first at every N, so a large global allocation cannot
    # perturb (allocator state, clocks) the cheaper measurements that follow it
    order = sorted(cfg["mechanisms"], key=lambda m: 0 if "window" in m else 1)
    for n in cfg["N"]:
        for mech in order:
            ws = cfg["windows"] if "window" in mech else [0]
            for w in ws:
                r = bench_one(mech, n, w, cfg, dev)
                r["exceeds_dedicated_vram"] = any(
                    float(r.get(k, 0) or 0) > frac * vram
                    for k in ("peak_extra_mem_fwd_bytes", "peak_extra_mem_fwdbwd_bytes"))
                rows.append(r)
                print(f"N={n:>6} {mech:<22} w={w:<4} {r['status']:<6} "
                      f"fwd={r.get('fwd_ms_median', float('nan')):9.3f} ms  "
                      f"fwd+bwd={r.get('fwdbwd_ms_median', float('nan')):9.3f} ms  "
                      f"peak fwd={r.get('peak_extra_mem_fwd_bytes', 0) / 2**20:10.2f} MiB  "
                      f"theory={r['theoretical_score_bytes'] / 2**20:10.2f} MiB", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out, "benchmark.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
