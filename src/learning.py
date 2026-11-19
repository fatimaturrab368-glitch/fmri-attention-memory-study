"""Option C: train matched masked Transformers on the paired-modular-sum task.

Protocol per run (architecture, distance d, seed):
  * data: fresh train / val / test splits from disjoint RNG streams;
  * identical model, optimiser, schedule and step budget for every architecture;
  * model selection (checkpoint) uses VALIDATION accuracy only;
  * the selected checkpoint is evaluated once on the test split, per position,
    and positions are grouped by exact reachability R_L[i, i + d].
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .reachability import reachability
from .synthetic_task import IGNORE, make_dataset
from .torch_models import MaskedTransformer, build_masks, n_params


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def per_position_accuracy(model, x, y, batch=1024):
    model.eval()
    correct = torch.zeros(x.shape[1], device=x.device)
    count = torch.zeros(x.shape[1], device=x.device)
    for a in range(0, x.shape[0], batch):
        xb, yb = x[a:a + batch], y[a:a + batch]
        pred = model(xb).argmax(-1)
        valid = yb != IGNORE
        correct += ((pred == yb) & valid).sum(0)
        count += valid.sum(0)
    model.train()
    return (correct / count.clamp(min=1)).cpu().numpy(), count.cpu().numpy()


def overall_accuracy(model, x, y):
    acc, cnt = per_position_accuracy(model, x, y)
    return float((acc * cnt).sum() / cnt.sum())


def lr_for(cfg: dict, arch: str) -> float:
    """Per-architecture learning rate selected on VALIDATION data (tuning seed),
    falling back to the shared default."""
    return float(cfg.get("lr_per_arch", {}).get(arch, cfg["lr"]))


def train_one(cfg: dict, arch: str, d: int, seed: int, device: torch.device, lr=None):
    n, L, w, s = cfg["n_tokens"], cfg["depth"], cfg["window"], cfg["shift"]
    lr = lr_for(cfg, arch) if lr is None else float(lr)
    data = make_dataset(cfg, d, seed)
    t = {k: (torch.from_numpy(v[0]).to(device), torch.from_numpy(v[1]).to(device))
         for k, v in data.items()}

    set_seed(seed)
    model = MaskedTransformer(cfg["vocab"], n, cfg["d_model"], cfg["n_heads"], L,
                              build_masks(arch, n, L, w, s), cfg["vocab"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"],
                            fused=device.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / cfg["warmup_steps"]))
    gen = torch.Generator(device="cpu").manual_seed(seed)

    xtr, ytr = t["train"]
    best = (-1.0, 0, None)
    curve = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for step in range(1, cfg["max_steps"] + 1):
        idx = torch.randint(0, xtr.shape[0], (cfg["batch_size"],), generator=gen).to(device)
        logits = model(xtr[idx])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ytr[idx].reshape(-1),
                               ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            va = overall_accuracy(model, *t["val"])
            curve.append({"step": step, "train_loss": float(loss), "val_acc": va})
            if va > best[0]:
                best = (va, step, copy.deepcopy(model.state_dict()))
    train_time = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else float("nan")

    model.load_state_dict(best[2])
    acc, cnt = per_position_accuracy(model, *t["test"])
    R = reachability(arch, n, L, w, s)[L]
    pos = np.arange(n - d)
    reach = R[pos, pos + d]
    acc, cnt = acc[: n - d], cnt[: n - d]

    def grp(mask):
        return float((acc[mask] * cnt[mask]).sum() / cnt[mask].sum()) if mask.any() else float("nan")

    summary = {
        "architecture": arch, "distance": d, "seed": seed, "lr": lr, "params": n_params(model),
        "best_step": best[1], "val_acc": best[0],
        "test_acc_all": grp(np.ones_like(reach)),
        "test_acc_reachable": grp(reach), "test_acc_unreachable": grp(~reach),
        "n_reachable_positions": int(reach.sum()), "n_unreachable_positions": int((~reach).sum()),
        "train_time_s": round(train_time, 2), "peak_train_mem_mib": round(peak_mb, 1),
        "device": str(device),
    }
    positions = [{"architecture": arch, "distance": d, "seed": seed, "position": int(i),
                  "reachable": bool(r), "test_acc": float(a), "n_test": int(c)}
                 for i, r, a, c in zip(pos, reach, acc, cnt)]
    curves = [dict(architecture=arch, distance=d, seed=seed, **c) for c in curve]
    return summary, positions, curves
