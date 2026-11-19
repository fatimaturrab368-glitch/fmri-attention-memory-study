"""Option B: measured GPU latency and memory of single-layer attention cores.

Measurement protocol
  * one attention core (no projections / MLP), B x H x N x Dh, given dtype;
  * inputs allocated first; memory reported is the PEAK EXTRA allocation
    beyond the inputs: torch.cuda.max_memory_allocated() after
    reset_peak_memory_stats() minus memory_allocated() before the call;
  * warm-up iterations, then timing with CUDA events and synchronisation,
    reporting median and inter-quartile range over repeats;
  * out-of-memory is caught and recorded as status "OOM" (never estimated).
Theoretical score storage (H * bytes * sum of window sizes^2) is reported in a
separate column and is NOT the measured value.
"""

from __future__ import annotations

import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .attention_ops import global_math, global_sdpa, shifted_region_mask, shifted_window, window_attention

BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH}


def make_op(mech: str, n: int, w: int, device):
    if mech == "global_math":
        return global_math, None
    if mech.startswith("global_sdpa_"):
        return global_sdpa, BACKENDS[mech.removeprefix("global_sdpa_")]
    if mech == "fixed_window":
        return (lambda q, k, v: window_attention(q, k, v, w)), None
    if mech == "shifted_window":
        mask = shifted_region_mask(n, w, w // 2, device)
        return (lambda q, k, v: shifted_window(q, k, v, w, w // 2, mask)), None
    raise ValueError(mech)


def theoretical_score_bytes(mech, n, w, heads, bytes_per):
    if mech.startswith("global"):
        return heads * bytes_per * n * n
    return heads * bytes_per * (n // w) * w * w  # cyclic implementation materialises full windows


def _time(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    q = statistics.quantiles(ts, n=4)
    return statistics.median(ts), q[0], q[2]


def _peak_extra(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def bench_one(mech, n, w, cfg, device):
    dtype = getattr(torch, cfg["dtype"])
    B, H, Dh = cfg["batch"], cfg["heads"], cfg["head_dim"]
    row = {"mechanism": mech, "N": n, "window": w if "window" in mech else "",
           "batch": B, "heads": H, "head_dim": Dh, "dtype": cfg["dtype"],
           "theoretical_score_bytes": theoretical_score_bytes(
               mech, n, w, H * B, torch.finfo(dtype).bits // 8)}
    op, backend = make_op(mech, n, w, device)
    ctx = (lambda: sdpa_kernel([backend])) if backend is not None else None
    try:
        g = torch.Generator(device=device).manual_seed(0)
        q, k, v = (torch.randn(B, H, n, Dh, device=device, dtype=dtype, generator=g)
                   for _ in range(3))
        qg, kg, vg = (t.clone().requires_grad_(True) for t in (q, k, v))
        dout = torch.randn_like(q)

        def fwd():
            with torch.no_grad():
                if ctx:
                    with ctx():
                        return op(q, k, v)
                return op(q, k, v)

        def fwdbwd():
            if ctx:
                with ctx():
                    out = op(qg, kg, vg)
            else:
                out = op(qg, kg, vg)
            out.backward(dout)
            qg.grad = kg.grad = vg.grad = None

        reps = cfg["repeats"] if n <= 8192 else max(5, cfg["repeats"] // 4)
        # warm up BEFORE measuring memory: the first calls allocate one-time state
        # (cuBLAS workspace, kernel selection) that is not part of the operation's footprint
        for _ in range(2):
            fwd()
            fwdbwd()
        row["peak_extra_mem_fwd_bytes"] = _peak_extra(fwd)
        row["fwd_ms_median"], row["fwd_ms_q1"], row["fwd_ms_q3"] = _time(fwd, cfg["warmup"], reps)
        row["peak_extra_mem_fwdbwd_bytes"] = _peak_extra(fwdbwd)
        row["fwdbwd_ms_median"], row["fwdbwd_ms_q1"], row["fwdbwd_ms_q3"] = _time(
            fwdbwd, cfg["warmup"], reps)
        row["fwd_tokens_per_s"] = B * n / (row["fwd_ms_median"] / 1000)
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    except RuntimeError as e:  # e.g. backend not available for this config
        row["status"] = "unavailable: " + str(e).splitlines()[0][:120]
    finally:
        torch.cuda.empty_cache()
    return row


def sdpa_backend_support(device, dtype):
    """Which SDPA backends actually run on this machine (probe, small input)."""
    q = torch.randn(1, 4, 256, 32, device=device, dtype=dtype)
    out = {}
    for name, be in BACKENDS.items():
        try:
            with sdpa_kernel([be]):
                F.scaled_dot_product_attention(q, q, q)
            out[name] = True
        except RuntimeError:
            out[name] = False
    return out
