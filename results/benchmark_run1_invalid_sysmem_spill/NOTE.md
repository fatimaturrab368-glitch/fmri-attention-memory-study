# Do not use these numbers

This is the first full benchmark run (2026-11-19 15:07). It is kept for transparency only.

At N = 32,768 the explicit global-attention runs (`global_math`, `global_sdpa_math`) recorded
peak allocations of 16,392 MiB and 53,312 MiB. The GPU has 16,376 MiB of dedicated VRAM. On
Windows (WDDM) the NVIDIA driver silently spilled those allocations into shared system RAM
instead of raising an out-of-memory error. Those rows therefore measure host paging, not GPU
memory or GPU compute (fwd+bwd 12.4 s).

The windowed measurements taken right after the spill also look perturbed: at N = 32,768,
fixed windows appeared about 8× slower than shifted windows.

Fix applied in `run_benchmark.py`:
- `torch.cuda.set_per_process_memory_fraction(0.95)`, so oversize configurations raise a genuine
  OOM;
- windowed mechanisms are measured before global ones at each N;
- an `exceeds_dedicated_vram` column is recorded.

The valid results are in `results/benchmark/`.
