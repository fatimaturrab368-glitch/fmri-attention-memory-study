# 04 — Results and interpretation

Every number below is read from a saved output file, named in each section. Figures are
generated from those files by `make_figures.py`.

## 0. Original study: re-verified

`results/results.csv` (N = 256, w = 4, s = 2, one head, FP16, theoretical):

| Quantity | Value |
|---|---|
| Global attention score storage per layer | 131,072 B |
| Window attention score storage per layer | 2,048 B (shifted layer: 2,032 B by actual partition) |
| Reduction | 64× (= N/w) |
| Fixed-window reachable pairs | 1,024 / 65,536 = 1.5625 % |
| Shifted-window reachable pairs after 16 layers | 15,360 / 65,536 = 23.4375 % |

## A. Exact reachability sweep: `results/sweep/` (Fig. U2)

2,240 rows covering N ∈ {16, …, 256}, w ∈ {4, 8}, s ∈ {1, 2, w/2}, masked and cyclic
boundaries, and L = 1–32. The laws below hold in **every** row of the grid and are enforced by
tests (`tests/test_upgrade.py::TestSweepLaws`):

1. **The shift does not change how many pairs are reachable.** For masked boundaries the
   reachable-pair count is identical for every s at the same (N, w, L): 320 / 320 groups.
2. **Interior receptive fields grow by one window per layer:**
   max receptive field = min(w·L, N).
3. **The farthest dependency grows by w/2 per layer:**
   max |i − j| = min((w/2)(L − 1) + (w − 1) + [L even]·(w/2 − s), N − 1) (800 / 800 rows).
   The shift only adds a constant offset on shifted layers.
   *This corrects the original README's wording "grows by the shift": with s = w/2 the two
   explanations were indistinguishable; the sweep separates them.*
4. **The cyclic boundary removes the edge deficit but creates wraparound:**
   reachable fraction = min(w·L/N, 1) exactly (800 / 800 rows). The masked boundary (Swin/SwiFT)
   is lower at the edges. Example, N = 256, w = 4, L = 16: masked 0.234, cyclic 0.250.
5. **Connected components and reachability disagree.** Shifted windows form one connected
   component from L = 2, while only 3.1 % of pairs are reachable at N = 256. A closure-type
   measure would hugely overstate access. Fixed windows have exactly N/w components at every
   depth.

At N = 256 and L = 32, the reachable fraction is 0.4375 for w = 4 and 0.75 for w = 8 (masked),
against 0.0156 and 0.0312 for fixed windows.

## C. Learning experiment: `results/learning/` (Fig. U4)

81 runs:
- 3 seeds × 8 distances × {global, fixed, shifted}, plus 3 seeds × 3 distances for `none`.
- 204,676 parameters in every run.
- Learning rate selected on validation data (seed 100).

Tables: `summary_by_distance.csv`, `summary_by_reachability.csv`.

**H1 — necessity: supported without exception.** Positions with `R_L[i, i+d] = False` score
0.249–0.251 in every run, for every architecture (pooled per-seed means: fixed 0.2499,
shifted 0.2502, none 0.2500; SD ≤ 0.0002). This is guaranteed by construction, since the
output cannot depend on x_{i+d}, so it serves as a check on masks and leakage rather than a
discovery.

**H2 — reachable dependencies are learned: supported for d ≤ 6; budget-limited at d = 8, 9.**

| Architecture | Distances | Reachable / scored positions | Test acc. on reachable (mean, min over seeds) |
|---|---|---|---|
| global | 1–9, 24 | all | 1.000, 1.000 (d = 24: min 0.9999) |
| global | 10 | 54/54 | 0.991, 0.974 |
| fixed | 1 | 48/63 | 1.000, 1.000 |
| fixed | 2 | 32/62 | 1.000, 1.000 |
| shifted | 1, 2, 4, 6 | all | 1.000, 1.000 |
| shifted | 8 | 28/56 | **0.833**, 0.500 |
| shifted | 9 | 14/55 | **0.844**, 0.532 |

- Fixed windows at d = 1 (0.822 overall) and d = 2 (0.637) score **exactly** what the
  reachability split predicts: the reachable positions are solved and the rest are at chance.
  For example, (48·1 + 15·0.25)/63 = 0.821.
- **The shifted model at d = 8 and 9** reaches these positions only at the full depth L = 4.
  Within each seed it is all-or-nothing: every reachable position is learned (2 of 3 seeds), or
  every one is stuck at about 0.5 (d = 8 seed 2; d = 9 seed 1). Validation curves stay at chance
  for about 4,000 steps, then jump. The failing seeds were still rising at the 6,000-step limit.
- **Post-hoc diagnostic** (`results/learning_followup/`, `configs/learning_followup.json`, not
  part of the pre-specified protocol). The same 6 shifted runs with 18,000 steps reach
  0.997–1.000 on reachable positions in **6 of 6** runs, with unreachable positions at
  0.249–0.251. This matches the prediction written into the config before running. It also
  revealed **instability**: validation accuracy sometimes fell back to chance and recovered
  (d = 8, seed 1: 0.62 → 0.25 → 0.63). Best-validation checkpointing hides this from the test
  number.

**Interpretation.** In this controlled setting, exact reachability was **necessary**. In the
pre-specified 6,000-step budget it was **sufficient** for all dependencies reachable within
3 layers, and for 2 of 3 seeds at the full depth. Dependencies that use the entire depth budget
are learned latest and least stably. Reachability predicts *where* learning is possible; it does
not predict *how fast*.

Global attention needed no depth to reach d = 24, but it was the slowest to optimise at short
distances in the pilot (0.32 accuracy at 1,500 steps for d = 1). Local windows act as a helpful
positional inductive bias. Its single sub-perfect run (d = 10, 0.974) was also a late
checkpoint (step 5,500).

## B. GPU benchmark: `results/benchmark/` (Fig. U3)

RTX 4070 Ti SUPER, 16 GiB, Windows 11, PyTorch 2.5.1 + CUDA 12.1. Attention core only: B = 1,
H = 4, Dh = 32, FP16. Allocator capped at 95 % of VRAM. Median latency over 40 repeats (10 for
N > 8,192) after warm-up. FlashAttention is **not available** in this PyTorch build. The
memory-efficient SDPA kernel, which is also exact global attention, was measured instead.

| N = 16,384 | Theoretical scores | **Measured** peak extra memory, fwd / fwd+bwd | **Measured** fwd+bwd latency |
|---|---|---|---|
| Global, explicit (math) | 2,048 MiB | 4,100 / 8,200 MiB | 63.2 ms |
| Global, memory-efficient SDPA | 2,048 MiB (not materialised) | **4 / 37 MiB** | 17.4 ms |
| Fixed windows, w = 4 | 0.5 MiB | 5 / 20 MiB | 0.81 ms |
| Shifted windows, w = 4 | 0.5 MiB | 17 / 25 MiB | 1.41 ms |
| Fixed windows, w = 256 | 32 MiB | 68 / 136 MiB | 1.06 ms |
| Shifted windows, w = 256 | 32 MiB | 80 / 148 MiB | 1.44 ms |

At N = 32,768:
- explicit global attention: **OOM**, as was the SDPA math backend;
- memory-efficient SDPA: 8 / 73 MiB, 86.9 ms;
- shifted windows (w = 256): 160 / 296 MiB, 3.5 ms.

**H3 — supported, with an important qualification.**
1. **Theory versus measurement.** Forward peak memory of explicit global attention and of
   w = 256 windows is about **2×** the theoretical score bytes (scores and softmax output are
   both stored). For tiny w = 4 windows it is about 10× (5 MiB versus 0.5 MiB at N = 16,384),
   because reshapes and intermediates dominate. Shifted windows add roll copies on top.
   Theoretical score storage is a lower bound, not "GPU memory".
2. **The memory advantage largely disappears against fused exact kernels.** In the forward
   pass, the memory-efficient global kernel used less memory than every windowed implementation
   (4 MiB versus 5–80 MiB at N = 16,384), while keeping full one-layer reachability. In
   fwd+bwd it sits between them: 37 MiB, versus 20–25 MiB for w = 4 windows and 136–148 MiB for
   w = 256 windows.
3. **The compute and latency advantage remains.** At N = 16,384, shifted windows (w = 256) were
   **12×** faster than the memory-efficient global kernel and **44×** faster than explicit
   global attention (fwd+bwd).
4. At N ≤ 1,024 all mechanisms take 0.1–0.7 ms. That is launch-overhead dominated, and no
   architectural conclusion can be drawn from it. Small irregularities at large N (e.g. shifted
   w = 4 at N = 32,768 slightly faster than at 16,384) are of the order of run-to-run noise
   (≈ 1 ms).

A first benchmark run was invalidated: Windows silently spilled oversize allocations into
system RAM. It is preserved with an explanation in `results/benchmark_run1_invalid_sysmem_spill/`.
A second, capped run measured memory before warm-up, so one-time cuBLAS workspace allocation
inflated the first configuration (8.5 MiB for w = 4 at N = 256). Its files were overwritten
after the fix and are not kept. Only the third run is reported.

## Synthesis

| Architecture | Reachability (exact) | Learned (this task) | Measured cost |
|---|---|---|---|
| Fixed windows | never beyond w | exactly the reachable pairs | lowest |
| Shifted windows | grows by w per layer; farthest dependency by w/2 per layer | reachable pairs, slowly and less stably at full depth | low; 1.4–1.7× the fixed-window fwd+bwd latency at N ≥ 16,384 |
| Global (exact kernels) | all pairs, one layer | all pairs (budget permitting) | highest compute; memory O(N) only with fused kernels |

**One-sentence answer to the research question:** exact depth-limited reachability determined
*which* dependencies matched window-attention networks could learn (never beyond it, and within
it given enough training). Once a fused exact kernel is available, the main measured cost
difference between window and global attention on this GPU is compute time (12–44× at
N = 16,384), not memory.
