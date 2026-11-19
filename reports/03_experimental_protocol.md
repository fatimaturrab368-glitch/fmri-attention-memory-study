# 03 — Experimental protocol

Three experiments, each with its own config, script and output folder. The original study
(`run.py`) is unchanged and still reproduces `results/results.csv` and `figures/fig1–3`
bit-for-bit.

| Exp. | Question answered | Config | Command | Output |
|---|---|---|---|---|
| A — exact reachability sweep | Which pairs *can* depend on each other, as a function of N, w, s, boundary and L? | `configs/sweep.json` | `python run_sweep.py` | `results/sweep/` |
| C — synthetic learning | Does a matched network *learn* exactly the reachable dependencies? | `configs/learning_tuning.json` → `configs/learning.json` | `python run_learning.py --config …` | `results/learning_tuning/`, `results/learning/` |
| B — GPU benchmark | What do the attention cores *measurably* cost? | `configs/benchmark.json` | `python run_benchmark.py` | `results/benchmark/` |

## A. Exact reachability sweep

- N ∈ {16, 32, 64, 128, 256}; w ∈ {4, 8}; s ∈ {1, 2, w/2} with 0 < s < w; depth 1–32 (the
  report quotes 2, 4, 8, 16, 32).
- Boundary conditions:
  - **masked**: Swin/SwiFT roll + region mask, which equals the non-cyclic partition;
  - **cyclic**: roll without the mask, i.e. a ring with wraparound. This is a comparison
    condition, not SwiFT.
- Metrics:
  - reachable ordered-pair fraction (self-pairs included);
  - min / mean / max receptive field;
  - max |i − j|;
  - number of connected components of the undirected union graph (union-find; a closure-type
    measure, reported to contrast with depth-limited reachability);
  - for each distance d, the first depth at which *some* / *all* pairs at distance d are
    reachable;
  - theoretical score bytes.
- No randomness, so there is nothing to seed.

## C. Synthetic long-range learning experiment

**Task (paired modular sum).** Input x ∈ {0,1,2,3}^64 i.i.d. uniform. Target at every
position i < 64 − d:

    y_i = (x_i + x_{i+d}) mod 4

Positions i ≥ 64 − d are not scored. y_i is marginally independent of x_i and of x_{i+d}
(tested). A model that cannot combine **both** tokens is at chance (0.25). There is no
single-token shortcut.

**Models.** The same pre-LN Transformer for every condition:
- 4 layers, d_model 64, 4 heads, MLP ratio 4;
- learned absolute positional embedding (as SwiFT);
- 204,676 parameters in every condition (computed; also recorded per run in `runs.csv`).

Conditions differ **only** in the per-layer Boolean attention mask:

| Condition | Mask | Role |
|---|---|---|
| `global` | all pairs | upper bound on access |
| `fixed` | w = 4 blocks every layer | no cross-window access |
| `shifted` | w = 4 blocks on odd layers, masked shift s = 2 on even layers (SwiFT schedule) | test condition |
| `none` | self only | no-communication control |

The per-layer masks are the exact `A_l` matrices of the reachability model. A unit test shows
that the network's Jacobian (logits_i with respect to embedding_j) is non-zero **iff**
`R_L[i,j]`.

**Distances.** d ∈ {1, 2, 4, 6, 8, 9, 10, 24} for `global`, `fixed` and `shifted`;
d ∈ {1, 8, 24} for the `none` control, whose chance-level result is guaranteed by construction.
With L = 4, shifted windows reach at most |i − j| = 9. Exact reachable / scored positions
(from `n_reachable_positions` in `runs.csv`):
- d = 1, 2, 4, 6: all positions (63/63, 62/62, 60/60, 58/58);
- d = 8: 28/56; d = 9: 14/55 (partly reachable; each such pair becomes reachable only at
  depth 4);
- d = 10, 24: none.

Fixed windows reach 48/63 positions at d = 1 and 32/62 at d = 2, and none at d ≥ 4. Global
reaches every position at every d. (An earlier draft of this protocol said "d = 2–9 partly
reachable"; that was wrong and is corrected here from the computed masks.) The grid was reduced
from 10 to 8 distances because of compute (about 75 s of GPU time per run; 81 runs in total).
The dropped distances (3, 12) add no new reachability regime.

**Data.**
- train 20,000 / validation 2,000 / test 10,000 sequences;
- each split is drawn from its own RNG stream (seed·10 + {1, 2, 3});
- tests confirm the splits share no sequence and that data generation is deterministic.

**Training (identical for all conditions).**
- AdamW, weight decay 0.01, linear warm-up over 100 steps then constant learning rate;
- batch 256, 6,000 steps;
- validation accuracy every 250 steps;
- the checkpoint with the best **validation** accuracy is kept and evaluated once on the test
  set.

**Hyper-parameter selection.**
- The learning rate is chosen per architecture from {3e-4, 1e-3, 3e-3}, by mean **validation**
  accuracy over d ∈ {1, 8}.
- Selection uses tuning seed **100**, which is never used again. Test accuracy plays no part.
- Selected values (from `results/learning_tuning/*/runs.csv`, mean validation accuracy at
  lr = 3e-4 / 1e-3 / 3e-3):

| Arch | 3e-4 | 1e-3 | 3e-3 | Selected |
|---|---|---|---|---|
| global | 0.658 | 1.000 | 1.000 | **1e-3** (tie → default) |
| fixed | 0.536 | 0.537 | 0.537 | **3e-3** |
| shifted | 0.813 | 0.813 | 0.813 | **1e-3** |
| none | 0.252 | 0.252 | 0.252 | **3e-3** |

- The tuning runs also printed test accuracy. It was not used for any decision.
- **Budget note.** A 1,500-step pilot left global attention at 0.32 validation accuracy even at
  d = 1, an optimisation-budget effect. The budget was therefore raised to 6,000 steps for all
  conditions **before** any final run. In tuning, the shifted model at d = 8 selected its
  checkpoint at step 5,750 of 6,000. Positions that are reachable only through multi-layer
  chains learn late, so 6,000 steps is near the edge for that condition.

**Seeds.** Final runs use seeds 0, 1, 2. The seed controls model initialisation, batch order and
data. Reported values are mean ± SD over seeds, with individual seeds plotted.

**Primary metric.** Test accuracy grouped by exact reachability `R_L[i, i+d]` (per position),
plus accuracy per distance.

**Determinism.** `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
and all seeds recorded in `config_used.json`.

## B. GPU benchmark

- One attention core per call, with no projections or MLP:
  - B = 1, H = 4, head dim 32, FP16;
  - N ∈ {256, 1,024, 4,096, 16,384, 32,768}.
- Mechanisms:
  - `global_math`: explicit QKᵀ, N × N materialised;
  - `global_sdpa_{flash, efficient, math}`: PyTorch SDPA with the backend forced; availability
    is probed and recorded;
  - `fixed_window` and `shifted_window`: Swin-style roll + mask; w ∈ {4, 256}. Here 256 is the
    token count of SwiFT's 4×4×4×4 window.
- Before timing, each windowed implementation is checked against a dense masked reference (unit
  test).
- Measurement:
  - inputs allocated first;
  - peak **extra** allocated memory, from `max_memory_allocated` after
    `reset_peak_memory_stats`, minus the baseline;
  - 10 warm-up calls, then 40 timed repeats (10 for N > 8,192) with CUDA events and
    synchronisation; median and IQR reported;
  - OOM is recorded as OOM, never estimated.
- Run alone on the GPU, with no other jobs.
- Theoretical score bytes are reported in a separate column and plotted in a separate panel.

## Leakage and fairness checklist

| Risk | Control |
|---|---|
| Label leakage / shortcuts | Target independent of each single token (tested); `none` control must be at chance |
| Train/test overlap | Disjoint RNG streams; overlap test |
| Tuning on test | Learning rate chosen on validation with a separate seed; test evaluated once per run |
| Capacity mismatch | Identical networks; parameter counts equal (tested) |
| Mask errors | Jacobian test; windowed ops match the dense reference |
| Cherry-picking | All seeds reported; per-seed points plotted |
| Benchmark contamination | Benchmark run with no concurrent GPU jobs; warm-up and synchronisation |
