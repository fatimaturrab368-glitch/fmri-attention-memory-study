# 01 — Audit of the existing project

Date: 2026-11-19. Scope: everything in the repository before the upgrade (`src/`, `tests/`,
`run.py`, `results/results.csv`, `results/swift_scale_illustration.csv`, `figures/fig1–3`,
`README.md`). The locally available papers (SwiFT, Swin, FlashAttention PDFs) and the three
official repositories at pinned commits were used as references.

## 1. What was actually implemented

| Component | Nature | Status |
|---|---|---|
| Boolean attention adjacency per layer (global / fixed windows / alternating shifted windows), w = 4, s = 2 | exact combinatorial model | verified |
| Depth-limited reachability `R_0 = I`, `R_l = A_l ⊙ R_(l−1)`, L = 1…16, N ∈ {16, 32, 64, 128, 256} | exact computation | verified |
| FP16 attention-score bytes (`2·N²`, `2·Σ n_i²`) | **theoretical calculation**, nothing measured | verified |
| SwiFT-scale storage illustration (81,920 tokens, 320× reduction) | theoretical arithmetic | verified |
| Learning experiments, GPU measurements | **none existed** | n/a |

## 2. Reported findings: re-run and checked

`python run.py` was re-run and every value was read back from `results/results.csv`:

| Reported | CSV value | Verdict |
|---|---|---|
| 256 tokens, global score storage 131,072 B/layer | `2·256² = 131,072` | **Reproduced** (theoretical, 1 head, FP16) |
| Window storage 2,048 B/layer | `2·64·4² = 2,048` (regular layer); shifted layer 2,032 B by actual partition, 2,048 B as Swin implements it | **Reproduced** (2,048 = peak per layer) |
| 64× reduction | 131,072 / 2,048 = 64 = N/w | **Reproduced** |
| Fixed-window reachability ≈ 1.6 % | 1,024 / 65,536 = **1.5625 %** exactly | Reproduced (approximate wording) |
| Shifted reachability after 16 layers ≈ 23.4 % | 15,360 / 65,536 = **23.4375 %** exactly | Reproduced |

Two runs produce byte-identical CSVs. All 16 tests pass.

## 3. Definitions (answers to the audit questions)

| Question | Answer (verified in code) |
|---|---|
| Measured or theoretical? | All byte values are **theoretical** score-matrix counts. Nothing was measured on hardware. |
| Bytes computed correctly? | Yes. `test_6` checks the analytic formula against an explicit count of unmasked entries in each partition. |
| Attention heads | **One head only.** Score storage scales linearly with heads H; the reduction factor is unchanged. Now stated explicitly (see fixes). |
| Fixed vs shifted comparable? | Yes. Same w, same N, same depth, and identical regular layers. They differ only on even layers. |
| Cyclic shift, masking or non-cyclic? | The shifted layer is the **non-cyclic** partition `[0,s),[s,s+w),…`. A test proves it is identical to Swin/SwiFT's cyclic roll + region mask. An unmasked cyclic version is used only as a negative control. |
| Reachability definition | Ordered pairs (i, j): output i after L layers can depend on input j. **Self-pairs counted** (denominator N²). |
| Ordered or unordered? | Ordered. The matrices are symmetric here because every window partition is symmetric, so the counts coincide. |
| Cumulative across layers? | Depth-limited product, not a union. Every layer has self-loops, so `R_L ⊇ R_(L−1)` (tested). The product therefore equals the cumulative union. |
| Respects layer order? | Yes. `R_l = A_l R_(l−1)`. A test shows `A₂A₁ ≠ A₁A₂` and matches a hand-derived N = 8 example. |
| Invalid boundary communication? | No. Tests show no 0 ↔ N−1 edge. A finite-difference test on a real masked-softmax network shows zero dependence outside `R_L`. |
| Reproducible? | Yes. Deterministic, no randomness in the experiment. |

## 4. Bugs and issues

No numerical or mathematical bugs were found. Issues found, by priority:

| # | Issue | Severity | Fix |
|---|---|---|---|
| 1 | The README (earlier version) said reach grows "by the shift, not the window size". Interior tokens actually gain w = 4 tokens per layer on alternating sides; only max\|i−j\| grows by 2 per layer. | Overstatement (already fixed in the previous audit) | Rewritten; exact laws locked into tests |
| 2 | Only one configuration (w = 4, s = 2, non-cyclic boundary) was studied, so "grows by s" and "grows by w/2" cannot be told apart. | Scope limitation | **Option A sweep** over w, s and boundary (this upgrade) |
| 3 | No evidence connects reachability to what a network can **learn**. | Major evidential gap | **Option C learning experiment** (this upgrade) |
| 4 | Storage was theoretical only; actual GPU memory and latency were unknown. | Evidential gap | **Option B benchmark**, kept strictly separate from theoretical numbers |
| 5 | The "single head" assumption appeared only in docstrings. | Clarity | Stated in reports and README |
| 6 | The validity of the "1D = temporal slice of 4D" claim rested on an argument, not a test. | Unverified claim | New test: 4D Swin-style mask equals the Kronecker product of 1D masks on a small grid |
| 7 | Not a git repository. Third-party PDFs sit in the project folder. | Publication hygiene | `.gitignore` excludes `*.pdf`; papers are linked, not redistributed |

## 5. Assumptions that remain

One head; FP16 scores only; batch 1; binary (hard) masks, whereas the source uses −100; MLP and
LayerNorm treated as per-token operations; 1D token axis; no patch merging; no final global
stage.

## 6. Prioritised fixes applied before expansion

1. Overstatement corrected (done previously; tests `test_4c`, `test_4d` added).
2. Original experiment preserved unchanged: `run.py`, `results/results.csv`, `figures/fig1–3`.
3. New work goes into separate modules, configs and output folders, so the original results stay
   reproducible bit-for-bit.
