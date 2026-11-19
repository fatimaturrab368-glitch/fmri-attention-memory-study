# 02 — Research direction

## Inputs to the decision

- **Audit (report 01):** the existing exact model is correct and reproducible. Its main gaps are
  (i) a single configuration, so the role of the shift could not be identified; (ii) no evidence
  linking reachability to learning; (iii) no measured hardware cost.
- **Hardware:** NVIDIA RTX 4070 Ti SUPER (16 GiB), PyTorch 2.5.1 + CUDA 12.1, i7-14700K, 64 GB
  RAM. Small Transformers train in about 1 minute per run, and benchmarking is feasible.

## Scoring (1–5; weights from the brief)

| Option | Rigor 25 % | SwiFT relevance 20 % | Evidence 20 % | Feasibility 15 % | Reproducibility 10 % | Application value 10 % | **Weighted** |
|---|---|---|---|---|---|---|---|
| A — reachability sweep | 5 | 3 | 3 | 5 | 5 | 3 | **4.00** |
| B — GPU benchmarking | 4 | 3 | 4 | 5 | 3 | 3 | **3.75** |
| C — synthetic long-range learning | 4 | 3 | 5 | 4 | 4 | 4 | **4.00** |
| D — 4D SwiFT-style simulation | 2 | 5 | 2 | 2 | 3 | 4 | **2.90** |
| E — focused hybrid (C as the core test, A as the predictor, B as cost context) | 5 | 4 | 5 | 4 | 4 | 5 | **4.55** |

**Rationale.**
- **A** is exact and cheap. On its own it only re-describes the graph; it cannot say whether
  reachability matters for learning.
- **B** produces real numbers, but timings depend on hardware and kernels. On its own it says
  nothing about information access.
- **C** gives the strongest new kind of evidence. Without an exact predictor, though, its results
  would be hard to interpret.
- **D** is the most SwiFT-flavoured but the least defensible. A faithful 4D reimplementation with
  patch merging cannot be verified against SwiFT without running SwiFT, and it would add no new
  evidence about learning. The one claim it would test, that a 1D axis is a valid slice of 4D
  window connectivity, is instead **proved directly by a test** (4D roll + region mask equals the
  Kronecker product of 1D masks).
- **E** combines them so that each part has a job. **A** supplies an exact, per-position
  prediction. **C** tests that prediction under matched conditions. **B** gives honest measured
  costs, clearly separated from theory.

Selected: **Option E**. This matches the preliminary preference, but it is justified by the audit
gaps rather than adopted by default. Option D is explicitly rejected.

## Research question

> For window-attention Transformers with identical parameters, does exact depth-limited
> reachability `R_L[i, j]` predict which input-to-output dependencies the network can learn, and
> what do the three attention patterns cost in measured GPU memory and latency?

## Hypotheses

- **H1 (necessity; must hold, serves as a validity check):** when `R_L[i, i+d]` is False, test
  accuracy at position i is at chance (1/K = 0.25) for every architecture.
- **H2 (sufficiency in practice; the empirical question):** when `R_L[i, i+d]` is True, a matched
  model trained under the same protocol reaches accuracy well above chance (≥ 0.9). In particular
  this applies to the shifted-window model at distances it reaches only through multi-layer
  chains.
- **H3 (cost):** measured peak memory of materialised global attention grows about as N², and of
  window attention about as N. Measured latency need not follow the theoretical storage ratio
  (kernel and launch effects), and fused exact global kernels (FlashAttention / memory-efficient
  SDPA) remove the N² memory term without changing connectivity.

## Interpretation plan

| Outcome | Interpretation |
|---|---|
| H1 holds and H2 holds | Reachability is necessary, and in this regime practically sufficient. Shifted windows buy access with depth. |
| H1 holds and H2 fails for some reachable pairs | Reachability is only an upper bound. Optimisation or inductive bias limits what is learned (a negative but informative result). |
| H1 fails | A bug (masking or leakage). Stop and investigate. |
| H3: windowed latency ≥ global SDPA at small N | Theoretical savings do not translate into speed at small N on this GPU. Report as is. |

## Baselines and controls

`global` (upper bound on access), `fixed` (no cross-window access), `shifted` (test condition),
and `none` (self-attention only; a no-communication control that rules out shortcuts). All four
are **the same network**, differing only in Boolean masks, so their parameter counts are
identical.
