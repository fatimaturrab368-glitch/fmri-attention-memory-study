# 05 — Limitations and next steps

## What this project is not

- **Not fMRI.** No neuroimaging data was used anywhere. Tokens are synthetic symbols on a 1D
  axis, an abstraction of SwiFT's temporal token axis.
- **Not SwiFT.** There is no patch merging, no 4D windows in the learning or benchmark
  experiments, no hierarchical stages and no final global stage. The 4D connection is limited to
  one proven statement: a test shows that 4D roll-and-mask connectivity is the Kronecker product
  of per-axis 1D connectivity.
- **Not a claim of novelty.** That windowed attention limits receptive fields, and that shifting
  windows restores cross-window flow, is the stated design rationale of Swin (Liu et al., 2021)
  and SwiFT (Kim et al., 2023). The receptive-field growth laws reported here are exact for
  this 1D setting. **No literature search was run to establish whether they appear elsewhere**,
  so they should be presented as "derived and verified here", not as "new".

## Limitations of each experiment

### A. Reachability sweep
- Reachability is an **upper bound** on dependence (a possible path), not information retention.
- Two window sizes (4, 8), shifts {1, 2, w/2}, and N ≤ 256. The laws are verified exactly only
  inside this grid. They are stated as observations on the grid, with no general proof.
- The "cyclic" boundary is a comparison condition (a ring), not what Swin/SwiFT compute.

### C. Learning experiment
- One synthetic task (paired modular sum, K = 4), one sequence length (64), one depth (4), one
  window (4), 3 seeds. Effects are large and consistent, but 3 seeds can only rule out gross
  instability, not small differences.
- **The training budget matters.** At 6,000 steps the shifted model learned the depth-4-only
  positions (d = 8, 9) in 2 of 3 seeds. A **post-hoc** run with 18,000 steps learned them in
  3 of 3. Because it was post hoc, it is reported separately and does not replace the main
  result.
- **Training is unstable** for these positions. Validation accuracy can collapse back to chance
  and then recover (e.g. d = 8, seed 1: 0.62 → 0.25 → 0.63). Results depend on best-validation
  checkpoint selection, which is applied identically to every condition.
- The learning rate was tuned per architecture on validation data with a separate seed. Other
  hyper-parameters (width, heads, batch, schedule) were fixed, not tuned. A different budget
  could change the global-attention picture: at 1,500 steps global attention had not learned
  even d = 1.
- The task requires an exact positional offset. Results may not transfer to tasks needing
  softer or content-based long-range integration, such as fMRI dynamics.

### B. GPU benchmark
- One GPU (RTX 4070 Ti SUPER), Windows, PyTorch 2.5.1. **This PyTorch build has no
  FlashAttention kernel.** The "efficient" SDPA backend (a memory-efficient exact kernel) was
  measured, so **FlashAttention itself was not measured**.
- Only an attention core is measured (B = 1, H = 4, Dh = 32, FP16). Projections, MLP, other
  layers and training state are excluded. Numbers are not end-to-end model memory.
- The first full run was invalidated by Windows/WDDM spilling oversize allocations into system
  RAM. It is preserved, with an explanation, in `results/benchmark_run1_invalid_sysmem_spill/`.
  The valid run caps the allocator at 95 % of VRAM.
- Small-N timings are near kernel-launch overhead and cannot separate the mechanisms.

## Next steps (in order of scientific value)

1. **Budget-robust learning test.** Repeat C with the longer budget for all architectures and
   more seeds (≥ 5), and report final-step as well as best-checkpoint accuracy, so the
   instability is quantified rather than hidden by checkpoint selection.
2. **Depth sweep in the learning task** (L = 2, 4, 6, 8). Check that the learnable distance
   tracks the exact reachability boundary as L changes, which is the strongest test of the
   predictor.
3. **Linux / FlashAttention benchmark** on the same configurations, to measure FlashAttention
   rather than infer it.
4. **Toward SwiFT:** add spatial patch merging to the reachability model (still exact), and only
   then consider a small 4D learning task. Verify against the official SwiFT code before any
   comparison claim.
5. **Literature check** (e.g. receptive-field analyses of Swin, sparse/local attention theory)
   before calling any growth law new.
