# 06 — Research application summary

## 1. Research question

In window-attention Transformers such as the Swin Transformer and SwiFT, does exact
depth-limited reachability (which input tokens *can* influence which outputs after L layers)
predict which long-range dependencies a network can learn? And what do fixed-window,
shifted-window and global attention actually cost on a GPU?

## 2. Why it matters

SwiFT (Kim et al., NeurIPS 2023) makes 4D fMRI tractable by restricting attention to local 4D
windows and shifting them between layers. Along time, the window is 4 frames with a 2-frame
shift in the released configuration. Whether a model like this can relate distant time points
therefore depends on depth, not just on the attention rule. Knowing exactly which dependencies
the architecture permits, and whether training actually uses them, is a prerequisite for
interpreting what such a model can and cannot learn from brain dynamics.

## 3. What I implemented

- **Exact reachability model.** Boolean per-layer attention graphs and depth-limited products
  `R_L = A_L ⋯ A_1`. The shifted layer is validated against an independent re-implementation of
  the Swin/SwiFT cyclic-roll + region-mask procedure. A test proves that 4D SwiFT-style windows
  factorise into per-axis 1D windows (Kronecker product).
- **Matched learning experiment.** Four Transformers that are identical (204,676 parameters)
  except for their attention masks: global, fixed, shifted, and none. They are trained on a
  synthetic task with no single-token shortcut. A gradient test confirms that each network's
  dependency graph equals `R_L` exactly.
- **GPU benchmark.** Explicit global attention, PyTorch's exact memory-efficient kernel, and
  fixed and shifted windows, with measured memory and latency kept separate from theoretical
  score storage.
- 29 automated tests, configs for every experiment, and fixed seeds. Every number is
  regenerated from saved CSVs.

## 4. Methods (brief)

- **Reachability sweep:** N up to 256, windows 4 and 8, three shifts, masked versus cyclic
  boundaries, depth up to 32.
- **Learning:** 64-token sequences, 4 layers, target y_i = (x_i + x_{i+d}) mod 4, 8 distances,
  3 seeds, 81 runs. Learning rates were selected on validation data with a separate seed, and
  each test set was evaluated once.
- **Benchmark:** RTX 4070 Ti SUPER; warm-up, CUDA synchronisation, median of repeats; the
  allocator capped so that out-of-memory is real.

## 5. Verified findings

1. **Reachability was necessary.** Every unreachable position, in every model and seed, scored
   at chance (0.249–0.251). This held across 81 runs.
2. **Reachable dependencies were learned.** All positions reachable within 3 layers reached
   1.000 test accuracy. Positions reachable only at the full depth were learned in 2 of 3 seeds
   at 6,000 steps. A post-hoc rerun with 18,000 steps learned them in 6 of 6 runs, but with
   unstable training.
3. **The window size, not the shift, sets how fast reach grows.** The farthest dependency grows
   by w/2 tokens per layer, and the shift does not change the number of reachable pairs. This
   held exactly in all 800 configurations tested, and it corrected an over-statement in my own
   first version.
4. **Measured cost differs from theory.**
   - Explicit global attention used about 2× its theoretical score memory and ran out of memory
     at 32,768 tokens.
   - PyTorch's exact memory-efficient kernel needed only 4–8 MiB in the forward pass (37–73 MiB
     with backward).
   - Windowed attention stayed 12× faster (at 16,384 tokens).
   - On this GPU, the practical advantage of windows was compute, not memory.

## 6. Limitations

Synthetic 1D tokens, not fMRI data. One task, one depth, 3 seeds. FlashAttention itself was
not available in this PyTorch build and was not measured. One result (the 18,000-step
diagnostic) is post hoc. No claim of reproducing SwiFT and no claim of novelty, since no
systematic literature search was done. Full list: `reports/05_limitations_and_next_steps.md`.

## 7. Relevance to SwiFT and neuroscience

SwiFT never merges the temporal axis. Its temporal receptive field therefore comes only from
stacked shifted windows and the final stage. The exact laws here give, for a single axis with
window 4 and shift 2, how far temporal dependencies can reach after each layer. The learning
experiment shows that such limits are hard limits for what a network can learn, while
dependencies inside them can still be slow to acquire. These are architecture-level facts to
check before reading a trained fMRI model's temporal sensitivity as a property of the brain.

---

## 60-second interview explanation

> "I wanted to understand what SwiFT's shifted-window attention can and cannot see across time.
> First I computed exactly which tokens can influence which outputs after each layer, and I
> checked that model against an independent re-implementation of the masking procedure in the
> official Swin and SwiFT code. That showed that reach
> grows by half a window per layer, regardless of the shift. Then I asked whether that
> prediction matters for learning. I trained four Transformers that were identical except for
> their attention masks, on a task where each output needs a token a fixed distance away. In
> 81 runs, every dependency the architecture could not reach stayed exactly at chance, and every
> reachable one within three layers was learned perfectly. The hardest reachable cases, which
> need the full depth, were learned only with longer training, and unstably. So reachability
> tells you where learning is possible, not how fast. I also benchmarked the attention cores on
> a GPU and found that a fused exact global kernel removes most of the memory advantage of
> windows, while windows stay over ten times faster. Everything is synthetic, not fMRI, and I'm
> explicit about that, but it gave me a precise picture of the trade-off SwiFT makes."
