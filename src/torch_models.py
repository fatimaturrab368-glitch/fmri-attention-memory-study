"""Small pre-LN Transformer whose ONLY architectural difference between
conditions is the per-layer Boolean attention mask (global / fixed / shifted /
none). All conditions therefore have identical parameter counts.

The masks come from `src.reachability.layer_adjacency`, so the dependency
graph of the network is exactly the one analysed by R_L. Embeddings, LayerNorm,
the MLP and the output head act per token and add no cross-token edges.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reachability import layer_adjacency


def build_masks(arch: str, n: int, depth: int, w: int, s: int) -> torch.Tensor:
    """(depth, n, n) bool tensor; True = query i may attend to key j."""
    return torch.from_numpy(np.stack([layer_adjacency(arch, l, n, w, s)
                                      for l in range(1, depth + 1)]))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
                                 nn.Linear(mlp_ratio * d_model, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, n, 3, self.n_heads, c // self.n_heads) \
                                       .permute(2, 0, 3, 1, 4)
        # explicit (math) attention: the boolean mask is applied exactly (-inf)
        att = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, n, c)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class MaskedTransformer(nn.Module):
    def __init__(self, vocab: int, n_tokens: int, d_model: int, n_heads: int, depth: int,
                 masks: torch.Tensor, n_classes: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))  # learned absolute (as SwiFT)
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.register_buffer("masks", masks, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok(x) + self.pos
        for l, blk in enumerate(self.blocks):
            h = blk(h, self.masks[l])
        return self.head(self.ln(h))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
