"""One-layer transformer for modular addition, following the setup of
Nanda et al., "Progress measures for grokking via mechanistic
interpretability" (ICLR 2023):

  * 1 transformer block, no LayerNorm, no biases (cleaner to interpret),
  * learned token + positional embeddings,
  * ReLU MLP,
  * logits read off the final ('=') position.

The forward pass optionally returns a cache of every intermediate
activation so the analysis code can inspect attention patterns, neuron
activations, and per-component logit contributions without hooks magic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GrokConfig:
    p: int = 113
    d_model: int = 128
    n_head: int = 4
    d_mlp: int = 512
    seq_len: int = 3


class GrokTransformer(nn.Module):
    def __init__(self, cfg: GrokConfig):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.d_model, cfg.p
        self.embed = nn.Parameter(torch.randn(p + 1, d) / d**0.5)   # W_E
        self.pos = nn.Parameter(torch.randn(cfg.seq_len, d) / d**0.5)
        self.W_Q = nn.Parameter(torch.randn(cfg.n_head, d, d // cfg.n_head) / d**0.5)
        self.W_K = nn.Parameter(torch.randn(cfg.n_head, d, d // cfg.n_head) / d**0.5)
        self.W_V = nn.Parameter(torch.randn(cfg.n_head, d, d // cfg.n_head) / d**0.5)
        self.W_O = nn.Parameter(torch.randn(cfg.n_head, d // cfg.n_head, d) / d**0.5)
        self.W_in = nn.Parameter(torch.randn(d, cfg.d_mlp) / d**0.5)   # MLP in
        self.W_out = nn.Parameter(torch.randn(cfg.d_mlp, d) / cfg.d_mlp**0.5)
        self.unembed = nn.Parameter(torch.randn(d, p) / d**0.5)        # W_U

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        # x: (B, 3) token ids
        resid = self.embed[x] + self.pos[None]                # (B, T, d)

        # --- attention (causal not needed: fixed 3-token task; use full) ---
        q = torch.einsum("btd,hde->bhte", resid, self.W_Q)
        k = torch.einsum("btd,hde->bhte", resid, self.W_K)
        v = torch.einsum("btd,hde->bhte", resid, self.W_V)
        scores = torch.einsum("bhte,bhse->bhts", q, k) / q.shape[-1] ** 0.5
        pattern = F.softmax(scores, dim=-1)                   # (B, H, T, T)
        z = torch.einsum("bhts,bhse->bhte", pattern, v)
        attn_out = torch.einsum("bhte,hed->btd", z, self.W_O)
        resid = resid + attn_out

        # --- MLP ---
        pre = resid @ self.W_in                               # (B, T, d_mlp)
        post = F.relu(pre)
        mlp_out = post @ self.W_out
        resid = resid + mlp_out

        logits = resid[:, -1] @ self.unembed                  # (B, p)
        if return_cache:
            cache = {
                "pattern": pattern.detach(),
                "neuron_acts": post[:, -1].detach(),   # neurons at '=' pos
                "attn_out": attn_out[:, -1].detach(),
                "mlp_out": mlp_out[:, -1].detach(),
            }
            return logits, cache
        return logits
