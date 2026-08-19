"""Modular addition dataset: sequences [a, b, =] -> label (a + b) mod p.

The full dataset is all p^2 pairs. Grokking appears when the model is
trained on a *fraction* of them with strong weight decay: train accuracy
hits 100% early (memorization) while test accuracy stays near chance for
thousands of steps, then abruptly jumps to ~100% (generalization).
"""

from __future__ import annotations

import torch


def make_dataset(p: int, frac_train: float, seed: int = 0):
    """Returns (train_x, train_y, test_x, test_y).

    x: (N, 3) LongTensor of [a, b, EQ] with EQ = p (extra token).
    y: (N,)   LongTensor of (a + b) % p.
    """
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    x = torch.stack([a, b, eq], dim=1)          # (p^2, 3)
    y = (a + b) % p

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    n_train = int(frac_train * p * p)
    tr, te = perm[:n_train], perm[n_train:]
    return x[tr], y[tr], x[te], y[te]
