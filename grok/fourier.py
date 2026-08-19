"""Mechanistic-interpretability tooling for the modular-addition network.

The reverse-engineered algorithm (Nanda et al. 2023): the network learns a
handful of "key frequencies" w = 2*pi*k/p and computes, in superposition,

    embed:   a -> (cos wa, sin wa),  b -> (cos wb, sin wb)
    attn+mlp:  multiply & add-angle:  cos w(a+b), sin w(a+b)
               via cos wa cos wb - sin wa sin wb, etc.
    unembed: logit(c)  ~=  sum_k  alpha_k * cos( w_k * (a + b - c) )

Constructive interference makes the correct residue c* = a+b mod p the
argmax. Everything in this file exists to *measure* that story:

  * `fourier_basis(p)`      — orthonormal DFT basis over Z_p
  * `embedding_spectrum`    — norm of W_E along each Fourier direction
  * `key_frequencies`       — dominant k's (the learned frequencies)
  * `cos_ab_c_basis` +
    `trig_logit_r2`         — fraction of logit variance explained by the
                              cos(w(a+b-c)) family: THE smoking gun.
                              Random/memorizing nets: ~0. Grokked: >0.9.
  * `restricted_accuracy`   — accuracy when logits are *projected onto*
                              the key-frequency trig components only
                              (a causal check: the story suffices).
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------- #
def fourier_basis(p: int) -> tuple[torch.Tensor, list[str]]:
    """Orthonormal Fourier basis over Z_p.

    Returns (basis, names): basis is (p, p); row 0 is the constant vector,
    rows 2k-1 / 2k are cos / sin at frequency k, k = 1..p//2.
    """
    x = torch.arange(p, dtype=torch.float64)
    rows, names = [torch.ones(p, dtype=torch.float64) / math.sqrt(p)], ["const"]
    for k in range(1, p // 2 + 1):
        c = torch.cos(2 * math.pi * k * x / p)
        s = torch.sin(2 * math.pi * k * x / p)
        rows += [c / c.norm(), s / s.norm()]
        names += [f"cos{k}", f"sin{k}"]
    basis = torch.stack(rows[:p])  # p rows exactly (sin at k=p/2 is 0 for even p)
    return basis, names[:p]


def embedding_spectrum(W_E: torch.Tensor, p: int) -> torch.Tensor:
    """Norm of the (numeric-token) embedding along each Fourier direction.

    W_E: (p+1, d) or (p, d). Returns (p,) — entry i is ||F_i @ W_E||_2.
    A grokked network concentrates almost all of this mass on a few
    cos/sin pairs; at initialization it is flat.
    """
    W = W_E[:p].to(torch.float64)          # drop the '=' token row
    basis, _ = fourier_basis(p)
    return (basis @ W).norm(dim=1)


def key_frequencies(W_E: torch.Tensor, p: int, top: int = 6) -> list[int]:
    """Dominant frequencies k, by combined cos_k/sin_k embedding energy."""
    spec = embedding_spectrum(W_E, p)
    energy = []
    for k in range(1, p // 2 + 1):
        idx_c, idx_s = 2 * k - 1, 2 * k
        e = spec[idx_c] ** 2 + (spec[idx_s] ** 2 if idx_s < p else 0.0)
        energy.append((e.item(), k))
    energy.sort(reverse=True)
    return [k for _, k in energy[:top]]


def spectrum_concentration(W_E: torch.Tensor, p: int, top: int = 6) -> float:
    """Fraction of non-constant embedding energy in the top-k frequencies.
    ~ top/(p/2) at init (flat spectrum), -> ~1.0 after grokking."""
    spec = embedding_spectrum(W_E, p) ** 2
    total = spec[1:].sum()
    ks = key_frequencies(W_E, p, top)
    mass = 0.0
    for k in ks:
        mass += spec[2 * k - 1]
        if 2 * k < p:
            mass += spec[2 * k]
    return (mass / total).item()


# --------------------------------------------------------------------- #
def cos_ab_c_basis(p: int, freqs: list[int], device="cpu") -> torch.Tensor:
    """Design matrix of the algorithm's predicted logit components.

    For each key frequency k, the theory says logits contain a term
    cos(w_k (a+b-c)) = cos(w_k(a+b))cos(w_k c) + sin(w_k(a+b))sin(w_k c).
    We build, for every input pair (a,b) and every output c, the two
    features per frequency. Returns (p*p, p, 2*len(freqs)).
    """
    a = torch.arange(p).repeat_interleave(p).double()
    b = torch.arange(p).repeat(p).double()
    c = torch.arange(p).double()
    feats = []
    for k in freqs:
        w = 2 * math.pi * k / p
        feats.append(torch.cos(w * ((a + b)[:, None] - c[None, :])))
        feats.append(torch.sin(w * ((a + b)[:, None] - c[None, :])))
    return torch.stack(feats, dim=-1).to(device)  # (p^2, p, 2F)


@torch.no_grad()
def trig_logit_r2(logits_all: torch.Tensor, p: int, freqs: list[int]) -> float:
    """Fraction of (centered) logit variance explained by the
    cos/sin(w(a+b-c)) family — least-squares fit, global coefficients.

    logits_all: (p*p, p), logits for every input pair in lexicographic
    (a-major) order.
    """
    X = cos_ab_c_basis(p, freqs)                       # (p^2, p, 2F)
    y = logits_all.double()
    y = y - y.mean(dim=1, keepdim=True)                # center per-input
    Xf = X.reshape(-1, X.shape[-1])                    # (p^2*p, 2F)
    yf = y.reshape(-1)
    beta = torch.linalg.lstsq(Xf, yf.unsqueeze(1)).solution.squeeze(1)
    resid = yf - Xf @ beta
    return (1.0 - resid.pow(2).sum() / yf.pow(2).sum()).item()


@torch.no_grad()
def restricted_accuracy(logits_all: torch.Tensor, p: int,
                        freqs: list[int]) -> float:
    """Causal sufficiency check: project logits onto the key-frequency trig
    family and measure accuracy of the *projection alone*. If the learned
    algorithm really is the Fourier story, this stays ~1.0."""
    X = cos_ab_c_basis(p, freqs)
    y = logits_all.double()
    y = y - y.mean(dim=1, keepdim=True)
    Xf = X.reshape(-1, X.shape[-1])
    beta = torch.linalg.lstsq(Xf, y.reshape(-1, 1)).solution
    proj = (Xf @ beta).reshape(p * p, p)
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    target = (a + b) % p
    return (proj.argmax(dim=1) == target).double().mean().item()


@torch.no_grad()
def excluded_loss(logits_all: torch.Tensor, p: int, freqs: list[int],
                  train_idx: torch.Tensor, targets_all: torch.Tensor) -> float:
    """Necessity test / leading progress measure (Nanda et al.).

    Remove the fitted key-frequency trig component from the logits and
    evaluate cross-entropy on the TRAIN split. While the model is purely
    memorizing, removing the (absent) Fourier component changes nothing
    and excluded loss ~= train loss ~= 0. As the Fourier circuit forms,
    train performance comes to DEPEND on it, so excluded train loss rises
    toward test-loss levels BEFORE test accuracy moves.

    logits_all: (p*p, p) a-major; train_idx: indices into the p*p axis;
    targets_all: (p*p,) labels for every pair.
    """
    X = cos_ab_c_basis(p, freqs)
    y = logits_all.double()
    y_c = y - y.mean(dim=1, keepdim=True)
    Xf = X.reshape(-1, X.shape[-1])
    beta = torch.linalg.lstsq(Xf, y_c.reshape(-1, 1)).solution
    proj = (Xf @ beta).reshape(p * p, p)
    excluded = y - proj                       # remove the trig component
    ce = torch.nn.functional.cross_entropy(
        excluded[train_idx], targets_all[train_idx], reduction="mean"
    )
    return ce.item()


@torch.no_grad()
def all_pair_logits(model, p: int, device="cpu") -> torch.Tensor:
    """Logits for every (a, b) pair, a-major order. (p*p, p)."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full_like(a, p)
    x = torch.stack([a, b, eq], dim=1).to(device)
    outs = []
    for i in range(0, x.shape[0], 4096):
        outs.append(model(x[i:i + 4096]).cpu())
    return torch.cat(outs)
