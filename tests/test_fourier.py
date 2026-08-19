import math

import pytest
import torch

from grok.fourier import (fourier_basis, embedding_spectrum, key_frequencies,
                          spectrum_concentration, cos_ab_c_basis,
                          trig_logit_r2, restricted_accuracy, excluded_loss)

torch.manual_seed(0)


@pytest.mark.parametrize("p", [53, 113])
def test_F1_basis_orthonormal(p):
    F, names = fourier_basis(p)
    assert F.shape == (p, p)
    eye = F @ F.T
    assert torch.allclose(eye, torch.eye(p, dtype=torch.float64), atol=1e-8)


def test_F2_planted_frequency_detected():
    p, d, k_true = 53, 32, 7
    a = torch.arange(p, dtype=torch.float64)
    col = torch.cos(2 * math.pi * k_true * a / p)
    W_E = col[:, None].repeat(1, d) + 1e-3 * torch.randn(p, d, dtype=torch.float64)
    W_E = torch.cat([W_E, torch.zeros(1, d, dtype=torch.float64)])  # EQ row
    assert key_frequencies(W_E, p, top=3)[0] == k_true
    assert spectrum_concentration(W_E, p, top=1) > 0.95


def test_F3_trig_identity_numeric():
    p, k = 113, 9
    w = 2 * math.pi * k / p
    a = torch.arange(p, dtype=torch.float64).repeat_interleave(p)
    b = torch.arange(p, dtype=torch.float64).repeat(p)
    lhs = torch.cos(w * (a + b))
    rhs = torch.cos(w * a) * torch.cos(w * b) - torch.sin(w * a) * torch.sin(w * b)
    assert torch.allclose(lhs, rhs, atol=1e-10)


def _synthetic_logits(p, freqs):
    a = torch.arange(p).repeat_interleave(p).double()
    b = torch.arange(p).repeat(p).double()
    c = torch.arange(p).double()
    logits = torch.zeros(p * p, p, dtype=torch.float64)
    for k in freqs:
        w = 2 * math.pi * k / p
        logits += torch.cos(w * ((a + b)[:, None] - c[None, :]))
    return logits


def test_F4_synthetic_logits_r2_and_restricted():
    p, true_freqs = 53, [3, 11]
    logits = _synthetic_logits(p, true_freqs)
    assert trig_logit_r2(logits, p, true_freqs) > 0.999
    assert trig_logit_r2(logits, p, [5, 20]) < 0.05
    assert restricted_accuracy(logits, p, true_freqs) == pytest.approx(1.0)


def test_F5_excluded_loss_on_synthetic():
    p, true_freqs = 53, [3, 11]
    logits = _synthetic_logits(p, true_freqs)
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    targets = (a + b) % p
    train_idx = torch.arange(0, p * p, 2)   # any subset works
    ce = excluded_loss(logits, p, true_freqs, train_idx, targets)
    assert abs(ce - math.log(p)) / math.log(p) < 0.05  # ~uniform after removal


def test_F6_interference_argmax_exact():
    p, freqs = 53, [3, 11, 17]
    logits = _synthetic_logits(p, freqs)
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    assert (logits.argmax(dim=1) == (a + b) % p).all()


def test_spectrum_flat_at_random_init():
    p, d = 113, 128
    W_E = torch.randn(p + 1, d)
    conc = spectrum_concentration(W_E, p, top=6)
    assert conc < 3 * (6 / (p / 2))  # near the null baseline, not concentrated
