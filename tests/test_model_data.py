import torch
import torch.nn as nn

from grok.data import make_dataset
from grok.model import GrokConfig, GrokTransformer


def test_M1_split_disjoint_and_complete():
    p = 53
    tr_x, tr_y, te_x, te_y = make_dataset(p, 0.5, seed=0)
    ids_tr = set((tr_x[:, 0] * p + tr_x[:, 1]).tolist())
    ids_te = set((te_x[:, 0] * p + te_x[:, 1]).tolist())
    assert ids_tr.isdisjoint(ids_te)
    assert len(ids_tr | ids_te) == p * p
    assert ((tr_x[:, 0] + tr_x[:, 1]) % p == tr_y).all()
    assert (tr_x[:, 2] == p).all()          # EQ token


def test_M2_shapes_and_cache():
    cfg = GrokConfig(p=53, d_model=64, n_head=4, d_mlp=256)
    model = GrokTransformer(cfg)
    x = torch.randint(0, 53, (8, 3)); x[:, 2] = 53
    logits, cache = model(x, return_cache=True)
    assert logits.shape == (8, 53)
    assert cache["pattern"].shape == (8, 4, 3, 3)
    assert torch.allclose(cache["pattern"].sum(-1),
                          torch.ones(8, 4, 3), atol=1e-5)
    assert cache["neuron_acts"].shape == (8, 256)
    assert (cache["neuron_acts"] >= 0).all()   # post-ReLU


def test_M3_no_biases_no_layernorm():
    model = GrokTransformer(GrokConfig(p=53))
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules())
    assert not any("bias" in name for name, _ in model.named_parameters())


def test_M4_determinism():
    def build():
        torch.manual_seed(0)
        return GrokTransformer(GrokConfig(p=53, d_model=64, d_mlp=128))
    m1, m2 = build(), build()
    x = torch.randint(0, 53, (16, 3)); x[:, 2] = 53
    with torch.no_grad():
        assert torch.equal(m1(x), m2(x))
