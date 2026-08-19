import csv
import json

import torch

from grok.train import TrainRun, train
from grok.model import GrokConfig, GrokTransformer


def test_T1_memorization_smoke(tmp_path):
    """Negative control: in the grokking regime the model memorizes FIRST.
    After 1500 full-batch steps at p=53: train acc high, test acc poor."""
    cfg = TrainRun(p=53, frac_train=0.5, steps=600, seed=0,
                   out_dir=str(tmp_path / "smoke"), log_every=100,
                   ckpt_steps=[0])
    out = train(cfg)
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert float(rows[-1]["train_acc"]) > 0.99, rows[-1]
    # memorize-first signature: at SOME logged step the train set is
    # (near-)solved while test accuracy is still poor
    gap = [r for r in rows
           if float(r["train_acc"]) > 0.99 and float(r["test_acc"]) < 0.60]
    assert gap, [(r["step"], r["train_acc"], r["test_acc"]) for r in rows]


def test_T2_checkpoint_roundtrip(tmp_path):
    cfg = TrainRun(p=53, frac_train=0.5, steps=100, seed=0,
                   out_dir=str(tmp_path / "rt"), log_every=50,
                   ckpt_steps=[100])
    out = train(cfg)
    model = GrokTransformer(GrokConfig(p=53))
    model.load_state_dict(torch.load(out / "ckpt_100.pt"))
    x = torch.randint(0, 53, (4, 3)); x[:, 2] = 53
    with torch.no_grad():
        y1 = model(x)
    model2 = GrokTransformer(GrokConfig(p=53))
    model2.load_state_dict(torch.load(out / "ckpt_100.pt"))
    with torch.no_grad():
        y2 = model2(x)
    assert torch.equal(y1, y2)


def test_T3_metrics_schema(tmp_path):
    cfg = TrainRun(p=53, frac_train=0.5, steps=100, seed=0,
                   out_dir=str(tmp_path / "schema"), log_every=100,
                   ckpt_steps=[0])
    out = train(cfg)
    header = open(out / "metrics.csv").readline().strip().split(",")
    assert header == ["step", "train_loss", "train_acc",
                      "test_loss", "test_acc", "weight_norm"]
    assert json.loads((out / "config.json").read_text())["p"] == 53
