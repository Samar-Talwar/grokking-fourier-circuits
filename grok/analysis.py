"""Post-hoc analysis over checkpoints (spec §4.5): never retrains.

For every ckpt_{step}.pt in a run directory, computes:
  key_frequencies, spectrum_concentration, trig_logit_r2,
  restricted_accuracy, excluded_loss
and writes them to metrics.json.

Usage:  PYTHONPATH=. python -m grok.analysis runs/full
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

from .data import make_dataset
from .model import GrokConfig, GrokTransformer
from .fourier import (all_pair_logits, key_frequencies,
                      spectrum_concentration, trig_logit_r2,
                      restricted_accuracy, excluded_loss)


def analyze_run(run_dir: str | Path, top: int = 6) -> Path:
    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text())
    p = cfg["p"]

    # rebuild the exact train split (same seed => same partition)
    tr_x, tr_y, _, _ = make_dataset(p, cfg["frac_train"], cfg["seed"])
    a_major_idx = tr_x[:, 0] * p + tr_x[:, 1]           # indices into p^2
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    targets_all = (a + b) % p

    ckpts = sorted(run.glob("ckpt_*.pt"),
                   key=lambda f: int(re.search(r"(\d+)", f.name).group(1)))
    results = []
    for f in ckpts:
        step = int(re.search(r"(\d+)", f.name).group(1))
        model = GrokTransformer(GrokConfig(p=p, d_model=cfg["d_model"],
                                           n_head=cfg["n_head"],
                                           d_mlp=cfg["d_mlp"]))
        model.load_state_dict(torch.load(f, map_location="cpu"))
        model.eval()

        W_E = model.embed.detach()
        freqs = key_frequencies(W_E, p, top=top)
        logits = all_pair_logits(model, p)
        entry = {
            "step": step,
            "key_freqs": freqs,
            "concentration": spectrum_concentration(W_E, p, top=top),
            "trig_r2": trig_logit_r2(logits, p, freqs),
            "restricted_acc": restricted_accuracy(logits, p, freqs),
            "excluded_loss": excluded_loss(logits, p, freqs,
                                           a_major_idx, targets_all),
        }
        results.append(entry)
        print(entry, flush=True)

    out = run / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    analyze_run(sys.argv[1] if len(sys.argv) > 1 else "runs/full")
