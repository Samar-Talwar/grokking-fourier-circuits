"""Full-batch training loop for the grokking experiment (spec §2.3, §4.4).

Deliberately dumb: no analysis happens here. It logs six scalars to
metrics.csv and saves log-spaced checkpoints so ALL interpretability is
post-hoc over checkpoints (grok/analysis.py).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import make_dataset
from .model import GrokConfig, GrokTransformer
from .fourier import spectrum_concentration


def default_ckpt_steps(total: int) -> list[int]:
    base = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    base += list(range(7500, total + 1, 2500))
    return sorted(set(s for s in base if s <= total))


@dataclass
class TrainRun:
    p: int = 113
    frac_train: float = 0.30
    steps: int = 40_000
    lr: float = 1e-3
    betas: tuple = (0.9, 0.98)
    wd: float = 1.0                  # load-bearing: drives the transition
    seed: int = 0
    d_model: int = 128
    n_head: int = 4
    d_mlp: int = 512
    out_dir: str = "runs/run"
    log_every: int = 100
    ckpt_steps: list = field(default_factory=list)
    # early stop: BOTH conditions (spec §2.3 — don't stop mid-cleanup)
    stop_test_acc: float = 0.999
    stop_concentration: float = 0.85
    resume: bool = False             # continue from state_last.pt if present


@torch.no_grad()
def _acc_loss(model, x, y) -> tuple[float, float]:
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return acc, loss


def train(cfg: TrainRun) -> Path:
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), default=list))

    tr_x, tr_y, te_x, te_y = make_dataset(cfg.p, cfg.frac_train, cfg.seed)
    model = GrokTransformer(GrokConfig(p=cfg.p, d_model=cfg.d_model,
                                       n_head=cfg.n_head, d_mlp=cfg.d_mlp))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=cfg.betas, weight_decay=cfg.wd)
    ckpts = set(cfg.ckpt_steps or default_ckpt_steps(cfg.steps))

    start_step = 0
    state_path = out / "state_last.pt"
    if cfg.resume and state_path.exists():
        st = torch.load(state_path)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_step = st["step"] + 1
        print(f"resuming from step {start_step}", flush=True)

    csv_path = out / "metrics.csv"
    mode = "a" if (cfg.resume and start_step > 0) else "w"
    with open(csv_path, mode, newline="") as f:
        wr = csv.writer(f)
        if mode == "w":
            wr.writerow(["step", "train_loss", "train_acc",
                         "test_loss", "test_acc", "weight_norm"])
        t0 = time.time()
        for step in range(start_step, cfg.steps + 1):
            if step in ckpts:
                torch.save(model.state_dict(), out / f"ckpt_{step}.pt")

            # ---- full-batch step -------------------------------------
            model.train()
            logits = model(tr_x)
            loss = F.cross_entropy(logits, tr_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % cfg.log_every == 0:
                model.eval()
                tr_acc, tr_loss = _acc_loss(model, tr_x, tr_y)
                te_acc, te_loss = _acc_loss(model, te_x, te_y)
                wnorm = sum(p_.pow(2).sum() for p_ in
                            model.parameters()).sqrt().item()
                wr.writerow([step, f"{tr_loss:.6f}", f"{tr_acc:.4f}",
                             f"{te_loss:.6f}", f"{te_acc:.4f}",
                             f"{wnorm:.3f}"])
                f.flush()
                if step % (cfg.log_every * 10) == 0:
                    print(f"step {step:6d} | train {tr_acc:.3f}/{tr_loss:.4f}"
                          f" | test {te_acc:.3f}/{te_loss:.4f}"
                          f" | ||W|| {wnorm:.1f} | {time.time()-t0:.0f}s",
                          flush=True)

                # early stop: generalized AND circuit cleaned up
                torch.save({"model": model.state_dict(),
                            "opt": opt.state_dict(), "step": step},
                           state_path)
                if te_acc >= cfg.stop_test_acc:
                    conc = spectrum_concentration(model.embed.detach(), cfg.p)
                    if conc >= cfg.stop_concentration:
                        torch.save(model.state_dict(),
                                   out / f"ckpt_{step}.pt")
                        (out / "DONE").write_text(str(step))
                        print(f"early stop at {step}: test_acc={te_acc:.4f},"
                              f" concentration={conc:.3f}", flush=True)
                        break
    return out
