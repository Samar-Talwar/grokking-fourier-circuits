"""The four headline figures (spec §4.6).

Usage: PYTHONPATH=. python plots/make_plots.py runs/full
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from grok.model import GrokConfig, GrokTransformer
from grok.fourier import embedding_spectrum


def load(run: Path):
    rows = list(csv.DictReader(open(run / "metrics.csv")))
    curves = {k: [float(r[k]) for r in rows] for k in rows[0]}
    metrics = json.loads((run / "metrics.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    return curves, metrics, cfg


def spectrum_of_ckpt(run: Path, step: int, cfg) -> torch.Tensor:
    model = GrokTransformer(GrokConfig(p=cfg["p"], d_model=cfg["d_model"],
                                       n_head=cfg["n_head"],
                                       d_mlp=cfg["d_mlp"]))
    model.load_state_dict(torch.load(run / f"ckpt_{step}.pt",
                                     map_location="cpu"))
    return embedding_spectrum(model.embed.detach(), cfg["p"])


def main(run_dir: str):
    run = Path(run_dir)
    curves, metrics, cfg = load(run)
    steps = curves["step"]

    # P1 — grokking curves
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(steps, curves["train_acc"], label="train accuracy", c="#4477aa")
    plt.plot(steps, curves["test_acc"], label="test accuracy", c="#cc6677")
    plt.xscale("log"); plt.xlabel("step (log)"); plt.ylabel("accuracy")
    plt.title(f"Grokking: modular addition mod {cfg['p']} "
              f"(frac_train={cfg['frac_train']}, wd={cfg['wd']})")
    plt.legend(); plt.tight_layout()
    plt.savefig(run / "P1_grokking_curves.png", dpi=150); plt.close()

    # P2 — embedding spectrum, init vs final
    ck_steps = sorted(int(f.stem.split("_")[1]) for f in run.glob("ckpt_*.pt"))
    s0 = spectrum_of_ckpt(run, ck_steps[0], cfg)
    s1 = spectrum_of_ckpt(run, ck_steps[-1], cfg)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True)
    for ax, s, t in ((axes[0], s0, f"init (step {ck_steps[0]})"),
                     (axes[1], s1, f"final (step {ck_steps[-1]})")):
        ax.bar(range(len(s)), s.numpy(), color="#4477aa")
        ax.set_title(f"embedding Fourier spectrum — {t}")
        ax.set_ylabel("||F_i W_E||")
    axes[1].set_xlabel("Fourier basis index (const, cos1, sin1, cos2, ...)")
    plt.tight_layout()
    plt.savefig(run / "P2_spectrum.png", dpi=150); plt.close()

    # P3 — progress measures lead the transition
    m_steps = [m["step"] for m in metrics]
    conc = [m["concentration"] for m in metrics]
    excl = [m["excluded_loss"] for m in metrics]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps, curves["test_acc"], c="#cc6677", label="test accuracy")
    ax1.plot(m_steps, conc, "o-", c="#4477aa",
             label="spectrum concentration")
    ax1.set_xscale("log"); ax1.set_xlabel("step (log)")
    ax1.set_ylabel("accuracy / concentration"); ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(m_steps, excl, "s--", c="#228833", label="excluded loss (train)")
    ax2.set_ylabel("excluded loss")
    # t_signal / t_grok annotations
    t_grok = next((s for s, a in zip(steps, curves["test_acc"]) if a > 0.95),
                  None)
    # leading indicator: excluded train loss lifting off its memorization
    # floor => train performance now depends on the forming Fourier circuit
    t_sig, run_min = None, float("inf")
    for s, e in zip(m_steps, excl):
        run_min = min(run_min, e)
        if s > 0 and e > 4 * run_min and run_min < 1.0:
            t_sig = s
            break
    for t, name, col in ((t_sig, "t_signal", "#4477aa"),
                         (t_grok, "t_grok", "#cc6677")):
        if t:
            ax1.axvline(t, color=col, ls=":", lw=1)
            ax1.text(t, 1.04, name, color=col, ha="center", fontsize=9)
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="center left")
    plt.title("Progress measures lead the grokking transition")
    plt.tight_layout()
    plt.savefig(run / "P3_progress_measures.png", dpi=150); plt.close()

    # P4 — trig R^2 and restricted accuracy over training
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(m_steps, [m["trig_r2"] for m in metrics], "o-",
             label="trig-basis logit $R^2$", c="#4477aa")
    plt.plot(m_steps, [m["restricted_acc"] for m in metrics], "s-",
             label="restricted accuracy", c="#228833")
    plt.xscale("log"); plt.xlabel("step (log)"); plt.ylim(-0.02, 1.05)
    plt.title("The Fourier algorithm explains the logits after grokking")
    plt.legend(); plt.tight_layout()
    plt.savefig(run / "P4_logit_r2.png", dpi=150); plt.close()

    print(f"wrote P1–P4 into {run}/  (t_signal={t_sig}, t_grok={t_grok})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/full")
