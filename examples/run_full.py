"""Headline grokking run. Default here: p=53, frac 0.40 (CPU-friendly,
groks in minutes). For the paper-scale p=113, frac 0.30 run (CPU
overnight or minutes on a free T4), pass --p 113 --frac 0.30."""
import argparse
from grok.train import TrainRun, train

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=53)
ap.add_argument("--frac", type=float, default=0.40)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--out", default="runs/full")
a = ap.parse_args()
train(TrainRun(p=a.p, frac_train=a.frac, steps=a.steps, out_dir=a.out))
