# CLAUDE.md — Implementation Specification
# Fourier-Circuit Reverse-Engineering of Grokking
# (Mechanistic Interpretability of Modular-Arithmetic Transformers)

This document is a complete, self-contained specification for implementing
Project 2 with Claude Code (Opus) inside Cursor. It contains all the
mathematics, module specifications with exact signatures, the test plan
with tolerances, reliable hyperparameters, milestones, and acceptance
criteria. Designed for MINIMUM compute: the entire project runs on CPU;
one optional GPU hour makes the headline run faster.

Reproduction targets: Power et al. 2022 (arXiv:2201.02177, grokking) and
Nanda et al., ICLR 2023 (arXiv:2301.05217, progress measures /
reverse-engineering). Reproduce FIRST; extend only after acceptance
criteria pass.

RULE FOR THE IMPLEMENTING AGENT: work in the milestone order of §8. Never
advance while any test of the current milestone fails. Analysis code
(Fourier tooling) is written and tested against SYNTHETIC constructions
BEFORE any real training run, so a failed measurement can never be
confused with a failed model.

---

## 1. Project summary

1. Train a one-layer transformer on modular addition (a + b) mod p from a
   fraction of all p² pairs, with full-batch AdamW and HIGH weight decay.
2. Reproduce **grokking**: train accuracy reaches 100% early while test
   accuracy stays near chance, then test accuracy abruptly rises to ~100%
   thousands of steps later.
3. **Reverse-engineer** the learned algorithm: show the network implements
   a discrete Fourier transform — embeddings become trig lookup tables at
   a sparse set of key frequencies; logits are explained by
   cos(w_k(a+b−c)) interference terms.
4. Build **progress measures** (spectrum concentration, restricted
   accuracy, excluded loss) that detect the generalizing circuit forming
   BEFORE test accuracy moves.
5. Produce four headline plots + a short technical report.

Environment: Python ≥3.10, `torch` (CPU build sufficient), `pytest`,
`matplotlib`, `numpy`. Nothing else.

---

## 2. Complete mathematics

### 2.1 Task and data

Fix prime p (default 113). Inputs are sequences [a, b, EQ] with
a, b ∈ {0..p−1} and EQ = p a special token; label y = (a + b) mod p.
Full dataset: all p² pairs. Train split: a uniformly random fraction
`frac_train` (default 0.30) chosen ONCE with a fixed seed; test = rest.
Training is FULL-BATCH on the entire train split every step (this
matters: grokking in this setup is cleanest without minibatch noise).

### 2.2 Model (Nanda-style one-layer transformer)

No LayerNorm. No biases anywhere. Learned token embedding
W_E ∈ R^{(p+1)×d} and positional embedding W_pos ∈ R^{3×d}.
One attention block (H heads, head dim d/H) with weights
W_Q, W_K, W_V ∈ R^{H×d×(d/H)}, W_O ∈ R^{H×(d/H)×d}; residual add.
One ReLU MLP: W_in ∈ R^{d×d_mlp}, W_out ∈ R^{d_mlp×d}; residual add.
Unembedding W_U ∈ R^{d×p}. Logits are read at the FINAL position (the
'=' token). Init: all weights ~ N(0, 1/√fan_in) (divide randn by √d or
√d_mlp as appropriate).

Defaults: d = 128, H = 4, d_mlp = 512, p = 113. Parameter count ≈ 2·10⁵.

Forward must support `return_cache=True` returning a dict with at least:
`pattern` (B,H,3,3) attention probabilities, `neuron_acts` (B,d_mlp)
post-ReLU activations at the '=' position, `attn_out`, `mlp_out` at the
'=' position. The cache is the raw material of all interpretability.

### 2.3 Training recipe that reliably groks (do not improvise)

    optimizer   AdamW, lr = 1e-3, betas = (0.9, 0.98), weight_decay = 1.0
    batch       full train split, every step
    loss        cross-entropy on the final-position logits
    steps       up to 40,000 (early-stop when test acc > 0.999 AND
                spectrum concentration > 0.85, but never before both)
    seed        fixed (0) for weights and split

Weight decay = 1.0 is the load-bearing hyperparameter: it is what drives
the transition from the memorizing solution to the smaller-norm Fourier
solution. Setting it to a "normal" 0.01 is the classic way to wait
forever and see nothing.

Fast smoke configuration (for tests and CI, minutes on CPU):
p = 53, frac_train = 0.5, d = 128, steps ≤ 15,000, same optimizer.

Checkpointing: save model state_dict at log-spaced steps
(0, 10, 20, 50, 100, 200, 500, 1k, 2k, 5k, then every 2.5k) so ALL
analysis is post-hoc over checkpoints — the training loop stays dumb and
never needs re-running to add a metric.

Metrics logged every 100 steps: train loss/acc, test loss/acc,
total weight L2 norm.

### 2.4 The discrete Fourier basis over Z_p

For x ∈ {0..p−1} and k ∈ {1..⌊p/2⌋} define w_k = 2πk/p and vectors
cos_k[x] = cos(w_k x), sin_k[x] = sin(w_k x). Together with the constant
vector 1/√p these form an ORTHONORMAL basis of R^p after normalizing each
cos_k, sin_k by its norm (for odd prime p each has norm √(p/2)).
Orthogonality identities the tests rely on:

    Σ_x cos(w_j x) cos(w_k x) = (p/2)·δ_jk     (j,k ≥ 1, j+k ≠ p)
    Σ_x sin(w_j x) sin(w_k x) = (p/2)·δ_jk
    Σ_x cos(w_j x) sin(w_k x) = 0
    Σ_x cos(w_k x) = Σ_x sin(w_k x) = 0        (k ≥ 1)

Assemble `fourier_basis(p) -> (F ∈ R^{p×p}, names)`: row 0 constant,
rows 2k−1 / 2k are normalized cos_k / sin_k. F F^T = I_p must hold to
atol 1e-8 (float64).

### 2.5 The learned algorithm (what the measurements are testing)

Post-grokking, the network is claimed to compute, in superposition over a
SPARSE set of key frequencies K (typically |K| = 4–6):

  1. **Embedding = trig lookup**: row a of W_E lies (almost) in the span
     of {cos_k, sin_k : k ∈ K} — i.e. token a is represented by
     (cos w_k a, sin w_k a) pairs.
  2. **Attention + MLP = angle addition**: the nonlinearity produces the
     product terms needed for the sum-of-angle identities
         cos w(a+b) = cos wa · cos wb − sin wa · sin wb
         sin w(a+b) = sin wa · cos wb + cos wa · sin wb
     so the residual stream at '=' contains cos w_k(a+b), sin w_k(a+b).
  3. **Unembedding = interference**: logit(c) ≈ Σ_{k∈K} α_k cos(w_k(a+b−c)),
     expanded via cos(w(a+b−c)) = cos(w(a+b))cos(wc) + sin(w(a+b))sin(wc).
     At c* = a+b mod p every term equals α_k (constructive interference);
     for c ≠ c* the phases decohere and the sum is small. Hence argmax
     logit = correct answer.

### 2.6 Measurements (each must be a function with a synthetic test)

M-a **Embedding spectrum**: spec_i = ||F_i · W_E[:p]||_2 for each basis
    row i (drop the EQ row; float64). Grokked nets concentrate spec on a
    few cos/sin pairs; at init spec is flat.
M-b **Key frequencies**: rank k by energy spec[2k−1]² + spec[2k]²; return
    top-n (default 6).
M-c **Spectrum concentration**: fraction of non-constant energy
    (Σ spec[1:]²) captured by the top-n key frequencies. Init ≈ n/(p/2);
    grokked > 0.85.
M-d **Trig-logit R²**: compute logits for ALL p² inputs (a-major order),
    center per-input (subtract row mean — the constant direction is
    unidentifiable), least-squares fit onto the 2|K| features
    {cos(w_k(a+b−c)), sin(w_k(a+b−c))}, report
    R² = 1 − ||resid||²/||y||². Init/memorizing: ≈ 0. Grokked: > 0.90.
M-e **Restricted accuracy** (sufficiency): accuracy of the PROJECTION of
    the logits onto that trig family alone. Grokked: > 0.999.
M-f **Excluded loss** (necessity — the leading progress measure): remove
    the fitted trig component from the logits (logits − projection) and
    evaluate cross-entropy on the TRAIN set. Signature of grokking:
    excluded train loss rises toward test-loss levels thousands of steps
    BEFORE test accuracy moves — the memorization circuit is being
    dismantled while train accuracy is still 100%.

All measurement code takes `logits_all ∈ R^{p²×p}` or `W_E` as plain
tensors — never a model — so it is trivially testable on synthetic data.

### 2.7 Progress-measure claim to verify (headline result)

Define t_grok = first step with test acc > 0.95, and t_signal = first
step with spectrum concentration > 2× its initial value (or excluded
loss > 2× its initial value). ACCEPTANCE: t_signal < t_grok by a wide
margin (Nanda et al. observe thousands of steps; require at least
t_signal ≤ 0.7·t_grok in your run).

---

## 3. Repository layout

    grokking/
    ├── CLAUDE.md
    ├── README.md                      # generated last (M6)
    ├── grok/
    │   ├── __init__.py
    │   ├── data.py                    # make_dataset(p, frac_train, seed)
    │   ├── model.py                   # GrokConfig, GrokTransformer
    │   ├── fourier.py                 # §2.4–2.6 measurement functions
    │   ├── train.py                   # full-batch loop + checkpointing
    │   └── analysis.py                # per-checkpoint sweep -> metrics.json
    ├── tests/
    │   ├── test_fourier.py
    │   ├── test_model_data.py
    │   └── test_training.py
    ├── examples/
    │   ├── run_smoke.py               # p=53 fast config
    │   └── run_full.py                # p=113 headline run
    └── plots/
        └── make_plots.py              # the four headline figures

---

## 4. Module specifications

### 4.1 `grok/data.py`

    make_dataset(p, frac_train, seed=0)
      -> (train_x (N,3) long, train_y (N,), test_x, test_y)
    x rows are [a, b, p] (EQ token id = p); y = (a+b) % p.
    Split by a seeded randperm of all p² indices. Deterministic.

### 4.2 `grok/model.py`

`GrokConfig` dataclass: p=113, d_model=128, n_head=4, d_mlp=512,
seq_len=3. `GrokTransformer(cfg)` exactly per §2.2, implemented with
explicit einsums (not nn.MultiheadAttention — the weights must be
directly inspectable as W_Q etc.). `forward(x, return_cache=False)`.

### 4.3 `grok/fourier.py` (exact signatures)

    fourier_basis(p) -> (Tensor (p,p) float64, list[str])
    embedding_spectrum(W_E, p) -> Tensor (p,) float64
    key_frequencies(W_E, p, top=6) -> list[int]
    spectrum_concentration(W_E, p, top=6) -> float
    cos_ab_c_basis(p, freqs, device="cpu") -> Tensor (p², p, 2|K|) float64
    trig_logit_r2(logits_all, p, freqs) -> float
    restricted_accuracy(logits_all, p, freqs) -> float
    excluded_loss(logits_all, p, freqs, train_idx, targets) -> float
    all_pair_logits(model, p, device="cpu") -> Tensor (p², p)  # batched

Numerics: all fits in float64 via `torch.linalg.lstsq`; center logits
per-input before fitting; document that the constant direction is
removed by centering.

### 4.4 `grok/train.py`

    train(cfg: TrainRun) -> Path  # run directory
`TrainRun` dataclass: p, frac_train, steps, lr=1e-3, betas=(0.9,0.98),
wd=1.0, seed=0, ckpt_steps=log-spaced list, out_dir. Full-batch loop per
§2.3; writes `metrics.csv` (step, train_loss, train_acc, test_loss,
test_acc, weight_norm) and `ckpt_{step}.pt` files. Early-stop rule of
§2.3. Must run the smoke config end-to-end in < 15 min on CPU.

### 4.5 `grok/analysis.py`

    analyze_run(run_dir) -> metrics.json
For every checkpoint: load model, compute all_pair_logits once, then
key_frequencies (from W_E), spectrum_concentration, trig_logit_r2,
restricted_accuracy, excluded_loss (train split reloaded with the same
seed). Append to a JSON list. Pure post-hoc; never retrains.

### 4.6 `plots/make_plots.py` — the four headline figures

    P1 grokking_curves.png    train & test accuracy vs step (log-x)
    P2 spectrum.png           embedding Fourier spectrum: init vs final
    P3 progress_measures.png  test acc + spectrum concentration +
                              excluded loss on one log-x axis, with
                              t_signal and t_grok annotated
    P4 logit_r2.png           trig R² and restricted accuracy vs step

---

## 5. Minimal-compute strategy

Everything is CPU-feasible. Model ≈ 2·10⁵ params; a full-batch step on
the p=113 train split (~3.8k examples) is a few ms of GEMMs.
  Stage A (CPU): M0–M4, all tests, smoke run (p=53) — minutes.
  Stage B (CPU or free T4): the p=113 headline run. CPU: order of an
  hour or two for 40k steps (acceptable overnight at worst). Colab/Kaggle
  T4: minutes. Analysis sweep over ~25 checkpoints: minutes on CPU
  (all_pair_logits is p²=12,769 forward passes, batched).
Total GPU budget: 0 required; ≤ 1 hour optional for comfort.

---

## 6. Test plan (every test MUST exist and pass; all CPU)

tests/test_fourier.py
  F1 basis_orthonormal — F F^T = I_p, float64, atol 1e-8 (p=113 and 53).
  F2 planted_frequency_detected — construct W_E rows = cos(w_7 a) (+ tiny
     noise): key_frequencies must rank 7 first; spectrum_concentration
     (top=1) > 0.95.
  F3 trig_identity_numeric — cos(w(a+b)) equals
     cos wa cos wb − sin wa sin wb over the full grid, atol 1e-10.
  F4 synthetic_logits_r2 — build logits_all := Σ_{k∈{3,11}} cos(w_k(a+b−c));
     trig_logit_r2 with freqs=[3,11] > 0.999; with WRONG freqs [5,20]
     < 0.05; restricted_accuracy with correct freqs == 1.0.
  F5 excluded_loss_on_synthetic — removing the true component from the
     synthetic logits must push excluded CE to ≈ uniform (ln p within 5%).
  F6 interference_argmax — for the synthetic logits of F4, argmax over c
     equals (a+b)%p for ALL p² inputs (the constructive-interference
     claim, exactly).

tests/test_model_data.py
  M1 split_disjoint_and_complete — train/test index sets partition p².
  M2 shapes_and_cache — forward (B,3)→(B,p); cache keys and shapes as
     specified in §2.2.
  M3 no_biases_no_layernorm — assert the module has no LayerNorm and no
     bias parameters (structural guard against silent drift from the
     Nanda setup).
  M4 determinism — two models built with seed 0 produce identical logits.

tests/test_training.py
  T1 memorization_smoke — p=53 config, 1,500 steps: train acc > 0.95
     while test acc < 0.6 (verifies we are IN the grokking regime:
     memorize-first). Runtime target < 3 min CPU.
  T2 checkpoint_roundtrip — save at step 100, reload, identical logits.
  T3 metrics_file_schema — metrics.csv has the six required columns.

Definition of done for Stage A: `PYTHONPATH=. pytest tests/ -q` all green.

---

## 7. Acceptance criteria (the whole project)

1. All tests green on CPU.
2. Headline run (p=113, frac 0.30, wd 1.0) exhibits grokking: train acc
   ≥ 0.999 early; test acc < 0.30 for at least 3× that many steps; test
   acc ≥ 0.999 by end. (Exact step counts vary by seed; the SHAPE is the
   criterion. If no transition by 40k steps, first suspect: weight decay
   not applied, LayerNorm/biases present, or minibatching instead of
   full-batch.)
3. Post-grokking: spectrum_concentration > 0.85 with ≤ 6 key
   frequencies; trig_logit_r2 > 0.90; restricted_accuracy > 0.999.
4. Progress-measure lead: t_signal ≤ 0.7·t_grok, visible in P3.
5. Init baselines recorded for contrast: concentration ≈ 6/(p/2),
   R² < 0.05.
6. Four plots + metrics.json committed; README documents the exact
   reproduction commands and seed.

---

## 8. Milestones (strict order)

M0 scaffold: tree of §3, deps, empty importable modules.
M1 data.py + model.py            → M1–M4 (model/data tests) pass.
M2 fourier.py                    → F1–F6 pass. (Analysis before training:
   the oracle is synthetic, so measurement bugs surface now.)
M3 train.py + smoke run          → T1–T3 pass; examples/run_smoke.py
   completes and memorizes.
M4 analysis.py + plots on the SMOKE run — pipeline works end-to-end on
   a run that has NOT grokked (all measures near baseline: this is the
   negative control).
M5 headline run p=113 (CPU overnight or free T4) → acceptance 2–5.
M6 README + 4–6 page report: derivations of §2.4–2.5 written out, the
   four figures, honest discussion of seed sensitivity.
M7 (extension, only after M6): pick ONE — different operation
   (a−b, a·b mod p), different p sweep, ablating individual frequencies
   (zero their embedding components and measure accuracy drop — a causal
   necessity test), or tracking WHICH step each frequency emerges.

---

## 9. Question bank

### A. Fourier mathematics
1.  Prove the orthogonality identities of §2.4 (hint: geometric sums of
    roots of unity).
2.  Why does the basis have exactly p vectors — count cos/sin pairs plus
    the constant for odd p.
3.  Why must logits be CENTERED per-input before the trig fit? What is
    unidentifiable otherwise?
4.  Derive cos(w(a+b−c)) = cos(w(a+b))cos(wc) + sin(w(a+b))sin(wc) and
    explain which network component supplies each factor.
5.  Prove the constructive-interference claim: for logits
    Σ_k α_k cos(w_k(a+b−c)) with α_k > 0, argmax_c is (a+b) mod p, and
    quantify the margin for |K| frequencies.
6.  Why do key frequencies come in cos/sin PAIRS with similar energy?
    What would a lone cosine (no sine) mean?
7.  Frequencies k and p−k alias: cos(w_{p−k}x) = cos(w_k x) on Z_p. How
    does your spectrum code avoid double-counting?
8.  What does the DFT of a RANDOM Gaussian embedding look like
    (expected spectrum flatness), and why is that the right null?
9.  Why analyze W_E in float64?
10. Relation between this discrete basis and the Fourier series you know
    from PDEs — what plays the role of the domain, and of frequency?

### B. Grokking dynamics
11. Define grokking precisely. Distinguish it from ordinary
    double descent and from simple underfitting.
12. Why is HIGH weight decay (≈1.0) essential? State the
    memorization-vs-generalization norm argument.
13. Why full-batch? What does minibatch noise do to the transition's
    sharpness and timing?
14. Why does frac_train matter? What happens at frac→1.0 and at
    frac→0.05?
15. What is the weight-norm trajectory during the three phases
    (memorize / circuit-formation / cleanup), and why does it fall
    through the transition?
16. Why does the network memorize FIRST rather than finding the Fourier
    solution immediately?
17. Is grokking a property of the task, the architecture, or the
    optimizer? Design the 3-factor experiment.
18. Your T1 test asserts memorization-with-poor-test at 1.5k steps. Why
    is that a necessary NEGATIVE control for the whole project?
19. What changes if you add LayerNorm? Biases? (Why does the spec forbid
    them?)
20. How seed-sensitive are t_grok and the key-frequency SET? How should
    the report state claims given that sensitivity?
21. Early-stopping on test accuracy alone would end the run mid-cleanup.
    Why does the spec's early-stop also require spectrum concentration?
22. Relate grokking to the "emergent capabilities" debate at LLM scale:
    what is analogous, and what does NOT transfer?

### C. Mechanistic interpretability methodology
23. Distinguish correlational evidence (R² fits) from causal evidence
    (ablations, restricted logits). Which of your measures is which?
24. Why is restricted_accuracy a SUFFICIENCY test and excluded_loss a
    NECESSITY test? Why do you need both?
25. Why is excluded loss evaluated on the TRAIN set? What exactly does
    its early rise demonstrate about the memorization circuit?
26. Why write measurement code against synthetic constructions before
    any training run (F2, F4, F5)? Name the failure mode this kills.
27. What is superposition, and where does this network exhibit it?
28. The claim "attention+MLP compute angle addition": what evidence in
    the neuron activations would support it (2-D periodicity), and what
    would falsify it?
29. Design the frequency-ablation experiment of M7: what do you zero,
    where, and what accuracy drop does the theory predict when ablating
    1 of 5 frequencies?
30. Why read logits only at the '=' position? What are the other
    positions' residual streams doing?
31. What is a progress measure formally, and what makes one USEFUL
    (leads the behavioral metric, cheap, mechanistically grounded)?
32. If trig R² were 0.6 post-grokking — neither ~0 nor >0.9 — what are
    the three hypotheses and how do you distinguish them?
33. Why keep the training loop metric-free and do all analysis post-hoc
    over checkpoints? (Reproducibility and adding metrics later.)
34. How would you test whether the SAME frequencies are learned across
    seeds, and what would either answer imply?
35. What does weight tying (NOT used here — W_E and W_U are separate)
    change about the interpretability story if you add it?

### D. Model, training, engineering
36. Why explicit einsum attention instead of nn.MultiheadAttention?
37. Count the parameters of the default config by hand.
38. Why is the EQ token embedding excluded from the spectrum analysis?
39. Why betas=(0.9, 0.98) — what does the higher β2 buy on this tiny,
    full-batch problem?
40. all_pair_logits runs p² = 12,769 sequences. Why batch it, and what
    batch size is sensible on CPU?
41. Why log-spaced checkpoints rather than uniform?
42. What memory does the full checkpoint sweep cost, and how would you
    shrink it (fp16 checkpoints? W_E-only snapshots for the spectrum)?
43. The lstsq design matrix in M-d has shape (p²·p, 2|K|) ≈ 1.4M×12 for
    p=113. Estimate its memory in float64 and justify feasibility.
44. Where does nondeterminism remain after seeding torch (threading,
    MKL reductions), and does it threaten any acceptance criterion?
45. Why is analysis in float64 but training in float32?

### E. Extensions and connections
46. For subtraction (a−b) mod p, what does the predicted logit family
    become? (cos(w(a−b−c))) — what changes in cos_ab_c_basis?
47. For multiplication mod p, why does the additive-frequency story need
    discrete logarithms, and what basis would you fit instead?
48. Predict how t_grok scales with p at fixed frac_train, and design the
    sweep within one CPU-day.
49. How would you detect the circuit forming with SAEs or probes instead
    of the Fourier prior, and what does the prior buy you here?
50. Connect excluded-loss-style leading indicators to what a frontier
    lab would want for capability forecasting. What breaks at scale?
51. If you could log ONE additional scalar during training to strengthen
    the causal story, what would it be and why?
52. Write the 90-second interview narration of the whole project:
    phenomenon → hypothesis → measurement → causal check → lead-time
    result.

---

## 10. Commands

    pip install torch pytest matplotlib numpy
    PYTHONPATH=. pytest tests/ -q                    # all green (Stage A)
    PYTHONPATH=. python examples/run_smoke.py        # p=53, minutes
    PYTHONPATH=. python examples/run_full.py         # p=113 headline run
    PYTHONPATH=. python -m grok.analysis runs/full   # metrics.json
    PYTHONPATH=. python plots/make_plots.py runs/full

END OF SPECIFICATION.
