# Report for review — Effective neighborhood size as the latent variable behind NO@k

*Addressed to Claude Fable 5. You have no prior context on this project; everything
you need is below. I'd like a critical read: is the claim supported by the data, are
the measures/definitions sound, and what would strengthen or break it?*

---

## 0. Background you need

We work on a research fork of **openTSNE** (t-SNE). We added one affinity knob, **γ**,
a row-wise power transform applied to each point's high-dimensional conditional
neighbor distribution *before* symmetrization:

```
p̃_{j|i} = p_{j|i}^γ / Σ_m p_{m|i}^γ        (γ<1 smooths, γ>1 sharpens, γ=1 = standard t-SNE)
```

The other knob is the usual t-SNE **perplexity ρ**. openTSNE keeps `k_neighbors = 3ρ`
neighbors per point, so each conditional row `p̃_{·|i}` is a distribution over 3ρ
neighbors that sums to 1.

Our quality metric is **NO@k** (Neighborhood Overlap at k, = QNX(k)): for each point,
the fraction of its k high-dimensional nearest neighbors (on PCA-50) that are also
among its k nearest neighbors in the 2-D embedding; averaged over points. Higher is
better.

**Prior result (Experiments 1–3):** sweeping γ and ρ, the NO@k optimum forms a
**ridge**, not a point — low-ρ+smooth-γ mimics high-ρ+sharp-γ. This suggested γ and ρ
are partially redundant. We *inferred* they might act through a single quantity, but
never measured it.

## 1. Hypothesis (what this experiment tests)

> NO@k is maximized when the **effective neighborhood size** — the number of neighbors
> carrying meaningful probability mass in the smoothed row `p̃_{·|i}` — equals ≈ c·k for
> some stable constant c. And γ and ρ influence NO@k *only* by setting this effective
> size.

Two testable predictions:
1. **Collapse:** plot NO@k against (effective size ÷ k), pooling all (γ, ρ, k). If the
   hypothesis holds, points from different (γ, ρ, k) and different datasets collapse
   onto **one hump peaking at a stable ratio**.
2. **Direction:** effective size should be a monotone function of ρ (and of γ) — a
   single latent size that both knobs move.

## 2. Method

Effective size is a property of the high-dimensional rows `p̃_{·|i}` **alone**, so
**no embeddings were run** for this experiment. We recomputed the conditional rows
`p̃` (the cheap Gaussian-perplexity fit + γ power transform — verified byte-identical
to what openTSNE builds internally) and reused the **already-cached NO@k** from a prior
(γ, ρ) grid.

- **Datasets (full):** MNIST (70k), Fashion-MNIST (70k), mouse cortex scRNA-seq
  (Tasic 2018, 23,822 cells). All reduced to PCA-50.
- **Grid:** 15 γ (evenly 0→3) × 10 ρ ({10,20,30,50,75,100,125,150,175,200}) × 5 seeds,
  NO@k measured at **k ∈ {10, 30, 100}**. (The 5 seeds affect only NO@k, via t-SNE init
  jitter; effective size is seed-independent.)
- **Effective-size measures** (per point, then averaged over all points). Given a row
  `p̃` of length `k_neighbors = 3ρ` (sums to 1), with `uniform = 1/k_neighbors`:
  1. `thr_c0.25` — #{j : p̃_j ≥ 0.25·uniform}
  2. `thr_c0.5`  — #{j : p̃_j ≥ 0.5·uniform}
  3. `thr_c1.0`  — #{j : p̃_j ≥ 1.0·uniform}
  4. `participation_ratio` — (Σp̃)²/Σp̃² = 1/Σp̃²  (threshold-free; the principled one)
  5. `two_pow_entropy` — 2^H, H = −Σ p̃ log₂ p̃  (= effective perplexity; a **baseline**
     for contrast, since it's the quantity perplexity already targets)

  The same definition is used for every k and dataset (no per-panel threshold changes).

- **Peak reporting (important):** the ridge is flat, so we do **not** take an argmax.
  We bin eff/k on a **log axis** (it spans ~3 decades because γ≈0 rows blow it up),
  compute per-bin mean & std of NO@k, define *collapse tightness* = median within-bin
  std of NO@k, and report the **peak as a band**: the contiguous eff/k range whose
  bin-mean NO@k is within one within-bin-std of the maximum. 5-seed variance is drawn
  as error bars on every point.

## 3. Results

### 3a. The collapse (prediction 1) — holds

NO@k vs (effective size ÷ k) collapses onto a single hump for **every** measure. On the
all-datasets overlay each dataset traces a clean hump; the humps sit at different
heights (peak NO@k: mnist≈0.37–0.40, fashion≈0.40, mouse≈0.52–0.61) but peak at the
**same eff/k location**. Within a dataset, the three k values (10/30/100) also collapse
onto one hump.

Pooled collapse tightness (lower = tighter; NO@k spans ~0.07–0.6):

| measure | pooled tightness | within-dataset median | peak eff/k band (pooled) |
|---|---|---|---|
| **participation_ratio** | **0.073** | 0.020 | [0.03, 3.10] |
| two_pow_entropy (baseline) | 0.078 | 0.024 | [0.06, 5.17] |
| thr_c0.5 | 0.079 | 0.024 | [0.08, 8.76] |
| thr_c0.25 | 0.082 | 0.026 | [0.06, 9.16] |
| thr_c1.0 | 0.084 | 0.027 | [0.04, 5.41] |

The **participation ratio gives the tightest collapse**. Note within-dataset tightness
(~0.02) is much tighter than pooled (~0.07): the *peak ratio* aligns across datasets,
but absolute NO@k height differs by dataset, which loosens the pool.

### 3b. Peak ratio stability across datasets — holds (as overlapping bands)

Per-dataset peak bands **overlap** for every measure (we test band overlap, not fragile
argmax centers):

| measure | mnist | fashion | mouse | overlap |
|---|---|---|---|---|
| participation_ratio | [0.32,1.24] | [0.32,1.97] | [0.03,3.14] | **[0.32, 1.24]** |
| two_pow_entropy | [0.56,2.12] | [0.36,2.13] | [0.04,3.37] | [0.56, 2.12] |
| thr_c1.0 | [0.39,2.25] | [0.25,2.26] | [0.05,2.30] | [0.39, 2.25] |
| thr_c0.5 | [0.67,3.73] | [0.44,3.74] | [0.13,3.80] | [0.67, 3.73] |
| thr_c0.25 | [0.75,3.97] | [0.50,6.05] | [0.22,4.05] | [0.75, 3.97] |

The headline reading: **the participation ratio peaks at eff/k ≈ [0.32, 1.24], i.e. the
effective number of neighbors ≈ k** at the NO@k optimum.

### 3c. Direction (prediction 2) — holds

For **every** measure, on **every** dataset: effective size **rises monotonically with
ρ** (at fixed γ ∈ {0.8, 1.0, 1.2}) and **falls monotonically with γ** (at fixed
ρ ∈ {30, 100}). No crossovers. The rise vs ρ is roughly linear; smaller γ rises faster.

### 3d. Interpretation

γ and ρ push the effective neighborhood size in opposite directions (ρ up, γ down), and
NO@k depends on (γ, ρ) essentially only through the resulting size, peaking when it hits
≈ k. This is a concrete mechanism for the (γ, ρ) **ridge** seen earlier: any (γ, ρ)
combination that lands the effective size on the same value gives the same NO@k.

## 4. Caveats / things I'd want you to poke at

- **Is participation ratio "peaks at ≈1" a real constant or an artifact of the band
  width?** The bands are wide (flat ridge). The overlap [0.32,1.24] straddles 1 but is
  not tight. Is band-overlap a fair way to claim "stable ratio"?
- **Pooled vs within-dataset tightness.** Should the collapse be judged per-dataset
  (tight, ~0.02) or pooled (looser, ~0.07)? The cross-dataset NO@k height offset is
  real (different datasets, different achievable NO@k). Is there a better normalization?
- **Baseline contrast.** 2^entropy (effective perplexity) is meant as the null "of
  course perplexity sets the size" contrast — yet it collapses almost as well as the
  principled measures. Does that weaken or support the story?
- **Log-binning choice.** eff/k spans ~0.02–60 (γ≈0 uniform rows). We log-bin; a prior
  linear-bin version mislocated the peak. Are there better estimators of the hump peak
  than binned means (e.g. LOWESS, or a parametric hump fit)?
- **k range is only {10, 30, 100}.** Enough to claim k-collapse?

## 5. Reproducibility

Driver: `experiments/no_k_effective_size.py` (openTSNE fork). Raw per-cell effective
sizes in `.../exp4_effective_size/effective_size_cells.csv`; merged eff-size↔NO@k in
`no_k_vs_effsize.csv`; cross-dataset summary + this finding in
`exp4_effective_size_summary/{FINDINGS.md, collapse_summary.csv, direction_summary.csv}`.
