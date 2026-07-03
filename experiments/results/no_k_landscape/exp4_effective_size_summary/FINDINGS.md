# Experiment 4 — Effective neighborhood size vs NO@k: findings

## 1. Which measure gives the tightest collapse

Collapse tightness = median within-bin std of NO@k on log-spaced eff/k bins (lower ⇒ tighter). NO@k spans ~0.07–0.6, so a tightness ~0.02 is a tight collapse.

Pooled (all datasets, all γ, ρ, k):

- **participation_ratio**: pooled tightness=0.0734 (within-dataset median 0.0200), peak eff/k ∈ [0.03, 3.10]
- **two_pow_entropy**: pooled tightness=0.0775 (within-dataset median 0.0235), peak eff/k ∈ [0.06, 5.17]
- **thr_c0.5**: pooled tightness=0.0791 (within-dataset median 0.0243), peak eff/k ∈ [0.08, 8.76]
- **thr_c0.25**: pooled tightness=0.0817 (within-dataset median 0.0260), peak eff/k ∈ [0.06, 9.16]
- **thr_c1.0**: pooled tightness=0.0844 (within-dataset median 0.0267), peak eff/k ∈ [0.04, 5.41]

➡ **Tightest pooled collapse: `participation_ratio`** (tightness 0.0734).
Note: within a single dataset the collapse is much tighter (~0.02) than pooled (~0.07) — the *peak ratio* aligns across datasets, but the absolute NO@k height differs (mnist≈0.37, fashion≈0.40, mouse≈0.52 at the peak), which loosens the pool.

## 2. Is the peak eff/k ratio stable within & across datasets

Peak is a **band** (bin-mean NO@k within one within-bin std of the max) — a flat ridge ⇒ wide band, so we test whether the three per-dataset bands *overlap* rather than comparing fragile argmax centers.

- **thr_c0.25**: mnist=[0.75,3.97], fashion_mnist=[0.50,6.05], mouse=[0.22,4.05]; bands **overlap** at eff/k ∈ [0.75, 3.97] ⇒ stable.
- **thr_c0.5**: mnist=[0.67,3.73], fashion_mnist=[0.44,3.74], mouse=[0.13,3.80]; bands **overlap** at eff/k ∈ [0.67, 3.73] ⇒ stable.
- **thr_c1.0**: mnist=[0.39,2.25], fashion_mnist=[0.25,2.26], mouse=[0.05,2.30]; bands **overlap** at eff/k ∈ [0.39, 2.25] ⇒ stable.
- **participation_ratio**: mnist=[0.32,1.24], fashion_mnist=[0.32,1.97], mouse=[0.03,3.14]; bands **overlap** at eff/k ∈ [0.32, 1.24] ⇒ stable.
- **two_pow_entropy**: mnist=[0.56,2.12], fashion_mnist=[0.36,2.13], mouse=[0.04,3.37]; bands **overlap** at eff/k ∈ [0.56, 2.12] ⇒ stable.

## 3. Direction of effective size vs ρ (per measure)

- **thr_c0.25**: all rise
- **thr_c0.5**: all rise
- **thr_c1.0**: all rise
- **participation_ratio**: all rise
- **two_pow_entropy**: all rise

Cross-measure agreement on ρ-direction: all measures rise with ρ (consistent)

## Guardrails honored

- One fixed definition per measure across all k and datasets (threshold denominator is always 1/k_neighbors).
- Peaks reported as bands (± one within-bin std), never a single argmax; 5-seed variance drawn as error bars on every scatter.
- Recomputed affinities use knn_seed=42 / the same effective perplexity as the cached NO@k, so eff-size and NO@k share a graph.
