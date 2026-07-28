# Experiment 2 — (γ, ρ) landscape optima (mode=fine)

## Per-target-k optima (these are the surfaces to optimise)

- **NO@10**: best at γ=1.071, ρ=20, NO=0.8473
- **NO@30**: best at γ=0.429, ρ=20, NO=0.8159
- **NO@100**: best at γ=0.000, ρ=30, NO=0.7281

## How the optimum moves with target k

- γ* ranges 0.00→1.07 as k grows; ρ* ranges 20→30.
- Trend: smooth-γ / higher-ρ as k grows (expected coupling direction). If γ* and ρ* trade off smoothly rather than landing on one cell, the surface is a **ridge** (γ and ρ are partially redundant); if each k pins a single (γ, ρ) cell, it is a **basin**.

## AUC-range summaries

- **AUC NO@k near-local (k=1..10)**: best at γ=1.500, ρ=20, AUC=0.8417
- **AUC NO@k mid-local (k=11..90)**: best at γ=0.429, ρ=20, AUC=0.7846
