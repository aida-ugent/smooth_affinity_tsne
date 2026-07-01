# Experiment 2 — (γ, ρ) landscape optima (mode=coarse)

## Per-target-k optima (these are the surfaces to optimise)

- **NO@10**: best at γ=1.125, ρ=15, NO=0.3973
- **NO@30**: best at γ=0.375, ρ=15, NO=0.3637
- **NO@100**: best at γ=0.000, ρ=15, NO=0.3742

## How the optimum moves with target k

- γ* ranges 0.00→1.12 as k grows; ρ* ranges 15→15.
- Trend: smooth-γ / higher-ρ as k grows (expected coupling direction). If γ* and ρ* trade off smoothly rather than landing on one cell, the surface is a **ridge** (γ and ρ are partially redundant); if each k pins a single (γ, ρ) cell, it is a **basin**.

## AUC-range summaries

- **AUC NO@k near-local (k=1..10)**: best at γ=1.500, ρ=15, AUC=0.4512
- **AUC NO@k mid-local (k=11..90)**: best at γ=0.375, ρ=15, AUC=0.3628
