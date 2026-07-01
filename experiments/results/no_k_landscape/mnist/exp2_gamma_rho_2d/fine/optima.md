# Experiment 2 — (γ, ρ) landscape optima (mode=fine)

## Per-target-k optima (these are the surfaces to optimise)

- **NO@10**: best at γ=0.643, ρ=10, NO=0.4025
- **NO@30**: best at γ=0.214, ρ=10, NO=0.3728
- **NO@100**: best at γ=0.214, ρ=20, NO=0.3740

## How the optimum moves with target k

- γ* ranges 0.21→0.64 as k grows; ρ* ranges 10→20.
- Trend: smooth-γ / higher-ρ as k grows (expected coupling direction). If γ* and ρ* trade off smoothly rather than landing on one cell, the surface is a **ridge** (γ and ρ are partially redundant); if each k pins a single (γ, ρ) cell, it is a **basin**.

## AUC-range summaries

- **AUC NO@k near-local (k=1..10)**: best at γ=1.286, ρ=10, AUC=0.4513
- **AUC NO@k mid-local (k=11..90)**: best at γ=0.214, ρ=10, AUC=0.3656
