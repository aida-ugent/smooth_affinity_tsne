# NO@k optimization-landscape experiments (γ and (γ, ρ))

These two experiments characterise the **shape of the NO@k objective** before we
build a derivative-free optimiser for γ (and later (γ, ρ)) at a *fixed target k*.

- **NO@k (= QNX(k))**: for each point, the fraction of its k high-dimensional
  nearest neighbours (on the 50-PC MNIST) also among its k nearest neighbours in
  the 2-D embedding; mean over all points. HD k-NN on 50 PCs, LD k-NN on the
  embedding, self excluded.
- **γ**: the repo's row-wise smoothing power transform on the conditional
  affinities (`openTSNE/affinity.py`); γ<1 smooths, γ>1 sharpens, γ=1 = upstream.
- **ρ**: the t-SNE perplexity.

All scripts reuse the paper's MNIST loading (pixels/255 → PCA-50) and t-SNE
settings (PCA init, early exaggeration 12 for 250 iters, then 750 iters), so
results are comparable to the paper. Variance across seeds comes from the PCA
init's per-seed jitter (paper-faithful).

## Files

| Script | Purpose |
|--------|---------|
| `../../no_k_landscape_common.py` | Shared helpers: data loading, neighbor-graph caching, γ→P (reuses `joint_probabilities_nn`), t-SNE-from-P, vectorised NO@k + one-pass full curve, figures, and a `--self_test`. |
| `../../no_k_gamma_sweep_1d.py` | **Experiment 1** — NO@k vs γ at fixed ρ for four (k, ρ) settings, with a ±1 std band over seeds. |
| `../../no_k_gamma_rho_2d.py` | **Experiment 2** — NO@k over the (γ, ρ) grid: per-target-k surfaces + AUC summaries. |

## How to run

```bash
conda activate graph_tsne_py310     # env with the editable openTSNE + deps

# 0) Correctness check (seconds): cached-P == fresh affinity; NO@k == brute force
python no_k_gamma_sweep_1d.py --self_test

# 1) Cheap smoke test on a subsample first
python no_k_gamma_sweep_1d.py --n_subsample 5000 --seeds 0 --n_gamma 9
python no_k_gamma_rho_2d.py   --mode coarse --n_subsample 5000 --seeds 0

# 2) Full Experiment 1 (50 γ × 5 seeds × 4 settings, full MNIST)
python no_k_gamma_sweep_1d.py
python no_k_gamma_sweep_1d.py --staircase   # + the piecewise-constant sanity check

# 3) Experiment 2: coarse pass first, inspect for a ridge/basin, then fine
python no_k_gamma_rho_2d.py --mode coarse
python no_k_gamma_rho_2d.py --mode fine

# Regenerate any figure from saved CSVs without recomputing embeddings
python no_k_gamma_sweep_1d.py --plot_only
python no_k_gamma_rho_2d.py --mode coarse --plot_only
```

## Changing the grid resolution

Edit the constants at the top of each script (no need to touch the logic):

- **Experiment 1** (`no_k_gamma_sweep_1d.py`): `N_GAMMA`, `GAMMA_MAX`, `SEEDS`,
  `SETTINGS` (list of `(k, ρ)`), `K_MAX`, and the `STAIRCASE_*` constants.
  CLI overrides: `--n_gamma`, `--gamma_max`, `--seeds`, `--n_subsample`.
- **Experiment 2** (`no_k_gamma_rho_2d.py`): `GAMMAS_COARSE`/`RHOS_COARSE`,
  `GAMMAS_FINE`/`RHOS_FINE`, `SEEDS`, `TARGET_KS`, `AUC_NEAR`/`AUC_MID`, `K_MAX`.
  CLI: `--mode {coarse,fine}`, `--seeds`, `--n_subsample`.

Recommended order: **coarse 2D first** → look at the per-k surfaces for a
ridge/basin → only then run the fine grid. The embedding count (one optimisation
per (ρ, γ, seed)) is the dominant cost on full 70k MNIST; the caching removes
redundant k-NN/affinity work but not the optimisations.

### Threads (`--n_jobs`)

These experiments run hundreds of *sequential* small embeddings. openTSNE and
sklearn do not scale past ~8–16 threads, so `n_jobs=-1` on a many-core machine
**oversubscribes and is dramatically slower** here (observed ~25 s/embedding at
`-1` vs a few seconds at `8` on a 256-core box). The default is therefore
`--n_jobs 8`. Embedding results are (essentially) independent of thread count, so
this only affects speed; pass `--n_jobs -1` for paper-exact threading on few-core
machines.

## Outputs

```
exp1_gamma_sweep_1d/
  no_k_gamma_sweep_1d.csv / .npz        raw per-(k, ρ, γ, seed) NO@k
  no_k{K}_rho{R}_gamma_sweep.png        one per (k, ρ): NO@k vs γ ±1 std
  no_k_gamma_sweep_1d_panel.png         2×2 combined panel
  no_k_gamma_sweep_1d_optima.csv        located peaks (γ*, interior/boundary, band)
  no_k_staircase.csv / .png             optional fine-grid staircase
  settings.json / settings.md           full reproducibility log
exp2_gamma_rho_2d/{coarse,fine}/
  no_k_gamma_rho_2d.csv / .npz          raw per-(ρ, γ, seed) NO@k + AUCs
  surface_no_at_{10,30,100}.png         per-target-k objective surfaces
  surface_auc_near.png / surface_auc_mid.png   AUC-range summaries
  surfaces_per_k_panel.png              combined per-k panel
  optima.csv / optima.md                best (γ, ρ) per k + ridge/basin note
  settings.json / settings.md
```
