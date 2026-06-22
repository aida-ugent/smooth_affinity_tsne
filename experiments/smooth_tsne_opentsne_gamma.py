#!/usr/bin/env python
"""
smooth_tsne_opentsne_gamma.py
=============================
Experiments 4–10 using a modified openTSNE where ``gamma`` is a native
affinity parameter.  The entire custom P-injection pipeline is replaced by a
single ``run_tsne`` call.

Experiments included:  1–10
Note: exp 1, 2, 3 require the conditional P from build_affinity().

Usage example
-------------
  python smooth_tsne_opentsne_gamma.py \\
      --dataset mnist \\
      --gamma_s 0.7 \\
      --n_runs 5

  # Regenerate figures only (no t-SNE):
  python smooth_tsne_opentsne_gamma.py --dataset mnist --plot_only
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Global font sizes — applied to every figure so titles, axis labels, ticks and
# legends are large and consistent across all experiments.
plt.rcParams.update({
    "axes.titlesize":        17,   # plot titles
    "axes.labelsize":        15,   # x / y axis labels
    "xtick.labelsize":       12,   # tick numbers
    "ytick.labelsize":       12,
    "legend.fontsize":       13,   # legend entries
    "legend.title_fontsize": 13,   # legend title
})
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

# ── Data loaders ──────────────────────────────────────────────────────────────
# Replace these imports with your own load functions if needed.
# Each must return (X_pca, y, ...) consistent with load_data() below.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "data"))
from load_data import load_mouse_data, load_mnist_data, load_adult_data


# =============================================================================
# Modified openTSNE API  — adjust attribute names to your build
# =============================================================================

def build_affinity(X_pca, perplexity, gamma=1.0, n_jobs=-1):
    """
    Build a modified openTSNE affinity object with gamma applied natively.
    Returns (P_joint, P_cond, knn_idx, knn_dist).

    Adjust the attribute names below to match your openTSNE modification:
      aff.P              — symmetrized joint P  (standard openTSNE attribute)
      aff.P_conditional  — conditional P, post-gamma  (you add this)
      aff.knn_indices    — shape (n, k_hd)  (expose from your build)
      aff.knn_distances  — shape (n, k_hd)  (expose from your build)
    """
    from openTSNE.affinity import PerplexityBasedNN
    aff      = PerplexityBasedNN(X_pca, perplexity=perplexity, gamma=gamma,
                                  n_jobs=n_jobs)
    P_joint  = aff.P
    P_cond   = aff.P_conditional
    knn_idx  = aff.knn_indices
    knn_dist = aff.knn_distances
    return P_joint, P_cond, knn_idx, knn_dist


def run_tsne_from_joint(X_pca, P_joint, random_state=42, n_jobs=-1,
                         n_iter_ee=250, n_iter_main=750, ee=12):
    """Run t-SNE optimization given a precomputed symmetrized joint P."""
    from openTSNE import TSNEEmbedding, initialization

    class _FixedAff:
        def __init__(self, P):
            self.P = P

    init = initialization.pca(X_pca, random_state=random_state)
    emb  = TSNEEmbedding(init, _FixedAff(P_joint), n_jobs=n_jobs)
    emb.optimize(n_iter_ee,   exaggeration=ee,  momentum=0.5, inplace=True)
    emb.optimize(n_iter_main, exaggeration=1.0, momentum=0.8, inplace=True)
    return np.array(emb)

def run_tsne(X_pca, perplexity, gamma=1.0, random_state=42,
             n_jobs=-1, n_iter_ee=250, n_iter_main=750, ee=12):
    """One-shot helper: build affinity with gamma then run optimization.

    TSNE.__init__ does not accept gamma, so we build PerplexityBasedNN
    (which does) and pass it as a precomputed affinities object to TSNE.fit().
    """
    from openTSNE import TSNE
    from openTSNE.affinity import PerplexityBasedNN
    aff = PerplexityBasedNN(X_pca, perplexity=perplexity, gamma=gamma,
                             n_jobs=n_jobs)
    return np.array(
        TSNE(
            n_jobs=n_jobs,
            random_state=random_state,
            early_exaggeration=ee,
            early_exaggeration_iter=n_iter_ee,
            n_iter=n_iter_main,
        ).fit(X_pca, affinities=aff)
    )


# =============================================================================
# Constants
# =============================================================================

C_STD       = "#2C3E50"
C_GREEN     = "#27AE60"
_CB_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]

# ─── Semantic colours for the γ comparison (Okabe–Ito, colourblind-safe) ─────
# One colour per role, used consistently across every discrete standard/smooth/
# sharp/matched plot (exp 1, 2, 7, 9, 10).  The two series compared most often —
# standard and smooth — are blue vs orange, the most reliably separable
# colourblind-safe pair (they differ in BOTH hue and luminance, so they also
# read clearly in greyscale).  "matched" is a standard run at a different
# perplexity, so it stays in the blue family (sky-blue) while orange keeps the
# key smooth line visually distinct.
#   NB: the continuous γ-sweep plots (exp 4, 6, 8) intentionally use a separate
#   cool→warm gradient (Blues for γ<1, Reds for γ>1) and are left untouched.
C_STANDARD = "#0072B2"   # blue            — standard t-SNE (γ=1.0)
C_SMOOTH   = "#E69F00"   # orange          — smooth   t-SNE (γ<1)
C_SHARP    = "#CC79A7"   # reddish-purple  — sharp    t-SNE (γ>1)
C_MATCHED  = "#56B4E9"   # sky-blue        — standard at matched perplexity (exp7)

# Exp8 fixed colors per perplexity value (colorblind-friendly)
_EXP8_COLORS = {
    30:  "#0072B2",  # blue
    50:  "#E69F00",  # orange
    100: "#009E73",  # green
    200: "#CC79A7",  # pink
}

# Per-dataset y-axis range for the exp8 global-Spearman plot.  Global Spearman ρ
# sits in a different band on each dataset, so a single shared range either
# clips or cramps; these are scaled to each dataset's observed values.
_EXP8_YLIM = {
    "mnist": (0.30, 0.45),
    "mouse": (0.40, 0.80),
    "adult": (0.20, 0.55),
}
_EXP8_YLIM_DEFAULT = (0.20, 0.60)

# Per-dataset y-axis range for the exp7 Neighborhood-Overlap comparison
# (mean NH@k curves), scaled to each dataset's observed band.
_EXP7_YLIM = {
    "mnist": (0.24, 0.44),
    "mouse": (0.20, 0.68),
    "adult": (0.35, 0.60),
}
_EXP7_YLIM_DEFAULT = (0.22, 0.45)

# Per-dataset y-axis range for the exp9 best-ρ envelope plot (09a).  The
# envelope (max over perplexities) sits higher than the single-ρ curves, so
# these bands differ from the exp7 ones.
_EXP9_YLIM = {
    "mnist": (0.33, 0.57),
    "mouse": (0.45, 0.70),
    "adult": (0.38, 0.65),
}
_EXP9_YLIM_DEFAULT = (0.22, 0.45)

_plot_save_dir = [None]
# Prefix prepended to every saved figure filename (e.g. "mouse_", "mnist_",
# "adult_"); set once in main()/plot_all_from_csv() from the dataset name.
_fig_name_prefix = [""]


# =============================================================================
# Helpers
# =============================================================================

def _save_fig(name, dpi=150):
    d = _plot_save_dir[0]
    if d is None:
        raise RuntimeError("_plot_save_dir[0] not set before calling _save_fig")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{_fig_name_prefix[0]}{name}.png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _global_spearman(X, Y, random_state=42, n_sample=5000):
    from scipy.spatial.distance import pdist
    n = len(X)
    if n > n_sample:
        idx = np.sort(np.random.default_rng(random_state).choice(n, size=n_sample, replace=False))
        X, Y = X[idx], Y[idx]
    rho, _ = spearmanr(pdist(X), pdist(Y))
    return float(rho)


def _nh_auc(nh_curve, k_values, k_lo, k_hi):
    mask = (k_values >= k_lo) & (k_values <= k_hi)
    if not mask.any():
        return float("nan")
    return float(nh_curve[mask].mean())


def _gamma_color(g, smooth_list, sharp_list):
    if np.isclose(g, 1.0):
        return C_STD
    if g < 1.0:
        idx  = smooth_list.index(g)
        blues = plt.cm.Blues(np.linspace(0.4, 0.85, max(len(smooth_list), 1)))
        return blues[idx]
    idx  = sharp_list.index(g)
    reds = plt.cm.Reds(np.linspace(0.4, 0.85, max(len(sharp_list), 1)))
    return reds[idx]


def _make_seeds(master_seed, n_runs):
    rng = np.random.default_rng(master_seed)
    return [int(s) for s in rng.integers(0, 2**31, size=n_runs)]


def _knn_indices(X, k, n_jobs=-1):
    """Return k-NN indices (excluding self), shape (n, k)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def _eff_perp_per_point(P_cond):
    """Effective perplexity 2^H per point from a sparse (n,n) P_conditional.

    Works directly on .data / .indptr — no Python loop over rows.
    """
    d = P_cond.data
    safe = np.where(d > 0, d, 1.0)
    neg_plogp = np.where(d > 0, -d * np.log2(safe), 0.0)
    H = np.add.reduceat(neg_plogp, P_cond.indptr[:-1])
    return 2.0 ** H


def _ranked_vals_matrix(P_cond, knn_indices):
    """Return (n, k) array of P(j|i) values in neighbor-rank order.

    Uses scipy sparse fancy indexing — no Python loop over rows.
    """
    n, k = knn_indices.shape
    row_idx = np.repeat(np.arange(n), k)
    col_idx = knn_indices.ravel()
    vals = np.asarray(P_cond[row_idx, col_idx]).ravel()
    return vals.reshape(n, k)


def _nh_curve_from_nbrs(hd_nbrs, ld_nbrs, k_values):
    """Compute NH@k for each k in k_values from pre-computed index arrays."""
    n = len(hd_nbrs)
    return np.array([
        np.mean([len(set(hd_nbrs[i, :k]) & set(ld_nbrs[i, :k])) / k
                 for i in range(n)])
        for k in k_values
    ])


def _heatmap(ax, pivot, title, cmap):
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns],
                       rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("γ")
    ax.set_ylabel("Perplexity")
    ax.set_title(title)
    vmin, vmax = pivot.values.min(), pivot.values.max()
    for ri in range(pivot.shape[0]):
        for ci in range(pivot.shape[1]):
            v  = pivot.values[ri, ci]
            tc = "white" if (v - vmin) / (vmax - vmin + 1e-9) < 0.6 else "black"
            ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color=tc)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def _heatmap_mean_std(ax, pivot_mean, pivot_std, title, cmap):
    """Heatmap coloured by mean; each cell annotated 'mean\\n±std'."""
    im = ax.imshow(pivot_mean.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot_mean.columns],
                       rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_mean.index)))
    ax.set_yticklabels(pivot_mean.index)
    ax.set_xlabel("γ")
    ax.set_ylabel("Perplexity")
    ax.set_title(title)
    for ri in range(pivot_mean.shape[0]):
        for ci in range(pivot_mean.shape[1]):
            m = pivot_mean.values[ri, ci]
            ax.text(ci, ri, f"{m:.3f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _tsne_bandwidths(knn_distances, perplexity, tol=1e-5, max_iter=50):
    """Binary-search sigma_i s.t. H(P_i) = log(perplexity). Used by exp3."""
    n      = knn_distances.shape[0]
    sigmas = np.empty(n)
    logU   = np.log(perplexity)
    for i in range(n):
        dist2 = knn_distances[i].astype(float) ** 2
        bmin, bmax, beta = -np.inf, np.inf, 1.0
        for _ in range(max_iter):
            P    = np.exp(-dist2 * beta)
            sumP = P.sum()
            if sumP <= 0:
                break
            H     = np.log(sumP) + beta * np.dot(dist2, P) / sumP
            Hdiff = H - logU
            if abs(Hdiff) < tol:
                break
            if Hdiff > 0:
                bmin = beta
                beta = 2 * beta if np.isinf(bmax) else 0.5 * (beta + bmax)
            else:
                bmax = beta
                beta = 0.5 * beta if np.isinf(bmin) else 0.5 * (beta + bmin)
        sigmas[i] = 1.0 / np.sqrt(2.0 * max(beta, 1e-300))
    return sigmas


# =============================================================================
# Shared: sensitivity grid  (used by Exp 2b, 6, and 8)
# =============================================================================

def _run_sensitivity_grid(X_pca, X_eval, gammas, perplexities, out_dir,
                           n_jobs=-1, master_seed=42, n_runs=1,
                           n_iter_ee=250, n_iter_main=750, ee=12):
    """
    For every (perplexity, gamma, seed): run t-SNE, compute AUC_1_10,
    AUC_11_90, global_spearman, median_eff_perp.

    Affinities are built once per (p, g) — deterministic — and reused across
    seeds.  Saves incrementally after each (perplexity, seed) pair.

    CSV columns: seed, perplexity, gamma, AUC_1_10, AUC_11_90,
                 global_spearman, median_eff_perp
    """
    print("\n  Running sensitivity grid ...")
    k_values = np.arange(1, 91)
    csv_path = os.path.join(out_dir, "sensitivity_grid.csv")
    seeds    = _make_seeds(master_seed, n_runs)

    existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    if not existing.empty and "seed" not in existing.columns:
        existing["seed"] = seeds[0]
    done_pairs = (set(zip(existing["perplexity"], existing["seed"]))
                  if not existing.empty else set())

    # Pre-compute HD k-NN once — independent of perplexity and gamma
    print("    Pre-computing HD KNN ...")
    hd_nbrs = _knn_indices(X_eval, k=int(k_values.max()), n_jobs=n_jobs)

    for p in sorted(perplexities):
        seeds_needed = [s for s in seeds if (p, s) not in done_pairs]
        if not seeds_needed:
            print(f"    ρ={p}: all seeds done, skipping.")
            continue

        # Build affinity once per (p, g) — P_cond is deterministic
        print(f"    ρ={p}: building affinities ...")
        aff_cache = {}  # g → (P_joint, median_eff_perp)
        for g in sorted(gammas):
            print(f"      aff γ={g:.2f} ...", end=" ", flush=True)
            P_joint, P_cond, _, _ = build_affinity(
                X_pca, perplexity=p, gamma=g, n_jobs=n_jobs)
            med_ep = float(np.median(_eff_perp_per_point(P_cond)))
            aff_cache[g] = (P_joint, med_ep)
            print(f"med_eff_perp={med_ep:.1f}")

        for seed in seeds_needed:
            print(f"      ρ={p}, seed={seed}")
            seed_rows = []
            for g in sorted(gammas):
                print(f"        γ={g:.2f} ...", end=" ", flush=True)
                P_joint, med_ep = aff_cache[g]
                Y = run_tsne_from_joint(X_pca, P_joint, random_state=seed,
                                        n_jobs=n_jobs, n_iter_ee=n_iter_ee,
                                        n_iter_main=n_iter_main, ee=ee)
                ld_nbrs = _knn_indices(np.asarray(Y), k=int(k_values.max()),
                                       n_jobs=n_jobs)
                nh = _nh_curve_from_nbrs(hd_nbrs, ld_nbrs, k_values)
                seed_rows.append({
                    "seed": seed, "perplexity": p, "gamma": g,
                    "AUC_1_10":        _nh_auc(nh, k_values, 1, 10),
                    "AUC_11_90":       _nh_auc(nh, k_values, 11, 90),
                    "global_spearman": _global_spearman(X_eval, np.asarray(Y),
                                                        random_state=master_seed, n_sample=5000),
                    "median_eff_perp": med_ep,
                })
                print("done")
            existing = pd.concat([existing, pd.DataFrame(seed_rows)],
                                 ignore_index=True)
            existing.to_csv(csv_path, index=False)
            done_pairs.add((p, seed))
            print(f"      checkpoint: ρ={p} seed={seed} saved → {csv_path}")

    print(f"  Grid complete ({len(existing)} rows) → {csv_path}")
    return existing


# =============================================================================
# Experiment 1 – Affinity row sharpness (single representative point)
# =============================================================================

def exp1_affinity_sharpness(P_cond, knn_idx, out_dir,
                              perplexity=30, gamma_s=0.5, top_ranks=20):
    """
    Grouped bar chart of P(j|i) vs neighbour rank for a single representative
    point: for each rank, two side-by-side bars compare
    (a) standard t-SNE  and  (b) smoothing with gamma_s.

    Only the ``top_ranks`` highest-ranked neighbours are drawn (the tail
    probabilities are vanishingly small and would clutter the chart); the CSV
    still records every rank.

    P_cond, knn_idx come from build_affinity(..., gamma=1.0).
    CSV: affinity_rows.csv
    """
    print("\n--- Exp 1: Affinity row sharpness ---")
    plots_dir = os.path.join(out_dir, "exp1_affinity_sharpness")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    n, k_hd = knn_idx.shape
    rows = _ranked_vals_matrix(P_cond, knn_idx)

    ranks = np.arange(1, k_hd + 1)
    top5  = rows[:, :5].sum(axis=1)
    idx_p = int(np.argmin(np.abs(top5 - top5.mean())))

    vals_std = rows[idx_p].copy()
    vals_smo = np.maximum(vals_std, 1e-300) ** gamma_s
    vals_smo /= vals_smo.sum()
    vals_smo = np.sort(vals_smo)[::-1]

    pd.DataFrame({
        "rank": ranks,
        "p_standard": vals_std,
        f"p_smooth_gamma{gamma_s}": vals_smo,
    }).to_csv(os.path.join(plots_dir, "affinity_rows.csv"), index=False)

    C_STD_1 = C_STANDARD
    C_SMO_1 = C_SMOOTH

    n_show = min(top_ranks, k_hd)
    x      = np.arange(n_show)
    width  = 0.4

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, vals_std[:n_show], width, color=C_STD_1,
           label=f"Standard  (ρ={perplexity},  γ=1.0)")
    ax.bar(x + width / 2, vals_smo[:n_show], width, color=C_SMO_1,
           label=f"Smooth  (ρ={perplexity},  γ={gamma_s})")
    ax.set_xlabel("Neighbor rank")
    ax.set_ylabel("Conditional probability  P(j|i)")
    ax.set_title(f"(a) Affinity row sharpness  (ρ={perplexity})")
    ax.set_xticks(x)
    ax.set_xticklabels(ranks[:n_show])
    ax.legend(frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("01_affinity_sharpness")
    print(f"  Exp 1 done → {plots_dir}")


# =============================================================================
# Experiment 2a – Effective perplexity distribution
# =============================================================================

def exp2_effective_perplexity(P_cond_base, P_cond_smooth, P_cond_sharp,
                                out_dir,
                                gamma_s=0.7, gamma_h=1.5, perplexity=30):
    """
    Distribution of effective perplexities for standard / smooth / sharp.
    P_cond_* come from build_affinity at gamma=1.0 / gamma_s / gamma_h.
    CSV: eff_perplexity_per_point.csv
    """
    print("\n--- Exp 2a: Effective perplexity distribution ---")
    plots_dir = os.path.join(out_dir, "exp2_eff_perplexity")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    _CONST_THR = 0.005

    ep_std = _eff_perp_per_point(P_cond_base)
    ep_smo = _eff_perp_per_point(P_cond_smooth)
    ep_shp = _eff_perp_per_point(P_cond_sharp)

    pd.DataFrame({
        "standard":          ep_std,
        f"smooth_g{gamma_s}": ep_smo,
        f"sharp_g{gamma_h}":  ep_shp,
    }).to_csv(os.path.join(plots_dir, "eff_perplexity_per_point.csv"),
              index=False)

    triples = [
        (ep_std, f"standard  (ρ={perplexity})", C_STANDARD),
        (ep_smo, f"smooth  γ={gamma_s}",        C_SMOOTH),
        (ep_shp, f"sharp  γ={gamma_h}",         C_SHARP),
    ]
    n_pts   = len(ep_std)
    varying = [d for d, _, _ in triples if d.std() >= _CONST_THR]
    lo = min(d.min() for d in varying) if varying else 0
    hi = max(d.max() for d in varying) if varying else 1
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    bins      = np.linspace(lo, hi, 51)
    max_count = max(
        (int(np.histogram(d, bins=bins)[0].max())
         for d, _, _ in triples if d.std() >= _CONST_THR),
        default=0,
    )
    max_count = max(max_count, n_pts)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylim(0, max_count * 1.25)
    for d, lbl, c in triples:
        if d.std() < _CONST_THR:
            ax.vlines(float(d.mean()), 0, n_pts, colors=c, lw=4, zorder=5,
                      label=lbl)
        else:
            ax.hist(d, bins=bins, alpha=0.65, color=c, label=lbl)
    ax.set_xlabel("Effective perplexity")
    ax.set_ylabel("Count")
    ax.set_title("(b) Effective perplexity distribution")
    ax.legend(frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("02a_eff_perp")
    print(f"  Exp 2a done → {plots_dir}")


def exp2b_median_eff_perp_heatmap(df_grid, out_dir):
    """Median effective perplexity heatmap from sensitivity_grid.csv."""
    print("\n--- Exp 2b: Median effective perplexity heatmap ---")
    plots_dir = os.path.join(out_dir, "exp2_eff_perplexity")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    df_mean = df_grid.groupby(["perplexity", "gamma"],
                               as_index=False)["median_eff_perp"].mean()
    pivot = df_mean.pivot(index="perplexity", columns="gamma",
                           values="median_eff_perp")
    fig, ax = plt.subplots(figsize=(8, 5))
    _heatmap(ax, pivot, "Median effective perplexity", "cividis")
    plt.tight_layout()
    _save_fig("02b_median_eff_perp_heatmap")
    print(f"  Exp 2b done → {plots_dir}")


# =============================================================================
# Experiment 3 – Per-point Δρ correlations
# =============================================================================

def exp3_delta_perp_correlations(P_cond_base, P_cond_smooth, knn_idx,
                                   knn_dist, out_dir,
                                   perplexity=30, gamma_s=0.7):
    """
    Hexbin: Δρ (change in effective perplexity) vs top-5 mass and σ_i.
    P_cond_* from build_affinity; knn_dist from build_affinity(..., gamma=1.0).
    CSV: delta_perp_per_point.csv
    """
    print("\n--- Exp 3: Δρ correlations ---")
    plots_dir = os.path.join(out_dir, "exp3_delta_perp")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    from scipy.stats import spearmanr as _sr

    delta_perp = _eff_perp_per_point(P_cond_smooth) - _eff_perp_per_point(P_cond_base)

    n         = knn_idx.shape[0]
    rows_mat  = _ranked_vals_matrix(P_cond_base, knn_idx)
    top5      = np.maximum(rows_mat[:, :5], 1e-300).sum(axis=1)

    sigmas = _tsne_bandwidths(knn_dist, perplexity)

    pd.DataFrame({
        "delta_rho": delta_perp,
        "top5_mass": top5,
        "sigma_i":   sigmas,
    }).to_csv(os.path.join(plots_dir, "delta_perp_per_point.csv"), index=False)

    candidates = [
        (top5,   "Top-5 conditional mass",   "03a_delta_rho_vs_top5"),
        (sigmas, r"t-SNE bandwidth  $\sigma_i$", "03b_delta_rho_vs_sigma"),
    ]
    for xvar, xlabel, fname in candidates:
        rho, _ = _sr(xvar, delta_perp)
        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hexbin(xvar, delta_perp, gridsize=45, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta$ perplexity")
        ax.set_title(f"Spearman ρ = {rho:.3f}")
        ax.set_ylim(bottom=0)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig(fname)
    print(f"  Exp 3 done → {plots_dir}")


# =============================================================================
# Experiment 4 – Neighborhood Overlap curves: fixed perplexity, varying γ
# =============================================================================

def exp4_nh_gamma_sweep(X_pca, X_eval, out_dir,
                          perplexity=30, k_max=200,
                          gammas=(0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0),
                          n_jobs=-1, master_seed=42, n_runs=1,
                          n_iter_ee=250, n_iter_main=750, ee=12):
    """
    Neighborhood Overlap for k=1…k_max: fixed perplexity, varying gamma.
    CSV (tidy): seed, k, gamma_0.00, gamma_0.30, ...
    """
    print("\n--- Exp 4: Neighborhood Overlap gamma sweep ---")
    plots_dir = os.path.join(out_dir, "exp4_neighborhood_overlap_gamma_sweep")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    gammas      = list(gammas)
    smooth_list = [g for g in gammas if g < 1.0]
    sharp_list  = [g for g in gammas if g > 1.0]
    k_values    = np.arange(1, k_max + 1)
    seeds       = _make_seeds(master_seed, n_runs)
    csv_path    = os.path.join(plots_dir, "neighborhood_overlap_gamma_sweep.csv")

    existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    if not existing.empty and "seed" not in existing.columns:
        existing["seed"] = seeds[0]
    done_seeds = set(existing["seed"].unique()) if not existing.empty else set()
    seeds_todo = [s for s in seeds if s not in done_seeds]

    if seeds_todo:
        print(f"  Pre-computing HD KNN at k={k_max} ...")
        hd_nbrs = _knn_indices(X_eval, k=k_max, n_jobs=n_jobs)

        for seed in seeds_todo:
            print(f"  seed={seed} ...")
            curves_seed = {}
            for g in gammas:
                print(f"    γ={g:.2f} ...", end=" ", flush=True)
                Y = run_tsne(X_pca, perplexity=perplexity, gamma=g,
                             random_state=seed, n_jobs=n_jobs,
                             n_iter_ee=n_iter_ee, n_iter_main=n_iter_main,
                             ee=ee)
                ld_nbrs      = _knn_indices(np.asarray(Y), k=k_max, n_jobs=n_jobs)
                curves_seed[g] = _nh_curve_from_nbrs(hd_nbrs, ld_nbrs, k_values)
                print("done")
            seed_rows = []
            for k_idx, k in enumerate(k_values):
                row = {"seed": seed, "k": k}
                for g in gammas:
                    row[f"gamma_{g:.2f}"] = curves_seed[g][k_idx]
                seed_rows.append(row)
            existing = pd.concat([existing, pd.DataFrame(seed_rows)],
                                 ignore_index=True)
            existing.to_csv(csv_path, index=False)
            print(f"  checkpoint: seed={seed} saved → {csv_path}")
    else:
        print(f"  All {len(seeds)} seeds already done, skipping computation.")

    out_df = existing

    # ── Plot: thin individual seeds + bold mean ───────────────────────────────
    k_vals_plot = np.sort(out_df["k"].unique())
    seed_list   = out_df["seed"].unique()

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in gammas:
        col     = f"gamma_{g:.2f}"
        c       = _gamma_color(g, smooth_list, sharp_list)
        lw_mean = 2.6 if np.isclose(g, 1.0) else 2.0
        ls      = "-" if np.isclose(g, 1.0) else ("--" if g < 1.0 else "-.")
        lbl     = f"γ={g:.2f}" + ("  (standard)" if np.isclose(g, 1.0) else "")
        for seed in seed_list:
            sub = out_df[out_df["seed"] == seed].sort_values("k")
            ax.plot(sub["k"].values, sub[col].values,
                    color=c, lw=0.6, ls=ls, alpha=0.2)
        mn = out_df.groupby("k")[col].mean().reindex(k_vals_plot).values
        ax.plot(k_vals_plot, mn, color=c, lw=lw_mean, ls=ls, label=lbl)
    ax.set_xlabel("k")
    ax.set_ylabel("Neighborhood Overlap")
    ax.set_title(f"Neighborhood Overlap  —  ρ={perplexity}, varying γ")
    ax.legend(ncol=2, fontsize=13, frameon=False, loc="lower right")
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("04_neighborhood_overlap_gamma_sweep")
    print(f"  Exp 4 done → {plots_dir}")


# =============================================================================
# Experiment 5 – Embedding comparison (standard vs smooth)
# =============================================================================

# Mouse legend for the three major cell-type groups.  The scatter points use
# the full Tasic cluster palette (dozens of colours); the legend collapses to
# three families.  Each swatch is a single representative colour and the label
# names the colour family it stands for, since a lone swatch cannot convey the
# whole palette of a group.
_MOUSE_MAJOR_LEGEND = [
    ("Excitatory (cool / greens)",     "#2ca02c"),
    ("Inhibitory (warm / reds)",       "#d62728"),
    ("Non-neuronal (neutral / greys)", "#7f7f7f"),
]


def _mouse_major_legend_handles():
    """Build the 3-family legend handles for the mouse embedding plots."""
    return [
        Line2D([0], [0], marker="o", color="w", markersize=8,
               markerfacecolor=col, label=lbl)
        for lbl, col in _MOUSE_MAJOR_LEGEND
    ]


def exp5_embedding_comparison(X_pca, y, out_dir,
                                dataset="mnist",
                                perplexity=30, gamma_s=0.5,
                                cluster_colors_arr=None,
                                n_jobs=-1, random_state=42,
                                n_iter_ee=250, n_iter_main=750, ee=12,
                                point_size=4, alpha=0.8):
    """Side-by-side scatter: standard (γ=1) vs smooth (γ=gamma_s) t-SNE."""
    print(f"\n--- Exp 5: Embedding comparison ({dataset}) ---")
    plots_dir = os.path.join(out_dir, "exp5_embedding")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    print(f"  Running standard t-SNE (γ=1.0, ρ={perplexity}) ...")
    Y_std = run_tsne(X_pca, perplexity=perplexity, gamma=1.0,
                     random_state=random_state, n_jobs=n_jobs,
                     n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee)
    print(f"  Running smooth t-SNE (γ={gamma_s}, ρ={perplexity}) ...")
    Y_smo = run_tsne(X_pca, perplexity=perplexity, gamma=gamma_s,
                     random_state=random_state, n_jobs=n_jobs,
                     n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee)

    Y_std_arr = np.asarray(Y_std)
    Y_smo_arr = np.asarray(Y_smo)

    # ── Colour/legend logic ───────────────────────────────────────────────────
    if dataset == "mouse" and cluster_colors_arr is not None:
        point_colors = cluster_colors_arr[y]

        def _get_major_class(name):
            if name.startswith(("Lamp5", "Vip", "Pvalb", "Sst")):
                return "Inhibitory"
            if name.startswith(("L2/3", "L5", "L6")):
                return "Excitatory"
            return "Non-neuronal"

        legend_handles = _mouse_major_legend_handles()
        legend_title = "Cell type"

    elif dataset == "mnist":
        palette      = plt.cm.tab10(np.linspace(0, 0.9, 10))
        digits       = np.array([int(d) for d in y])
        point_colors = palette[digits]
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markersize=8,
                   markerfacecolor=palette[d], label=str(d))
            for d in range(10)
        ]
        legend_title = "Digit"

    else:
        pal          = {0: "#4393C3", 1: "#D6604D"}
        point_colors = [pal[int(yi)] for yi in y]
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markersize=8,
                   markerfacecolor=pal[v],
                   label="Income ≤50K" if v == 0 else "Income >50K")
            for v in [0, 1]
        ]
        legend_title = "Income"

    # ── Save embedding CSVs (for plot_only mode) ─────────────────────────────
    # Persist the per-point colour too, so --plot_only reproduces the exact
    # colours (notably the Tasic cluster colours for mouse, which cannot be
    # recovered from the integer label alone).
    y_str      = [str(yi) for yi in y]
    color_hex  = [matplotlib.colors.to_hex(c) for c in point_colors]
    pd.DataFrame({"x": Y_std_arr[:, 0], "y": Y_std_arr[:, 1],
                  "label": y_str, "color": color_hex}).to_csv(
        os.path.join(plots_dir, "embedding_standard.csv"), index=False)
    pd.DataFrame({"x": Y_smo_arr[:, 0], "y": Y_smo_arr[:, 1],
                  "label": y_str, "color": color_hex}).to_csv(
        os.path.join(plots_dir, "embedding_smooth.csv"), index=False)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Y, ttl in [
        (axes[0], Y_std_arr, f"Standard t-SNE  (ρ={perplexity},  γ=1.0)"),
        (axes[1], Y_smo_arr, f"Smooth t-SNE  (ρ={perplexity},  γ={gamma_s})"),
    ]:
        ax.scatter(Y[:, 0], Y[:, 1], c=point_colors,
                   s=point_size, alpha=alpha, edgecolors="none", rasterized=True)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    if legend_handles:
        axes[1].legend(handles=legend_handles, title=legend_title,
                       fontsize=13, title_fontsize=13, frameon=False,
                       loc="upper left",
                       bbox_to_anchor=(1.02, 1.0),
                       bbox_transform=axes[1].transAxes)
    plt.tight_layout()
    _save_fig("05_embedding")
    print(f"  Exp 5 done → {plots_dir}")


# =============================================================================
# Experiment 6 – Neighborhood Overlap sensitivity heatmaps
# =============================================================================

def exp6_sensitivity_heatmaps(df, out_dir, dataset="mnist"):
    """
    Heatmaps of AUC_1_10 and AUC_11_90 over γ × perplexity.
    Data comes from _run_sensitivity_grid.
    """
    print("\n--- Exp 6: Neighborhood Overlap sensitivity heatmaps ---")
    plots_dir = os.path.join(out_dir, "exp6_sensitivity")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    csv_path = os.path.join(plots_dir, "sensitivity_grid.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Grid saved → {csv_path}")

    df_mean = df.groupby(["perplexity", "gamma"], as_index=False).mean(
        numeric_only=True)
    n_seeds = df["seed"].nunique() if "seed" in df.columns else 1
    df_std  = (df.groupby(["perplexity", "gamma"], as_index=False)
                 .std(numeric_only=True, ddof=1).fillna(0))

    _p = "" if dataset in ("mouse", "adult") else "(a) "
    _q = "" if dataset in ("mouse", "adult") else "(b) "
    specs = [
        ("AUC_1_10",  f"{_p}Neighborhood Overlap  AUC  (k=1–10)",
         "06a_neighborhood_overlap_auc_1_10",  "RdYlGn"),
        ("AUC_11_90", f"{_q}Neighborhood Overlap  AUC  (k=11–90)",
         "06b_neighborhood_overlap_auc_11_90", "RdYlGn"),
    ]
    for metric, title, fname, cmap in specs:
        pivot_mean = df_mean.pivot(index="perplexity", columns="gamma",
                                   values=metric)
        pivot_std  = (df_std.pivot(index="perplexity", columns="gamma",
                                   values=metric)
                      if n_seeds > 1 else None)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _heatmap_mean_std(ax, pivot_mean, pivot_std, title, cmap)
        plt.tight_layout()
        _save_fig(fname)

    print(f"  Exp 6 done → {plots_dir}")


# =============================================================================
# Experiment 7 – Neighborhood Overlap: standard / smooth / sharp
# =============================================================================

def exp7_nh_comparison(X_pca, X_eval, out_dir,
                         perplexity=30, gamma_s=0.5, gamma_h=1.5,
                         P_cond_smooth=None,
                         k_max=200, dataset="mnist",
                         n_jobs=-1, master_seed=42, n_runs=1,
                         n_iter_ee=250, n_iter_main=750, ee=12):
    """
    Compare NH for four variants: standard, smooth, sharp, matched-perplexity.
    P_cond_smooth, knn_idx_base: from build_affinity (used for matched ρ).
    If P_cond_smooth is None, matched-perplexity line is omitted.
    CSV (tidy): seed, k, <label_std>, <label_smo>, [<label_mat>]
    """
    print("\n--- Exp 7: Neighborhood Overlap comparison ---")
    plots_dir = os.path.join(out_dir, "exp7_neighborhood_overlap_comparison")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    k_values  = np.arange(1, k_max + 1)
    seeds     = _make_seeds(master_seed, n_runs)
    csv_path  = os.path.join(plots_dir, "neighborhood_overlap_comparison.csv")
    C_STD7    = C_STANDARD
    C_SMO7    = C_SMOOTH
    C_MAT7    = C_MATCHED

    label_std = f"standard  ρ={perplexity}"
    label_smo = f"smooth  γ={gamma_s}"

    # Compute matched perplexity from smooth conditional P
    p_matched = None
    label_mat = None
    if P_cond_smooth is not None:
        p_matched = max(
            int(round(float(np.median(_eff_perp_per_point(P_cond_smooth))))),
            perplexity + 1,
        )
        label_mat = f"standard  ρ={p_matched}  (matched)"
        print(f"  ρ_matched (smooth γ={gamma_s}) = {p_matched}")

    # Build affinities once (deterministic)
    print(f"  Building affinities ...")
    P_jnt_std, _, _, _ = build_affinity(X_pca, perplexity, gamma=1.0,
                                         n_jobs=n_jobs)
    P_jnt_smo, _, _, _ = build_affinity(X_pca, perplexity, gamma=gamma_s,
                                         n_jobs=n_jobs)
    P_jnt_mat = None
    if p_matched is not None:
        P_jnt_mat, _, _, _ = build_affinity(X_pca, p_matched, gamma=1.0,
                                              n_jobs=n_jobs)

    existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    if not existing.empty and "seed" not in existing.columns:
        existing["seed"] = seeds[0]
    done_seeds = set(existing["seed"].unique()) if not existing.empty else set()
    seeds_todo = [s for s in seeds if s not in done_seeds]

    if seeds_todo:
        print(f"  Pre-computing HD KNN at k={k_max} ...")
        hd_nbrs = _knn_indices(X_eval, k=k_max, n_jobs=n_jobs)

        for seed in seeds_todo:
            print(f"  seed={seed} ...")
            curves = {}
            for lbl, P_jnt in [
                (label_std, P_jnt_std),
                (label_smo, P_jnt_smo),
            ]:
                print(f"    {lbl[:12]} ...", end=" ", flush=True)
                Y = run_tsne_from_joint(X_pca, P_jnt, random_state=seed,
                                        n_jobs=n_jobs, n_iter_ee=n_iter_ee,
                                        n_iter_main=n_iter_main, ee=ee)
                ld_nbrs = _knn_indices(np.asarray(Y), k=k_max, n_jobs=n_jobs)
                curves[lbl] = _nh_curve_from_nbrs(hd_nbrs, ld_nbrs, k_values)
                print("done")
            if P_jnt_mat is not None:
                print(f"    matched ...", end=" ", flush=True)
                Y = run_tsne_from_joint(X_pca, P_jnt_mat, random_state=seed,
                                        n_jobs=n_jobs, n_iter_ee=n_iter_ee,
                                        n_iter_main=n_iter_main, ee=ee)
                ld_nbrs = _knn_indices(np.asarray(Y), k=k_max, n_jobs=n_jobs)
                curves[label_mat] = _nh_curve_from_nbrs(hd_nbrs, ld_nbrs,
                                                         k_values)
                print("done")
            seed_rows = []
            for k_idx, k in enumerate(k_values):
                row = {"seed": seed, "k": k,
                       label_std: curves[label_std][k_idx],
                       label_smo: curves[label_smo][k_idx]}
                if label_mat is not None:
                    row[label_mat] = curves[label_mat][k_idx]
                seed_rows.append(row)
            existing = pd.concat([existing, pd.DataFrame(seed_rows)],
                                 ignore_index=True)
            existing.to_csv(csv_path, index=False)
            print(f"  checkpoint: seed={seed} saved → {csv_path}")
    else:
        print(f"  All seeds done, skipping computation.")

    out_df = existing

    # ── Plot: thin individual seeds + bold mean ───────────────────────────────
    k_vals_plot = np.sort(out_df["k"].unique())
    seed_list   = out_df["seed"].unique()
    val_cols    = [c for c in out_df.columns if c not in ("k", "seed")]

    def _style7(lbl):
        if "smooth"  in lbl.lower(): return C_SMO7,    "--", 2.2
        if "matched" in lbl.lower(): return C_MAT7,     "-", 2.0
        return C_STD7, "-", 2.6

    val_cols = [c for c in val_cols if "sharp" not in c.lower()]

    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl in val_cols:
        color, ls, lw = _style7(lbl)
        for seed in seed_list:
            sub = out_df[out_df["seed"] == seed].sort_values("k")
            ax.plot(sub["k"].values, sub[lbl].values,
                    color=color, lw=0.6, ls=ls, alpha=0.2)
        mn = out_df.groupby("k")[lbl].mean().reindex(k_vals_plot).values
        ax.plot(k_vals_plot, mn, color=color, ls=ls, lw=lw, label=lbl)
    ax.set_xlabel("k")
    ax.set_ylabel("Neighborhood Overlap")
    ax.set_ylim(*_EXP7_YLIM.get(dataset, _EXP7_YLIM_DEFAULT))
    _p7 = "(a) " if dataset == "mnist" else ""
    ax.set_title(f"{_p7}Neighborhood Overlap  (ρ={perplexity})")
    ax.legend(fontsize=13, frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("07_neighborhood_overlap_comparison")
    print(f"  Exp 7 done → {plots_dir}")


# =============================================================================
# Experiment 8 – Global Spearman vs γ
# =============================================================================

def exp8_global_spearman_vs_gamma(X_pca, X_eval, out_dir,
                                    gammas, perplexities, dataset="mnist",
                                    n_jobs=-1, master_seed=42, n_runs=1,
                                    n_iter_ee=250, n_iter_main=750, ee=12):
    print("\n--- Exp 8: Global Spearman vs γ ---")
    plots_dir = os.path.join(out_dir, "exp8_global_spearman")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    gammas       = sorted(gammas)
    perplexities = sorted(perplexities)
    seeds        = _make_seeds(master_seed, n_runs)
    csv_path     = os.path.join(plots_dir, "global_spearman_vs_gamma.csv")

    existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    if not existing.empty and "seed" not in existing.columns:
        existing["seed"] = seeds[0]
    done_pairs = (set(zip(existing["perplexity"], existing["seed"]))
                  if not existing.empty else set())

    for p in perplexities:
        seeds_needed = [s for s in seeds if (p, s) not in done_pairs]
        if not seeds_needed:
            continue
        for seed in seeds_needed:
            print(f"  ρ={p}, seed={seed} ...")
            seed_rows = []
            for g in gammas:
                print(f"    γ={g:.2f} ...", end=" ", flush=True)
                Y = run_tsne(X_pca, perplexity=p, gamma=g,
                             random_state=seed, n_jobs=n_jobs,
                             n_iter_ee=n_iter_ee, n_iter_main=n_iter_main,
                             ee=ee)
                rho = _global_spearman(X_eval, np.asarray(Y),
                                       random_state=master_seed)
                seed_rows.append({
                    "seed": seed, "perplexity": p,
                    "gamma": g, "global_spearman": rho,
                })
                print(f"ρ={rho:.4f}")
            existing = pd.concat([existing, pd.DataFrame(seed_rows)],
                                 ignore_index=True)
            existing.to_csv(csv_path, index=False)
            done_pairs.add((p, seed))
            print(f"  checkpoint: ρ={p} seed={seed} saved")

    out_df = existing

    # ── Plot: mean ± std ─────────────────────────────────────────────────────
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("white")
    for p, m in zip(perplexities, markers):
        c   = _EXP8_COLORS.get(p, _CB_PALETTE[perplexities.index(p) % len(_CB_PALETTE)])
        sub = out_df[out_df["perplexity"] == p]
        grp = sub.groupby("gamma")["global_spearman"]
        mn  = grp.mean().reindex(gammas)
        sd  = grp.std(ddof=1).fillna(0).reindex(gammas)
        ax.plot(gammas, mn.values, color=c, marker=m, markersize=7,
                lw=2.2, label=f"ρ={p}")
        ax.fill_between(gammas, mn.values - sd.values, mn.values + sd.values,
                        color=c, alpha=0.15)
    ax.set_xlabel("γ")
    ax.set_ylabel("Global Spearman ρ")
    # Per-dataset y-range, scaled to each dataset's band of Global Spearman ρ.
    ax.set_ylim(*_EXP8_YLIM.get(dataset, _EXP8_YLIM_DEFAULT))
    _prefix = "" if dataset in ("mouse", "adult") else "(a) "
    ax.set_title(f"{_prefix}Effect of γ on global structure preservation")
    ax.legend(fontsize=13, frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("08_global_spearman_vs_gamma")
    print(f"  Exp 8 done → {plots_dir}")
    return out_df


# =============================================================================
# Experiment 9 – Smooth t-SNE vs point-wise max over standard perplexities
# =============================================================================

def exp9_smooth_vs_envelope(X_pca, X_eval, out_dir,
                              gamma_s=0.7,
                              standard_perplexities=(5, 10, 20, 30, 50, 70,
                                                     100, 120, 150, 200, 300),
                              k_max=200, dataset="mnist",
                              n_jobs=-1, master_seed=42, n_runs=1,
                              n_iter_ee=250, n_iter_main=750, ee=12):
    """
    For each perplexity run both standard (γ=1) and smooth (γ=gamma_s) t-SNE.
    Plot the point-wise max (envelope) over all perplexities.

    CSV (tidy): seed, k, standard_p5, smooth_p5, standard_p10, ...
    """
    print("\n--- Exp 9: Smooth vs standard envelope (all perplexities) ---")
    plots_dir = os.path.join(out_dir, "exp9_smooth_vs_envelope")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    perps    = list(standard_perplexities)
    k_values = np.arange(1, k_max + 1)
    seeds    = _make_seeds(master_seed, n_runs)
    csv_path = os.path.join(plots_dir, "smooth_vs_envelope.csv")

    std_cols = [f"standard_p{p}" for p in perps]
    smo_cols = [f"smooth_p{p}"   for p in perps]

    existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    if not existing.empty and "seed" not in existing.columns:
        existing["seed"] = seeds[0]

    def _is_perp_seed_done(df, p, seed):
        col = f"standard_p{p}"
        if df.empty or col not in df.columns:
            return False
        sub = df[df["seed"] == seed]
        return not sub.empty and sub[col].notna().all()

    rows_by_seed_k = {}
    if not existing.empty:
        for _, row in existing.iterrows():
            key = (int(row["seed"]), int(row["k"]))
            rows_by_seed_k[key] = row.to_dict()

    for p in perps:
        seeds_needed = [s for s in seeds
                        if not _is_perp_seed_done(existing, p, s)]
        if not seeds_needed:
            print(f"  ρ={p}: all seeds done, skipping")
            continue
        print(f"\n  ρ={p}: pre-computing HD KNN at k={k_max} ...")
        hd_nbrs = _knn_indices(X_eval, k=k_max, n_jobs=n_jobs)
        for seed in seeds_needed:
            print(f"    seed={seed} standard (γ=1.0) ...")
            Y_std = run_tsne(X_pca, perplexity=p, gamma=1.0,
                             random_state=seed, n_jobs=n_jobs,
                             n_iter_ee=n_iter_ee, n_iter_main=n_iter_main,
                             ee=ee)
            ld_std = _knn_indices(np.asarray(Y_std), k=k_max, n_jobs=n_jobs)
            nh_std = _nh_curve_from_nbrs(hd_nbrs, ld_std, k_values)

            print(f"    seed={seed} smooth (γ={gamma_s}) ...")
            Y_smo = run_tsne(X_pca, perplexity=p, gamma=gamma_s,
                             random_state=seed, n_jobs=n_jobs,
                             n_iter_ee=n_iter_ee, n_iter_main=n_iter_main,
                             ee=ee)
            ld_smo = _knn_indices(np.asarray(Y_smo), k=k_max, n_jobs=n_jobs)
            nh_smo = _nh_curve_from_nbrs(hd_nbrs, ld_smo, k_values)

            for k_idx, k in enumerate(k_values):
                key = (seed, int(k))
                if key not in rows_by_seed_k:
                    rows_by_seed_k[key] = {"seed": seed, "k": int(k)}
                rows_by_seed_k[key][f"standard_p{p}"] = nh_std[k_idx]
                rows_by_seed_k[key][f"smooth_p{p}"]   = nh_smo[k_idx]

        checkpoint_df = pd.DataFrame(list(rows_by_seed_k.values()))
        checkpoint_df.to_csv(csv_path, index=False)
        existing = checkpoint_df
        print(f"  checkpoint: ρ={p} saved → {csv_path}")

    out_df = pd.DataFrame(list(rows_by_seed_k.values())) if rows_by_seed_k else existing
    if out_df.empty:
        print("  No data available for plotting — nothing to do.")
        return

    # ── Compute per-seed envelopes ────────────────────────────────────────────
    C_STD_LOC   = C_STANDARD
    C_GREEN_LOC = C_SMOOTH
    env_std_per_seed   = []
    env_smo_per_seed   = []
    std_winner_per_seed = []
    smo_winner_per_seed = []

    for seed in seeds:
        sub = out_df[out_df["seed"] == seed].sort_values("k")
        if sub.empty:
            continue
        avail_std = [c for c in std_cols if c in sub.columns]
        avail_smo = [c for c in smo_cols if c in sub.columns]
        nh_std_mat = np.stack([sub[c].values for c in avail_std], axis=0)
        nh_smo_mat = np.stack([sub[c].values for c in avail_smo], axis=0)
        env_std_per_seed.append(np.max(nh_std_mat, axis=0))
        env_smo_per_seed.append(np.max(nh_smo_mat, axis=0))
        perps_std = [int(c.split("_p")[-1]) for c in avail_std]
        perps_smo = [int(c.split("_p")[-1]) for c in avail_smo]
        std_winner_per_seed.append(
            np.array([perps_std[i] for i in np.argmax(nh_std_mat, axis=0)]))
        smo_winner_per_seed.append(
            np.array([perps_smo[i] for i in np.argmax(nh_smo_mat, axis=0)]))

    first_seed = out_df["seed"].iloc[0]
    k_ax = out_df[out_df["seed"] == first_seed].sort_values("k")["k"].values
    mn_std = np.mean(env_std_per_seed, axis=0)
    mn_smo = np.mean(env_smo_per_seed, axis=0)

    # Plot 09a: thin individual seed envelopes + bold mean
    fig, ax = plt.subplots(figsize=(8, 5))
    for env in env_std_per_seed:
        ax.plot(k_ax, env, color=C_STD_LOC,   lw=0.6, ls="-",  alpha=0.2)
    for env in env_smo_per_seed:
        ax.plot(k_ax, env, color=C_GREEN_LOC, lw=0.6, ls="--", alpha=0.2)
    ax.plot(k_ax, mn_std, color=C_STD_LOC,   lw=2.6, ls="-",
            label="standard t-SNE  (best ρ per k)")
    ax.plot(k_ax, mn_smo, color=C_GREEN_LOC, lw=2.2, ls="--",
            label=f"smooth  γ={gamma_s}  (best ρ per k)")
    ax.set_xlabel("k")
    ax.set_ylabel("Neighborhood Overlap")
    ax.set_ylim(*_EXP9_YLIM.get(dataset, _EXP9_YLIM_DEFAULT))
    ax.set_title(f"Smooth vs standard t-SNE — best ρ envelope  (γ={gamma_s})")
    ax.legend(fontsize=13, frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("09a_smooth_vs_envelope")

    # Plot 09b: winning perplexity per k (mode over seeds)
    from scipy.stats import mode as scipy_mode
    std_winner_mode = np.array([
        scipy_mode(np.array([sw[ki] for sw in std_winner_per_seed]),
                   keepdims=False).mode
        for ki in range(len(k_ax))
    ])
    smo_winner_mode = np.array([
        scipy_mode(np.array([sw[ki] for sw in smo_winner_per_seed]),
                   keepdims=False).mode
        for ki in range(len(k_ax))
    ])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(k_ax, std_winner_mode, where="mid", color=C_STD_LOC,   lw=2.0,
            label="standard")
    ax.step(k_ax, smo_winner_mode, where="mid", color=C_GREEN_LOC, lw=2.0,
            ls="--", label=f"smooth γ={gamma_s}")
    ax.set_xlabel("k")
    ax.set_ylabel("Best ρ")
    ax.set_title("Winning perplexity per k")
    ax.set_yticks(perps)
    ax.legend(fontsize=13, frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("09b_winner_perplexity")
    print(f"  Exp 9 done → {plots_dir}")


# =============================================================================
# Experiment 10 – Neighborhood Overlap vs perplexity
# =============================================================================

def exp10_no_vs_perplexity(X_pca, X_eval, out_dir,
                            gamma_s=0.7,
                            perplexities=(5, 10, 15, 20, 25, 30,
                                          40, 50, 60, 70, 80, 90, 100),
                            k_eval=30, dataset="mnist",
                            n_jobs=-1, master_seed=42, n_runs=1,
                            n_iter_ee=250, n_iter_main=750, ee=12):
    """
    For each perplexity run standard (γ=1) and smooth (γ=gamma_s) t-SNE.
      10a  NO@k_eval vs perplexity
      10b  AUC (mean NO over k=1..k_eval) vs perplexity

    CSV: seed, perplexity, standard_no, smooth_gamma{s}_no,
                           standard_auc, smooth_gamma{s}_auc
    """
    print("\n--- Exp 10: NO@k and AUC vs perplexity ---")
    plots_dir = os.path.join(out_dir, "exp10_no_vs_perplexity")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    perps       = list(perplexities)
    k_values    = np.arange(1, k_eval + 1)
    seeds       = _make_seeds(master_seed, n_runs)
    csv_path    = os.path.join(plots_dir, "no_vs_perplexity.csv")
    smo_no_col  = f"smooth_gamma{gamma_s}_no"
    smo_auc_col = f"smooth_gamma{gamma_s}_auc"

    existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    if not existing.empty and "seed" not in existing.columns:
        existing["seed"] = seeds[0]
    done_pairs = (set(zip(existing["perplexity"], existing["seed"]))
                  if not existing.empty else set())

    for p in perps:
        seeds_needed = [s for s in seeds if (p, s) not in done_pairs]
        if not seeds_needed:
            continue
        print(f"\n  ρ={p}: pre-computing HD KNN at k={k_eval} ...")
        hd_nbrs = _knn_indices(X_eval, k=k_eval, n_jobs=n_jobs)
        for seed in seeds_needed:
            print(f"    seed={seed} standard (γ=1.0) ...")
            Y_std = run_tsne(X_pca, perplexity=p, gamma=1.0,
                             random_state=seed, n_jobs=n_jobs,
                             n_iter_ee=n_iter_ee, n_iter_main=n_iter_main,
                             ee=ee)
            ld_std    = _knn_indices(np.asarray(Y_std), k=k_eval, n_jobs=n_jobs)
            curve_std = _nh_curve_from_nbrs(hd_nbrs, ld_std, k_values)

            print(f"    seed={seed} smooth (γ={gamma_s}) ...")
            Y_smo = run_tsne(X_pca, perplexity=p, gamma=gamma_s,
                             random_state=seed, n_jobs=n_jobs,
                             n_iter_ee=n_iter_ee, n_iter_main=n_iter_main,
                             ee=ee)
            ld_smo    = _knn_indices(np.asarray(Y_smo), k=k_eval, n_jobs=n_jobs)
            curve_smo = _nh_curve_from_nbrs(hd_nbrs, ld_smo, k_values)

            new_row = pd.DataFrame([{
                "seed": seed, "perplexity": p,
                "standard_no":  float(curve_std[-1]),
                smo_no_col:     float(curve_smo[-1]),
                "standard_auc": float(curve_std.mean()),
                smo_auc_col:    float(curve_smo.mean()),
            }])
            existing = pd.concat([existing, new_row], ignore_index=True)
            existing.to_csv(csv_path, index=False)
            done_pairs.add((p, seed))
            print(f"  checkpoint: ρ={p} seed={seed} saved")

    out_df = existing
    if out_df.empty:
        print("  No data available for plotting — nothing to do.")
        return

    # ── Plot: thin seeds + bold mean ─────────────────────────────────────────
    C_STD_LOC   = C_STANDARD
    C_GREEN_LOC = C_SMOOTH

    def _plot10(std_col, smo_col, ylabel, title, fname):
        p_ax      = np.sort(out_df["perplexity"].unique())
        seed_list = out_df["seed"].unique()
        mn        = out_df.groupby("perplexity")[[std_col, smo_col]].mean()
        fig, ax   = plt.subplots(figsize=(7, 5))
        for seed in seed_list:
            sub = out_df[out_df["seed"] == seed].sort_values("perplexity")
            ax.plot(sub["perplexity"].values, sub[std_col].values,
                    color=C_STD_LOC,   lw=0.6, ls="-",  alpha=0.25)
            ax.plot(sub["perplexity"].values, sub[smo_col].values,
                    color=C_GREEN_LOC, lw=0.6, ls="--", alpha=0.25)
        ax.plot(p_ax, mn.loc[p_ax, std_col].values,
                color=C_STD_LOC,   lw=2.4, ls="-",  marker="o", markersize=6,
                label="standard t-SNE")
        ax.plot(p_ax, mn.loc[p_ax, smo_col].values,
                color=C_GREEN_LOC, lw=2.2, ls="--", marker="s", markersize=6,
                label=f"smooth  γ={gamma_s}")
        ax.set_xlabel("Perplexity")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(p_ax)
        ax.tick_params(axis="x", labelsize=11)
        ax.legend(fontsize=13, frameon=False)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig(fname)

    _p10a = "(b) " if dataset == "mnist" else ""
    _plot10("standard_no", smo_no_col,
            f"Neighborhood Overlap at k={k_eval}",
            f"{_p10a}NO@{k_eval} vs perplexity",
            "10a_no_vs_perplexity")
    _plot10("standard_auc", smo_auc_col,
            f"AUC  (mean NO,  k=1–{k_eval})",
            f"AUC@{k_eval} vs perplexity",
            "10b_auc_vs_perplexity")
    print(f"  Exp 10 done → {plots_dir}")


# =============================================================================
# Data loading
# =============================================================================

def load_data(dataset, args):
    """
    Returns (X_pca, X_eval, y, extra).
    X_eval is the same as X_pca (used for metric computation).
    Replace with your own loading logic if using a different dataset.
    """
    extra = {}
    if dataset == "mouse":
        import pickle
        data_dir = args.mouse_data_dir or os.path.join(_SCRIPT_DIR, "data")
        pkl_path = os.path.join(data_dir, "tasic2018.pickle")
        print(f"Loading mouse cortex from {pkl_path} ...")
        X_pca, _, y, _, _ = load_mouse_data(pkl_path, data_dir=data_dir,
                                              return_highdim=True)
        with open(pkl_path, "rb") as fh:
            tasic = pickle.load(fh)
        extra["cluster_colors_arr"] = tasic["clusterColors"]
        if args.mouse_subsample and args.mouse_subsample < len(X_pca):
            rng = np.random.default_rng(args.random_state)
            sel = np.sort(rng.choice(len(X_pca), size=args.mouse_subsample,
                                      replace=False))
            X_pca, y = X_pca[sel], y[sel]

    elif dataset == "mnist":
        print("Loading MNIST (PCA-50) ...")
        X_pca, y, _ = load_mnist_data(n_pca=50,
                                       random_state=args.random_state)
        if args.mnist_subsample and args.mnist_subsample < len(X_pca):
            rng = np.random.default_rng(args.random_state)
            sel = np.sort(rng.choice(len(X_pca), size=args.mnist_subsample,
                                      replace=False))
            X_pca, y = X_pca[sel], y[sel]

    elif dataset == "adult":
        print("Loading Adult census data ...")
        X_pca, y = load_adult_data(max_rows=args.adult_max_rows,
                                    random_state=args.random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    X_eval = X_pca
    print(f"  n={len(X_pca)}  d={X_pca.shape[1]}")
    return X_pca, X_eval, y, extra


# =============================================================================
# Plot-only mode  (regenerate figures from saved CSVs)
# =============================================================================

def _csv(plots_dir, fname):
    path = os.path.join(plots_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            "Run without --plot_only first to generate the data."
        )
    return path


def plot_all_from_csv(args):
    """Regenerate all figures from previously saved CSVs (no t-SNE runs)."""
    out_dir = args.out_dir
    print(f"\n{'='*60}")
    print("  PLOT-ONLY mode: reading CSVs from", out_dir)
    print(f"{'='*60}\n")

    # ── Exp 1 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp1:
        plots_dir = os.path.join(out_dir, "exp1_affinity_sharpness")
        _plot_save_dir[0] = plots_dir
        df1      = pd.read_csv(_csv(plots_dir, "affinity_rows.csv"))
        ranks    = df1["rank"].values
        vals_std = df1["p_standard"].values
        smo_col  = [c for c in df1.columns if c.startswith("p_smooth")][0]
        vals_smo = df1[smo_col].values
        gamma_s1 = float(smo_col.split("gamma")[-1])
        C_STD_1 = C_STANDARD
        C_SMO_1 = C_SMOOTH
        n_show = min(args.exp1_top_ranks, len(ranks))
        x      = np.arange(n_show)
        width  = 0.4
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - width / 2, vals_std[:n_show], width, color=C_STD_1,
               label=f"Standard  (ρ={args.perplexity},  γ=1.0)")
        ax.bar(x + width / 2, vals_smo[:n_show], width, color=C_SMO_1,
               label=f"Smooth  (ρ={args.perplexity},  γ={gamma_s1})")
        ax.set_xlabel("Neighbor rank")
        ax.set_ylabel("Conditional probability  P(j|i)")
        ax.set_title(f"(a) Affinity row sharpness  (ρ={args.perplexity})")
        ax.set_xticks(x)
        ax.set_xticklabels(ranks[:n_show])
        ax.legend(frameon=False)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig("01_affinity_sharpness")
        print("  Exp 1 replotted.")

    # ── Exp 2 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp2:
        plots_dir = os.path.join(out_dir, "exp2_eff_perplexity")
        _plot_save_dir[0] = plots_dir
        df2     = pd.read_csv(_csv(plots_dir, "eff_perplexity_per_point.csv"))
        col_std = "standard"
        col_smo = [c for c in df2.columns if "smooth" in c][0]
        col_shp = [c for c in df2.columns if "sharp"  in c][0]
        gs = float(col_smo.split("_g")[-1])
        gh = float(col_shp.split("_g")[-1])
        _CONST_THR = 0.005
        triples = [
            (df2[col_std].values, f"standard  (ρ={args.perplexity})", C_STANDARD),
            (df2[col_smo].values, f"smooth  γ={gs}",                  C_SMOOTH),
            (df2[col_shp].values, f"sharp  γ={gh}",                   C_SHARP),
        ]
        n_pts   = len(triples[0][0])
        varying = [d for d, _, _ in triples if d.std() >= _CONST_THR]
        lo = min(d.min() for d in varying) if varying else 0
        hi = max(d.max() for d in varying) if varying else 1
        bins = np.linspace(lo, hi, 51)
        max_count = max(
            (int(np.histogram(d, bins=bins)[0].max())
             for d, _, _ in triples if d.std() >= _CONST_THR), default=0)
        max_count = max(max_count, n_pts)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_ylim(0, max_count * 1.25)
        for d, lbl, c in triples:
            if d.std() < _CONST_THR:
                ax.vlines(float(d.mean()), 0, n_pts, colors=c, lw=4, zorder=5,
                          label=lbl)
            else:
                ax.hist(d, bins=bins, alpha=0.65, color=c, label=lbl)
        ax.set_xlabel("Effective perplexity")
        ax.set_ylabel("Count")
        ax.set_title("(b) Effective perplexity distribution")
        ax.legend(frameon=False)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig("02a_eff_perp")
        # exp2b: median eff perp heatmap from sensitivity_grid
        grid_csv = os.path.join(out_dir, "sensitivity_grid", "sensitivity_grid.csv")
        if os.path.exists(grid_csv):
            df_grid = pd.read_csv(grid_csv)
            if "seed" not in df_grid.columns:
                df_grid["seed"] = 0
            if "median_eff_perp" in df_grid.columns:
                exp2b_median_eff_perp_heatmap(df_grid, out_dir)
        print("  Exp 2 replotted.")

    # ── Exp 3 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp3:
        plots_dir = os.path.join(out_dir, "exp3_delta_perp")
        _plot_save_dir[0] = plots_dir
        df3 = pd.read_csv(_csv(plots_dir, "delta_perp_per_point.csv"))
        from scipy.stats import spearmanr as _sr3
        candidates = [
            (df3["top5_mass"].values, "Top-5 conditional mass",
             "03a_delta_rho_vs_top5"),
            (df3["sigma_i"].values,   r"t-SNE bandwidth  $\sigma_i$",
             "03b_delta_rho_vs_sigma"),
        ]
        for xvar, xlabel, fname in candidates:
            rho, _ = _sr3(xvar, df3["delta_rho"].values)
            fig, ax = plt.subplots(figsize=(6, 5))
            hb = ax.hexbin(xvar, df3["delta_rho"].values,
                           gridsize=45, cmap="Blues", mincnt=1)
            plt.colorbar(hb, ax=ax, label="Count")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\Delta$ perplexity")
            ax.set_title(f"Spearman ρ = {rho:.3f}")
            ax.set_ylim(bottom=0)
            _clean_axes(ax)
            plt.tight_layout()
            _save_fig(fname)
        print("  Exp 3 replotted.")

    # ── Exp 4 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp4:
        plots_dir = os.path.join(out_dir, "exp4_neighborhood_overlap_gamma_sweep")
        _plot_save_dir[0] = plots_dir
        df4 = pd.read_csv(_csv(plots_dir, "neighborhood_overlap_gamma_sweep.csv"))
        if "seed" not in df4.columns:
            df4["seed"] = 0
        gammas = sorted([float(c.split("_", 1)[-1])
                         for c in df4.columns if c.startswith("gamma_")])
        smooth_list = [g for g in gammas if g < 1.0]
        sharp_list  = [g for g in gammas if g > 1.0]
        k_vals_plot = np.sort(df4["k"].unique())
        seed_list   = df4["seed"].unique()
        fig, ax = plt.subplots(figsize=(8, 5))
        for g in gammas:
            col     = f"gamma_{g:.2f}"
            c       = _gamma_color(g, smooth_list, sharp_list)
            lw_mean = 2.6 if np.isclose(g, 1.0) else 2.0
            ls      = "-" if np.isclose(g, 1.0) else ("--" if g < 1.0 else "-.")
            lbl     = f"γ={g:.2f}" + ("  (standard)" if np.isclose(g, 1.0) else "")
            for seed in seed_list:
                sub = df4[df4["seed"] == seed].sort_values("k")
                ax.plot(sub["k"].values, sub[col].values,
                        color=c, lw=0.6, ls=ls, alpha=0.2)
            mn = df4.groupby("k")[col].mean().reindex(k_vals_plot).values
            ax.plot(k_vals_plot, mn, color=c, lw=lw_mean, ls=ls, label=lbl)
        ax.set_xlabel("k")
        ax.set_ylabel("Neighborhood Overlap")
        ax.set_title(f"Neighborhood Overlap  —  ρ={args.perplexity}, varying γ")
        ax.legend(ncol=2, fontsize=13, frameon=False, loc="lower right")
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig("04_neighborhood_overlap_gamma_sweep")
        print("  Exp 4 replotted.")

    # ── Exp 5 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp5:
        plots_dir = os.path.join(out_dir, "exp5_embedding")
        _plot_save_dir[0] = plots_dir
        df_std = pd.read_csv(_csv(plots_dir, "embedding_standard.csv"))
        df_smo = pd.read_csv(_csv(plots_dir, "embedding_smooth.csv"))
        labels = df_std["label"].values
        dataset = args.dataset
        if dataset == "mnist":
            palette      = plt.cm.tab10(np.linspace(0, 0.9, 10))
            digits       = np.array([int(d) for d in labels])
            point_colors = palette[digits]
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markersize=8,
                       markerfacecolor=palette[d], label=str(d))
                for d in range(10)
            ]
            legend_title = "Digit"
        elif dataset == "adult":
            pal          = {0: "#4393C3", 1: "#D6604D"}
            point_colors = [pal[int(v)] for v in labels]
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markersize=8,
                       markerfacecolor=pal[v],
                       label="Income ≤50K" if v == 0 else "Income >50K")
                for v in [0, 1]
            ]
            legend_title = "Income"
        else:  # mouse
            # Prefer the per-point colour persisted in the CSV; fall back to
            # reloading the Tasic cluster colours and indexing by cluster id
            # (for CSVs written before the "color" column existed).
            if "color" in df_std.columns:
                point_colors = df_std["color"].values
            else:
                import pickle
                data_dir = args.mouse_data_dir or os.path.join(
                    _SCRIPT_DIR, "data")
                with open(os.path.join(data_dir, "tasic2018.pickle"), "rb") as fh:
                    cluster_colors_arr = pickle.load(fh)["clusterColors"]
                point_colors = cluster_colors_arr[labels.astype(int)]

            legend_handles = _mouse_major_legend_handles()
            legend_title = "Cell type"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, df_emb, ttl in [
            (axes[0], df_std, f"Standard t-SNE  (ρ={args.perplexity},  γ=1.0)"),
            (axes[1], df_smo,
             f"Smooth t-SNE  (ρ={args.perplexity},  γ={args.exp5_gamma})"),
        ]:
            ax.scatter(df_emb["x"], df_emb["y"], c=point_colors,
                       s=4, alpha=0.8, edgecolors="none", rasterized=True)
            ax.set_title(ttl)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        if legend_handles:
            axes[1].legend(handles=legend_handles, title=legend_title,
                           fontsize=13, title_fontsize=13, frameon=False,
                           loc="upper left",
                           bbox_to_anchor=(1.02, 1.0),
                           bbox_transform=axes[1].transAxes)
        plt.tight_layout()
        _save_fig("05_embedding")
        print("  Exp 5 replotted.")

    # ── Exp 6 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp6:
        plots_dir = os.path.join(out_dir, "exp6_sensitivity")
        _plot_save_dir[0] = plots_dir
        df6 = pd.read_csv(_csv(plots_dir, "sensitivity_grid.csv"))
        if "seed" not in df6.columns:
            df6["seed"] = 0
        exp6_sensitivity_heatmaps(df6, out_dir, dataset=args.dataset)
        print("  Exp 6 replotted.")

    # ── Exp 7 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp7:
        plots_dir = os.path.join(out_dir, "exp7_neighborhood_overlap_comparison")
        _plot_save_dir[0] = plots_dir
        df7 = pd.read_csv(_csv(plots_dir, "neighborhood_overlap_comparison.csv"))
        if "seed" not in df7.columns:
            df7["seed"] = 0
        C_STD7    = C_STANDARD
        C_SMO7    = C_SMOOTH
        C_MATCHED7 = C_MATCHED
        val_cols  = [c for c in df7.columns
                     if c not in ("k", "seed") and "sharp" not in c.lower()]
        k_vals_plot = np.sort(df7["k"].unique())
        seed_list   = df7["seed"].unique()

        def _style7(lbl):
            if "smooth"  in lbl.lower(): return C_SMO7,     "--", 2.2
            if "matched" in lbl.lower(): return C_MATCHED7,  "-", 2.0
            return C_STD7, "-", 2.6

        fig, ax = plt.subplots(figsize=(8, 5))
        for lbl in val_cols:
            c, ls, lw = _style7(lbl)
            for seed in seed_list:
                sub = df7[df7["seed"] == seed].sort_values("k")
                ax.plot(sub["k"].values, sub[lbl].values,
                        color=c, lw=0.6, ls=ls, alpha=0.2)
            mn = df7.groupby("k")[lbl].mean().reindex(k_vals_plot).values
            ax.plot(k_vals_plot, mn, color=c, ls=ls, lw=lw, label=lbl)
        ax.set_xlabel("k")
        ax.set_ylabel("Neighborhood Overlap")
        ax.set_ylim(*_EXP7_YLIM.get(args.dataset, _EXP7_YLIM_DEFAULT))
        _p7 = "(a) " if args.dataset == "mnist" else ""
        ax.set_title(f"{_p7}Neighborhood Overlap  (ρ={args.perplexity})")
        ax.legend(fontsize=13, frameon=False)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig("07_neighborhood_overlap_comparison")
        print("  Exp 7 replotted.")

    # ── Exp 8 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp8:
        plots_dir = os.path.join(out_dir, "exp8_global_spearman")
        _plot_save_dir[0] = plots_dir
        df8 = pd.read_csv(_csv(plots_dir, "global_spearman_vs_gamma.csv"))
        if "seed" not in df8.columns:
            df8["seed"] = 0
        perplexities8 = sorted(df8["perplexity"].unique())
        gammas8       = sorted(df8["gamma"].unique())
        markers       = ["o", "s", "^", "D"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor("white")
        for p, m in zip(perplexities8, markers):
            c   = _EXP8_COLORS.get(p, _CB_PALETTE[perplexities8.index(p) % len(_CB_PALETTE)])
            sub = df8[df8["perplexity"] == p]
            grp = sub.groupby("gamma")["global_spearman"]
            mn  = grp.mean().reindex(gammas8)
            sd  = grp.std(ddof=1).fillna(0).reindex(gammas8)
            ax.plot(gammas8, mn.values, color=c, marker=m, markersize=7,
                    lw=2.2, label=f"ρ={p}")
            ax.fill_between(gammas8, mn.values - sd.values,
                            mn.values + sd.values, color=c, alpha=0.15)
        ax.set_xlabel("γ")
        ax.set_ylabel("Global Spearman ρ")
        # Per-dataset y-range, scaled to each dataset's band of Global Spearman ρ.
        ax.set_ylim(*_EXP8_YLIM.get(args.dataset, _EXP8_YLIM_DEFAULT))
        _prefix8 = "" if args.dataset in ("mouse", "adult") else "(a) "
        ax.set_title(f"{_prefix8}Effect of γ on global structure preservation")
        ax.legend(fontsize=13, frameon=False)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig("08_global_spearman_vs_gamma")
        print("  Exp 8 replotted.")

    # ── Exp 9 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp9:
        plots_dir = os.path.join(out_dir, "exp9_smooth_vs_envelope")
        _plot_save_dir[0] = plots_dir
        df9 = pd.read_csv(_csv(plots_dir, "smooth_vs_envelope.csv"))
        if "seed" not in df9.columns:
            df9["seed"] = 0
        gamma_s9  = args.gamma_s
        C_STD_LOC   = C_STANDARD
        C_GREEN_LOC = C_SMOOTH
        std_cols9 = [c for c in df9.columns if c.startswith("standard_p")]
        smo_cols9 = [c for c in df9.columns if c.startswith("smooth_p")]
        perps9    = sorted([int(c.split("_p")[-1]) for c in std_cols9])

        env_std_list, env_smo_list = [], []
        std_winner_list, smo_winner_list = [], []
        for seed in df9["seed"].unique():
            sub = df9[df9["seed"] == seed].sort_values("k")
            nh_std_mat = np.stack([sub[c].values for c in std_cols9], axis=0)
            nh_smo_mat = np.stack([sub[c].values for c in smo_cols9], axis=0)
            env_std_list.append(np.max(nh_std_mat, axis=0))
            env_smo_list.append(np.max(nh_smo_mat, axis=0))
            std_winner_list.append(
                np.array([perps9[i] for i in np.argmax(nh_std_mat, axis=0)]))
            smo_winner_list.append(
                np.array([perps9[i] for i in np.argmax(nh_smo_mat, axis=0)]))

        k_ax   = df9[df9["seed"] == df9["seed"].iloc[0]].sort_values("k")["k"].values
        mn_std = np.mean(env_std_list, axis=0)
        mn_smo = np.mean(env_smo_list, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        for env in env_std_list:
            ax.plot(k_ax, env, color=C_STD_LOC,   lw=0.6, ls="-",  alpha=0.2)
        for env in env_smo_list:
            ax.plot(k_ax, env, color=C_GREEN_LOC, lw=0.6, ls="--", alpha=0.2)
        ax.plot(k_ax, mn_std, color=C_STD_LOC,   lw=2.6, ls="-",
                label="standard t-SNE  (best ρ per k)")
        ax.plot(k_ax, mn_smo, color=C_GREEN_LOC, lw=2.2, ls="--",
                label=f"smooth  γ={gamma_s9}  (best ρ per k)")
        ax.set_xlabel("k"); ax.set_ylabel("Neighborhood Overlap")
        ax.set_ylim(*_EXP9_YLIM.get(args.dataset, _EXP9_YLIM_DEFAULT))
        ax.set_title(f"Smooth vs standard — best ρ envelope  (γ={gamma_s9})")
        ax.legend(fontsize=13, frameon=False)
        _clean_axes(ax); plt.tight_layout()
        _save_fig("09a_smooth_vs_envelope")

        from scipy.stats import mode as scipy_mode
        std_wm = np.array([
            scipy_mode(np.array([sw[ki] for sw in std_winner_list]),
                       keepdims=False).mode
            for ki in range(len(k_ax))
        ])
        smo_wm = np.array([
            scipy_mode(np.array([sw[ki] for sw in smo_winner_list]),
                       keepdims=False).mode
            for ki in range(len(k_ax))
        ])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(k_ax, std_wm, where="mid", color=C_STD_LOC,   lw=2.0,
                label="standard")
        ax.step(k_ax, smo_wm, where="mid", color=C_GREEN_LOC, lw=2.0,
                ls="--", label=f"smooth γ={gamma_s9}")
        ax.set_xlabel("k"); ax.set_ylabel("Best ρ")
        ax.set_title("Winning perplexity per k")
        ax.set_yticks(perps9); ax.legend(fontsize=13, frameon=False)
        _clean_axes(ax); plt.tight_layout()
        _save_fig("09b_winner_perplexity")
        print("  Exp 9 replotted.")

    # ── Exp 10 ────────────────────────────────────────────────────────────────
    if not args.skip_exp10:
        plots_dir = os.path.join(out_dir, "exp10_no_vs_perplexity")
        _plot_save_dir[0] = plots_dir
        df10 = pd.read_csv(_csv(plots_dir, "no_vs_perplexity.csv"))
        if "seed" not in df10.columns:
            df10["seed"] = 0
        smo_no_col10  = [c for c in df10.columns
                         if c.startswith("smooth_gamma") and c.endswith("_no")][0]
        smo_auc_col10 = [c for c in df10.columns
                         if c.startswith("smooth_gamma") and c.endswith("_auc")][0]
        gamma_s10 = float(smo_no_col10.replace("smooth_gamma", "").replace("_no", ""))
        k_eval10  = args.exp10_k_eval
        C_STD_LOC   = C_STANDARD
        C_GREEN_LOC = C_SMOOTH

        def _plot10_only(std_col, smo_col, ylabel, title, fname):
            seed_list = df10["seed"].unique()
            mn  = df10.groupby("perplexity")[[std_col, smo_col]].mean()
            sd  = df10.groupby("perplexity")[[std_col, smo_col]].std(ddof=1).fillna(0)
            p_ax = mn.index.values
            fig, ax = plt.subplots(figsize=(7, 5))
            for seed in seed_list:
                sub = df10[df10["seed"] == seed].sort_values("perplexity")
                ax.plot(sub["perplexity"].values, sub[std_col].values,
                        color=C_STD_LOC,   lw=0.6, ls="-",  alpha=0.25)
                ax.plot(sub["perplexity"].values, sub[smo_col].values,
                        color=C_GREEN_LOC, lw=0.6, ls="--", alpha=0.25)
            ax.plot(p_ax, mn[std_col].values, color=C_STD_LOC,   lw=2.4, ls="-",
                    marker="o", markersize=6, label="standard t-SNE")
            ax.plot(p_ax, mn[smo_col].values, color=C_GREEN_LOC, lw=2.2, ls="--",
                    marker="s", markersize=6, label=f"smooth  γ={gamma_s10}")
            ax.set_xlabel("Perplexity"); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.set_xticks(p_ax); ax.tick_params(axis="x", labelsize=11)
            ax.legend(fontsize=13, frameon=False)
            _clean_axes(ax); plt.tight_layout()
            _save_fig(fname)

        _p10a = "(b) " if args.dataset == "mnist" else ""
        _plot10_only("standard_no", smo_no_col10,
                     f"Neighborhood Overlap at k={k_eval10}",
                     f"{_p10a}NO@{k_eval10} vs perplexity",
                     "10a_no_vs_perplexity")
        _plot10_only("standard_auc", smo_auc_col10,
                     f"AUC  (mean NO,  k=1–{k_eval10})",
                     f"AUC@{k_eval10} vs perplexity",
                     "10b_auc_vs_perplexity")
        print("  Exp 10 replotted.")

    print(f"\nAll plots regenerated in: {out_dir}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="t-SNE experiments with modified openTSNE (native gamma).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",          required=True,
                   choices=["mnist", "mouse", "adult"])
    p.add_argument("--out_dir",          default=None)
    p.add_argument("--mouse_data_dir",   default=None)
    p.add_argument("--mouse_subsample",  type=int, default=None)
    p.add_argument("--mnist_subsample",  type=int, default=0)
    p.add_argument("--adult_max_rows",   type=int, default=None)
    p.add_argument("--perplexity",       type=int,   default=30)
    p.add_argument("--gamma_s",          type=float, default=0.7)
    p.add_argument("--gamma_h",          type=float, default=1.5)
    p.add_argument("--exp4_gammas",      type=float, nargs="+",
                   default=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0])
    p.add_argument("--exp5_gamma",       type=float, default=0.5)
    p.add_argument("--exp_grid_gammas",  type=float, nargs="+",
                   default=[0.0, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0])
    p.add_argument("--exp_grid_perps",   type=int,   nargs="+",
                   default=[30, 50, 100, 200])
    p.add_argument("--k_max",            type=int,   default=200)
    p.add_argument("--exp4_k_max",       type=int,   default=200)
    p.add_argument("--n_iter_ee",        type=int,   default=250)
    p.add_argument("--n_iter_main",      type=int,   default=750)
    p.add_argument("--ee",               type=float, default=12.0)
    p.add_argument("--n_jobs",           type=int,   default=-1)
    p.add_argument("--random_state",     type=int,   default=42)
    p.add_argument("--exp9_perplexities", type=int,  nargs="+",
                   default=[5, 10, 20, 30, 50, 70, 100, 120, 150, 200, 300])
    p.add_argument("--exp10_perplexities", type=int, nargs="+",
                   default=[5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100])
    p.add_argument("--exp10_k_eval",      type=int,  default=30)
    p.add_argument("--exp1_gamma",        type=float, default=0.5,
                   help="gamma_s shown in exp1 bar chart.")
    p.add_argument("--exp1_top_ranks",    type=int,   default=20,
                   help="Number of top neighbour ranks to draw in the exp1 "
                        "grouped bar chart.")
    p.add_argument("--n_runs",           type=int,   default=10,
                   help="Number of random seeds per experiment.")
    for i in range(1, 11):
        p.add_argument(f"--skip_exp{i}", action="store_true")
    p.add_argument("--plot_only", action="store_true",
                   help="Skip computation; regenerate figures from saved CSVs.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(_SCRIPT_DIR, "results",
                                    f"{args.dataset}_gamma_tsne")
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # Prefix every saved figure with the dataset name (mouse_/mnist_/adult_).
    _fig_name_prefix[0] = f"{args.dataset}_"

    print(f"\n{'='*60}")
    print(f"  dataset    : {args.dataset}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"  perplexity : {args.perplexity}")
    print(f"  gamma_s/h  : {args.gamma_s} / {args.gamma_h}")
    print(f"  n_runs     : {args.n_runs}")
    print(f"  plot_only  : {args.plot_only}")
    print(f"{'='*60}\n")

    if args.plot_only:
        plot_all_from_csv(args)
        return

    nj = args.n_jobs
    X_pca, X_eval, y, extra = load_data(args.dataset, args)

    # ── Shared base affinities (exp 1, 2, 3, 7) ──────────────────────────────
    need_base = not any([args.skip_exp1, args.skip_exp2,
                         args.skip_exp3, args.skip_exp7])
    P_jnt_base = P_cnd_base = P_jnt_smo = P_cnd_smo = None
    P_jnt_shp  = P_cnd_shp  = knn_idx_base = knn_dist_base = None
    if need_base:
        print(f"Building base affinities (ρ={args.perplexity}) ...")
        P_jnt_base, P_cnd_base, knn_idx_base, knn_dist_base = build_affinity(
            X_pca, args.perplexity, gamma=1.0,      n_jobs=nj)
        P_jnt_smo,  P_cnd_smo,  _,            _             = build_affinity(
            X_pca, args.perplexity, gamma=args.gamma_s, n_jobs=nj)
        P_jnt_shp,  P_cnd_shp,  _,            _             = build_affinity(
            X_pca, args.perplexity, gamma=args.gamma_h, n_jobs=nj)

    if not args.skip_exp1:
        exp1_affinity_sharpness(
            P_cnd_base, knn_idx_base, args.out_dir,
            perplexity=args.perplexity, gamma_s=args.exp1_gamma,
            top_ranks=args.exp1_top_ranks,
        )

    if not args.skip_exp2:
        exp2_effective_perplexity(
            P_cnd_base, P_cnd_smo, P_cnd_shp, args.out_dir,
            gamma_s=args.gamma_s, gamma_h=args.gamma_h,
            perplexity=args.perplexity,
        )

    if not args.skip_exp3:
        exp3_delta_perp_correlations(
            P_cnd_base, P_cnd_smo, knn_idx_base, knn_dist_base, args.out_dir,
            perplexity=args.perplexity, gamma_s=args.gamma_s,
        )

    # ── Sensitivity grid (shared by Exp 2b, 6, and 8) ────────────────────────
    df_grid   = None
    need_grid = not args.skip_exp2 or not args.skip_exp6
    if need_grid:
        print("Computing sensitivity grid ...")
        grid_dir = os.path.join(args.out_dir, "sensitivity_grid")
        os.makedirs(grid_dir, exist_ok=True)
        df_grid = _run_sensitivity_grid(
            X_pca, X_eval,
            gammas=args.exp_grid_gammas,
            perplexities=args.exp_grid_perps,
            out_dir=grid_dir,
            n_jobs=nj, master_seed=args.random_state, n_runs=args.n_runs,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main,
            ee=args.ee,
        )

    if not args.skip_exp4:
        exp4_nh_gamma_sweep(
            X_pca, X_eval, args.out_dir,
            perplexity=args.perplexity,
            k_max=args.exp4_k_max,
            gammas=args.exp4_gammas,
            n_jobs=nj, master_seed=args.random_state, n_runs=args.n_runs,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    if not args.skip_exp5:
        exp5_embedding_comparison(
            X_pca, y, args.out_dir,
            dataset=args.dataset,
            perplexity=args.perplexity,
            gamma_s=args.exp5_gamma,
            cluster_colors_arr=extra.get("cluster_colors_arr"),
            n_jobs=nj, random_state=args.random_state,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    if not args.skip_exp2 and df_grid is not None:
        exp2b_median_eff_perp_heatmap(df_grid, args.out_dir)

    if not args.skip_exp6 and df_grid is not None:
        exp6_sensitivity_heatmaps(df_grid, args.out_dir, dataset=args.dataset)

    if not args.skip_exp7:
        exp7_nh_comparison(
            X_pca, X_eval, args.out_dir,
            perplexity=args.perplexity,
            gamma_s=args.gamma_s,
            gamma_h=args.gamma_h,
            P_cond_smooth=P_cnd_smo,
            k_max=args.k_max, dataset=args.dataset,
            n_jobs=nj, master_seed=args.random_state, n_runs=args.n_runs,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    if not args.skip_exp8:
        exp8_global_spearman_vs_gamma(
            X_pca, X_eval, args.out_dir,
            gammas=args.exp_grid_gammas,
            perplexities=args.exp_grid_perps,
            dataset=args.dataset,
            n_jobs=nj, master_seed=args.random_state, n_runs=args.n_runs,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    if not args.skip_exp9:
        exp9_smooth_vs_envelope(
            X_pca, X_eval, args.out_dir,
            gamma_s=args.gamma_s,
            standard_perplexities=args.exp9_perplexities,
            k_max=args.k_max, dataset=args.dataset,
            n_jobs=nj, master_seed=args.random_state, n_runs=args.n_runs,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    if not args.skip_exp10:
        exp10_no_vs_perplexity(
            X_pca, X_eval, args.out_dir,
            gamma_s=args.gamma_s,
            perplexities=args.exp10_perplexities,
            k_eval=args.exp10_k_eval, dataset=args.dataset,
            n_jobs=nj, master_seed=args.random_state, n_runs=args.n_runs,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS DONE")
    print(f"Results in: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
