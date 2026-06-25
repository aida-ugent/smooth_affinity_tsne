#!/usr/bin/env python
"""
exp.py  —  Curriculum-gamma t-SNE experiment
============================================

Question
--------
We added ``gamma`` as a native affinity parameter to openTSNE.  Earlier
experiments established a trade-off:

  * SHARP gamma (γ ≈ 1.5–2)  →  best *local* neighbor preservation
  * SMOOTH gamma (γ ≲ 1)     →  best *global* structure preservation

Standard t-SNE (γ=1) sits in between and is dominated on both ends.

Idea (inspired by PaCMAP's 3-phase weight schedule)
---------------------------------------------------
Instead of a single fixed γ, anneal γ across optimization in three stages,
so each scale of structure is formed by the γ that is best at it:

  Stage 1 — GLOBAL : smooth γ (≤1), with early exaggeration, to lay out the
                     coarse global arrangement of clusters.
  Stage 2 — MID    : γ ≈ 1, to settle coarse neighborhoods.
  Stage 3 — LOCAL  : sharp γ (>1), to tighten the exact fine-grained
                     nearest-neighbor structure.

Because γ is baked into the high-dimensional joint affinity matrix P, we
realise the schedule by recomputing P with a new γ between optimization
phases (reusing the *same* kNN graph), and continuing gradient descent.

What this script does
---------------------
1. Loads a dataset (mnist / mouse / adult), optionally subsampled.
2. Runs several MODELS, all with an identical total iteration budget for a
   fair comparison:
      - baselines: standard (γ=1), smooth, sharp
      - a small sweep of 3-stage curriculum configurations
3. Evaluates each across the full neighbourhood-size range with two CURVES
   (saved to curves.csv, drawn as line charts over k=1..K_MAX):
      - NO@k    : Neighbourhood Overlap — fraction of the true k-NN preserved
      - Trust@k : Trustworthiness — penalty for false (intruding) neighbours
   plus three SCALAR metrics (saved to results.csv, drawn as grouped bars):
      - knn_acc_10     : label kNN purity   (fair local quality)
      - global_spearman: Spearman ρ of pairwise distances (global)
      - triplet_acc    : random-triplet distance-ordering accuracy (fair global)
4. Aggregates over seeds, writes results.csv + curves.csv, draws the plots,
   and prints an automatic VERDICT on whether the curriculum idea works.

Usage
-----
  # quick smoke test (small subsample, 1 seed, fewer iters)
  python exp.py --dataset mnist --quick

  # full run
  python exp.py --dataset mnist --n_samples 10000 --n_seeds 3

  # re-plot / re-verdict from an existing results.csv (no t-SNE)
  python exp.py --dataset mnist --plot_only
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

from openTSNE import TSNEEmbedding, initialization
from openTSNE.affinity import PerplexityBasedNN, joint_probabilities_nn

import baselines  # external DR methods (pacmap / umap / trimap / vae)

# data loaders live in ../experiments/data
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "experiments", "data")
sys.path.insert(0, _EXP_DATA_DIR)
from load_data import load_mnist_data, load_mouse_data, load_adult_data  # noqa: E402


# =============================================================================
# Metrics
# =============================================================================
# Each embedding is characterised at EVERY neighbourhood size k = 1..K with two
# curves, plus three scalar summary metrics:
#
#   curves (curves.csv → line charts)
#     * NO@k    — Neighbourhood Overlap: fraction of each point's k high-D
#                 nearest neighbours that are still among its k low-D nearest
#                 neighbours (1 = perfect local preservation).  The HD kNN graph
#                 is also the t-SNE family's optimisation target, so NO is
#                 "home-field" for t-SNE — read it for the γ-family story, not as
#                 a neutral cross-family local score.
#     * Trust@k — Trustworthiness: penalises low-D neighbours that are actually
#                 far in high-D.  Objective-independent → a fair local metric.
#
#   scalars (results.csv → grouped bars)
#     * knn_acc_10      — label kNN purity        (fair, supervised local)
#     * global_spearman — Spearman ρ of HD vs LD pairwise distances (global)
#     * triplet_acc     — random-triplet distance-ordering accuracy (fair global)

def _knn_indices(X, k, n_jobs=-1):
    """k nearest-neighbour indices (self excluded), shape (n, k)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def neighborhood_overlap_curve(hd_nbrs, ld_nbrs, K):
    """NO@k for k=1..K: mean over points of |HD_k ∩ LD_k| / k.

    Computed for the whole curve at once: a common element hd[j]=ld[r] lies in
    both top-k sets for every k > max(j, r), so each contributes a suffix to the
    cumulative-intersection count, summed over points then divided by k·n.
    """
    n = hd_nbrs.shape[0]
    diff = np.zeros(K + 1)              # suffix-difference accumulator over k-idx
    kidx = np.arange(K)
    for i in range(n):
        hd = hd_nbrs[i, :K]
        ld = ld_nbrs[i, :K]
        order = np.argsort(ld, kind="stable")
        sorted_ld = ld[order]
        pos = np.searchsorted(sorted_ld, hd)
        ranks = np.full(K, K, dtype=np.int64)
        ok = pos < K
        hit = np.zeros(K, dtype=bool)
        hit[ok] = sorted_ld[pos[ok]] == hd[ok]
        ranks[hit] = order[pos[hit]]
        thr = np.maximum(kidx, ranks)  # element enters both sets at this k-index
        thr = thr[thr < K]
        np.add.at(diff, thr, 1.0)
    counts = np.cumsum(diff[:K])
    return counts / (np.arange(1, K + 1) * n)


def hd_rank_matrix(X_sub):
    """rank[i, j] = position of j in the HD distance ordering from i (self = 0).

    O(m²); computed once on a fixed subsample and reused by every embedding's
    trustworthiness curve.
    """
    d = pairwise_distances(X_sub, metric="euclidean")
    order = np.argsort(d, axis=1, kind="stable")
    m = d.shape[0]
    rank = np.empty((m, m), dtype=np.int32)
    rank[np.arange(m)[:, None], order] = np.arange(m)[None, :]
    return rank


def trustworthiness_curve(hd_rank, Y_sub, K, n_jobs=-1):
    """Trust@k for k=1..K on a subsample (sklearn's definition, vectorised)."""
    m = hd_rank.shape[0]
    ld_order = _knn_indices(Y_sub, k=K, n_jobs=n_jobs)        # (m, K) LD nbrs
    hr = np.take_along_axis(hd_rank, ld_order, axis=1)        # their HD ranks
    T = np.full(K, np.nan)
    for k in range(1, K + 1):
        denom = m * k * (2 * m - 3 * k - 1)
        if denom <= 0:
            break                                            # k too large for m
        pen = np.maximum(hr[:, :k] - k, 0).sum()
        T[k - 1] = 1.0 - (2.0 / denom) * pen
    return T


def label_knn_accuracy(ld_nbrs, y, k):
    """Mean fraction of each point's k LD-neighbours sharing its true label.

    Reference is the ground-truth labels, NOT any method's HD kNN graph, so it
    is fair across method families (unlike NO).
    """
    y = np.asarray(y)
    nb = ld_nbrs[:, :k]
    return float(np.mean((y[nb] == y[:, None]).mean(axis=1)))


def global_spearman(X, Y, random_state=42, n_sample=5000):
    """Spearman ρ between HD and LD pairwise distances (on a subsample)."""
    n = len(X)
    if n > n_sample:
        idx = np.sort(np.random.default_rng(random_state).choice(
            n, size=n_sample, replace=False))
        X, Y = X[idx], Y[idx]
    rho, _ = spearmanr(pdist(X), pdist(Y))
    return float(rho)


def triplet_accuracy(X, Y, n_per_point=5, random_state=42):
    """Fraction of random triplets whose HD distance ordering survives in LD.

    For each anchor i we draw ``n_per_point`` random pairs (j, l) and check
    whether sign(d(i,j) − d(i,l)) agrees between HD and LD.  Method-agnostic and
    a standard global-structure score (used by TriMap / PaCMAP).
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    i = np.repeat(np.arange(n), n_per_point)
    j = rng.integers(0, n, size=i.shape)
    l = rng.integers(0, n, size=i.shape)
    bad = (j == i) | (l == i) | (j == l)
    j[bad] = (j[bad] + 1) % n
    l[bad] = (l[bad] + 2) % n
    dij_hd = np.linalg.norm(X[i] - X[j], axis=1)
    dil_hd = np.linalg.norm(X[i] - X[l], axis=1)
    dij_ld = np.linalg.norm(Y[i] - Y[j], axis=1)
    dil_ld = np.linalg.norm(Y[i] - Y[l], axis=1)
    return float(np.mean((dij_hd < dil_hd) == (dij_ld < dil_ld)))


def evaluate(Y, X_eval, hd_nbrs, K, trust_sub, hd_rank, y=None,
             master_seed=42, n_per_point=5, n_jobs=-1):
    """Return ({scalar metrics}, {curve arrays}) for an embedding Y."""
    Y = np.asarray(Y)
    ld_nbrs = _knn_indices(Y, k=K, n_jobs=n_jobs)
    curves = {
        "no":    neighborhood_overlap_curve(hd_nbrs, ld_nbrs, K),
        "trust": trustworthiness_curve(hd_rank, Y[trust_sub], K, n_jobs=n_jobs),
    }
    scalars = {
        "knn_acc_10": (label_knn_accuracy(ld_nbrs, y, 10)
                       if y is not None else np.nan),
        "global_spearman": global_spearman(X_eval, Y, random_state=master_seed),
        "triplet_acc": triplet_accuracy(X_eval, Y, n_per_point=n_per_point,
                                        random_state=master_seed),
    }
    return scalars, curves


# =============================================================================
# Affinity / optimization machinery
# =============================================================================

class _MutableAff:
    """Minimal affinities object exposing a swappable ``.P`` for TSNEEmbedding."""
    def __init__(self, P):
        self.P = P


def build_knn(X_pca, perplexity, n_jobs=-1, verbose=False):
    """Compute the kNN graph once (γ=1); return (knn_idx, knn_dist, eff_perp)."""
    aff = PerplexityBasedNN(X_pca, perplexity=perplexity, gamma=1.0,
                            n_jobs=n_jobs, verbose=verbose)
    return aff.knn_indices, aff.knn_distances, aff.effective_perplexity_


def joint_P_for_gamma(knn_idx, knn_dist, eff_perp, gamma, n_jobs=-1):
    """Build the symmetrized joint P for a given γ from a fixed kNN graph.

    Reusing the same neighbors/distances guarantees every γ-variant and every
    curriculum stage shares one kNN graph — only the power transform changes.
    """
    P, _ = joint_probabilities_nn(
        knn_idx, knn_dist, [eff_perp],
        symmetrize=True, normalization="pair-wise",
        n_jobs=n_jobs, gamma=gamma,
    )
    return P


def run_single(P_joint, init, n_iter_ee, n_iter_main, ee, n_jobs=-1,
               random_state=42):
    """Standard 2-phase t-SNE: early exaggeration then main, fixed P."""
    emb = TSNEEmbedding(init, _MutableAff(P_joint), n_jobs=n_jobs,
                        random_state=random_state)
    emb.optimize(n_iter_ee,   exaggeration=ee,  momentum=0.5, inplace=True)
    emb.optimize(n_iter_main, exaggeration=1.0, momentum=0.8, inplace=True)
    return np.array(emb)


def run_curriculum(stages, init, knn_idx, knn_dist, eff_perp, n_jobs=-1,
                   random_state=42):
    """Multi-stage t-SNE with a per-stage γ.

    ``stages`` is a list of dicts:
        {gamma, n_iter, exaggeration, momentum}
    Between stages we recompute P with the stage's γ (same kNN graph) and
    continue gradient descent from the current embedding.
    """
    aff = _MutableAff(joint_P_for_gamma(knn_idx, knn_dist, eff_perp,
                                        stages[0]["gamma"], n_jobs=n_jobs))
    emb = TSNEEmbedding(init, aff, n_jobs=n_jobs, random_state=random_state)
    for i, st in enumerate(stages):
        if i > 0:  # stage 0's P is already loaded above
            aff.P = joint_P_for_gamma(knn_idx, knn_dist, eff_perp,
                                      st["gamma"], n_jobs=n_jobs)
        emb.optimize(st["n_iter"], exaggeration=st["exaggeration"],
                     momentum=st["momentum"], inplace=True)
    return np.array(emb)


# =============================================================================
# Model definitions
# =============================================================================

def make_models(total_iter, ee_iter, ee, gamma_smooth, gamma_sharp):
    """Define every model to compare.

    All models consume the SAME total number of gradient-descent iterations
    (``total_iter``) so differences reflect the γ schedule, not compute.

    A baseline = (ee_iter early-exag iters) + (main iters), main = total-ee.
    A curriculum splits ``total_iter`` across three γ stages, with the global
    stage carrying the early-exaggeration phase.
    """
    main_iter = total_iter - ee_iter

    models = {}

    # ── Baselines: single fixed γ ────────────────────────────────────────────
    models["baseline_standard"] = {
        "kind": "single", "gamma": 1.0,
        "n_iter_ee": ee_iter, "n_iter_main": main_iter, "ee": ee,
        "group": "baseline",
    }
    models["baseline_smooth"] = {
        "kind": "single", "gamma": gamma_smooth,
        "n_iter_ee": ee_iter, "n_iter_main": main_iter, "ee": ee,
        "group": "baseline",
    }
    models["baseline_sharp"] = {
        "kind": "single", "gamma": gamma_sharp,
        "n_iter_ee": ee_iter, "n_iter_main": main_iter, "ee": ee,
        "group": "baseline",
    }

    # ── Curriculum sweep ──────────────────────────────────────────────────────
    # Stage 1 (global) always carries early exaggeration (ee, momentum 0.5).
    # The remaining (main_iter) iters are split between mid (γ=1) and local
    # (sharp γ).  We sweep the global γ, the sharp γ, and the mid/local split.
    rest = main_iter

    def curric(g_global, g_sharp, frac_local):
        n_local = int(round(rest * frac_local))
        n_mid = rest - n_local
        return {
            "kind": "curriculum",
            "group": "curriculum",
            "stages": [
                {"gamma": g_global, "n_iter": ee_iter,
                 "exaggeration": ee,  "momentum": 0.5},   # GLOBAL  + early exag
                {"gamma": 1.0,       "n_iter": n_mid,
                 "exaggeration": 1.0, "momentum": 0.8},   # MID
                {"gamma": g_sharp,   "n_iter": n_local,
                 "exaggeration": 1.0, "momentum": 0.8},   # LOCAL
            ],
        }

    for g_global in (gamma_smooth, 1.0):
        gtag = f"glob{g_global:g}"
        for g_sharp in (gamma_sharp,):
            for frac_local, ftag in ((0.5, "split5050"), (0.66, "localheavy")):
                name = f"curric_{gtag}_sharp{g_sharp:g}_{ftag}"
                models[name] = curric(g_global, g_sharp, frac_local)

    # A 2-stage ablation: smooth-global → sharp-local, skipping the γ=1 mid
    # stage, to test whether the mid stage matters.
    n_local = int(round(rest * 0.5))
    models[f"curric_glob{gamma_smooth:g}_sharp{gamma_sharp:g}_no_mid"] = {
        "kind": "curriculum", "group": "curriculum",
        "stages": [
            {"gamma": gamma_smooth, "n_iter": ee_iter,
             "exaggeration": ee,  "momentum": 0.5},
            {"gamma": gamma_sharp, "n_iter": rest,
             "exaggeration": 1.0, "momentum": 0.8},
        ],
    }

    return models


def make_external_models(methods="all", vae_epochs=150):
    """External DR baselines (pacmap / umap / trimap / vae).

    ``methods`` is "all" or a comma-separated subset. Only methods whose
    library is installed are returned, so the run degrades gracefully.
    """
    avail = baselines.available_methods()
    if methods != "all":
        wanted = {m.strip() for m in methods.split(",") if m.strip()}
        avail = {k: v for k, v in avail.items() if k in wanted}

    models = {}
    for name, fn in avail.items():
        if name == "vae":
            # bind epochs without leaking the loop variable
            def _vae(X, random_state, n_jobs=-1, _fn=fn, _ep=vae_epochs):
                return _fn(X, random_state=random_state, epochs=_ep,
                           n_jobs=n_jobs)
            models[name] = {"kind": "external", "group": "external", "fn": _vae}
        else:
            models[name] = {"kind": "external", "group": "external", "fn": fn}
    return models


# =============================================================================
# Driver
# =============================================================================

def load_dataset(name, n_samples, random_state, args):
    """Return (X_pca, y) for the requested dataset, optionally subsampled."""
    if name == "mnist":
        X, y, _ = load_mnist_data(n_pca=50, random_state=random_state)
    elif name == "mouse":
        if not args.mouse_pickle:
            raise SystemExit("--mouse_pickle is required for --dataset mouse")
        X, y, _, _ = load_mouse_data(args.mouse_pickle)
    elif name == "adult":
        X, y = load_adult_data(max_rows=max(n_samples or 0, 10000),
                               random_state=random_state)
    else:
        raise SystemExit(f"unknown dataset {name}")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if n_samples and n_samples < len(X):
        idx = np.sort(np.random.default_rng(random_state).choice(
            len(X), size=n_samples, replace=False))
        X, y = X[idx], y[idx]
    return X, y


def run_all(args, out_dir):
    csv_path = os.path.join(out_dir, "results.csv")
    curve_path = os.path.join(out_dir, "curves.csv")
    rng_seeds = np.random.default_rng(args.master_seed).integers(
        0, 2**31, size=args.n_seeds)
    seeds = [int(s) for s in rng_seeds]

    models = make_models(args.total_iter, args.ee_iter, args.ee,
                         args.gamma_smooth, args.gamma_sharp)
    if not args.no_external:
        ext = make_external_models(args.methods, vae_epochs=args.vae_epochs)
        models.update(ext)
        print(f"  external baselines: {list(ext)}")
    print(f"Models ({len(models)}): {list(models)}")

    print(f"\nLoading {args.dataset} (n_samples={args.n_samples}) ...")
    X_pca, y = load_dataset(args.dataset, args.n_samples, args.master_seed, args)
    X_eval = X_pca
    n = len(X_eval)
    print(f"  data shape = {X_pca.shape}")

    # neighbourhood-size axis for the curves (clamp to n-2 so k is valid)
    K = int(min(args.k_max, n - 2))
    print(f"Pre-computing evaluation HD-kNN (k={K}) ...")
    hd_nbrs = _knn_indices(X_eval, K, n_jobs=args.n_jobs)

    # fixed subsample + HD rank matrix for the O(m^2) trustworthiness curve
    m = int(min(args.trust_subsample, n))
    trust_sub = (np.sort(np.random.default_rng(args.master_seed).choice(
        n, size=m, replace=False)) if m < n else np.arange(n))
    print(f"Pre-computing HD rank matrix for trustworthiness (m={m}) ...")
    hd_rank = hd_rank_matrix(X_eval[trust_sub])

    existing = (pd.read_csv(csv_path)
                if (os.path.exists(csv_path) and not args.fresh)
                else pd.DataFrame())
    existing_curves = (pd.read_csv(curve_path)
                       if (os.path.exists(curve_path) and not args.fresh)
                       else pd.DataFrame())
    done = (set(zip(existing["model"], existing["seed"]))
            if not existing.empty else set())
    i10 = min(9, K - 1)  # index of NO@10 / trust@10 for the progress line

    rows, crows = [], []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        # kNN graph is γ-independent and seed-independent for the data, but the
        # PCA init depends on data only; recompute init per seed for variety.
        knn_idx, knn_dist, eff_perp = build_knn(
            X_pca, args.perplexity, n_jobs=args.n_jobs, verbose=False)
        init = initialization.pca(X_pca, random_state=seed)

        for name, spec in models.items():
            if (name, seed) in done:
                print(f"  {name}: cached, skip")
                continue
            t0 = time.time()
            if spec["kind"] == "single":
                P = joint_P_for_gamma(knn_idx, knn_dist, eff_perp,
                                      spec["gamma"], n_jobs=args.n_jobs)
                Y = run_single(P, init.copy(), spec["n_iter_ee"],
                               spec["n_iter_main"], spec["ee"],
                               n_jobs=args.n_jobs, random_state=seed)
            elif spec["kind"] == "external":
                # external methods run on the same PCA features and seed, but
                # use their own initialization / optimizer internally.
                Y = spec["fn"](X_pca, random_state=seed, n_jobs=args.n_jobs)
            else:
                Y = run_curriculum(spec["stages"], init.copy(),
                                   knn_idx, knn_dist, eff_perp,
                                   n_jobs=args.n_jobs, random_state=seed)
            scal, curves = evaluate(
                Y, X_eval, hd_nbrs, K, trust_sub, hd_rank, y=y,
                master_seed=args.master_seed,
                n_per_point=args.triplet_per_point, n_jobs=args.n_jobs)
            dt = time.time() - t0
            rows.append({"model": name, "group": spec["group"], "seed": seed,
                         "time_s": round(dt, 1), **scal})
            crows.append(pd.DataFrame({
                "model": name, "group": spec["group"], "seed": seed,
                "k": np.arange(1, K + 1),
                "no": curves["no"], "trust": curves["trust"]}))
            print(f"  {name:42s} NO@10={curves['no'][i10]:.3f}  "
                  f"trust@10={curves['trust'][i10]:.3f}  "
                  f"knn={scal['knn_acc_10']:.3f}  "
                  f"ρ={scal['global_spearman']:.3f}  "
                  f"trip={scal['triplet_acc']:.3f}  ({dt:.1f}s)")

            # checkpoint after every (model, seed)
            pd.concat([existing, pd.DataFrame(rows)], ignore_index=True
                      ).to_csv(csv_path, index=False)
            pd.concat([existing_curves] + crows, ignore_index=True
                      ).to_csv(curve_path, index=False)

    final = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) \
        if rows else existing
    final.to_csv(csv_path, index=False)
    final_c = pd.concat([existing_curves] + crows, ignore_index=True) \
        if crows else existing_curves
    final_c.to_csv(curve_path, index=False)
    print(f"\nResults → {csv_path}\nCurves  → {curve_path}")
    return final, final_c


# =============================================================================
# Analysis, plots, verdict
# =============================================================================

# colour-blind-safe qualitative palette (Paul Tol 'muted' + a few extensions),
# one colour per model; group is additionally encoded by marker + line style so
# the figures stay legible in greyscale and to all colour-vision types.
_PALETTE = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
            "#DDCC77", "#CC6677", "#882255", "#AA4499", "#661100",
            "#6699CC", "#000000"]
# marker + line style per group (used by exp.py's 3 groups and by the schedule
# families that tune_curriculum.py adds); .get(..) fallbacks keep it open-ended.
_GROUP_LS = {"baseline": (0, (5, 2)), "curriculum": "solid",
             "external": (0, (1, 1.2)),
             "reference": (0, (5, 2)), "ramp": "solid",
             "duration": (0, (3, 1, 1, 1)), "fine": (0, (1, 1.2))}
_GROUP_MARKER = {"baseline": "s", "curriculum": "o", "external": "^",
                 "reference": "s", "ramp": "o", "duration": "D", "fine": "."}
_GROUP_ORDER = {"baseline": 0, "curriculum": 1, "external": 2,
                "reference": 0, "ramp": 1, "duration": 2, "fine": 3}


def _pretty(name):
    return name.replace("baseline_", "").replace("curric_", "")


def _groups_present(df):
    """Unique groups in df, ordered by _GROUP_ORDER (unknown groups last)."""
    return sorted(df["group"].unique().tolist(),
                  key=lambda g: _GROUP_ORDER.get(g, 99))


def _model_order(df):
    info = df[["model", "group"]].drop_duplicates().copy()
    info["go"] = info["group"].map(_GROUP_ORDER).fillna(9)
    return info.sort_values(["go", "model"])["model"].tolist()


def summarize_scalars(df):
    return df.groupby(["model", "group"], as_index=False).agg(
        knn_acc_mean=("knn_acc_10", "mean"), knn_acc_std=("knn_acc_10", "std"),
        spearman_mean=("global_spearman", "mean"),
        spearman_std=("global_spearman", "std"),
        triplet_mean=("triplet_acc", "mean"), triplet_std=("triplet_acc", "std"),
        n=("seed", "nunique"),
    ).fillna(0.0)


def summarize_curves(cdf):
    return cdf.groupby(["model", "group", "k"], as_index=False).agg(
        no_mean=("no", "mean"), no_std=("no", "std"),
        trust_mean=("trust", "mean"), trust_std=("trust", "std"),
    ).fillna(0.0)


def _auc(curve_sum, value, model, k_lo, k_hi):
    sub = curve_sum[(curve_sum["model"] == model) &
                    (curve_sum["k"] >= k_lo) & (curve_sum["k"] <= k_hi)]
    return float(sub[value].mean()) if len(sub) else float("nan")


def _plot_curve(curve_sum, value, std, ylabel, title, note, out_path):
    """One line per model, mean over seeds with a faint ±1 SD band."""
    models = _model_order(curve_sum)
    cmap = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}
    K = int(curve_sum["k"].max())
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    for m in models:
        sub = curve_sum[curve_sum["model"] == m].sort_values("k")
        grp = sub["group"].iloc[0]
        ax.fill_between(sub["k"], sub[value] - sub[std], sub[value] + sub[std],
                        color=cmap[m], alpha=0.08, linewidth=0)
        ax.plot(sub["k"], sub[value], color=cmap[m], linewidth=1.9,
                linestyle=_GROUP_LS.get(grp, "solid"),
                marker=_GROUP_MARKER.get(grp, "o"),
                markevery=max(1, K // 12), markersize=5,
                markeredgecolor="white", markeredgewidth=0.4,
                label=_pretty(m))
    ax.set_xlim(0, K)
    ax.set_xlabel("neighbourhood size  k", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, style="italic", color="#555555")
    # reserve right margin for the per-model legend so it is never clipped
    fig.subplots_adjust(left=0.07, right=0.74, top=0.92, bottom=0.10)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=9, title="model", handlelength=2.8)
    leg.get_title().set_fontweight("bold")
    ax.add_artist(leg)
    # group-encoding key (marker + line style) inside the axes
    handles = [plt.Line2D([0], [0], color="#444444",
                          linestyle=_GROUP_LS.get(g, "solid"),
                          marker=_GROUP_MARKER.get(g, "o"),
                          markersize=6, label=g)
               for g in _groups_present(curve_sum)]
    key = ax.legend(handles=handles, loc="lower left", frameon=False,
                    fontsize=9, title="family", handlelength=2.8)
    key.get_title().set_fontweight("bold")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_scalars(scal_sum, out_path, dataset):
    """Grouped horizontal bars for the three fair scalar metrics."""
    models = _model_order(scal_sum)
    cmap = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}
    s = scal_sum.set_index("model").reindex(models)
    colors = [cmap[m] for m in models]
    panels = [
        ("knn_acc_mean", "knn_acc_std",
         "kNN label accuracy @10\n(fair local — higher better)"),
        ("spearman_mean", "spearman_std",
         "Global Spearman ρ\n(distance ranks — higher better)"),
        ("triplet_mean", "triplet_std",
         "Random-triplet accuracy\n(global ordering — higher better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    ypos = np.arange(len(models))
    for ax, (mcol, scol, title) in zip(axes, panels):
        ax.barh(ypos, s[mcol].values, xerr=s[scol].values,
                color=colors, edgecolor="black", linewidth=0.4,
                error_kw=dict(ecolor="#444444", lw=0.8))
        ax.set_yticks(ypos)
        ax.set_yticklabels([_pretty(m) for m in models], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        xmax = float(np.nanmax(s[mcol].values)) if len(s) else 1.0
        for yi, v in zip(ypos, s[mcol].values):
            ax.text(v + 0.01 * xmax, yi, f"{v:.3f}", va="center", fontsize=8)
    fig.suptitle(f"Fair cross-method scalar metrics — {dataset}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff(scal_sum, curve_sum, out_path, dataset):
    """Local–global trade-off scatter: NO@1-10 (x) vs Spearman ρ (y)."""
    models = _model_order(scal_sum)
    cmap = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}
    s = scal_sum.set_index("model")
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    for m in models:
        x = _auc(curve_sum, "no_mean", m, 1, 10)
        y = s.loc[m, "spearman_mean"]
        grp = s.loc[m, "group"]
        ax.scatter(x, y, s=170, c=cmap[m], marker=_GROUP_MARKER.get(grp, "o"),
                   edgecolors="black", linewidth=0.6, zorder=3)
        ax.annotate(_pretty(m), (x, y), fontsize=8, xytext=(5, 4),
                    textcoords="offset points")
    handles = [plt.Line2D([0], [0], marker=_GROUP_MARKER.get(g, "o"), color="w",
                          markerfacecolor="#888888", markersize=11,
                          markeredgecolor="black", label=g)
               for g in _groups_present(scal_sum)]
    ax.legend(handles=handles, frameon=False, loc="best", title="family")
    ax.set_xlabel("Local — Neighbourhood Overlap NO@1–10  →  better", fontsize=12)
    ax.set_ylabel("Global — Spearman ρ  →  better", fontsize=12)
    ax.set_title(f"Local–global trade-off — {dataset}\n"
                 "top-right escapes the trade-off (good local AND global)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_plots(scal_sum, curve_sum, out_dir, dataset):
    p_no = os.path.join(out_dir, f"{dataset}_NO_curve.png")
    p_tr = os.path.join(out_dir, f"{dataset}_trustworthiness_curve.png")
    p_sc = os.path.join(out_dir, f"{dataset}_scalar_metrics.png")
    p_td = os.path.join(out_dir, f"{dataset}_tradeoff_scatter.png")
    _plot_curve(
        curve_sum, "no_mean", "no_std",
        "Neighbourhood Overlap   NO@k",
        f"Neighbourhood Overlap vs k — {dataset}",
        "higher = more of the true k-NN preserved   •   shaded = ±1 SD over seeds",
        p_no)
    _plot_curve(
        curve_sum, "trust_mean", "trust_std",
        "Trustworthiness   T@k",
        f"Trustworthiness vs k — {dataset}",
        "higher = fewer false neighbours intruding   •   shaded = ±1 SD over seeds",
        p_tr)
    _plot_scalars(scal_sum, p_sc, dataset)
    _plot_tradeoff(scal_sum, curve_sum, p_td, dataset)
    print(f"Plots → {p_no}\n        {p_tr}\n        {p_sc}\n        {p_td}")


def verdict(scal_sum, curve_sum, out_dir, dataset):
    """Decide whether the curriculum idea works and write a markdown report.

    The within-family local story uses NO@1-10 (t-SNE home-field, valid only
    among the γ models); the cross-method comparison uses ONLY the fair,
    method-agnostic metrics (trust@10, knn@10, spearman, triplet).
    """
    s = scal_sum.set_index("model")
    no10 = {m: _auc(curve_sum, "no_mean", m, 1, 10) for m in s.index}
    tr10 = {m: _auc(curve_sum, "trust_mean", m, 10, 10) for m in s.index}
    tbl = pd.DataFrame({
        "NO@1-10":  pd.Series(no10),
        "trust@10": pd.Series(tr10),
        "knn@10":   s["knn_acc_mean"],
        "spearman": s["spearman_mean"],
        "triplet":  s["triplet_mean"],
        "group":    s["group"],
    })
    cols = ["NO@1-10", "trust@10", "knn@10", "spearman", "triplet"]
    base = tbl[tbl["group"] == "baseline"]
    curr = tbl[tbl["group"] == "curriculum"].copy()
    ext = tbl[tbl["group"] == "external"]

    lines = [f"# Curriculum-γ t-SNE — verdict ({dataset})\n"]
    lines.append(
        "Metrics: **NO@1-10** = local neighbour overlap (t-SNE-family "
        "home-field — comparable *only* within the γ models); **trust@10 / "
        "knn@10** = fair local quality; **spearman / triplet** = fair global "
        "structure.\n")
    lines.append("## Baselines\n")
    lines.append(base[cols].round(4).to_string())
    lines.append("\n## Curriculum models\n")
    lines.append(curr[cols].round(4).to_string())

    std = base.loc["baseline_standard"]
    best_local = base["NO@1-10"].max()
    best_global = base["spearman"].max()

    def cap(val, lo, hi):
        if hi - lo <= 1e-9:
            return 1.0
        return (val - lo) / (hi - lo)

    curr["local_capture"] = curr["NO@1-10"].apply(
        lambda v: cap(v, std["NO@1-10"], best_local))
    curr["global_capture"] = curr["spearman"].apply(
        lambda v: cap(v, std["spearman"], best_global))
    curr["min_capture"] = curr[["local_capture", "global_capture"]].min(axis=1)
    bc = curr.sort_values("min_capture", ascending=False).iloc[0]

    lines.append("\n## Headroom capture "
                 "(local=NO@1-10, global=spearman; vs standard, "
                 "normalised to best baseline)\n")
    lines.append(curr[["local_capture", "global_capture", "min_capture"]]
                 .round(3).to_string())

    dom = curr[(curr["NO@1-10"] >= std["NO@1-10"]) &
               (curr["trust@10"] >= std["trust@10"]) &
               (curr["spearman"] >= std["spearman"])]

    lines.append("\n## Verdict\n")
    works = False
    if not dom.empty:
        lines.append(f"- **{len(dom)} curriculum config(s)** beat standard "
                     "t-SNE on local (NO), trust@10 AND global (spearman) "
                     "simultaneously.")
        works = True
    else:
        lines.append("- No curriculum config beats standard t-SNE on local, "
                     "trust and global at once.")
    lines.append(
        f"- Best trade-off config: **{bc.name}** — captures "
        f"{bc['local_capture']*100:.0f}% of the local headroom and "
        f"{bc['global_capture']*100:.0f}% of the global headroom that the "
        f"specialist baselines offer.")
    if bc["min_capture"] >= 0.5 and bc["local_capture"] > 0 \
            and bc["global_capture"] > 0:
        lines.append(
            "- **CONCLUSION: the curriculum idea WORKS.** A single γ schedule "
            "reaches most of the sharp specialist's local quality AND most of "
            "the smooth specialist's global quality — escaping the trade-off "
            "any single fixed γ is stuck on.")
        works = True
    elif works:
        lines.append(
            "- **CONCLUSION: partial success.** Beats standard t-SNE across "
            "scales but does not strongly capture both specialists' headroom. "
            "Worth tuning (γ values / stage split).")
    else:
        lines.append(
            "- **CONCLUSION: not convincing yet.** No config simultaneously "
            "improves on standard across scales. Try a wider sweep (stronger "
            "sharp γ, smoother global γ, longer local stage).")

    # ── Comparison against external DR methods (FAIR metrics only) ───────────
    if not ext.empty:
        lines.append("\n## vs external DR methods (FAIR metrics only)\n")
        lines.append(
            "NO@1-10 is *excluded* here — it scores preservation of the exact "
            "Euclidean kNN graph that t-SNE optimises directly, so it is not a "
            "neutral cross-family metric. trust@10 / knn@10 / spearman / "
            "triplet are method-agnostic.\n")
        fair = ["trust@10", "knn@10", "spearman", "triplet"]
        lines.append(ext[fair].round(4).to_string())
        lines.append("\nBest curriculum (**" + str(bc.name) + "**): " +
                     ", ".join(f"{c}={bc[c]:.4f}" for c in fair) + ".\n")
        for mname, r in ext.iterrows():
            wins = [c for c in fair if bc[c] >= r[c]]
            txt = ("ties/beats on " + ", ".join(wins)) if wins \
                else "loses on all fair metrics"
            lines.append(f"- vs **{mname}**: curriculum {txt}.")

    report = "\n".join(lines)
    path = os.path.join(out_dir, f"{dataset}_VERDICT.md")
    with open(path, "w") as f:
        f.write(report + "\n")
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    print(f"\nVerdict report → {path}")
    return works


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "mouse", "adult"])
    ap.add_argument("--mouse_pickle", default=None,
                    help="path to mouse pickle (required for --dataset mouse)")
    ap.add_argument("--n_samples", type=int, default=10000,
                    help="subsample size (0 = use all)")
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--master_seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)

    # evaluation knobs
    ap.add_argument("--k_max", type=int, default=200,
                    help="max neighbourhood size k for the NO / trust curves")
    ap.add_argument("--trust_subsample", type=int, default=4000,
                    help="subsample size for the O(m^2) trustworthiness curve")
    ap.add_argument("--triplet_per_point", type=int, default=5,
                    help="random triplets sampled per anchor for triplet_acc")

    # γ schedule knobs
    ap.add_argument("--gamma_smooth", type=float, default=0.5)
    ap.add_argument("--gamma_sharp", type=float, default=2.0)

    # iteration budget (shared by every model)
    ap.add_argument("--total_iter", type=int, default=1000)
    ap.add_argument("--ee_iter", type=int, default=250,
                    help="early-exaggeration iterations (== global stage length)")
    ap.add_argument("--ee", type=float, default=12.0,
                    help="early-exaggeration factor")

    # external DR baselines
    ap.add_argument("--methods", default="all",
                    help="external baselines: 'all' or comma-separated subset "
                         "of pacmap,umap,trimap,vae")
    ap.add_argument("--no_external", action="store_true",
                    help="skip external baselines, run only γ models")
    ap.add_argument("--vae_epochs", type=int, default=150,
                    help="training epochs for the VAE baseline")

    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any cached results.csv and recompute")
    ap.add_argument("--plot_only", action="store_true",
                    help="skip t-SNE, just (re)plot + verdict from results.csv")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test: small subsample, 1 seed, fewer iters")
    args = ap.parse_args()

    if args.quick:
        args.n_samples = min(args.n_samples, 3000) if args.n_samples else 3000
        args.n_seeds = 1
        args.total_iter = 400
        args.ee_iter = 100
        args.trust_subsample = min(args.trust_subsample, 1500)

    out_dir = args.out_dir or os.path.join(
        _SCRIPT_DIR, "results", f"{args.dataset}_curriculum")
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        csv_path = os.path.join(out_dir, "results.csv")
        curve_path = os.path.join(out_dir, "curves.csv")
        if not (os.path.exists(csv_path) and os.path.exists(curve_path)):
            raise SystemExit(
                f"Need both results.csv and curves.csv in {out_dir}; "
                "run without --plot_only first.")
        df = pd.read_csv(csv_path)
        cdf = pd.read_csv(curve_path)
    else:
        df, cdf = run_all(args, out_dir)

    scal_sum = summarize_scalars(df)
    curve_sum = summarize_curves(cdf)
    scal_sum.to_csv(os.path.join(out_dir, "summary_scalars.csv"), index=False)
    curve_sum.to_csv(os.path.join(out_dir, "summary_curves.csv"), index=False)
    make_plots(scal_sum, curve_sum, out_dir, args.dataset)
    verdict(scal_sum, curve_sum, out_dir, args.dataset)


if __name__ == "__main__":
    main()
