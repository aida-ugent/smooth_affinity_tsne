#!/usr/bin/env python
"""
no_k_landscape_common.py
========================
Shared helpers for the NO@k optimization-landscape experiments
(``no_k_gamma_sweep_1d.py`` and ``no_k_gamma_rho_2d.py``).

These experiments characterise the shape of the NO@k (= QNX(k)) objective as a
function of the smoothing parameter ``gamma`` (and the perplexity ``rho``) so we
can later optimise gamma for a *fixed target k* with a derivative-free method.

Everything here deliberately reuses the existing repo conventions:

* MNIST loading / PCA-50 ........ ``data/load_data.py::load_mnist_data``
* gamma smoothing ............... ``openTSNE.affinity.joint_probabilities_nn``
  (the *same* module-level function ``set_perplexity`` calls; we vary gamma
  instead of perplexity — see openTSNE/affinity.py:275-282).
* t-SNE optimisation from a precomputed joint P .... mirrors
  ``smooth_tsne_opentsne_gamma.py::run_tsne_from_joint`` (PCA init + two-phase
  early-exaggeration schedule).
* NO@k set-intersection metric .... same definition as
  ``smooth_tsne_opentsne_gamma.py::_nh_curve_from_nbrs`` (here vectorised so it
  can be evaluated for many embeddings cheaply).

Key efficiency fact exploited by the drivers:
    P depends only on (rho, gamma) — NOT on the seed.
    The neighbor graph depends only on rho.
    The HD k-NN depends on neither.
So the drivers compute the HD k-NN once, the neighbor graph once per rho, the
joint P once per (rho, gamma), and only repeat the (cheap-init) optimisation per
seed.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Make the sibling ``data/`` loaders importable exactly like the other drivers.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
sys.path.insert(0, _DATA_DIR)
from load_data import (load_mnist_data, load_mouse_data, load_adult_data,   # noqa: E402
                       load_coil20_data, load_fashion_mnist_data,
                       load_swiss_roll_data, load_pbmc3k_data)

DATASETS = ["mnist", "fashion_mnist", "mouse", "adult", "coil20",
            "swiss_roll", "pbmc3k"]


# =============================================================================
# Paper-faithful t-SNE settings (logged into every run's settings.md)
# =============================================================================
TSNE_SETTINGS = {
    "init": "pca",                 # openTSNE default; per-seed jitter -> variance
    "early_exaggeration": 12.0,
    "early_exaggeration_iter": 250,
    "n_iter": 750,
    "ee_momentum": 0.5,
    "main_momentum": 0.8,
    "metric": "euclidean",
    "pca_dims": 50,
}


# =============================================================================
# Data
# =============================================================================
def load_dataset(name="mnist", random_state=42, n_subsample=None):
    """Load any supported dataset as ``(X_pca, y)``, reusing the repo's loaders
    in ``data/load_data.py`` exactly as driver A does (same preprocessing /
    PCA-50). ``n_subsample`` returns a deterministic random subset.

    Supported: mnist, fashion_mnist, mouse, adult, coil20, swiss_roll, pbmc3k.
    """
    npca = TSNE_SETTINGS["pca_dims"]
    if name == "mnist":
        X, y, _ = load_mnist_data(n_pca=npca, random_state=random_state)
    elif name == "fashion_mnist":
        X, y, _ = load_fashion_mnist_data(n_pca=npca, random_state=random_state)
    elif name == "coil20":
        X, y, _ = load_coil20_data(n_pca=npca, random_state=random_state)
    elif name == "swiss_roll":
        X, y, _ = load_swiss_roll_data(random_state=random_state)
    elif name == "adult":
        X, y = load_adult_data(random_state=random_state)
    elif name == "mouse":
        pkl = os.path.join(_DATA_DIR, "tasic2018.pickle")
        X, _, y, _, _ = load_mouse_data(pkl, data_dir=_DATA_DIR,
                                        return_highdim=True)
    elif name == "pbmc3k":
        X, y = load_pbmc3k_data(
            os.path.join(_DATA_DIR, "pbmc3k_processed.h5ad"), n_pca=npca)
    else:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if n_subsample is not None and n_subsample < len(X):
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(len(X), size=n_subsample, replace=False))
        X, y = X[idx], y[idx]
    return X, y


def load_mnist_pca50(random_state=42, n_subsample=None):
    """Backwards-compatible MNIST loader (delegates to :func:`load_dataset`)."""
    return load_dataset("mnist", random_state=random_state,
                        n_subsample=n_subsample)


# =============================================================================
# k-NN helpers
# =============================================================================
def compute_hd_knn(X, k_max, n_jobs=-1):
    """High-dimensional k-NN indices on the 50-PC data (self excluded).

    Returns an ``(n, k_max)`` int array. This is independent of gamma and rho,
    so compute it once for the largest k you need and slice for smaller k.
    Mirrors ``smooth_tsne_opentsne_gamma.py::_knn_indices``.
    """
    nn = NearestNeighbors(n_neighbors=k_max + 1, metric="euclidean",
                          n_jobs=n_jobs)
    nn.fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def compute_ld_knn(Y, k_max, n_jobs=-1):
    """Low-dimensional k-NN indices on a 2-D embedding (self excluded).

    Compute once at ``k_max`` per embedding; NO@k for every smaller target k is
    derived from this single list via :func:`no_at_k`.
    """
    return compute_hd_knn(np.asarray(Y), k_max, n_jobs=n_jobs)


# =============================================================================
# NO@k = QNX(k)
# =============================================================================
def _intersection_counts(a, b, chunk=2000):
    """Per-row |set(a[i]) & set(b[i])| for two equal-shaped int arrays.

    Vectorised + row-chunked to bound memory. Neighbor indices within a row are
    unique, so counting how many of ``a``'s entries appear anywhere in ``b`` is
    exactly the intersection size.
    """
    n, k = a.shape
    counts = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        sa = a[s:s + chunk]
        sb = b[s:s + chunk]
        # (m, k, k): does each a-neighbor equal any b-neighbor in the same row?
        match = (sa[:, :, None] == sb[:, None, :]).any(axis=2)
        counts[s:s + chunk] = match.sum(axis=1)
    return counts


def no_at_k(hd_nbrs, ld_nbrs, k, chunk=2000):
    """NO@k (= QNX(k)): mean over points of |HD@k ∩ LD@k| / k.

    ``hd_nbrs``/``ld_nbrs`` are ``(n, k_max)`` index arrays with k_max >= k.
    Same definition as ``_nh_curve_from_nbrs``, vectorised.
    """
    counts = _intersection_counts(hd_nbrs[:, :k], ld_nbrs[:, :k], chunk=chunk)
    return float(counts.mean() / k)


def no_at_ks(hd_nbrs, ld_nbrs, k_list, chunk=2000):
    """NO@k for every k in ``k_list`` from a single shared pair of k-NN lists."""
    return {int(k): no_at_k(hd_nbrs, ld_nbrs, int(k), chunk=chunk)
            for k in k_list}


def no_full_curve(hd_nbrs, ld_nbrs, chunk=1000):
    """NO@k for *every* k in 1..k_max in a single O(n·k_max) pass.

    A neighbor j contributes to NO@k iff it is in both the HD and LD top-k, i.e.
    for all k >= max(hd_rank(j), ld_rank(j)). So we histogram those thresholds
    once and cumulative-sum to get the whole curve — far cheaper than evaluating
    the set intersection separately for each k (used for the AUC summaries and
    the per-target-k surfaces in Experiment 2).

    Returns a float array ``curve`` of length k_max where ``curve[k-1] == NO@k``.
    """
    n, kmax = hd_nbrs.shape
    ranks = np.arange(1, kmax + 1)
    hist = np.zeros(kmax + 1, dtype=np.int64)   # hist[t] = #pairs with threshold t
    for s in range(0, n, chunk):
        a = hd_nbrs[s:s + chunk]                # (m, kmax) HD ids, col j -> rank j+1
        b = ld_nbrs[s:s + chunk]                # (m, kmax) LD ids
        m = a.shape[0]
        eq = (a[:, :, None] == b[:, None, :])   # (m, kmax_hd, kmax_ld)
        common_mask = eq.any(axis=2)            # HD neighbor also present in LD
        ld_pos = (eq * ranks[None, None, :]).max(axis=2)        # 1-based LD rank
        hd_pos = np.broadcast_to(ranks[None, :], (m, kmax))     # 1-based HD rank
        t = np.maximum(hd_pos, ld_pos)[common_mask]             # thresholds
        if t.size:
            np.add.at(hist, t, 1)
    cum = np.cumsum(hist)                        # cum[k] = total intersections @k
    return cum[1:kmax + 1] / (n * ranks)


def no_curve(hd_nbrs, ld_nbrs, k_values, chunk=2000):
    """NO@k over an explicit list of k (kept for parity with the repo's API)."""
    return np.array([no_at_k(hd_nbrs, ld_nbrs, int(k), chunk=chunk)
                     for k in k_values])


def auc_range(curve, k_values, k_lo, k_hi):
    """Mean NO@k over the inclusive k-window [k_lo, k_hi] (paper Fig. 6 AUC).

    Mirrors ``smooth_tsne_opentsne_gamma.py::_nh_auc``.
    """
    k_values = np.asarray(k_values)
    mask = (k_values >= k_lo) & (k_values <= k_hi)
    if not mask.any():
        return float("nan")
    return float(np.asarray(curve)[mask].mean())


# =============================================================================
# Affinity caching + t-SNE from a precomputed P
# =============================================================================
def build_neighbor_cache(X_pca, rho, n_jobs=-1, knn_seed=42, verbose=False):
    """Build the perplexity-``rho`` neighbor graph once and cache it.

    Returns ``(neighbors, distances, eff_perplexity)``. The neighbor graph (and
    hence these arrays) is independent of gamma, so the caller reuses it across
    every gamma. ``knn_seed`` is fixed so the (approximate) graph is
    reproducible and seed-independent — variance across the 5 run seeds comes
    from the PCA-init jitter, not from the k-NN search.
    """
    from openTSNE.affinity import PerplexityBasedNN
    aff = PerplexityBasedNN(X_pca, perplexity=rho, gamma=1.0,
                            n_jobs=n_jobs, random_state=knn_seed,
                            verbose=verbose)
    return (np.asarray(aff.knn_indices),
            np.asarray(aff.knn_distances),
            float(aff.effective_perplexity_))


def joint_P_for_gamma(neighbors, distances, eff_perplexity, gamma, n_jobs=-1):
    """Symmetrised joint P for a given gamma, reusing a cached neighbor graph.

    Thin wrapper over the repo's own ``joint_probabilities_nn`` — this is the
    *same* call ``PerplexityBasedNN.set_perplexity`` makes (affinity.py:275-282),
    so the gamma power-transform is the library's, not reimplemented here. The
    result is byte-equivalent to ``PerplexityBasedNN(perplexity=rho,
    gamma=gamma).P`` (verified by the self-test in the drivers).
    """
    from openTSNE.affinity import joint_probabilities_nn
    P, _ = joint_probabilities_nn(
        neighbors,
        distances,
        [eff_perplexity],
        symmetrize=True,
        n_jobs=n_jobs,
        gamma=gamma,
    )
    return P


def embed_from_P(X_pca, P_joint, seed, n_jobs=-1,
                 ee=None, ee_iter=None, n_iter=None):
    """Run t-SNE from a precomputed joint P with paper-faithful settings.

    Mirrors ``smooth_tsne_opentsne_gamma.py::run_tsne_from_joint``: PCA init
    (seeded by ``seed`` -> per-run jitter) followed by the two-phase
    early-exaggeration schedule. Returns the 2-D embedding as an ndarray.
    """
    from openTSNE import TSNEEmbedding, initialization

    if ee is None:
        ee = TSNE_SETTINGS["early_exaggeration"]
    if ee_iter is None:
        ee_iter = TSNE_SETTINGS["early_exaggeration_iter"]
    if n_iter is None:
        n_iter = TSNE_SETTINGS["n_iter"]

    class _FixedAff:
        def __init__(self, P):
            self.P = P

    init = initialization.pca(X_pca, random_state=seed)
    emb = TSNEEmbedding(init, _FixedAff(P_joint), n_jobs=n_jobs)
    emb.optimize(ee_iter, exaggeration=ee,
                 momentum=TSNE_SETTINGS["ee_momentum"], inplace=True)
    emb.optimize(n_iter, exaggeration=1.0,
                 momentum=TSNE_SETTINGS["main_momentum"], inplace=True)
    return np.array(emb)


# =============================================================================
# Figures
# =============================================================================
def save_fig(path, dpi=150):
    """Save the current matplotlib figure (PNG, tight bbox) like the repo does."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


# =============================================================================
# Self-test: prove the cached-P path matches a fresh PerplexityBasedNN(gamma)
# =============================================================================
def run_self_test(n=800, rho=30.0, gammas=(0.0, 0.5, 1.0, 1.7), n_jobs=1):
    """Verify (a) cached-P == fresh-affinity-P for each gamma, and
    (b) the vectorised NO@k matches a brute-force set-intersection.

    Uses a small random matrix so it runs in a second or two. Returns True on
    success and raises AssertionError otherwise.
    """
    from openTSNE.affinity import PerplexityBasedNN

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 50))

    neighbors, distances, eff = build_neighbor_cache(
        X, rho, n_jobs=n_jobs, knn_seed=42)

    for g in gammas:
        P_cached = joint_P_for_gamma(neighbors, distances, eff, g, n_jobs=n_jobs)
        aff = PerplexityBasedNN(X, perplexity=rho, gamma=g,
                                n_jobs=n_jobs, random_state=42)
        diff = abs(P_cached - aff.P)
        max_abs = diff.max() if diff.nnz else 0.0
        assert max_abs < 1e-10, (
            f"cached P != fresh affinity P at gamma={g} (max abs diff {max_abs})")

    # NO@k cross-check vs brute-force set intersection on random index arrays.
    k_max = 20
    a = np.argsort(rng.standard_normal((200, 60)), axis=1)[:, :k_max]
    b = np.argsort(rng.standard_normal((200, 60)), axis=1)[:, :k_max]
    curve = no_full_curve(a, b)
    for k in (5, 10, 20):
        fast = no_at_k(a, b, k)
        brute = np.mean([len(set(a[i, :k]) & set(b[i, :k])) / k
                         for i in range(len(a))])
        assert abs(fast - brute) < 1e-12, (
            f"vectorised NO@{k}={fast} != brute {brute}")
        assert abs(curve[k - 1] - brute) < 1e-12, (
            f"full-curve NO@{k}={curve[k - 1]} != brute {brute}")

    print("Self-test passed: cached-P matches fresh affinity for all gammas, "
          "and vectorised NO@k matches brute force.")
    return True


if __name__ == "__main__":
    run_self_test()
