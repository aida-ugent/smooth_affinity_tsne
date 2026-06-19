"""
compare_affinity_variants.py

Compare 8 affinity configurations for t-SNE using
Neighborhood Overlap @ k (NH@k) as the quality metric.

NH@k measures how well the local HD structure is preserved in the embedding:
for each point, it counts the fraction of its k HD neighbors that also appear
among its k LD neighbors, then averages over all points.

Methods compared
----------------
1. γ=1.0        PerplexityBasedNN(perplexity=30, gamma=1.0)   — standard t-SNE
2. γ=0.7        PerplexityBasedNN(perplexity=30, gamma=0.7)   — smooth t-SNE
3. γ=0.0        PerplexityBasedNN(perplexity=30, gamma=0.0)   — extreme smoothing,
                uniform weight over perplexity-determined neighbors
4. γ=1.5        PerplexityBasedNN(perplexity=30, gamma=1.5)   — sharp t-SNE
5. FixedSigmaNN FixedSigmaNN(sigma=sigma_med, k=3*perplexity) — fixed bandwidth
6. MultiscaleMix MultiscaleMixture(perplexities=[30,50,100,200])
7. Multiscale    Multiscale(perplexities=[30,50,100,200])
8. Uniform       Uniform(k_neighbors=perplexity)              — binary kNN kernel
9. tt-SNE        StudentTNN(perplexity=30, dof='auto')        — twice Student tt-SNE:
                 heavy-tailed Student-t kernel in the HD space (de Bodt et al.,
                 ESANN 2018), dof = estimated intrinsic dimensionality M'

Usage
-----
    python compare_affinity_variants.py --dataset mnist --n_runs 1 --out_dir results/aff_comparison/mnist
    python compare_affinity_variants.py --dataset mouse --out_dir results/aff_comparison/mouse
    python compare_affinity_variants.py --dataset adult --out_dir results/aff_comparison/adult
    python compare_affinity_variants.py --dataset mnist --out_dir results/aff_comparison/mnist --plot_only
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

plt.rcParams.update({
    "axes.titlesize":        17,
    "axes.labelsize":        15,
    "xtick.labelsize":       12,
    "ytick.labelsize":       12,
    "legend.fontsize":       13,
    "legend.title_fontsize": 13,
})

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "data"))


# =============================================================================
# Plotting style  — defined at module level so plot_only can use it
# =============================================================================

# Each entry: (legend_label, color, linestyle, linewidth)
#
# Line style logic:
#   γ=1.0 (standard)  — solid:     the reference baseline
#   γ variants        — dashed:    same family, not the reference
#   Multiscale group  — dotted:    different kernel family
#   Uniform           — dash-dot:  binary kernel baseline
#   FixedSigmaNN      — long-dash: fixed-bandwidth baseline
#   tt-SNE            — dense dash-dot-dot: Student-t HD kernel (own family)
_METHOD_STYLE = {
    "γ=1.0":         ("γ=1.0",         "#000000", "-",                2.4),
    "γ=0.7":         ("γ=0.7",         "#009E73", "--",               2.2),
    "γ=0.0":         ("γ=0.0",         "#D55E00", "--",               2.0),
    "γ=1.5":         ("γ=1.5",         "#CC79A7", "--",               2.0),
    "MultiscaleMix": ("MultiscaleMix", "#0072B2", ":",                2.2),
    "Multiscale":    ("Multiscale",    "#56B4E9", ":",                2.2),
    "Uniform":       ("Uniform",       "#9B59B6", "-.",               2.0),
    "FixedSigmaNN":  ("FixedSigmaNN",  "#E69F00", (0, (5, 1)),        2.0),
    "tt-SNE":        ("tt-SNE",        "#C0392B", (0, (3, 1, 1, 1, 1, 1)), 2.4),
}

# Canonical order for the legend
_METHOD_ORDER = [
    "γ=1.0", "γ=0.7", "γ=0.0", "γ=1.5",
    "MultiscaleMix", "Multiscale",
    "Uniform", "FixedSigmaNN", "tt-SNE",
]


# =============================================================================
# Data loading
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_mnist(subsample, random_state):
    from load_data import load_mnist_data
    X_pca, y, _ = load_mnist_data(n_pca=50, random_state=random_state)
    if subsample and subsample < len(X_pca):
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(len(X_pca), size=subsample, replace=False))
        X_pca, y = X_pca[idx], y[idx]
    return X_pca, y


def load_mouse(pickle_path, data_dir, subsample, random_state):
    from load_data import load_mouse_data
    X_pca, y, _, _ = load_mouse_data(
        pickle_path, data_dir=data_dir, return_highdim=False,
    )
    if subsample and subsample < len(X_pca):
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(len(X_pca), size=subsample, replace=False))
        X_pca, y = X_pca[idx], y[idx]
    return X_pca, y


def load_adult(max_rows, random_state):
    from load_data import load_adult_data
    X, y = load_adult_data(max_rows=max_rows, random_state=random_state)
    return X, y


def load_dataset(args):
    """Return (X_pca, y) for the requested dataset."""
    if args.dataset == "mnist":
        print(f"\nLoading MNIST (subsample={args.mnist_subsample or 'full'}) ...")
        return load_mnist(args.mnist_subsample, args.random_state)
    elif args.dataset == "mouse":
        pkl = args.mouse_pickle or os.path.join(_SCRIPT_DIR, "data", "tasic2018.pickle")
        data_dir = args.mouse_data_dir or os.path.join(_SCRIPT_DIR, "data")
        print(f"\nLoading mouse cortex from {pkl} ...")
        return load_mouse(pkl, data_dir, args.mouse_subsample, args.random_state)
    elif args.dataset == "adult":
        print(f"\nLoading Adult dataset (max_rows={args.adult_max_rows}) ...")
        return load_adult(args.adult_max_rows, args.random_state)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset!r}")


# =============================================================================
# Sigma calibration for FixedSigmaNN
# =============================================================================

def calibrate_sigma(X_pca, perplexity):
    """
    Median distance to the perplexity-th nearest neighbor — puts the Gaussian
    bandwidth in the same ballpark as PerplexityBasedNN's adaptive sigma.
    """
    nn = NearestNeighbors(n_neighbors=perplexity, algorithm="ball_tree").fit(X_pca)
    dists, _ = nn.kneighbors(X_pca)
    return float(np.median(dists[:, perplexity - 1]))


# =============================================================================
# Affinity builders
# =============================================================================

def make_affinity_builders(X_pca, perplexity, gamma_s, gamma_h,
                           multiscale_perps, ttsne_dof, n_jobs, random_state):
    """
    Return an ordered dict ``{label: thunk}`` where each thunk is a zero-argument
    callable that builds and returns that method's affinity object on demand.

    Building is deferred (rather than constructing every affinity up front) so
    that, when resuming, only the methods that still have pending (method, seed)
    work pay the cost of building their affinity matrix and kNN graph.  All
    methods use the same X_pca and perplexity so comparisons stay controlled,
    and affinities are deterministic across seeds, so each thunk is called at
    most once and the result is cached by the caller.
    """
    from openTSNE.affinity import (
        PerplexityBasedNN, FixedSigmaNN, MultiscaleMixture, Multiscale, Uniform,
        StudentTNN,
    )

    k_fixed   = min(len(X_pca) - 1, 3 * perplexity)   # match PerplexityBasedNN exactly
    k_uniform = min(len(X_pca) - 1, 3 * perplexity)

    def build_standard():
        print("  γ=1.0  (standard) ...")
        return PerplexityBasedNN(
            X_pca, perplexity=perplexity, gamma=1.0,
            n_jobs=n_jobs, random_state=random_state,
        )

    def build_smooth():
        print(f"  γ={gamma_s}  (smooth) ...")
        return PerplexityBasedNN(
            X_pca, perplexity=perplexity, gamma=gamma_s,
            n_jobs=n_jobs, random_state=random_state,
        )

    def build_uniform_weight():
        print("  γ=0.0  (uniform-weight over perplexity neighbours) ...")
        return PerplexityBasedNN(
            X_pca, perplexity=perplexity, gamma=0.0,
            n_jobs=n_jobs, random_state=random_state,
        )

    def build_sharp():
        print(f"  γ={gamma_h}  (sharp) ...")
        return PerplexityBasedNN(
            X_pca, perplexity=perplexity, gamma=gamma_h,
            n_jobs=n_jobs, random_state=random_state,
        )

    def build_fixedsigma():
        sigma = calibrate_sigma(X_pca, perplexity)
        print(f"  FixedSigmaNN  (sigma={sigma:.4f}, k={k_fixed}) ...")
        return FixedSigmaNN(
            X_pca, sigma=sigma, k=k_fixed, gamma=1.0,
            n_jobs=n_jobs, random_state=random_state,
        )

    def build_msmix():
        print(f"  MultiscaleMix  (perplexities={multiscale_perps}) ...")
        return MultiscaleMixture(
            X_pca, perplexities=multiscale_perps,
            n_jobs=n_jobs, random_state=random_state,
        )

    def build_ms():
        print(f"  Multiscale  (perplexities={multiscale_perps}) ...")
        return Multiscale(
            X_pca, perplexities=multiscale_perps,
            n_jobs=n_jobs, random_state=random_state,
        )

    def build_uniform():
        print(f"  Uniform  (k_neighbors={k_uniform}, symmetrize=mean) ...")
        return Uniform(
            X_pca, k_neighbors=k_uniform, symmetrize="mean",
            n_jobs=n_jobs, random_state=random_state,
        )

    def build_ttsne():
        print(f"  tt-SNE  (twice Student, dof={ttsne_dof}) ...")
        aff = StudentTNN(
            X_pca, perplexity=perplexity, dof=ttsne_dof,
            n_jobs=n_jobs, random_state=random_state,
        )
        print(f"    tt-SNE dof (intrinsic dim M') = {aff.dof:.3f}")
        return aff

    # Keep insertion order aligned with _METHOD_ORDER.
    return {
        "γ=1.0":         build_standard,
        "γ=0.7":         build_smooth,
        "γ=0.0":         build_uniform_weight,
        "γ=1.5":         build_sharp,
        "FixedSigmaNN":  build_fixedsigma,
        "MultiscaleMix": build_msmix,
        "Multiscale":    build_ms,
        "Uniform":       build_uniform,
        "tt-SNE":        build_ttsne,
    }


# =============================================================================
# t-SNE runner
# =============================================================================

def run_tsne(aff, X_pca, random_state, n_jobs, n_iter_ee, n_iter_main, ee):
    """Run t-SNE with a precomputed affinity object."""
    from openTSNE import TSNE
    embedding = TSNE(
        n_jobs=n_jobs,
        random_state=random_state,
        early_exaggeration=ee,
        early_exaggeration_iter=n_iter_ee,
        n_iter=n_iter_main,
    ).fit(X_pca, affinities=aff)
    return np.array(embedding)


# =============================================================================
# Neighborhood Overlap — full curve k = 1 … k_max
# =============================================================================

def knn_indices(X, k, n_jobs=-1):
    """Return (n, k) array of kNN indices, self excluded."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean",
                          n_jobs=n_jobs, algorithm="ball_tree")
    nn.fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def nh_curve(hd_nbrs, ld_nbrs, k_max):
    """
    Compute NH@k for every k from 1 to k_max given pre-built index arrays.

    Uses the same set-intersection approach as exp4 in the main script.
    hd_nbrs, ld_nbrs : (n, k_max) int arrays
    Returns           : (k_max,) float array, result[k-1] = mean NH@k
    """
    n = len(hd_nbrs)
    return np.array([
        np.mean([
            len(set(hd_nbrs[i, :k]) & set(ld_nbrs[i, :k])) / k
            for i in range(n)
        ])
        for k in range(1, k_max + 1)
    ])


# =============================================================================
# Plotting
# =============================================================================

def plot_nh_curves(df, out_dir, k_max=None, dataset="mnist"):
    """
    One line per method.
    - If only 1 seed: just plot the mean line (no band).
    - If >1 seed: thin per-seed lines (alpha=0.2) + bold mean, same as exp4.
    Colors and line styles come from _METHOD_STYLE.
    """
    seed_list = sorted(df["seed"].unique())
    n_seeds   = len(seed_list)
    k_vals    = np.sort(df["k"].unique())
    if k_max is not None:
        k_vals = k_vals[k_vals <= k_max]

    # Methods present in data, in canonical order
    present = [m for m in _METHOD_ORDER if m in df["method"].unique()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("white")

    for method in present:
        _, color, ls, lw = _METHOD_STYLE[method]
        sub = df[df["method"] == method]

        if n_seeds > 1:
            for seed in seed_list:
                s = sub[sub["seed"] == seed].sort_values("k")
                s = s[s["k"].isin(k_vals)]
                ax.plot(s["k"].values, s["nh_score"].values,
                        color=color, lw=0.6, ls=ls, alpha=0.2)

        mn = sub.groupby("k")["nh_score"].mean().reindex(k_vals)
        ax.plot(k_vals, mn.values, color=color, lw=lw, ls=ls, label=method)

    ax.set_xlabel("k  (neighbourhood size)")
    ax.set_ylabel("NH@k  (Neighbourhood Overlap)")
    _prefix = "(b) " if dataset == "mnist" else ""
    ax.set_title(f"{_prefix}Affinity variants — NH@k on {dataset.upper()}")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_nh_affinity_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Figure saved → {path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare affinity variants via NH@k",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--dataset",          choices=["mnist", "mouse", "adult"],
                   default="mnist")
    # MNIST options
    p.add_argument("--mnist_subsample",  type=int,   default=0,
                   help="Subsample size for MNIST (0 = full 70k)")
    # Mouse options
    p.add_argument("--mouse_pickle",     default=None,
                   help="Path to tasic2018.pickle (default: data/tasic2018.pickle)")
    p.add_argument("--mouse_data_dir",   default=None,
                   help="Dir containing importantGenesTasic2018.npy")
    p.add_argument("--mouse_subsample",  type=int,   default=0,
                   help="Subsample size for mouse (0 = all ~3000)")
    # Adult options
    p.add_argument("--adult_max_rows",   type=int,   default=10000,
                   help="Max rows for the Adult dataset")
    # Shared affinity / t-SNE options
    p.add_argument("--perplexity",       type=int,   default=30)
    p.add_argument("--gamma_s",          type=float, default=0.7,
                   help="gamma for the Smooth method")
    p.add_argument("--gamma_h",          type=float, default=1.5,
                   help="gamma for the Sharp method")
    p.add_argument("--multiscale_perps", type=int,   nargs="+",
                   default=[30, 50, 100, 200])
    p.add_argument("--ttsne_dof",        default="auto",
                   help="Degrees of freedom (intrinsic dim M') for the tt-SNE "
                        "Student-t HD kernel. 'auto' estimates it from the data.")
    p.add_argument("--k_max",            type=int,   default=200,
                   help="Compute NH@k for k = 1 … k_max")
    p.add_argument("--n_runs",           type=int,   default=1,
                   help="Number of random seeds")
    p.add_argument("--random_state",     type=int,   default=42)
    p.add_argument("--n_iter_ee",        type=int,   default=250)
    p.add_argument("--n_iter_main",      type=int,   default=750)
    p.add_argument("--ee",               type=float, default=12.0)
    p.add_argument("--n_jobs",           type=int,   default=-1)
    p.add_argument("--out_dir",          default=None,
                   help="Output directory (default: results/aff_comparison/<dataset>)")
    p.add_argument("--plot_only",        action="store_true",
                   help="Skip computation; regenerate figures from saved CSV")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = args.out_dir or os.path.join("results", "aff_comparison", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "nh_scores.csv")

    # ── Plot-only ─────────────────────────────────────────────────────────────
    if args.plot_only:
        df = pd.read_csv(csv_path)
        plot_nh_curves(df, out_dir, dataset=args.dataset)
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    X_pca, y = load_dataset(args)
    print(f"  X_pca: {X_pca.shape}")
    X_eval = X_pca   # evaluate NH@k in PCA space (the actual t-SNE input)

    # ── Seeds ─────────────────────────────────────────────────────────────────
    rng    = np.random.default_rng(args.random_state)
    seeds  = [int(s) for s in rng.integers(0, 2**31, size=args.n_runs)]
    print(f"Seeds: {seeds}")

    # ── Resume ────────────────────────────────────────────────────────────────
    if os.path.exists(csv_path):
        existing    = pd.read_csv(csv_path)
        done_pairs  = set(zip(existing["method"], existing["seed"]))
        print(f"Resuming: {len(done_pairs)} (method, seed) pairs already done.")
    else:
        existing   = pd.DataFrame()
        done_pairs = set()

    # ── Affinity builders (deferred — built lazily, only when needed) ─────────
    ttsne_dof = args.ttsne_dof if args.ttsne_dof == "auto" else float(args.ttsne_dof)
    builders = make_affinity_builders(
        X_pca,
        perplexity=args.perplexity,
        gamma_s=args.gamma_s,
        gamma_h=args.gamma_h,
        multiscale_perps=args.multiscale_perps,
        ttsne_dof=ttsne_dof,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    # Which methods still have pending (method, seed) work?  Only these get built.
    pending = [m for m in builders
               if any((m, s) not in done_pairs for s in seeds)]
    if not pending:
        print("\nAll (method, seed) pairs already computed — nothing to run.")
    else:
        print(f"\nPending methods (will build affinities for these only): {pending}")

        # ── Build HD kNN once — reused for every method and seed ──────────────
        print(f"\nBuilding HD kNN (k={args.k_max}) ...")
        hd_nbrs = knn_indices(X_eval, k=args.k_max, n_jobs=args.n_jobs)

        # ── Main loop ─────────────────────────────────────────────────────────
        all_rows = []
        built = {}   # cache: affinities are deterministic across seeds

        for seed in seeds:
            print(f"\n=== Seed {seed} ===")
            for method_name, thunk in builders.items():
                if (method_name, seed) in done_pairs:
                    print(f"  {method_name}: already done, skipping.")
                    continue

                aff = built.get(method_name)
                if aff is None:
                    aff = thunk()
                    built[method_name] = aff

                print(f"  {method_name}: t-SNE ...", end=" ", flush=True)
                Y = run_tsne(
                    aff, X_pca,
                    random_state=seed,
                    n_jobs=args.n_jobs,
                    n_iter_ee=args.n_iter_ee,
                    n_iter_main=args.n_iter_main,
                    ee=args.ee,
                )

                print("NH@k ...", end=" ", flush=True)
                ld_nbrs = knn_indices(np.asarray(Y), k=args.k_max, n_jobs=args.n_jobs)
                curve   = nh_curve(hd_nbrs, ld_nbrs, args.k_max)

                for k_idx in range(args.k_max):
                    all_rows.append({
                        "method":   method_name,
                        "seed":     seed,
                        "k":        k_idx + 1,
                        "nh_score": curve[k_idx],
                    })
                print("done.")

                # Checkpoint after every (method, seed)
                new_df = pd.concat([existing, pd.DataFrame(all_rows)],
                                   ignore_index=True)
                new_df.to_csv(csv_path, index=False)
                done_pairs.add((method_name, seed))
                all_rows = []
                existing = new_df
                print(f"    → checkpoint saved")

    # ── Summary + figure ──────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    print("\n=== Mean NH@k per method at selected k values ===")
    summary = (df[df["k"].isin([10, 30, 50, 100, 200])]
                 .groupby(["method", "k"])["nh_score"]
                 .mean()
                 .unstack("k")
                 .round(4))
    print(summary.to_string())

    plot_nh_curves(df, out_dir, dataset=args.dataset)
    print("\nDone.")


if __name__ == "__main__":
    main()
