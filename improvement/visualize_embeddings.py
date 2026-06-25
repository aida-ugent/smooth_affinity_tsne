#!/usr/bin/env python
"""
visualize_embeddings.py  —  sanity-check / paper-style embedding grid
=====================================================================

Concern: PaCMAP / UMAP / TriMap score low on ``global_spearman`` in exp.py, and
we want to verify the methods themselves are running correctly (not a setup
bug) before trusting that number.

This renders the canonical "compare DR methods on a labelled dataset" figure
(the same kind every DR paper shows): a grid of 2-D embeddings coloured by
class, on the EXACT same PCA-50 input / subsample / seed that exp.py evaluates.
If PaCMAP/UMAP/TriMap produce clean, well-separated clusters here, they are
working — and their low global_spearman is a metric-choice artifact (they don't
optimise pairwise-distance preservation), not a pipeline error.

Each panel's title also shows the three scalar metrics (knn@10 / spearman /
triplet) so the picture and the numbers sit side by side.

Usage
-----
  python visualize_embeddings.py --dataset mnist --n_samples 10000
  python visualize_embeddings.py --dataset mnist --quick
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import exp          # reuse loaders, kNN/affinity machinery, metrics
import baselines    # external DR wrappers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "mouse", "adult"])
    ap.add_argument("--mouse_pickle", default=None)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--vae_epochs", type=int, default=150)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_samples = min(args.n_samples, 3000) if args.n_samples else 3000
        args.vae_epochs = 40

    out_dir = args.out_dir or os.path.join(
        exp._SCRIPT_DIR, "results", f"{args.dataset}_embeddings")
    os.makedirs(out_dir, exist_ok=True)

    # Same data exp.py evaluates (subsample uses master_seed=42 there).
    X, y = exp.load_dataset(args.dataset, args.n_samples, 42, args)
    print(f"data {X.shape}")
    k_values = np.arange(1, 91)
    hd_nbrs = exp._knn_indices(X, k=int(k_values.max()))

    # γ-models share one kNN graph + PCA init.
    knn_idx, knn_dist, eff_perp = exp.build_knn(X, args.perplexity,
                                                n_jobs=args.n_jobs)
    init = exp.initialization.pca(X, random_state=args.seed)

    def tsne_standard():
        P = exp.joint_P_for_gamma(knn_idx, knn_dist, eff_perp, 1.0,
                                  n_jobs=args.n_jobs)
        return exp.run_single(P, init.copy(), 250, 750, 12,
                              n_jobs=args.n_jobs, random_state=args.seed)

    def tsne_frontheavy():
        spec = exp.make_models(1000, 250, 12, 0.5, 2.0)[
            "sched_glob0.5_sharp2_frontheavy"]
        return exp.run_schedule(spec["stages"], init.copy(), knn_idx,
                                  knn_dist, eff_perp, n_jobs=args.n_jobs,
                                  random_state=args.seed)

    avail = baselines.available_methods()
    panels = [
        ("t-SNE (γ=1)", tsne_standard),
        ("schedule front-heavy", tsne_frontheavy),
    ]
    for name in ("pacmap", "umap", "trimap", "vae"):
        if name not in avail:
            continue
        fn = avail[name]
        if name == "vae":
            panels.append((name, lambda fn=fn: fn(
                X, random_state=args.seed, epochs=args.vae_epochs,
                n_jobs=args.n_jobs)))
        else:
            panels.append((name, lambda fn=fn: fn(
                X, random_state=args.seed, n_jobs=args.n_jobs)))

    # colour by class label
    yi = np.asarray(y)
    classes = np.unique(yi)
    try:
        order = np.argsort(classes.astype(float))
        classes = classes[order]
    except ValueError:
        pass
    cmap = plt.cm.tab10 if len(classes) <= 10 else plt.cm.tab20
    cidx = {c: i for i, c in enumerate(classes)}
    colors = np.array([cmap(cidx[c] % cmap.N) for c in yi])

    n = len(panels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, fn) in zip(axes, panels):
        print(f"running {name} ...", flush=True)
        Y = np.asarray(fn())
        # three scalar metrics for the title (same definitions as exp.py)
        ld10 = exp._knn_indices(Y, k=10, n_jobs=args.n_jobs)
        knn = exp.label_knn_accuracy(ld10, yi, 10)
        rho = exp.global_spearman(X, Y, random_state=42)
        trip = exp.triplet_accuracy(X, Y, random_state=42)
        ax.scatter(Y[:, 0], Y[:, 1], c=colors, s=3, alpha=0.6,
                   edgecolors="none", rasterized=True)
        ax.set_title(f"{name}\nknn@10={knn:.3f}  ρ={rho:.3f}  trip={trip:.3f}",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(f"{args.dataset} — DR method embeddings (coloured by class)",
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{args.dataset}_embeddings_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGrid → {path}")


if __name__ == "__main__":
    main()
