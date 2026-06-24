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
3. Evaluates each with three metrics that capture the three scales:
      - LOCAL  : neighborhood-overlap AUC over k=1..10
      - MID    : neighborhood-overlap AUC over k=11..90
      - GLOBAL : Spearman correlation of pairwise distances (HD vs LD)
4. Aggregates over seeds, writes results.csv, draws summary plots, and
   prints an automatic VERDICT on whether the curriculum idea works.

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

from openTSNE import TSNEEmbedding, initialization
from openTSNE.affinity import PerplexityBasedNN, joint_probabilities_nn

# data loaders live in ../experiments/data
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "experiments", "data")
sys.path.insert(0, _EXP_DATA_DIR)
from load_data import load_mnist_data, load_mouse_data, load_adult_data  # noqa: E402


# =============================================================================
# Metrics  (identical formulas to experiments/smooth_tsne_opentsne_gamma.py)
# =============================================================================

def _knn_indices(X, k, n_jobs=-1):
    """k-NN indices excluding self, shape (n, k)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def _nh_curve_from_nbrs(hd_nbrs, ld_nbrs, k_values):
    """Neighborhood Overlap NH@k for each k: |HD_k ∩ LD_k| / k, averaged."""
    n = len(hd_nbrs)
    return np.array([
        np.mean([len(set(hd_nbrs[i, :k]) & set(ld_nbrs[i, :k])) / k
                 for i in range(n)])
        for k in k_values
    ])


def _nh_auc(nh_curve, k_values, k_lo, k_hi):
    mask = (k_values >= k_lo) & (k_values <= k_hi)
    if not mask.any():
        return float("nan")
    return float(nh_curve[mask].mean())


def _global_spearman(X, Y, random_state=42, n_sample=5000):
    """Spearman ρ between HD and LD pairwise distances (on a subsample)."""
    n = len(X)
    if n > n_sample:
        idx = np.sort(np.random.default_rng(random_state).choice(
            n, size=n_sample, replace=False))
        X, Y = X[idx], Y[idx]
    rho, _ = spearmanr(pdist(X), pdist(Y))
    return float(rho)


def evaluate(Y, X_eval, hd_nbrs, k_values, master_seed=42):
    """Return dict of the three scale metrics for an embedding Y."""
    ld_nbrs = _knn_indices(np.asarray(Y), k=int(k_values.max()))
    nh = _nh_curve_from_nbrs(hd_nbrs, ld_nbrs, k_values)
    return {
        "local_auc_1_10":  _nh_auc(nh, k_values, 1, 10),
        "mid_auc_11_90":   _nh_auc(nh, k_values, 11, 90),
        "global_spearman": _global_spearman(X_eval, np.asarray(Y),
                                            random_state=master_seed),
    }


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
    rng_seeds = np.random.default_rng(args.master_seed).integers(
        0, 2**31, size=args.n_seeds)
    seeds = [int(s) for s in rng_seeds]

    k_values = np.arange(1, 91)
    models = make_models(args.total_iter, args.ee_iter, args.ee,
                         args.gamma_smooth, args.gamma_sharp)
    print(f"Models ({len(models)}): {list(models)}")

    print(f"\nLoading {args.dataset} (n_samples={args.n_samples}) ...")
    X_pca, y = load_dataset(args.dataset, args.n_samples, args.master_seed, args)
    X_eval = X_pca
    print(f"  data shape = {X_pca.shape}")

    print(f"Pre-computing evaluation HD-kNN (k={k_values.max()}) ...")
    hd_nbrs = _knn_indices(X_eval, k=int(k_values.max()))

    existing = (pd.read_csv(csv_path)
                if (os.path.exists(csv_path) and not args.fresh)
                else pd.DataFrame())
    done = (set(zip(existing["model"], existing["seed"]))
            if not existing.empty else set())

    rows = []
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
            else:
                Y = run_curriculum(spec["stages"], init.copy(),
                                   knn_idx, knn_dist, eff_perp,
                                   n_jobs=args.n_jobs, random_state=seed)
            m = evaluate(Y, X_eval, hd_nbrs, k_values,
                         master_seed=args.master_seed)
            dt = time.time() - t0
            row = {"model": name, "group": spec["group"], "seed": seed,
                   "time_s": round(dt, 1), **m}
            rows.append(row)
            print(f"  {name:42s} local={m['local_auc_1_10']:.4f}  "
                  f"mid={m['mid_auc_11_90']:.4f}  "
                  f"global={m['global_spearman']:.4f}  ({dt:.1f}s)")

            # checkpoint after every (model, seed)
            cur = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            cur.to_csv(csv_path, index=False)

    final = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) \
        if rows else existing
    final.to_csv(csv_path, index=False)
    print(f"\nResults → {csv_path}")
    return final


# =============================================================================
# Analysis, plots, verdict
# =============================================================================

METRICS = [
    ("local_auc_1_10",  "Local (NH AUC k=1–10)",  True),
    ("mid_auc_11_90",   "Mid (NH AUC k=11–90)",   True),
    ("global_spearman", "Global (Spearman ρ)",    True),
]


def summarize(df):
    g = df.groupby(["model", "group"], as_index=False).agg(
        local_mean=("local_auc_1_10", "mean"),
        local_std=("local_auc_1_10", "std"),
        mid_mean=("mid_auc_11_90", "mean"),
        mid_std=("mid_auc_11_90", "std"),
        global_mean=("global_spearman", "mean"),
        global_std=("global_spearman", "std"),
        n=("seed", "nunique"),
    ).fillna(0.0)
    return g


def make_plots(df, summary, out_dir, dataset):
    # 1) grouped bar chart of the three metrics per model
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    order = (summary.sort_values("group")["model"].tolist())
    s = summary.set_index("model").reindex(order)
    colors = ["#888888" if g == "baseline" else "#0072B2"
              for g in s["group"]]
    specs = [("local_mean", "local_std", "Local  NH AUC (k=1–10)"),
             ("mid_mean", "mid_std", "Mid  NH AUC (k=11–90)"),
             ("global_mean", "global_std", "Global  Spearman ρ")]
    for ax, (mcol, scol, title) in zip(axes, specs):
        ax.barh(range(len(s)), s[mcol].values, xerr=s[scol].values,
                color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"Curriculum-γ t-SNE — {dataset}  "
                 f"(grey=baseline, blue=curriculum)", fontsize=14)
    plt.tight_layout()
    p1 = os.path.join(out_dir, f"{dataset}_metrics_bars.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # 2) trade-off scatter: local vs global, sized by mid
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, r in summary.iterrows():
        is_base = r["group"] == "baseline"
        ax.scatter(r["local_mean"], r["global_mean"],
                   s=140, c=("#888888" if is_base else "#0072B2"),
                   marker=("s" if is_base else "o"),
                   edgecolors="black", zorder=3)
        ax.annotate(r["model"].replace("baseline_", "").replace("curric_", ""),
                    (r["local_mean"], r["global_mean"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Local  NH AUC (k=1–10)   →  better")
    ax.set_ylabel("Global  Spearman ρ   →  better")
    ax.set_title(f"Local–Global trade-off — {dataset}\n"
                 "top-right = escapes the trade-off (good local AND global)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p2 = os.path.join(out_dir, f"{dataset}_tradeoff_scatter.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots → {p1}\n        {p2}")


def verdict(summary, out_dir, dataset):
    """Decide whether the curriculum idea works and write a markdown report."""
    base = summary[summary["group"] == "baseline"].set_index("model")
    curr = summary[summary["group"] == "curriculum"]

    std = base.loc["baseline_standard"]
    # best baseline value per metric (the specialist each scale is best at)
    best_local_base = base["local_mean"].max()
    best_mid_base = base["mid_mean"].max()
    best_global_base = base["global_mean"].max()

    lines = []
    lines.append(f"# Curriculum-γ t-SNE — verdict ({dataset})\n")
    lines.append("## Baselines\n")
    lines.append(base[["local_mean", "mid_mean", "global_mean"]]
                 .round(4).to_string())
    lines.append("\n## Curriculum models\n")
    lines.append(curr.set_index("model")[["local_mean", "mid_mean",
                 "global_mean"]].round(4).to_string())

    # Criterion A: does any curriculum DOMINATE standard t-SNE on all 3 metrics?
    dom = curr[(curr["local_mean"] >= std["local_mean"]) &
               (curr["mid_mean"] >= std["mid_mean"]) &
               (curr["global_mean"] >= std["global_mean"])]

    # Criterion B: does any curriculum capture most of BOTH specialists' edge,
    # i.e. local close to the sharp specialist AND global close to the smooth
    # specialist — escaping the trade-off rather than landing in the middle?
    def frac_of_best(val, std_val, best_val):
        # fraction of the baseline's available headroom (std→best) captured
        if best_val - std_val <= 1e-9:
            return 1.0  # no headroom to capture
        return (val - std_val) / (best_val - std_val)

    curr = curr.copy()
    curr["local_capture"] = curr["local_mean"].apply(
        lambda v: frac_of_best(v, std["local_mean"], best_local_base))
    curr["global_capture"] = curr["global_mean"].apply(
        lambda v: frac_of_best(v, std["global_mean"], best_global_base))
    # combined: must capture a healthy share of both ends simultaneously
    curr["min_capture"] = curr[["local_capture", "global_capture"]].min(axis=1)

    best_combo = curr.sort_values("min_capture", ascending=False).iloc[0]

    lines.append("\n## Headroom capture (vs standard, normalized to best baseline)\n")
    lines.append(curr.set_index("model")[
        ["local_capture", "global_capture", "min_capture"]].round(3).to_string())

    lines.append("\n## Verdict\n")
    works = False
    if not dom.empty:
        winner = dom.sort_values("min_capture", ascending=False).iloc[0] \
            if "min_capture" in dom else dom.iloc[0]
        lines.append(
            f"- **{len(dom)} curriculum config(s) DOMINATE standard t-SNE** "
            f"(≥ on local, mid, and global simultaneously).")
        works = True
    else:
        lines.append("- No curriculum config dominates standard t-SNE on all "
                     "three metrics at once.")

    bc = best_combo
    lines.append(
        f"- Best trade-off config: **{bc['model']}** — captures "
        f"{bc['local_capture']*100:.0f}% of the local headroom and "
        f"{bc['global_capture']*100:.0f}% of the global headroom that the "
        f"specialist baselines offer.")

    # "works" (escapes trade-off) if it grabs a meaningful share of BOTH ends
    if bc["min_capture"] >= 0.5 and bc["local_capture"] > 0 and bc["global_capture"] > 0:
        lines.append(
            "- **CONCLUSION: the curriculum idea WORKS.** A single 3-stage γ "
            "schedule reaches most of the local quality of the sharp specialist "
            "AND most of the global quality of the smooth specialist — escaping "
            "the trade-off that any single fixed γ is stuck on.")
        works = True
    elif works:
        lines.append(
            "- **CONCLUSION: partial success.** A config beats standard t-SNE "
            "on all metrics, but does not strongly capture both specialists' "
            "headroom. Worth tuning further (γ values / stage iteration split).")
    else:
        lines.append(
            "- **CONCLUSION: not convincing yet.** No config simultaneously "
            "improves on standard t-SNE across scales. Try a wider sweep "
            "(stronger sharp γ, smoother global γ, longer local stage), or "
            "the trade-off may be fundamental for this data.")

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

    # γ schedule knobs
    ap.add_argument("--gamma_smooth", type=float, default=0.5)
    ap.add_argument("--gamma_sharp", type=float, default=2.0)

    # iteration budget (shared by every model)
    ap.add_argument("--total_iter", type=int, default=1000)
    ap.add_argument("--ee_iter", type=int, default=250,
                    help="early-exaggeration iterations (== global stage length)")
    ap.add_argument("--ee", type=float, default=12.0,
                    help="early-exaggeration factor")

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

    out_dir = args.out_dir or os.path.join(
        _SCRIPT_DIR, "results", f"{args.dataset}_curriculum")
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        csv_path = os.path.join(out_dir, "results.csv")
        if not os.path.exists(csv_path):
            raise SystemExit(f"No results.csv at {csv_path}; run without --plot_only first.")
        df = pd.read_csv(csv_path)
    else:
        df = run_all(args, out_dir)

    summary = summarize(df)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    make_plots(df, summary, out_dir, args.dataset)
    verdict(summary, out_dir, args.dataset)


if __name__ == "__main__":
    main()
