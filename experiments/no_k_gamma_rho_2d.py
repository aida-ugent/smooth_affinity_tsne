#!/usr/bin/env python
"""
no_k_gamma_rho_2d.py  —  Experiment 2: the (γ, ρ) objective landscape
=====================================================================
We sweep both γ and the perplexity ρ on a grid and measure NO@k. The goal is to
see whether the optimum over (γ, ρ) is a single basin, multiple peaks, or a
**ridge** — γ and ρ are coupled because γ rescales the effective perplexity, so
low-ρ + smooth-γ can mimic high-ρ. That tells us whether joint optimisation is
meaningful or collapses to roughly one parameter.

Outputs (per mode):
  * Per-target-k heatmaps of NO@k over (γ, ρ) for k ∈ {10, 30, 100} — each IS the
    objective surface we would optimise for that k; the max cell is annotated.
  * AUC-summary heatmaps (paper Fig. 6 style): AUC of NO@k over near-local
    (k=1..10) and mid-local (k=11..90) — a summary on top of, not a replacement
    for, the per-k surfaces.
  * An optima log: per target k, the best (γ, ρ) and how it moves across k
    (stable point vs ridge).

Efficiency: HD k-NN once; neighbor graph once per ρ; joint P once per (ρ, γ),
reused across seeds; the full NO@k curve (k=1..k_max) is computed in one pass per
embedding (see no_k_landscape_common.no_full_curve), giving the target-k values
and both AUC ranges at once.

Usage
-----
  python no_k_gamma_rho_2d.py --mode coarse                 # cheap first pass
  python no_k_gamma_rho_2d.py --mode fine                   # after inspecting
  python no_k_gamma_rho_2d.py --mode coarse --n_subsample 5000 --seeds 0
  python no_k_gamma_rho_2d.py --mode coarse --plot_only     # rebuild figures
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
import no_k_landscape_common as common  # noqa: E402


# =============================================================================
# Grid parameters — edit here to start cheap and scale up
# =============================================================================
GAMMA_MIN, GAMMA_MAX = 0.0, 3.0

# Coarse pass: inspect whether a ridge/basin appears before paying for the fine grid.
GAMMAS_COARSE = np.linspace(GAMMA_MIN, GAMMA_MAX, 9)
RHOS_COARSE = [15, 30, 50, 100, 150, 200]

# Fine pass.
GAMMAS_FINE = np.linspace(GAMMA_MIN, GAMMA_MAX, 15)
RHOS_FINE = [10, 20, 30, 50, 75, 100, 125, 150, 175, 200]

SEEDS = [0, 1, 2, 3, 4]
TARGET_KS = [10, 30, 100]          # per-k objective surfaces
AUC_NEAR = (1, 10)                 # near-local AUC window
AUC_MID = (11, 90)                 # mid-local AUC window
K_MAX = 100                        # largest k -> HD/LD k-NN and full curve
# Default thread count. Hundreds of sequential small embeddings; openTSNE/sklearn
# don't scale past ~8-16 threads, so n_jobs=-1 on a many-core box oversubscribes
# and is far slower. Results are (essentially) thread-count independent. Pass
# --n_jobs -1 for paper-exact threading on few-core machines.
N_JOBS = 8
KNN_SEED = 42
DATA_SEED = 42


def _grid_for_mode(mode):
    if mode == "coarse":
        return np.asarray(GAMMAS_COARSE), list(RHOS_COARSE)
    return np.asarray(GAMMAS_FINE), list(RHOS_FINE)


# =============================================================================
# Compute
# =============================================================================
def run_experiment(args, out_dir):
    gammas, rhos = _grid_for_mode(args.mode)
    seeds = args.seeds

    print(f"Loading {args.dataset} (PCA-50){' subsample=' + str(args.n_subsample) if args.n_subsample else ''} ...")
    X_pca, _ = common.load_dataset(args.dataset, random_state=DATA_SEED,
                                   n_subsample=args.n_subsample)
    n = len(X_pca)
    print(f"  X_pca: {X_pca.shape}")
    print(f"Mode={args.mode}: {len(gammas)} γ × {len(rhos)} ρ × {len(seeds)} seeds "
          f"= {len(gammas) * len(rhos) * len(seeds)} embeddings")

    print(f"Computing HD k-NN once at k_max={K_MAX} ...")
    hd = common.compute_hd_knn(X_pca, K_MAX, n_jobs=args.n_jobs)

    records = []
    done = 0
    total = len(gammas) * len(rhos) * len(seeds)
    for rho in rhos:
        print(f"[ρ={rho}] building neighbor graph once ...")
        neighbors, distances, eff = common.build_neighbor_cache(
            X_pca, float(rho), n_jobs=args.n_jobs, knn_seed=KNN_SEED)
        for gamma in gammas:
            P = common.joint_P_for_gamma(neighbors, distances, eff, float(gamma),
                                         n_jobs=args.n_jobs)
            for seed in seeds:
                Y = common.embed_from_P(X_pca, P, int(seed), n_jobs=args.n_jobs)
                ld = common.compute_ld_knn(Y, K_MAX, n_jobs=args.n_jobs)
                curve = common.no_full_curve(hd, ld)           # NO@k for k=1..K_MAX
                rec = {"rho": rho, "gamma": float(gamma), "seed": int(seed)}
                for k in TARGET_KS:
                    rec[f"no_at_{k}"] = float(curve[k - 1])
                rec["auc_near"] = common.auc_range(
                    curve, np.arange(1, K_MAX + 1), *AUC_NEAR)
                rec["auc_mid"] = common.auc_range(
                    curve, np.arange(1, K_MAX + 1), *AUC_MID)
                records.append(rec)
                done += 1
            print(f"  [ρ={rho}] γ={gamma:.3f} done ({done}/{total})")

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out_dir, "no_k_gamma_rho_2d.csv")
    df.to_csv(csv_path, index=False)
    np.savez_compressed(os.path.join(out_dir, "no_k_gamma_rho_2d.npz"),
                        **{c: df[c].values for c in df.columns})
    print(f"\nSaved raw results -> {csv_path}")
    _write_settings(args, out_dir, gammas, rhos, seeds, n)
    return df


# =============================================================================
# Plotting
# =============================================================================
def _pivot_mean_std(df, value_col):
    """Return (gammas, rhos, mean_grid, std_grid) with rho rows, gamma cols."""
    gammas = np.array(sorted(df["gamma"].unique()))
    rhos = np.array(sorted(df["rho"].unique()))
    g = df.groupby(["rho", "gamma"])[value_col]
    mean = g.mean().unstack("gamma").reindex(index=rhos, columns=gammas)
    std = g.std(ddof=0).unstack("gamma").reindex(index=rhos, columns=gammas)
    return gammas, rhos, mean.values, std.values


def _heatmap(ax, gammas, rhos, grid, title, cbar_label):
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([f"{v:.2f}" for v in gammas], rotation=45, ha="right")
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels([str(int(r)) for r in rhos])
    ax.set_xlabel("γ")
    ax.set_ylabel("ρ (perplexity)")
    ax.set_title(title)
    # Annotate the max cell.
    j, i = np.unravel_index(np.nanargmax(grid), grid.shape)  # (row=rho, col=gamma)
    ax.scatter([i], [j], marker="*", s=260, color="#D55E00", edgecolor="white",
               zorder=5)
    ax.text(0.03, 0.97, f"max @ γ={gammas[i]:.2f}, ρ={int(rhos[j])}\n{grid[j, i]:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            color="#D55E00",
            bbox=dict(boxstyle="round", fc="white", ec="#D55E00", alpha=0.85))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    return gammas[i], int(rhos[j]), float(grid[j, i])


def plot_all(df, out_dir, mode):
    optima = []

    # Per-target-k surfaces.
    for k in TARGET_KS:
        col = f"no_at_{k}"
        if col not in df.columns:
            continue
        gammas, rhos, mean, _std = _pivot_mean_std(df, col)
        fig, ax = plt.subplots(figsize=(8, 6))
        gstar, rstar, vstar = _heatmap(
            ax, gammas, rhos, mean,
            f"NO@{k} objective surface  (mode={mode})", f"NO@{k}")
        common.save_fig(os.path.join(out_dir, f"surface_no_at_{k}.png"))
        optima.append({"target": f"NO@{k}", "best_gamma": gstar,
                       "best_rho": rstar, "best_value": vstar})

    # AUC-summary surfaces (paper Fig. 6 style).
    for col, label in [("auc_near", f"AUC NO@k near-local (k={AUC_NEAR[0]}..{AUC_NEAR[1]})"),
                       ("auc_mid", f"AUC NO@k mid-local (k={AUC_MID[0]}..{AUC_MID[1]})")]:
        if col not in df.columns:
            continue
        gammas, rhos, mean, _std = _pivot_mean_std(df, col)
        fig, ax = plt.subplots(figsize=(8, 6))
        gstar, rstar, vstar = _heatmap(
            ax, gammas, rhos, mean, f"{label}  (mode={mode})", label)
        common.save_fig(os.path.join(out_dir, f"surface_{col}.png"))
        optima.append({"target": label, "best_gamma": gstar,
                       "best_rho": rstar, "best_value": vstar})

    # Combined per-k panel for quick comparison.
    present = [k for k in TARGET_KS if f"no_at_{k}" in df.columns]
    if len(present) >= 2:
        fig, axes = plt.subplots(1, len(present), figsize=(7 * len(present), 6))
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, present):
            gammas, rhos, mean, _ = _pivot_mean_std(df, f"no_at_{k}")
            _heatmap(ax, gammas, rhos, mean, f"NO@{k}", f"NO@{k}")
        fig.suptitle(f"NO@k objective surfaces over (γ, ρ) — mode={mode}",
                     fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        common.save_fig(os.path.join(out_dir, "surfaces_per_k_panel.png"))

    _write_optima(df, out_dir, mode, optima)


def _write_optima(df, out_dir, mode, optima):
    opt_df = pd.DataFrame(optima)
    opt_df.to_csv(os.path.join(out_dir, "optima.csv"), index=False)

    lines = [f"# Experiment 2 — (γ, ρ) landscape optima (mode={mode})\n",
             "## Per-target-k optima (these are the surfaces to optimise)\n"]
    per_k = [o for o in optima if o["target"].startswith("NO@")]
    for o in per_k:
        lines.append(f"- **{o['target']}**: best at γ={o['best_gamma']:.3f}, "
                     f"ρ={o['best_rho']}, NO={o['best_value']:.4f}")
    if len(per_k) >= 2:
        gs = [o["best_gamma"] for o in per_k]
        rs = [o["best_rho"] for o in per_k]
        lines.append("\n## How the optimum moves with target k\n")
        lines.append(f"- γ* ranges {min(gs):.2f}→{max(gs):.2f} as k grows; "
                     f"ρ* ranges {min(rs)}→{max(rs)}.")
        trend = ("smooth-γ / higher-ρ as k grows (expected coupling direction)"
                 if (per_k[-1]["best_gamma"] <= per_k[0]["best_gamma"])
                 else "sharp-γ as k grows")
        lines.append(f"- Trend: {trend}. If γ* and ρ* trade off smoothly rather "
                     "than landing on one cell, the surface is a **ridge** "
                     "(γ and ρ are partially redundant); if each k pins a single "
                     "(γ, ρ) cell, it is a **basin**.")
    lines.append("\n## AUC-range summaries\n")
    for o in [o for o in optima if not o["target"].startswith("NO@")]:
        lines.append(f"- **{o['target']}**: best at γ={o['best_gamma']:.3f}, "
                     f"ρ={o['best_rho']}, AUC={o['best_value']:.4f}")
    with open(os.path.join(out_dir, "optima.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


# =============================================================================
# Settings log
# =============================================================================
def _write_settings(args, out_dir, gammas, rhos, seeds, n):
    info = {
        "experiment": "no_k_gamma_rho_2d",
        "mode": args.mode,
        "dataset": args.dataset,
        "n_points": int(n),
        "n_subsample": args.n_subsample,
        "gammas": [float(g) for g in gammas],
        "rhos": [int(r) for r in rhos],
        "seeds": list(seeds),
        "target_ks": TARGET_KS,
        "auc_near": AUC_NEAR,
        "auc_mid": AUC_MID,
        "k_max": K_MAX,
        "knn_seed": KNN_SEED,
        "data_seed": DATA_SEED,
        "tsne_settings": common.TSNE_SETTINGS,
    }
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out_dir, "settings.md"), "w") as f:
        f.write(f"# Experiment 2 — (γ, ρ) landscape ({args.mode})\n\n")
        f.write("```json\n" + json.dumps(info, indent=2) + "\n```\n")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 2: NO@k objective landscape over (γ, ρ).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=common.DATASETS)
    p.add_argument("--mode", default="coarse", choices=["coarse", "fine"])
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_jobs", type=int, default=N_JOBS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--n_subsample", type=int, default=None,
                   help="Use a random subset of MNIST (cheap smoke test).")
    p.add_argument("--plot_only", action="store_true",
                   help="Skip computation; rebuild figures from saved CSV.")
    p.add_argument("--self_test", action="store_true",
                   help="Run the cached-P / NO@k correctness self-test and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        common.run_self_test()
        return

    out_dir = args.out_dir or os.path.join(
        _SCRIPT_DIR, "results", "no_k_landscape", args.dataset,
        "exp2_gamma_rho_2d", args.mode)
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        df = pd.read_csv(os.path.join(out_dir, "no_k_gamma_rho_2d.csv"))
        plot_all(df, out_dir, args.mode)
        return

    df = run_experiment(args, out_dir)
    plot_all(df, out_dir, args.mode)


if __name__ == "__main__":
    main()
