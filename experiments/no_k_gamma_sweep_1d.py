#!/usr/bin/env python
"""
no_k_gamma_sweep_1d.py  —  Experiment 1: objective shape of NO@k vs gamma
========================================================================
For each (k, rho) setting we sweep gamma on a fine grid, run t-SNE at each
gamma, and measure NO@k for the target k. Repeating over several seeds gives a
±1 std band, so we can read off three things per setting:

  * Is NO@k(gamma) unimodal (one clear peak)?
  * Is the peak interior (an interesting gamma) or at the boundary (trivial)?
  * How large is the seed noise relative to the differences we care about?

Efficiency (see no_k_landscape_common.py): the HD k-NN is computed once; the
neighbor graph is built once per rho; the joint P is built once per (rho, gamma)
and reused across seeds; only the (cheap-init) optimisation repeats per seed.

Usage
-----
  # Full paper run (50 gammas x 5 seeds x 4 settings on full MNIST):
  python no_k_gamma_sweep_1d.py --dataset mnist

  # Cheap smoke test:
  python no_k_gamma_sweep_1d.py --n_subsample 5000 --seeds 0 --n_gamma 9

  # Regenerate figures from saved CSV (no t-SNE):
  python no_k_gamma_sweep_1d.py --plot_only

  # Optional piecewise-constant "staircase" sanity check (k=30, rho=30):
  python no_k_gamma_sweep_1d.py --staircase
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
# Grid / run parameters — edit here to start cheap and scale up
# =============================================================================
GAMMA_MIN, GAMMA_MAX = 0.0, 3.0
N_GAMMA = 50                                   # gammas linspace(0, 3, N_GAMMA)
SEEDS = [0, 1, 2, 3, 4]                        # run-to-run variance
SETTINGS = [(30, 30), (30, 100), (100, 30), (100, 100)]  # (k, rho)
K_MAX = 100                                    # largest target k -> HD/LD k-NN
# Default thread count. These experiments run hundreds of *sequential* small
# embeddings; openTSNE/sklearn do not scale past ~8-16 threads, so n_jobs=-1 on a
# many-core box oversubscribes and is dramatically slower here. Results are
# (essentially) independent of thread count. Pass --n_jobs -1 for paper-exact
# threading on machines with few cores.
N_JOBS = 8
KNN_SEED = 42                                  # fixed neighbor-graph seed
DATA_SEED = 42                                 # MNIST load / PCA / subsample seed

# Staircase sanity check (single setting, single seed, very fine gamma grid).
STAIRCASE_SETTING = (30, 30)                   # (k, rho)
STAIRCASE_N = 200
STAIRCASE_HALFWIDTH = 0.4                      # window peak ± halfwidth, clipped


# =============================================================================
# Compute
# =============================================================================
def run_experiment(args, out_dir):
    gammas = np.linspace(GAMMA_MIN, args.gamma_max, args.n_gamma)
    seeds = args.seeds
    settings = SETTINGS

    print(f"Loading {args.dataset} (PCA-50){' subsample=' + str(args.n_subsample) if args.n_subsample else ''} ...")
    X_pca, _ = common.load_dataset(args.dataset, random_state=DATA_SEED,
                                   n_subsample=args.n_subsample)
    n = len(X_pca)
    print(f"  X_pca: {X_pca.shape}")

    print(f"Computing HD k-NN once at k_max={K_MAX} ...")
    hd = common.compute_hd_knn(X_pca, K_MAX, n_jobs=args.n_jobs)

    # Which target k's must be evaluated for each rho (settings may share a rho).
    rhos = sorted({r for (_, r) in settings})
    rho_to_ks = {r: sorted({k for (k, rr) in settings if rr == r}) for r in rhos}

    n_embeddings = len(rhos) * len(gammas) * len(seeds)  # not exact if ks differ
    print(f"Sweeping: rhos={rhos}, n_gamma={len(gammas)}, seeds={seeds}")
    print(f"  ~{len(rhos) * len(gammas) * len(seeds)} embeddings total\n")

    records = []
    done = 0
    for rho in rhos:
        print(f"[rho={rho}] building neighbor graph once ...")
        neighbors, distances, eff = common.build_neighbor_cache(
            X_pca, float(rho), n_jobs=args.n_jobs, knn_seed=KNN_SEED)
        ks_for_rho = rho_to_ks[rho]
        for gi, gamma in enumerate(gammas):
            P = common.joint_P_for_gamma(neighbors, distances, eff, float(gamma),
                                         n_jobs=args.n_jobs)
            for seed in seeds:
                Y = common.embed_from_P(X_pca, P, int(seed), n_jobs=args.n_jobs)
                ld = common.compute_ld_knn(Y, K_MAX, n_jobs=args.n_jobs)
                nos = common.no_at_ks(hd, ld, ks_for_rho)
                for k, v in nos.items():
                    records.append({"k": k, "rho": rho, "gamma": float(gamma),
                                    "seed": int(seed), "no_at_k": v})
                done += 1
            if (gi + 1) % max(1, len(gammas) // 10) == 0:
                print(f"  [rho={rho}] gamma {gi + 1}/{len(gammas)} "
                      f"(={gamma:.3f}); {done} embeddings done")

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out_dir, "no_k_gamma_sweep_1d.csv")
    df.to_csv(csv_path, index=False)
    np.savez_compressed(os.path.join(out_dir, "no_k_gamma_sweep_1d.npz"),
                        k=df["k"].values, rho=df["rho"].values,
                        gamma=df["gamma"].values, seed=df["seed"].values,
                        no_at_k=df["no_at_k"].values)
    print(f"\nSaved raw results -> {csv_path}")

    _write_settings(args, out_dir, gammas, seeds, settings, n)
    return df


def run_staircase(args, out_dir, peak_gamma):
    """Very fine gamma grid (single seed) around the peak to expose the
    piecewise-constant NO@k staircase that the macro curve averages out."""
    k, rho = STAIRCASE_SETTING
    lo = max(GAMMA_MIN, peak_gamma - STAIRCASE_HALFWIDTH)
    hi = min(args.gamma_max, peak_gamma + STAIRCASE_HALFWIDTH)
    gammas = np.linspace(lo, hi, STAIRCASE_N)
    seed = args.seeds[0]
    print(f"\n[staircase] k={k} rho={rho} seed={seed}: "
          f"{STAIRCASE_N} gammas in [{lo:.3f}, {hi:.3f}] ...")

    X_pca, _ = common.load_dataset(args.dataset, random_state=DATA_SEED,
                                   n_subsample=args.n_subsample)
    hd = common.compute_hd_knn(X_pca, K_MAX, n_jobs=args.n_jobs)
    neighbors, distances, eff = common.build_neighbor_cache(
        X_pca, float(rho), n_jobs=args.n_jobs, knn_seed=KNN_SEED)

    rows = []
    for gi, gamma in enumerate(gammas):
        P = common.joint_P_for_gamma(neighbors, distances, eff, float(gamma),
                                     n_jobs=args.n_jobs)
        Y = common.embed_from_P(X_pca, P, int(seed), n_jobs=args.n_jobs)
        ld = common.compute_ld_knn(Y, K_MAX, n_jobs=args.n_jobs)
        rows.append({"k": k, "rho": rho, "gamma": float(gamma),
                     "seed": int(seed), "no_at_k": common.no_at_k(hd, ld, k)})
        if (gi + 1) % 20 == 0:
            print(f"  staircase {gi + 1}/{STAIRCASE_N}")

    df = pd.DataFrame.from_records(rows)
    df.to_csv(os.path.join(out_dir, "no_k_staircase.csv"), index=False)
    print(f"Saved staircase -> {os.path.join(out_dir, 'no_k_staircase.csv')}")
    return df


# =============================================================================
# Plotting (loads from CSV; reused by --plot_only)
# =============================================================================
def _agg(df, k, rho):
    sub = df[(df["k"] == k) & (df["rho"] == rho)].sort_values("gamma")
    g = sub.groupby("gamma")["no_at_k"]
    gammas = np.array(sorted(sub["gamma"].unique()))
    mean = g.mean().reindex(gammas).values
    std = g.std(ddof=0).reindex(gammas).values
    return gammas, mean, std


def _peak_info(gammas, mean):
    i = int(np.nanargmax(mean))
    g_star = gammas[i]
    edge = 1e-9
    if g_star <= gammas[0] + edge:
        loc = "boundary (low)"
    elif g_star >= gammas[-1] - edge:
        loc = "boundary (high)"
    else:
        loc = "interior"
    return g_star, mean[i], loc, i


def _plot_one(ax, gammas, mean, std, k, rho):
    ax.plot(gammas, mean, color="#0072B2", lw=2, label="mean NO@k")
    ax.fill_between(gammas, mean - std, mean + std, color="#0072B2",
                    alpha=0.2, label="±1 std (seeds)")
    g_star, m_star, loc, _ = _peak_info(gammas, mean)
    ax.axvline(g_star, color="#D55E00", ls="--", lw=1.5)
    ax.scatter([g_star], [m_star], color="#D55E00", zorder=5)
    ax.annotate(f"peak γ={g_star:.2f}\n({loc})\nNO@{k}={m_star:.3f}",
                xy=(g_star, m_star), xytext=(0.55, 0.12),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#D55E00"),
                fontsize=11, color="#D55E00")
    ax.axvline(1.0, color="0.5", ls=":", lw=1)   # gamma=1 = upstream t-SNE
    ax.set_xlabel("γ")
    ax.set_ylabel(f"NO@{k}")
    ax.set_title(f"NO@{k} vs γ  (ρ={rho})")
    common.clean_axes(ax)
    return g_star, m_star, loc


def plot_all(df, out_dir, staircase_df=None):
    summary = []
    for (k, rho) in SETTINGS:
        if df[(df["k"] == k) & (df["rho"] == rho)].empty:
            continue
        gammas, mean, std = _agg(df, k, rho)
        fig, ax = plt.subplots(figsize=(7, 5))
        g_star, m_star, loc = _plot_one(ax, gammas, mean, std, k, rho)
        ax.legend(loc="upper right")
        common.save_fig(os.path.join(out_dir, f"no_k{k}_rho{rho}_gamma_sweep.png"))
        max_band = float(np.nanmax(std))
        summary.append({"k": k, "rho": rho, "peak_gamma": float(g_star),
                        "peak_no": float(m_star), "peak_location": loc,
                        "max_std_band": max_band})

    # 2x2 combined panel.
    present = [(k, rho) for (k, rho) in SETTINGS
               if not df[(df["k"] == k) & (df["rho"] == rho)].empty]
    if len(present) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        for ax, (k, rho) in zip(axes.ravel(), SETTINGS):
            gammas, mean, std = _agg(df, k, rho)
            _plot_one(ax, gammas, mean, std, k, rho)
        axes.ravel()[0].legend(loc="upper right")
        fig.suptitle("NO@k vs γ across (k, ρ) settings", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        common.save_fig(os.path.join(out_dir, "no_k_gamma_sweep_1d_panel.png"))

    if staircase_df is not None and not staircase_df.empty:
        k, rho = STAIRCASE_SETTING
        s = staircase_df.sort_values("gamma")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(s["gamma"], s["no_at_k"], color="#009E73", lw=1,
                drawstyle="steps-mid")
        ax.set_xlabel("γ")
        ax.set_ylabel(f"NO@{k} (single seed)")
        ax.set_title(f"NO@{k} staircase (ρ={rho}, 1 seed, {len(s)} γ points)\n"
                     "piecewise-constant: changes only when a neighbor crosses k")
        common.clean_axes(ax)
        common.save_fig(os.path.join(out_dir, "no_k_staircase.png"))

    if summary:
        pd.DataFrame(summary).to_csv(
            os.path.join(out_dir, "no_k_gamma_sweep_1d_optima.csv"), index=False)
        print("\nLocated optima:")
        for s in summary:
            print(f"  k={s['k']:>3} rho={s['rho']:>3}: peak γ={s['peak_gamma']:.3f}"
                  f" ({s['peak_location']}), NO={s['peak_no']:.4f},"
                  f" max std band={s['max_std_band']:.4f}")


# =============================================================================
# Settings log
# =============================================================================
def _write_settings(args, out_dir, gammas, seeds, settings, n):
    info = {
        "experiment": "no_k_gamma_sweep_1d",
        "dataset": args.dataset,
        "n_points": int(n),
        "n_subsample": args.n_subsample,
        "gammas": [float(g) for g in gammas],
        "n_gamma": len(gammas),
        "gamma_range": [GAMMA_MIN, float(args.gamma_max)],
        "seeds": list(seeds),
        "settings_k_rho": settings,
        "k_max": K_MAX,
        "knn_seed": KNN_SEED,
        "data_seed": DATA_SEED,
        "tsne_settings": common.TSNE_SETTINGS,
    }
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out_dir, "settings.md"), "w") as f:
        f.write("# Experiment 1 — NO@k vs γ (1D sweep)\n\n")
        f.write("```json\n" + json.dumps(info, indent=2) + "\n```\n")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 1: NO@k objective shape vs γ (fixed ρ).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=common.DATASETS)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_jobs", type=int, default=N_JOBS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--n_gamma", type=int, default=N_GAMMA,
                   help="Number of γ grid points (linspace from 0).")
    p.add_argument("--gamma_max", type=float, default=GAMMA_MAX)
    p.add_argument("--n_subsample", type=int, default=None,
                   help="Use a random subset of MNIST (cheap smoke test).")
    p.add_argument("--staircase", action="store_true",
                   help="Also run the very-fine single-seed staircase check.")
    p.add_argument("--plot_only", action="store_true",
                   help="Skip computation; rebuild figures from saved CSVs.")
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
        "exp1_gamma_sweep_1d")
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        df = pd.read_csv(os.path.join(out_dir, "no_k_gamma_sweep_1d.csv"))
        sc_path = os.path.join(out_dir, "no_k_staircase.csv")
        sc = pd.read_csv(sc_path) if os.path.exists(sc_path) else None
        plot_all(df, out_dir, staircase_df=sc)
        return

    df = run_experiment(args, out_dir)
    plot_all(df, out_dir)

    if args.staircase:
        k, rho = STAIRCASE_SETTING
        gammas, mean, _ = _agg(df, k, rho)
        peak_gamma = float(gammas[int(np.nanargmax(mean))])
        sc = run_staircase(args, out_dir, peak_gamma)
        plot_all(df, out_dir, staircase_df=sc)


if __name__ == "__main__":
    main()
