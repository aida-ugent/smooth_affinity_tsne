#!/usr/bin/env python
"""
no_k_perplexity_sweep_1d.py  —  Experiment 3: NO@k vs perplexity at fixed gamma
==============================================================================
The mirror image of Experiment 1 (no_k_gamma_sweep_1d.py): instead of fixing the
perplexity rho and sweeping the smoothing gamma, here we **fix gamma and sweep
the perplexity rho** (the x-axis). For each (k, gamma) we run t-SNE across a grid
of perplexities, measure NO@k, and repeat over seeds for a +/-1 std band.

Settings (6 curves):
  NO@30  with gamma in {0.8, 1.0, 1.2}
  NO@100 with gamma in {0.8, 1.0, 1.2}
Produces 6 individual figures plus one combined 2x3 panel.

Each plot lets us read off, for a fixed gamma, the perplexity that maximises
NO@k for that target k, and whether that optimum is interior or at the grid edge.

Efficiency (see no_k_landscape_common.py): HD k-NN once; the neighbor graph is
built once per rho and reused across the 3 gammas; the joint P is built once per
(rho, gamma) and reused across seeds; both target k's come from one embedding via
the one-pass NO@k curve.

Usage
-----
  python no_k_perplexity_sweep_1d.py --dataset mnist
  python no_k_perplexity_sweep_1d.py --dataset mnist --n_subsample 5000 --seeds 0
  python no_k_perplexity_sweep_1d.py --dataset mnist --plot_only
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
# Perplexity grid (x-axis). Geometric spacing spans local -> global evenly.
PERPS = [5, 8, 12, 18, 25, 35, 50, 70, 95, 125, 160, 200]
GAMMAS = [0.8, 1.0, 1.2]        # one curve family per gamma
TARGET_KS = [30, 100]           # NO@30 and NO@100
SEEDS = [0, 1, 2, 3, 4]
K_MAX = 100                     # largest target k -> HD/LD k-NN
N_JOBS = 8                      # see note in no_k_gamma_sweep_1d.py
KNN_SEED = 42
DATA_SEED = 42

# Fixed (k, gamma) settings, in panel order (row = k, col = gamma).
SETTINGS = [(k, g) for k in TARGET_KS for g in GAMMAS]


# =============================================================================
# Compute
# =============================================================================
def run_experiment(args, out_dir):
    perps = args.perps
    seeds = args.seeds

    print(f"Loading {args.dataset} (PCA-50)"
          f"{' subsample=' + str(args.n_subsample) if args.n_subsample else ''} ...")
    X_pca, _ = common.load_dataset(args.dataset, random_state=DATA_SEED,
                                   n_subsample=args.n_subsample)
    n = len(X_pca)
    print(f"  X_pca: {X_pca.shape}")

    print(f"Computing HD k-NN once at k_max={K_MAX} ...")
    hd = common.compute_hd_knn(X_pca, K_MAX, n_jobs=args.n_jobs)

    print(f"Sweeping: perps={perps}, gammas={GAMMAS}, seeds={seeds}")
    print(f"  ~{len(perps) * len(GAMMAS) * len(seeds)} embeddings total\n")

    records = []
    done = 0
    for rho in perps:
        print(f"[ρ={rho}] building neighbor graph once ...")
        neighbors, distances, eff = common.build_neighbor_cache(
            X_pca, float(rho), n_jobs=args.n_jobs, knn_seed=KNN_SEED)
        for gamma in GAMMAS:
            P = common.joint_P_for_gamma(neighbors, distances, eff, float(gamma),
                                         n_jobs=args.n_jobs)
            for seed in seeds:
                Y = common.embed_from_P(X_pca, P, int(seed), n_jobs=args.n_jobs)
                ld = common.compute_ld_knn(Y, K_MAX, n_jobs=args.n_jobs)
                curve = common.no_full_curve(hd, ld)
                for k in TARGET_KS:
                    records.append({"k": k, "gamma": float(gamma), "rho": int(rho),
                                    "seed": int(seed),
                                    "no_at_k": float(curve[k - 1])})
                done += 1
        print(f"  [ρ={rho}] done ({done}/{len(perps) * len(GAMMAS) * len(seeds)})")

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out_dir, "no_k_perplexity_sweep_1d.csv")
    df.to_csv(csv_path, index=False)
    np.savez_compressed(os.path.join(out_dir, "no_k_perplexity_sweep_1d.npz"),
                        **{c: df[c].values for c in df.columns})
    print(f"\nSaved raw results -> {csv_path}")
    _write_settings(args, out_dir, perps, seeds, n)
    return df


# =============================================================================
# Plotting (loads from CSV; reused by --plot_only and the parallel runner)
# =============================================================================
def _agg(df, k, gamma):
    sub = df[(df["k"] == k) & (np.isclose(df["gamma"], gamma))].sort_values("rho")
    g = sub.groupby("rho")["no_at_k"]
    rhos = np.array(sorted(sub["rho"].unique()))
    mean = g.mean().reindex(rhos).values
    std = g.std(ddof=0).reindex(rhos).values
    return rhos, mean, std


def _peak_info(rhos, mean):
    i = int(np.nanargmax(mean))
    r_star = rhos[i]
    if i == 0:
        loc = "boundary (low)"
    elif i == len(rhos) - 1:
        loc = "boundary (high)"
    else:
        loc = "interior"
    return r_star, mean[i], loc


def _plot_one(ax, rhos, mean, std, k, gamma):
    ax.plot(rhos, mean, color="#0072B2", lw=2, marker="o", ms=3,
            label="mean NO@k")
    ax.fill_between(rhos, mean - std, mean + std, color="#0072B2", alpha=0.2,
                    label="±1 std (seeds)")
    r_star, m_star, loc = _peak_info(rhos, mean)
    ax.axvline(r_star, color="#D55E00", ls="--", lw=1.5)
    ax.scatter([r_star], [m_star], color="#D55E00", zorder=5)
    ax.annotate(f"peak ρ={r_star}\n({loc})\nNO@{k}={m_star:.3f}",
                xy=(r_star, m_star), xytext=(0.55, 0.12),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#D55E00"),
                fontsize=10, color="#D55E00")
    ax.set_xlabel("perplexity ρ")
    ax.set_ylabel(f"NO@{k}")
    ax.set_title(f"NO@{k} vs ρ  (γ={gamma:g})")
    common.clean_axes(ax)
    return r_star, m_star, loc


def plot_all(df, out_dir):
    summary = []
    for (k, gamma) in SETTINGS:
        if df[(df["k"] == k) & (np.isclose(df["gamma"], gamma))].empty:
            continue
        rhos, mean, std = _agg(df, k, gamma)
        fig, ax = plt.subplots(figsize=(7, 5))
        r_star, m_star, loc = _plot_one(ax, rhos, mean, std, k, gamma)
        ax.legend(loc="upper right")
        common.save_fig(os.path.join(
            out_dir, f"no_k{k}_gamma{gamma:g}_perp_sweep.png"))
        summary.append({"k": k, "gamma": gamma, "peak_rho": int(r_star),
                        "peak_no": float(m_star), "peak_location": loc,
                        "max_std_band": float(np.nanmax(std))})

    # Combined 2x3 panel: rows = target k, cols = gamma.
    present = [(k, g) for (k, g) in SETTINGS
               if not df[(df["k"] == k) & (np.isclose(df["gamma"], g))].empty]
    if len(present) == 6:
        fig, axes = plt.subplots(len(TARGET_KS), len(GAMMAS), figsize=(18, 10))
        for ax, (k, gamma) in zip(axes.ravel(), SETTINGS):
            rhos, mean, std = _agg(df, k, gamma)
            _plot_one(ax, rhos, mean, std, k, gamma)
        axes.ravel()[0].legend(loc="upper right")
        fig.suptitle("NO@k vs perplexity ρ across fixed γ", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        common.save_fig(os.path.join(out_dir, "no_k_perplexity_sweep_1d_panel.png"))

    if summary:
        pd.DataFrame(summary).to_csv(
            os.path.join(out_dir, "no_k_perplexity_sweep_1d_optima.csv"),
            index=False)
        print("\nLocated optima:")
        for s in summary:
            print(f"  k={s['k']:>3} γ={s['gamma']:g}: peak ρ={s['peak_rho']:>3}"
                  f" ({s['peak_location']}), NO={s['peak_no']:.4f},"
                  f" max std band={s['max_std_band']:.4f}")


# =============================================================================
# Settings log
# =============================================================================
def _write_settings(args, out_dir, perps, seeds, n):
    info = {
        "experiment": "no_k_perplexity_sweep_1d",
        "dataset": args.dataset,
        "n_points": int(n),
        "n_subsample": args.n_subsample,
        "perplexities": list(perps),
        "gammas": GAMMAS,
        "target_ks": TARGET_KS,
        "seeds": list(seeds),
        "k_max": K_MAX,
        "knn_seed": KNN_SEED,
        "data_seed": DATA_SEED,
        "tsne_settings": common.TSNE_SETTINGS,
    }
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out_dir, "settings.md"), "w") as f:
        f.write("# Experiment 3 — NO@k vs perplexity (fixed γ)\n\n")
        f.write("```json\n" + json.dumps(info, indent=2) + "\n```\n")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 3: NO@k objective shape vs perplexity (fixed γ).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=common.DATASETS)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_jobs", type=int, default=N_JOBS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--perps", type=int, nargs="+", default=PERPS,
                   help="Perplexity grid (x-axis).")
    p.add_argument("--n_subsample", type=int, default=None,
                   help="Use a random subset (cheap smoke test).")
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
        "exp3_perplexity_sweep_1d")
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        df = pd.read_csv(os.path.join(out_dir, "no_k_perplexity_sweep_1d.csv"))
        plot_all(df, out_dir)
        return

    df = run_experiment(args, out_dir)
    plot_all(df, out_dir)


if __name__ == "__main__":
    main()
