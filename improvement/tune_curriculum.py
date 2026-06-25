#!/usr/bin/env python
"""
tune_curriculum.py  —  schedule search for curriculum-γ t-SNE
=============================================================

`exp.py` compares a small, fixed set of curriculum schedules against baselines
and external DR.  THIS script instead *explores the schedule space itself* to
find which curriculum design is best, before committing to one.

A curriculum schedule has three kinds of knob, all swept here:

  * how many stages          (2, 3, 4, 5, ... up to a near-continuous ramp)
  * the γ at each stage       (from very smooth γ≈0 up to very sharp γ≈3)
  * the duration of each stage (equal / front-heavy / back-heavy splits)

All schedules anneal γ monotonically from low→high: smooth γ first lays out the
coarse global arrangement (under early exaggeration), then γ rises to sharpen
the fine local neighbour structure.

Every schedule reuses the SAME kNN graph, PCA init, iteration budget and
evaluation metrics as `exp.py` (imported directly), so the numbers are directly
comparable to that experiment.  Single seed by design — this is a search, not a
significance test; re-run the winner through `exp.py` for the multi-seed story.

Outputs (→ results/<dataset>_curriculum_tuning/):
  results.csv / curves.csv          scalar + curve metrics per schedule
  summary_scalars.csv / summary_curves.csv
  <dataset>_NO_curve.png            NO@k line chart  (same style as exp.py)
  <dataset>_trustworthiness_curve.png
  <dataset>_scalar_metrics.png
  <dataset>_tradeoff_scatter.png
  <dataset>_TUNING.md               ranked best-schedule report

Usage
-----
  python tune_curriculum.py --dataset mnist                # full search
  python tune_curriculum.py --dataset mnist --quick        # fast smoke test
  python tune_curriculum.py --dataset mnist --plot_only     # re-plot from csv
"""

import os
import time
import argparse

import numpy as np
import pandas as pd

import exp  # reuse data loading, t-SNE machinery, metrics and plotting


# =============================================================================
# Schedule construction
# =============================================================================

def gamma_ramp(g0, g1, n):
    """n γ values rising linearly from g0 to g1 (n=1 → [g0])."""
    if n <= 1:
        return [float(g0)]
    return [float(g) for g in np.linspace(g0, g1, n)]


def build_stages(gammas, weights, ee_iter, total_iter, ee):
    """Turn a (γ-list, duration-weights) schedule into optimizer stages.

    ``total_iter`` is split across the stages by ``weights``; the first stage
    additionally carries the early-exaggeration warm-up (its first ``ee_iter``
    iterations run with exaggeration ``ee`` and low momentum).  Every later
    stage runs plain (exaggeration 1, momentum 0.8).  Stages with 0 iterations
    are dropped.
    """
    S = len(gammas)
    sw = float(sum(weights)) or 1.0
    counts, alloc = [], 0
    for i, w in enumerate(weights):
        c = (total_iter - alloc) if i == S - 1 else int(round(total_iter * w / sw))
        counts.append(c)
        if i < S - 1:
            alloc += c

    stages = []
    for i, (g, c) in enumerate(zip(gammas, counts)):
        if c <= 0:
            continue
        if i == 0:
            ee_c = min(ee_iter, c)
            stages.append({"gamma": float(g), "n_iter": ee_c,
                           "exaggeration": ee, "momentum": 0.5})
            if c - ee_c > 0:
                stages.append({"gamma": float(g), "n_iter": c - ee_c,
                               "exaggeration": 1.0, "momentum": 0.8})
        else:
            stages.append({"gamma": float(g), "n_iter": c,
                           "exaggeration": 1.0, "momentum": 0.8})
    return stages


def make_configs():
    """Define every schedule to search. Returns name -> spec dict."""
    cfg = {}

    # ── single-γ references (context: the fixed points the curriculum must beat)
    for g, tag in [(0.5, "smooth"), (1.0, "standard"), (2.0, "sharp")]:
        cfg[f"ref_{tag}"] = {"kind": "single", "group": "reference", "gamma": g}

    # ── N-stage linear ramps, equal duration, over several γ ranges ───────────
    for (g0, g1) in [(0.5, 2.0), (0.25, 2.5), (0.0, 3.0)]:
        for n in (2, 3, 4, 5):
            cfg[f"ramp_g{g0:g}-{g1:g}_s{n}"] = {
                "kind": "curriculum", "group": "ramp",
                "gammas": gamma_ramp(g0, g1, n), "weights": [1.0] * n}

    # ── duration splits on the canonical 3-stage 0.5 → 1 → 2 schedule ─────────
    for tag, w in [("frontheavy", [0.5, 0.3, 0.2]),
                   ("equal",      [1.0, 1.0, 1.0]),
                   ("backheavy",  [0.2, 0.3, 0.5]),
                   ("localheavy", [0.2, 0.2, 0.6])]:
        cfg[f"dur_3s_{tag}"] = {
            "kind": "curriculum", "group": "duration",
            "gammas": [0.5, 1.0, 2.0], "weights": w}

    # ── fine / near-continuous ramps (many small γ steps) ─────────────────────
    for (g0, g1) in [(0.0, 3.0), (0.5, 2.5)]:
        for n in (8, 12):
            cfg[f"fine_g{g0:g}-{g1:g}_s{n}"] = {
                "kind": "curriculum", "group": "fine",
                "gammas": gamma_ramp(g0, g1, n), "weights": [1.0] * n}

    return cfg


# =============================================================================
# Driver
# =============================================================================

def run_search(args, out_dir):
    csv_path = os.path.join(out_dir, "results.csv")
    curve_path = os.path.join(out_dir, "curves.csv")
    seed = args.seed

    cfg = make_configs()
    print(f"Schedules ({len(cfg)}): {list(cfg)}")

    print(f"\nLoading {args.dataset} (n_samples={args.n_samples}) ...")
    X, y = exp.load_dataset(args.dataset, args.n_samples, args.master_seed, args)
    n = len(X)
    print(f"  data shape = {X.shape}")

    K = int(min(args.k_max, n - 2))
    print(f"Pre-computing evaluation HD-kNN (k={K}) ...")
    hd_nbrs = exp._knn_indices(X, K, n_jobs=args.n_jobs)

    m = int(min(args.trust_subsample, n))
    trust_sub = (np.sort(np.random.default_rng(args.master_seed).choice(
        n, size=m, replace=False)) if m < n else np.arange(n))
    print(f"Pre-computing HD rank matrix for trustworthiness (m={m}) ...")
    hd_rank = exp.hd_rank_matrix(X[trust_sub])

    print("Building kNN graph + PCA init ...")
    knn_idx, knn_dist, eff_perp = exp.build_knn(
        X, args.perplexity, n_jobs=args.n_jobs, verbose=False)
    init = exp.initialization.pca(X, random_state=seed)

    existing = (pd.read_csv(csv_path)
                if (os.path.exists(csv_path) and not args.fresh)
                else pd.DataFrame())
    existing_curves = (pd.read_csv(curve_path)
                       if (os.path.exists(curve_path) and not args.fresh)
                       else pd.DataFrame())
    done = set(existing["model"]) if not existing.empty else set()
    i10 = min(9, K - 1)

    rows, crows = [], []
    for name, spec in cfg.items():
        if name in done:
            print(f"  {name}: cached, skip")
            continue
        t0 = time.time()
        if spec["kind"] == "single":
            P = exp.joint_P_for_gamma(knn_idx, knn_dist, eff_perp,
                                      spec["gamma"], n_jobs=args.n_jobs)
            Y = exp.run_single(P, init.copy(), args.ee_iter,
                               args.total_iter - args.ee_iter, args.ee,
                               n_jobs=args.n_jobs, random_state=seed)
        else:
            stages = build_stages(spec["gammas"], spec["weights"],
                                  args.ee_iter, args.total_iter, args.ee)
            Y = exp.run_curriculum(stages, init.copy(), knn_idx, knn_dist,
                                   eff_perp, n_jobs=args.n_jobs,
                                   random_state=seed)
        scal, curves = exp.evaluate(
            Y, X, hd_nbrs, K, trust_sub, hd_rank, y=y,
            master_seed=args.master_seed,
            n_per_point=args.triplet_per_point, n_jobs=args.n_jobs)
        dt = time.time() - t0
        rows.append({"model": name, "group": spec["group"], "seed": seed,
                     "time_s": round(dt, 1), **scal})
        crows.append(pd.DataFrame({
            "model": name, "group": spec["group"], "seed": seed,
            "k": np.arange(1, K + 1),
            "no": curves["no"], "trust": curves["trust"]}))
        print(f"  {name:28s} NO@10={curves['no'][i10]:.3f}  "
              f"trust@10={curves['trust'][i10]:.3f}  "
              f"knn={scal['knn_acc_10']:.3f}  "
              f"ρ={scal['global_spearman']:.3f}  "
              f"trip={scal['triplet_acc']:.3f}  ({dt:.1f}s)")
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
# Ranking report
# =============================================================================

def tuning_report(scal_sum, curve_sum, out_dir, dataset):
    s = scal_sum.set_index("model")
    no10 = {mname: exp._auc(curve_sum, "no_mean", mname, 1, 10) for mname in s.index}
    tr10 = {mname: exp._auc(curve_sum, "trust_mean", mname, 10, 10) for mname in s.index}
    tbl = pd.DataFrame({
        "NO@1-10":  pd.Series(no10),
        "trust@10": pd.Series(tr10),
        "knn@10":   s["knn_acc_mean"],
        "spearman": s["spearman_mean"],
        "triplet":  s["triplet_mean"],
        "group":    s["group"],
    })

    # Balanced score, robust to how close the references sit: min-max normalise
    # the local axis (NO@1-10) and the global axis (spearman) across ALL configs
    # to [0, 1], then take their geometric mean.  A schedule scores high only if
    # it is near the best on BOTH axes — i.e. it escapes the local/global
    # trade-off rather than excelling at one end.
    def minmax(col):
        v = tbl[col].astype(float)
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi - lo > 1e-12 else v * 0.0 + 0.5

    tbl["local_n"] = minmax("NO@1-10")
    tbl["global_n"] = minmax("spearman")
    tbl["balanced"] = np.sqrt(tbl["local_n"].clip(lower=0) *
                              tbl["global_n"].clip(lower=0))

    refs = tbl[tbl["group"] == "reference"]
    sched = tbl[tbl["group"] != "reference"].copy().sort_values(
        "balanced", ascending=False)

    cols = ["NO@1-10", "trust@10", "knn@10", "spearman", "triplet"]
    lines = [f"# Curriculum-γ schedule search — {dataset} (single seed)\n"]
    lines.append("`balanced` = geometric mean of min-max-normalised local "
                 "(NO@1-10) and global (spearman) scores across all configs. "
                 "High only when a schedule is near-best on BOTH axes (escapes "
                 "the trade-off). `local_n`/`global_n` are the normalised "
                 "components.\n")

    lines.append("## Reference points (single fixed γ)\n")
    lines.append(refs[cols + ["local_n", "global_n", "balanced"]]
                 .round(3).to_string())

    lines.append("\n## Schedules ranked by balanced score\n")
    lines.append(sched[cols + ["local_n", "global_n", "balanced"]]
                 .round(3).to_string())

    def best_of(metric):
        r = sched.sort_values(metric, ascending=False).iloc[0]
        return r.name, r[metric]

    lines.append("\n## Winners\n")
    top = sched.iloc[0]
    lines.append(f"- **Best balanced schedule: `{top.name}`** "
                 f"(balanced={top['balanced']:.3f}; local_n={top['local_n']:.2f}, "
                 f"global_n={top['global_n']:.2f}).")
    for label, metric in [("local (NO@1-10)", "NO@1-10"),
                          ("global (spearman)", "spearman"),
                          ("fair-local (knn@10)", "knn@10"),
                          ("global (triplet)", "triplet")]:
        nm, vl = best_of(metric)
        lines.append(f"- Best {label}: `{nm}` = {vl:.4f}")

    report = "\n".join(lines)
    path = os.path.join(out_dir, f"{dataset}_TUNING.md")
    with open(path, "w") as f:
        f.write(report + "\n")
    print("\n" + "=" * 70 + "\n" + report + "\n" + "=" * 70)
    print(f"\nTuning report → {path}")


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "mouse", "adult"])
    ap.add_argument("--mouse_pickle", default=None)
    ap.add_argument("--n_samples", type=int, default=5000,
                    help="subsample size (0 = all); smaller keeps the search fast")
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42,
                    help="single optimization/init seed for the search")
    ap.add_argument("--master_seed", type=int, default=42,
                    help="seed for data subsample + metric subsamples (fixed)")
    ap.add_argument("--n_jobs", type=int, default=-1)

    # iteration budget (shared by every schedule)
    ap.add_argument("--total_iter", type=int, default=750)
    ap.add_argument("--ee_iter", type=int, default=200,
                    help="early-exaggeration iters at the start (within stage 1)")
    ap.add_argument("--ee", type=float, default=12.0)

    # evaluation knobs (mirror exp.py)
    ap.add_argument("--k_max", type=int, default=200)
    ap.add_argument("--trust_subsample", type=int, default=3000)
    ap.add_argument("--triplet_per_point", type=int, default=5)

    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test: tiny subsample, fewer iters")
    args = ap.parse_args()

    if args.quick:
        args.n_samples = min(args.n_samples, 2000) if args.n_samples else 2000
        args.total_iter = 400
        args.ee_iter = 100
        args.trust_subsample = min(args.trust_subsample, 1500)

    out_dir = args.out_dir or os.path.join(
        exp._SCRIPT_DIR, "results", f"{args.dataset}_curriculum_tuning")
    os.makedirs(out_dir, exist_ok=True)

    if args.plot_only:
        csv_path = os.path.join(out_dir, "results.csv")
        curve_path = os.path.join(out_dir, "curves.csv")
        if not (os.path.exists(csv_path) and os.path.exists(curve_path)):
            raise SystemExit(f"Need results.csv + curves.csv in {out_dir}; "
                             "run without --plot_only first.")
        df = pd.read_csv(csv_path)
        cdf = pd.read_csv(curve_path)
    else:
        df, cdf = run_search(args, out_dir)

    scal_sum = exp.summarize_scalars(df)
    curve_sum = exp.summarize_curves(cdf)
    scal_sum.to_csv(os.path.join(out_dir, "summary_scalars.csv"), index=False)
    curve_sum.to_csv(os.path.join(out_dir, "summary_curves.csv"), index=False)
    exp.make_plots(scal_sum, curve_sum, out_dir, args.dataset)
    tuning_report(scal_sum, curve_sum, out_dir, args.dataset)


if __name__ == "__main__":
    main()
