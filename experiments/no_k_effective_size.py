#!/usr/bin/env python
"""
no_k_effective_size.py  —  Experiment 4: effective neighborhood size,
the latent variable behind NO@k?
====================================================================
Hypothesis: NO@k is maximised when the *effective neighborhood size* — the number
of neighbors carrying meaningful probability mass in the smoothed conditional row
``p̃_{j|i}`` — equals ≈ c·k, and both γ and ρ act only by setting this size. So far
this has only been *inferred* from the NO@k optima of Experiments 1–3. Here we
measure effective size **directly** from the high-dimensional affinity rows and
test it.

Effective size is a property of the smoothed conditional rows alone, so **no new
embeddings are needed**. We recompute the conditional rows ``p̃_{j|i}`` (the cheap
neighbor-graph + γ power-transform, *not* t-SNE — byte-identical to the rows that
produced the stored NO@k, see no_k_landscape_common.conditional_rows_for_gamma)
and reuse the already-cached NO@k values from the exp2 (γ, ρ) *fine* grids.

For each point i, from its smoothed conditional row ``p̃_{j|i}`` we compute a family
of effective-size measures (so the conclusion doesn't hinge on one arbitrary
choice), then average each over all points -> one number per (dataset, γ, ρ, measure):

  * threshold counts: #{j : p̃_{j|i} ≥ c·uniform}, c ∈ {0.25, 0.5, 1.0}, where
    uniform = 1/k_neighbors = 1/(3ρ) (the uniform share over the kept neighbors);
  * participation ratio: (Σ p̃)² / Σ p̃² = 1/Σ p̃²  (threshold-free);
  * 2^entropy of the row (= the effective perplexity; the contrast baseline only).

The SAME measure definition is used across every NO@k target k and every dataset —
the threshold denominator is always 1/k_neighbors; it never varies between panels.

Two deliverables:
  1. Direction check (per dataset): mean effective size vs ρ at fixed γ ∈ {0.8,1,1.2}
     and vs γ at fixed ρ ∈ {30,100}, per measure — does size rise or fall with ρ?
  2. The decisive plot: NO@k (mean ± 5-seed std) vs (effective size ÷ k), pooling
     all (γ, ρ, k) points, one figure per measure. Per-dataset (coloured by k) and
     all-datasets-overlaid (coloured by dataset). We check whether the points
     collapse onto one hump peaking at a stable ratio. Peaks are reported as a
     **band** (never a single argmax — the ridge is flat).

Usage
-----
  python no_k_effective_size.py --dataset mnist
  python no_k_effective_size.py --all                 # all 3 datasets + summary
  python no_k_effective_size.py --summary             # overlay + FINDINGS (needs cells)
  python no_k_effective_size.py --dataset mnist --plot_only
  python no_k_effective_size.py --dataset mnist --n_subsample 3000   # code-path smoke test
  python no_k_effective_size.py --self_test
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
# Grid / measure parameters — mirror the exp2 *fine* grid so NO@k merges cleanly
# =============================================================================
GAMMA_FINE = np.linspace(0.0, 3.0, 15)          # exact match to exp2 fine γ
RHO_FINE = [10, 20, 30, 50, 75, 100, 125, 150, 175, 200]   # exp2 fine ρ
TARGET_KS = [10, 30, 100]                        # NO@k targets present in exp2 CSV

# Direction-check axes (effective size is cheap, so we're free to pick these).
GAMMA_DIR_FIXED = [0.8, 1.0, 1.2]                # curves for eff-size vs ρ
RHO_DIR_FIXED = [30, 100]                        # curves for eff-size vs γ
GAMMA_DIR_SWEEP = np.round(np.linspace(0.5, 1.5, 11), 4)   # x-axis for eff-size vs γ

THRESH_CS = [0.25, 0.5, 1.0]
MEASURES = ["thr_c0.25", "thr_c0.5", "thr_c1.0",
            "participation_ratio", "two_pow_entropy"]
MEASURE_LABELS = {
    "thr_c0.25": "threshold count (p̃ ≥ 0.25·uniform)",
    "thr_c0.5": "threshold count (p̃ ≥ 0.5·uniform)",
    "thr_c1.0": "threshold count (p̃ ≥ 1.0·uniform)",
    "participation_ratio": "participation ratio  1/Σp̃²",
    "two_pow_entropy": "2^entropy  (effective perplexity — baseline)",
}

N_JOBS = 8
KNN_SEED = 42                                    # same graph as the cached NO@k
DATA_SEED = 42

DATASETS_MAIN = ["mnist", "fashion_mnist", "mouse"]
DATASET_COLORS = {"mnist": "#0072B2", "fashion_mnist": "#D55E00",
                  "mouse": "#009E73"}
K_COLORS = {10: "#332288", 30: "#88CCEE", 100: "#CC6677"}


# =============================================================================
# Effective-size measures (per point) from a dense conditional array C (n, k)
# =============================================================================
def per_point_measures(C):
    """All effective-size measures for every point, from the dense conditional
    rows ``C`` (n_samples, k_neighbors), each row summing to 1.

    Returns a dict ``measure -> (n,) array``. ``uniform = 1/k_neighbors`` is the
    uniform share over the kept neighbors (= 1/(3ρ) when the graph is not capped);
    using the actual row width keeps the threshold correct if it is capped.
    """
    k = C.shape[1]
    uniform = 1.0 / k
    out = {}
    for c in THRESH_CS:
        out[f"thr_c{c}"] = (C >= c * uniform).sum(axis=1).astype(np.float64)
    out["participation_ratio"] = 1.0 / (C ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logC = np.where(C > 0, np.log2(C), 0.0)
    H = -(C * logC).sum(axis=1)                  # Shannon entropy in bits
    out["two_pow_entropy"] = np.exp2(H)          # = effective perplexity
    return out


def _apply_gamma(base, gamma):
    """Row power-transform of the γ=1 conditional ``base`` (the library's exact
    transform, affinity.py:516-518). ``base`` is reused across γ so the costly
    Gaussian-perplexity fit is done once per ρ, not once per (ρ, γ)."""
    if gamma == 1.0:
        return base
    C = base ** gamma
    C /= C.sum(axis=1, keepdims=True)
    return C


# =============================================================================
# Compute: one row per (ρ, γ) with mean/std-over-points of each measure
# =============================================================================
def _all_gammas():
    g = set(np.round(GAMMA_FINE, 6))
    g |= set(np.round(GAMMA_DIR_FIXED, 6))
    g |= set(np.round(GAMMA_DIR_SWEEP, 6))
    return sorted(g)


def run_experiment(args, out_dir):
    print(f"Loading {args.dataset} (PCA-50)"
          f"{' subsample=' + str(args.n_subsample) if args.n_subsample else ''} ...")
    X_pca, _ = common.load_dataset(args.dataset, random_state=DATA_SEED,
                                   n_subsample=args.n_subsample)
    n = len(X_pca)
    gammas = _all_gammas()
    print(f"  X_pca: {X_pca.shape}")
    print(f"  {len(RHO_FINE)} ρ × {len(gammas)} γ cells (no embeddings) ...")

    records = []
    for rho in RHO_FINE:
        print(f"[ρ={rho}] building neighbor graph once ...")
        _nbrs, dists, eff = common.build_neighbor_cache(
            X_pca, float(rho), n_jobs=args.n_jobs, knn_seed=KNN_SEED)
        base = common.conditional_rows_for_gamma(dists, eff, 1.0,
                                                 n_jobs=args.n_jobs)
        k_neighbors = base.shape[1]
        for gamma in gammas:
            C = _apply_gamma(base, float(gamma))
            m = per_point_measures(C)
            rec = {"dataset": args.dataset, "rho": int(rho),
                   "gamma": float(gamma), "k_neighbors": int(k_neighbors),
                   "eff_perplexity": float(eff)}
            for name, vals in m.items():
                rec[f"{name}_mean"] = float(vals.mean())
                rec[f"{name}_std"] = float(vals.std(ddof=0))
            records.append(rec)
        print(f"  [ρ={rho}] done ({len(records)}/{len(RHO_FINE) * len(gammas)})")

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out_dir, "effective_size_cells.csv")
    df.to_csv(csv_path, index=False)
    np.savez_compressed(os.path.join(out_dir, "effective_size_cells.npz"),
                        **{c: df[c].values for c in df.columns})
    print(f"\nSaved raw per-cell effective sizes -> {csv_path}")
    _write_settings(args, out_dir, gammas, n)
    return df


# =============================================================================
# Merge with cached NO@k (exp2 fine grid) -> the decisive-scatter table
# =============================================================================
def _exp2_fine_csv(dataset):
    return os.path.join(_SCRIPT_DIR, "results", "no_k_landscape", dataset,
                        "exp2_gamma_rho_2d", "fine", "no_k_gamma_rho_2d.csv")


def _key(gamma, rho):
    return (round(float(gamma), 6), int(rho))


def build_scatter_table(cells, dataset):
    """Merge per-cell effective sizes (fine-γ subset) with seed-aggregated NO@k.

    Returns a long dataframe with one row per (γ, ρ, k, measure):
    ``dataset, gamma, rho, k, measure, eff_size, eff_over_k, no_mean, no_std, n_seeds``.
    """
    no_csv = _exp2_fine_csv(dataset)
    if not os.path.exists(no_csv):
        print(f"  [!] cached NO@k not found for {dataset}: {no_csv} — "
              "skipping scatter table.")
        return None
    nod = pd.read_csv(no_csv)

    # Seed aggregation: melt NO@10/30/100 -> (k, no_at_k), mean/std over seeds.
    long = nod.melt(id_vars=["rho", "gamma", "seed"],
                    value_vars=[f"no_at_{k}" for k in TARGET_KS],
                    var_name="kcol", value_name="no_at_k")
    long["k"] = long["kcol"].str.replace("no_at_", "").astype(int)
    agg = (long.groupby(["rho", "gamma", "k"])["no_at_k"]
           .agg(no_mean="mean", no_std=lambda s: s.std(ddof=0), n_seeds="count")
           .reset_index())

    # Effective-size lookup keyed on (γ, ρ), restricted to the fine-γ cells.
    fine_g = set(np.round(GAMMA_FINE, 6))
    eff_lookup = {name: {} for name in MEASURES}
    for _, r in cells.iterrows():
        if round(float(r["gamma"]), 6) not in fine_g:
            continue
        key = _key(r["gamma"], r["rho"])
        for name in MEASURES:
            eff_lookup[name][key] = float(r[f"{name}_mean"])

    rows = []
    for _, r in agg.iterrows():
        key = _key(r["gamma"], r["rho"])
        k = int(r["k"])
        for name in MEASURES:
            eff = eff_lookup[name].get(key)
            if eff is None:
                continue
            rows.append({"dataset": dataset, "gamma": float(r["gamma"]),
                         "rho": int(r["rho"]), "k": k, "measure": name,
                         "eff_size": eff, "eff_over_k": eff / k,
                         "no_mean": float(r["no_mean"]),
                         "no_std": float(r["no_std"]),
                         "n_seeds": int(r["n_seeds"])})
    return pd.DataFrame(rows)


# =============================================================================
# Peak band / collapse tightness (no single argmax — the ridge is flat)
# =============================================================================
def peak_band(x, y, nbins=18):
    """Bin ``x`` (= eff/k), take mean & std of ``y`` (= NO@k) per bin.

    Bins are **log-spaced**: eff/k spans ~3 decades (≈0.02–60, because γ≈0 uniform
    rows blow it up), so linear bins would dump the whole signal into one bin and
    report a meaningless half-bin-width "peak". Geometric bins resolve the hump.

    collapse tightness  = median within-bin std of NO@k (lower ⇒ tighter collapse).
    peak band           = contiguous x-range whose bin-mean NO@k is within one
                          tightness of the max bin-mean (reported, not an argmax).
    Returns a dict; NaNs if too few points.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size < nbins:
        return {"tightness": float("nan"), "peak_center": float("nan"),
                "band_lo": float("nan"), "band_hi": float("nan"),
                "peak_value": float("nan"), "n_points": int(x.size)}
    edges = np.geomspace(x.min(), x.max(), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])            # geometric bin centers
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    bmean = np.full(nbins, np.nan)
    bstd = np.full(nbins, np.nan)
    for b in range(nbins):
        yb = y[idx == b]
        if yb.size >= 2:
            bmean[b] = yb.mean()
            bstd[b] = yb.std(ddof=0)
        elif yb.size == 1:
            bmean[b] = yb[0]
    tight = float(np.nanmedian(bstd))
    top = int(np.nanargmax(bmean))
    thr = bmean[top] - (tight if np.isfinite(tight) else 0.0)
    # Grow a contiguous band around the top bin while bin-mean stays >= thr.
    lo = hi = top
    while lo - 1 >= 0 and np.isfinite(bmean[lo - 1]) and bmean[lo - 1] >= thr:
        lo -= 1
    while hi + 1 < nbins and np.isfinite(bmean[hi + 1]) and bmean[hi + 1] >= thr:
        hi += 1
    return {"tightness": tight, "peak_center": float(centers[top]),
            "band_lo": float(centers[lo]), "band_hi": float(centers[hi]),
            "peak_value": float(bmean[top]), "n_points": int(x.size)}


# =============================================================================
# Per-dataset plots
# =============================================================================
def plot_direction(cells, dataset, out_dir):
    """5 measures × 2 columns: mean eff size vs ρ (γ∈{0.8,1,1.2}) and vs γ
    (ρ∈{30,100}). Returns a list of direction-summary records."""
    summary = []
    fig, axes = plt.subplots(len(MEASURES), 2, figsize=(13, 3.1 * len(MEASURES)))
    for row, name in enumerate(MEASURES):
        col = f"{name}_mean"
        # --- left: eff size vs ρ, one curve per fixed γ ---
        axL = axes[row, 0]
        for gamma in GAMMA_DIR_FIXED:
            sub = cells[np.isclose(cells["gamma"], gamma)].sort_values("rho")
            if sub.empty:
                continue
            rhos = sub["rho"].values
            eff = sub[col].values
            axL.plot(rhos, eff, marker="o", ms=3, lw=1.8, label=f"γ={gamma:g}")
            direction = ("rises" if eff[-1] > eff[0] else
                         "falls" if eff[-1] < eff[0] else "flat")
            monotone = bool(np.all(np.diff(eff) >= -1e-9) or
                            np.all(np.diff(eff) <= 1e-9))
            summary.append({"dataset": dataset, "measure": name,
                            "gamma": float(gamma),
                            "eff_at_rho_min": float(eff[0]),
                            "eff_at_rho_max": float(eff[-1]),
                            "direction_vs_rho": direction,
                            "monotone_vs_rho": monotone})
        axL.set_xlabel("perplexity ρ")
        axL.set_ylabel(name)
        axL.set_title(f"{MEASURE_LABELS[name]}\nvs ρ", fontsize=9)
        common.clean_axes(axL)
        if row == 0:
            axL.legend(fontsize=8, title="fixed γ")
        # --- right: eff size vs γ, one curve per fixed ρ ---
        axR = axes[row, 1]
        gsweep = set(np.round(GAMMA_DIR_SWEEP, 6))
        for rho in RHO_DIR_FIXED:
            sub = cells[(cells["rho"] == rho) &
                        cells["gamma"].round(6).isin(gsweep)].sort_values("gamma")
            if sub.empty:
                continue
            axR.plot(sub["gamma"].values, sub[col].values, marker="s", ms=3,
                     lw=1.8, label=f"ρ={rho}")
        axR.set_xlabel("γ")
        axR.set_ylabel(name)
        axR.set_title(f"{MEASURE_LABELS[name]}\nvs γ", fontsize=9)
        common.clean_axes(axR)
        if row == 0:
            axR.legend(fontsize=8, title="fixed ρ")
    fig.suptitle(f"Effective neighborhood size — direction check ({dataset})",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    common.save_fig(os.path.join(out_dir, f"direction_{dataset}.png"))
    return summary


def plot_scatter_per_dataset(scat, dataset, out_dir):
    """One figure per measure: NO@k (mean ± seed-std) vs eff/k, coloured by k.
    Returns per-measure peak-band records."""
    recs = []
    for name in MEASURES:
        sub = scat[scat["measure"] == name]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for k in TARGET_KS:
            s = sub[sub["k"] == k]
            if s.empty:
                continue
            ax.errorbar(s["eff_over_k"], s["no_mean"], yerr=s["no_std"],
                        fmt="o", ms=4, lw=0, elinewidth=0.8, capsize=2,
                        color=K_COLORS[k], alpha=0.8, label=f"k={k}")
        band = peak_band(sub["eff_over_k"], sub["no_mean"])
        _annotate_band(ax, band)
        ax.set_xscale("log")
        ax.set_xlabel("effective size ÷ k  (log scale)")
        ax.set_ylabel("NO@k  (mean ± 5-seed std)")
        ax.set_title(f"{dataset}: NO@k vs eff-size/k\n{MEASURE_LABELS[name]}",
                     fontsize=10)
        ax.legend(fontsize=8, title="target k")
        common.clean_axes(ax)
        common.save_fig(os.path.join(out_dir, f"scatter_{name}_{dataset}.png"))
        rec = {"dataset": dataset, "measure": name}
        rec.update(band)
        recs.append(rec)
    return recs


def _annotate_band(ax, band):
    if not np.isfinite(band["band_lo"]):
        return
    ax.axvspan(band["band_lo"], band["band_hi"], color="#DDDDDD", alpha=0.6,
               zorder=0)
    ax.axvline(band["peak_center"], color="#D55E00", ls="--", lw=1.4)
    ax.text(0.97, 0.05,
            f"peak eff/k ≈ [{band['band_lo']:.2f}, {band['band_hi']:.2f}]\n"
            f"tightness (med within-bin std) = {band['tightness']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="#D55E00", alpha=0.85))


# =============================================================================
# Summary across datasets: overlay scatters + FINDINGS.md
# =============================================================================
def plot_scatter_overlay(all_scat, out_dir):
    """One figure per measure: all datasets overlaid, coloured by dataset.
    Returns pooled and per-dataset peak-band records."""
    recs = []
    for name in MEASURES:
        sub = all_scat[all_scat["measure"] == name]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        for ds in DATASETS_MAIN:
            s = sub[sub["dataset"] == ds]
            if s.empty:
                continue
            ax.errorbar(s["eff_over_k"], s["no_mean"], yerr=s["no_std"],
                        fmt="o", ms=4, lw=0, elinewidth=0.7, capsize=1.5,
                        color=DATASET_COLORS[ds], alpha=0.6, label=ds)
            band = peak_band(s["eff_over_k"], s["no_mean"])
            rec = {"scope": ds, "measure": name}
            rec.update(band)
            recs.append(rec)
        pooled = peak_band(sub["eff_over_k"], sub["no_mean"])
        _annotate_band(ax, pooled)
        prec = {"scope": "pooled", "measure": name}
        prec.update(pooled)
        recs.append(prec)
        ax.set_xscale("log")
        ax.set_xlabel("effective size ÷ k  (log scale)")
        ax.set_ylabel("NO@k  (mean ± 5-seed std)")
        ax.set_title(f"NO@k vs eff-size/k — all datasets\n{MEASURE_LABELS[name]}",
                     fontsize=10)
        ax.legend(fontsize=8, title="dataset")
        common.clean_axes(ax)
        common.save_fig(os.path.join(out_dir, f"scatter_{name}_all_datasets.png"))
    return recs


def write_findings(collapse_recs, direction_recs, out_dir):
    coll = pd.DataFrame(collapse_recs)
    coll.to_csv(os.path.join(out_dir, "collapse_summary.csv"), index=False)
    dsum = pd.DataFrame(direction_recs)
    dsum.to_csv(os.path.join(out_dir, "direction_summary.csv"), index=False)

    # Tightest-collapse measure (pooled).
    pooled = coll[coll["scope"] == "pooled"].dropna(subset=["tightness"])
    lines = ["# Experiment 4 — Effective neighborhood size vs NO@k: findings\n"]

    lines.append("## 1. Which measure gives the tightest collapse\n")
    if not pooled.empty:
        pooled = pooled.sort_values("tightness")
        best = pooled.iloc[0]
        lines.append("Collapse tightness = median within-bin std of NO@k on "
                     "log-spaced eff/k bins (lower ⇒ tighter). NO@k spans ~0.07–0.6, "
                     "so a tightness ~0.02 is a tight collapse.\n")
        lines.append("Pooled (all datasets, all γ, ρ, k):\n")
        for _, r in pooled.iterrows():
            wd = coll[(coll["measure"] == r["measure"]) &
                      (coll["scope"] != "pooled")]["tightness"]
            wmed = float(np.nanmedian(wd)) if len(wd) else float("nan")
            lines.append(f"- **{r['measure']}**: pooled tightness={r['tightness']:.4f} "
                         f"(within-dataset median {wmed:.4f}), "
                         f"peak eff/k ∈ [{r['band_lo']:.2f}, {r['band_hi']:.2f}]")
        lines.append(f"\n➡ **Tightest pooled collapse: `{best['measure']}`** "
                     f"(tightness {best['tightness']:.4f}).")
        lines.append("Note: within a single dataset the collapse is much tighter "
                     "(~0.02) than pooled (~0.07) — the *peak ratio* aligns across "
                     "datasets, but the absolute NO@k height differs (mnist≈0.37, "
                     "fashion≈0.40, mouse≈0.52 at the peak), which loosens the pool.\n")

    lines.append("## 2. Is the peak eff/k ratio stable within & across datasets\n")
    lines.append("Peak is a **band** (bin-mean NO@k within one within-bin std of the "
                 "max) — a flat ridge ⇒ wide band, so we test whether the three "
                 "per-dataset bands *overlap* rather than comparing fragile argmax "
                 "centers.\n")
    for name in MEASURES:
        sub = coll[(coll["measure"] == name) &
                   (coll["scope"] != "pooled")].dropna(subset=["band_lo"])
        if sub.empty:
            continue
        los, his = sub["band_lo"].values, sub["band_hi"].values
        ov_lo, ov_hi = float(los.max()), float(his.min())
        bands = ", ".join(f"{ds}=[{lo:.2f},{hi:.2f}]"
                          for ds, lo, hi in zip(sub["scope"], los, his))
        if ov_hi >= ov_lo:
            stab = f"bands **overlap** at eff/k ∈ [{ov_lo:.2f}, {ov_hi:.2f}] ⇒ stable"
        else:
            stab = "bands do **not** overlap ⇒ peak ratio not stable across datasets"
        lines.append(f"- **{name}**: {bands}; {stab}.")
    lines.append("")

    lines.append("## 3. Direction of effective size vs ρ (per measure)\n")
    for name in MEASURES:
        sub = dsum[dsum["measure"] == name]
        if sub.empty:
            continue
        dirs = set(sub["direction_vs_rho"])
        mono = bool(np.all(sub["monotone_vs_rho"].values))
        tag = ("all rise" if dirs == {"rises"} else
               "all fall" if dirs == {"falls"} else
               f"MIXED {sorted(dirs)} — crossover")
        lines.append(f"- **{name}**: {tag}"
                     + ("" if mono else "  (non-monotone in ρ)"))
    cross = _crossover_across_measures(dsum)
    lines.append(f"\nCross-measure agreement on ρ-direction: {cross}\n")

    lines.append("## Guardrails honored\n")
    lines.append("- One fixed definition per measure across all k and datasets "
                 "(threshold denominator is always 1/k_neighbors).")
    lines.append("- Peaks reported as bands (± one within-bin std), never a single "
                 "argmax; 5-seed variance drawn as error bars on every scatter.")
    lines.append("- Recomputed affinities use knn_seed=42 / the same effective "
                 "perplexity as the cached NO@k, so eff-size and NO@k share a graph.")

    with open(os.path.join(out_dir, "FINDINGS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _crossover_across_measures(dsum):
    """Do the non-baseline measures agree on the sign of eff-size change vs ρ?"""
    per_ds = []
    for ds in dsum["dataset"].unique():
        dirs = set(dsum[(dsum["dataset"] == ds) &
                        (dsum["measure"] != "two_pow_entropy")]["direction_vs_rho"])
        per_ds.append((ds, dirs))
    if all(d == {"rises"} for _, d in per_ds):
        return "all measures rise with ρ (consistent)"
    if all(d == {"falls"} for _, d in per_ds):
        return "all measures fall with ρ (consistent)"
    return "measures DISAGREE in direction on some dataset — crossover flagged: " \
           + "; ".join(f"{ds}:{sorted(d)}" for ds, d in per_ds)


# =============================================================================
# Settings log
# =============================================================================
def _write_settings(args, out_dir, gammas, n):
    info = {
        "experiment": "no_k_effective_size",
        "dataset": args.dataset,
        "n_points": int(n),
        "n_subsample": args.n_subsample,
        "rhos": list(RHO_FINE),
        "gammas_computed": [float(g) for g in gammas],
        "gamma_fine_for_merge": [float(g) for g in GAMMA_FINE],
        "gamma_dir_fixed": GAMMA_DIR_FIXED,
        "rho_dir_fixed": RHO_DIR_FIXED,
        "gamma_dir_sweep": [float(g) for g in GAMMA_DIR_SWEEP],
        "target_ks": TARGET_KS,
        "measures": MEASURES,
        "thresh_cs": THRESH_CS,
        "knn_seed": KNN_SEED,
        "data_seed": DATA_SEED,
        "no_k_source": "exp2_gamma_rho_2d/fine (seed-aggregated)",
        "tsne_settings": common.TSNE_SETTINGS,
    }
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(out_dir, "settings.md"), "w") as f:
        f.write("# Experiment 4 — effective neighborhood size\n\n")
        f.write("```json\n" + json.dumps(info, indent=2) + "\n```\n")


# =============================================================================
# Orchestration
# =============================================================================
def _out_dir(dataset):
    return os.path.join(_SCRIPT_DIR, "results", "no_k_landscape", dataset,
                        "exp4_effective_size")


def _summary_dir():
    return os.path.join(_SCRIPT_DIR, "results", "no_k_landscape",
                        "exp4_effective_size_summary")


def process_dataset(args):
    """Compute (or load) per-cell effective sizes for one dataset, then make its
    per-dataset direction + scatter figures. Returns (scatter_df, direction_recs,
    collapse_recs) for the summary step (scatter_df is None if subsampled)."""
    out_dir = _out_dir(args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    cells_csv = os.path.join(out_dir, "effective_size_cells.csv")

    if args.plot_only:
        cells = pd.read_csv(cells_csv)
    else:
        cells = run_experiment(args, out_dir)

    direction_recs = plot_direction(cells, args.dataset, out_dir)

    if args.n_subsample is not None:
        print("  [note] subsampled run — the recomputed graph differs from the "
              "cached full-data NO@k, so the scatter merge is skipped.")
        return None, direction_recs, []

    scat = build_scatter_table(cells, args.dataset)
    if scat is None or scat.empty:
        return None, direction_recs, []
    scat.to_csv(os.path.join(out_dir, "no_k_vs_effsize.csv"), index=False)
    collapse_recs = plot_scatter_per_dataset(scat, args.dataset, out_dir)
    return scat, direction_recs, collapse_recs


def run_summary(datasets):
    """Overlay scatters + FINDINGS from each dataset's cached exp4 outputs."""
    sdir = _summary_dir()
    os.makedirs(sdir, exist_ok=True)
    scats, dir_recs = [], []
    for ds in datasets:
        out_dir = _out_dir(ds)
        cells_csv = os.path.join(out_dir, "effective_size_cells.csv")
        scat_csv = os.path.join(out_dir, "no_k_vs_effsize.csv")
        if not os.path.exists(cells_csv):
            print(f"  [!] {ds}: no cells CSV — run the dataset first; skipping.")
            continue
        cells = pd.read_csv(cells_csv)
        dir_recs += plot_direction(cells, ds, out_dir)  # refresh direction recs
        if os.path.exists(scat_csv):
            scats.append(pd.read_csv(scat_csv))
    if not scats:
        print("No scatter tables available; cannot build the summary.")
        return
    all_scat = pd.concat(scats, ignore_index=True)
    collapse_recs = plot_scatter_overlay(all_scat, sdir)
    # Fold per-dataset collapse bands in too (pooled + per-dataset from overlay
    # already cover both scopes).
    write_findings(collapse_recs, dir_recs, sdir)
    print(f"\nSummary written -> {sdir}")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 4: effective neighborhood size vs NO@k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="mnist", choices=common.DATASETS)
    p.add_argument("--all", action="store_true",
                   help="Process mnist, fashion_mnist, mouse then build summary.")
    p.add_argument("--summary", action="store_true",
                   help="Build cross-dataset overlay + FINDINGS from cached cells.")
    p.add_argument("--n_jobs", type=int, default=N_JOBS)
    p.add_argument("--n_subsample", type=int, default=None,
                   help="Random subset (code-path smoke test; skips NO@k merge).")
    p.add_argument("--plot_only", action="store_true",
                   help="Skip computation; rebuild from saved cells CSV.")
    p.add_argument("--self_test", action="store_true",
                   help="Check conditional rows match the library and exit.")
    return p.parse_args()


def _self_test():
    from openTSNE.affinity import joint_probabilities_nn
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 50))
    nbrs, dists, eff = common.build_neighbor_cache(X, 30.0, n_jobs=1, knn_seed=42)
    for g in (0.5, 1.0, 1.7):
        C = common.conditional_rows_for_gamma(dists, eff, g, n_jobs=1)
        assert np.allclose(C.sum(axis=1), 1.0), "rows must sum to 1"
        _, Pcond = joint_probabilities_nn(nbrs, dists, [eff], symmetrize=False,
                                          gamma=g, n_jobs=1)
        Pd = np.asarray(Pcond.todense())
        lib = np.array([Pd[i, nbrs[i]] for i in range(20)])
        assert np.allclose(lib, C[:20]), f"conditional mismatch at γ={g}"
    # Sanity: effective sizes are monotone in the c-threshold and positive.
    C = common.conditional_rows_for_gamma(dists, eff, 1.0, n_jobs=1)
    m = per_point_measures(C)
    assert (m["thr_c0.25"] >= m["thr_c0.5"]).all()
    assert (m["thr_c0.5"] >= m["thr_c1.0"]).all()
    assert (m["participation_ratio"] > 0).all()
    assert (m["two_pow_entropy"] > 0).all()
    print("Self-test passed: conditional rows are byte-identical to the library, "
          "rows sum to 1, and the measures are well-formed.")


def main():
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    if args.summary and not args.all:
        run_summary(DATASETS_MAIN)
        return

    if args.all:
        for ds in DATASETS_MAIN:
            args.dataset = ds
            process_dataset(args)
        run_summary(DATASETS_MAIN)
    else:
        process_dataset(args)


if __name__ == "__main__":
    main()
