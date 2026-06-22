"""Compute the across-seed standard deviation for every multi-seed experiment.

Each result CSV that contains a `seed` column was produced by re-running the
same configuration over several random seeds. This script groups every CSV by
its parameter columns (everything that is *not* the seed and *not* a metric),
computes the standard deviation of each metric across the seeds, and writes a
tidy summary table.

Run from the `experiments/` directory:

    python seed_variation.py
"""

import glob
import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_PATH = os.path.join(RESULTS_DIR, "seed_variation_summary.csv")

# Columns that describe the configuration of a run (not a measured metric).
PARAM_COLS = {"seed", "perplexity", "gamma", "k", "method", "rank"}


def summarise(csv_path):
    df = pd.read_csv(csv_path)
    if "seed" not in df.columns:
        return None  # not a multi-seed experiment

    n_seeds = df["seed"].nunique()
    if n_seeds < 2:
        return None  # nothing to vary over

    # Group by the parameter columns present (minus the seed itself); the
    # remaining numeric columns are the metrics whose seed-variation we want.
    group_cols = [c for c in df.columns if c in PARAM_COLS and c != "seed"]
    metric_cols = [
        c
        for c in df.columns
        if c not in PARAM_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        return None

    if group_cols:
        # std across seeds within each configuration, then averaged over configs
        per_config_std = df.groupby(group_cols)[metric_cols].std(ddof=1)
    else:
        per_config_std = df[metric_cols].std(ddof=1).to_frame().T

    rel = os.path.relpath(csv_path, RESULTS_DIR)
    dataset = rel.split(os.sep)[0]
    experiment = os.path.dirname(rel)

    rows = []
    for metric in metric_cols:
        std_vals = per_config_std[metric].dropna()
        rows.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "metric": metric,
                "n_seeds": n_seeds,
                "n_configs": len(per_config_std),
                "mean_std_across_seeds": std_vals.mean(),
                "max_std_across_seeds": std_vals.max(),
            }
        )
    return pd.DataFrame(rows)


def main():
    pieces = []
    for csv_path in sorted(glob.glob(os.path.join(RESULTS_DIR, "**", "*.csv"), recursive=True)):
        out = summarise(csv_path)
        if out is not None:
            pieces.append(out)

    summary = pd.concat(pieces, ignore_index=True)
    summary = summary.sort_values(["dataset", "experiment", "metric"]).reset_index(drop=True)
    summary.to_csv(OUT_PATH, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print(summary.to_string(index=False))
    print(f"\nWrote {len(summary)} rows to {os.path.relpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
