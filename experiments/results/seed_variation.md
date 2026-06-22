# Variation across random seeds

Every experiment that carries a `seed` column was re-run over multiple random
seeds (five seeds for all experiments, ten for the MNIST global-Spearman run).
For each one we group the rows by their configuration (perplexity, gamma, k, …)
and compute the standard deviation of every metric across the seeds; the
per-configuration spread is then averaged and the worst case reported. The full
table is in [`seed_variation_summary.csv`](seed_variation_summary.csv), produced
by [`../seed_variation.py`](../seed_variation.py).

The headline result is that the embeddings are remarkably stable to the choice
of seed. For all neighborhood-overlap–based metrics — the gamma sweep
(exp4), the standard/smooth/sharp comparison (exp7), the smooth-vs-envelope
curves (exp9), the NO/AUC-vs-perplexity scores (exp10) and the AUC columns of
the sensitivity grid (exp6) — the across-seed standard deviation is typically
between 0.0002 and 0.0012 on a 0–1 scale, i.e. well under one part in a
thousand, with worst-case spreads still under ~0.003. In practice this means
the neighborhood-preservation scores reported in the plots are essentially
reproducible: re-running with a different seed moves them by far less than the
differences we are comparing between gamma settings or perplexities. The
`median_eff_perp` column has exactly zero variation, as expected, since the
effective perplexity is a property of the high-dimensional affinities and does
not depend on the embedding's random initialization. The single most
seed-sensitive quantity is the global Spearman correlation (exp8): its mean
across-seed std is roughly 0.003 (adult), 0.009 (mnist) and 0.007–0.009
(mouse), and individual configurations reach up to ~0.02–0.06. This is still a
small absolute spread, but it confirms that global-structure preservation is
the metric most affected by random initialization, so its trends should be read
as means over seeds rather than from any single run.
