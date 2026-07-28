# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research fork of [openTSNE](https://github.com/pavlin-policar/openTSNE). The library was modified to add **one new affinity parameter, `gamma`** (a row-wise power transform on the high-dimensional neighbor distribution before symmetrization), plus a heavy-tailed `StudentTNN` ("tt-SNE") affinity. Everything else in the repo is experiment code comparing this "smooth/sharp" affinity family against standard t-SNE and external DR methods. `README.md` is the fork's readme; `README.rst` is the original upstream readme kept for reference.

The library and the research code are deliberately separate:
- `openTSNE/` — the compiled library. **Only `affinity.py` was modified** relative to upstream.
- `experiments/`, `improvement/`, `benchmarks/` — independent research drivers that import the installed library.

## Build, test, lint

```bash
pip install -e .                 # editable install; compiles the Cython extensions (required after touching any .pyx)
pip install numpy pandas scipy scikit-learn matplotlib   # extra deps for the experiment scripts

pytest -v                        # full test suite
pytest tests/test_affinities.py -v                       # one test file
pytest tests/test_affinities.py::TestName::test_x -v     # one test

# CI's syntax gate (matches azure-pipelines.yml):
flake8 . --count --select=E901,E999,F821,F822,F823 --show-source --statistics
```

The `*.cpp` and `*.so` files in `openTSNE/` are build artifacts checked into the tree. Editing a `.pyx` (`_tsne.pyx`, `quad_tree.pyx`, `kl_divergence.pyx`) requires re-running `pip install -e .` to recompile.

## The `gamma` parameter — key architectural fact

`gamma` is **not** a `TSNE.__init__` argument. It lives only on the affinity classes. To use it you must build the affinity object yourself and pass it to `fit`:

```python
from openTSNE import TSNE
from openTSNE.affinity import PerplexityBasedNN
aff = PerplexityBasedNN(X, perplexity=30, gamma=0.7)   # gamma<1 smooths, >1 sharpens, =1 is upstream-identical
embedding = TSNE().fit(X, affinities=aff)
```

- The transform is `p_{j|i} -> p_{j|i}^gamma / sum_m p_{m|i}^gamma`, applied **before** symmetrization. See [openTSNE/affinity.py:480](openTSNE/affinity.py#L480) (docstring) and [openTSNE/affinity.py:512](openTSNE/affinity.py#L512-L518) (implementation). `gamma == 1.0` is special-cased to skip the transform entirely, so the default path is byte-identical to upstream.
- `gamma` is threaded through the module-level helpers `joint_probabilities_nn` / the FixedSigma variant, and accepted by `PerplexityBasedNN`, `StudentTNN`, `FixedSigmaNN`, `MultiscaleMixture`, `Multiscale`, and `Uniform`.
- Recomputing `P` with a new `gamma` while reusing the same kNN graph is the mechanism behind the curriculum experiment (`improvement/exp.py`) — it swaps `P` between optimization phases.

## Research drivers (each is a standalone `argparse` CLI)

All accept `--dataset {mnist, mouse, adult}` and a `--plot_only` flag that rebuilds figures from already-written CSVs without re-running t-SNE. They locate `data/` and write `results/` relative to their own file location, so they can be launched from any working directory.

- **`experiments/smooth_tsne_opentsne_gamma.py`** (Driver A) — Experiments 1–10: standard vs. smooth vs. sharp t-SNE. Output → `experiments/results/<dataset>_gamma_tsne/expN_*/`. See `parse_args()` near the bottom of the file.
- **`experiments/compare_affinity_variants.py`** (Driver B) — compares 9 affinity kernels by Neighborhood Overlap@k. Output → `experiments/results/affinity_comparison_<dataset>/`.
- **`experiments/seed_variation.py`** — post-processing: scans every result CSV containing a `seed` column, groups by config columns, and writes across-seed std-devs to `experiments/results/seed_variation_summary.csv`. Run from `experiments/`.
- **`improvement/exp.py`** — curriculum-gamma experiment: anneals `gamma` across 3 optimization stages (smooth→mid→sharp). Evaluates LOCAL / MID / GLOBAL metrics and prints an automatic VERDICT. `--quick` for a smoke test.
- **`improvement/baselines.py`** — external DR baselines (PaCMAP, UMAP, TriMap, autoencoder) for `exp.py`. Availability is detected at import; `available_methods()` returns only those whose deps are installed, so missing libraries degrade gracefully.

Metrics used throughout the research code: **Neighborhood Overlap @ k** (fraction of HD k-NN preserved in LD) and **global Spearman ρ** (rank correlation of pairwise distances). All methods are fed the same PCA-50 input for fair comparison.

## Data

`experiments/data/load_data.py` provides the loaders. `mnist` and `adult` download on first run (OpenML). `mouse` (Tasic 2018 cortex scRNA-seq) loads from `experiments/data/tasic2018.pickle`; this pickle and the `.h5ad` files are **git-ignored and large** — verify they exist before running the `mouse` dataset. Rebuild the pickle from raw Allen Institute downloads via `experiments/data/build_mouse_pickle.py` (see README §3 for the download URLs).

## Git & Commits

- Create commits after completing each logical unit of work
- Do not push to the remote repository unless asked
- Use conventional commit messages (e.g. "feat:", "fix:", "refactor:")