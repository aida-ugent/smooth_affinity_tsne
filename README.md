# smooth-affinity t-SNE

A fork of [openTSNE](https://github.com/pavlin-policar/openTSNE) that adds a
single new affinity parameter, **`gamma`**, and a set of experiments comparing
this "smooth/sharp" affinity family against standard t-SNE and several other
affinity kernels.

The hypothesis: re-weighting the high-dimensional neighbor probabilities with a
power transform before optimization changes how local vs. global structure is
preserved — and a value of `gamma < 1` ("smooth t-SNE") can preserve
neighborhoods across a wider range of scales than a single fixed perplexity.

---

## 1. What was added to the library

Standard t-SNE builds a conditional neighbor distribution `p_{j|i}` for each
point from a Gaussian kernel calibrated to a target perplexity. This fork adds a
**row-wise power transform** applied to that distribution *before*
symmetrization:

```
p_{j|i}  ->  p_{j|i}^gamma / sum_m p_{m|i}^gamma
```

- `gamma = 1.0` (default) — identity; behaves exactly like upstream openTSNE.
- `gamma > 1.0` — **sharpens**: nearest neighbors get even more weight.
- `0 < gamma < 1.0` — **smooths**: weight spreads more evenly over the
  perplexity-determined neighborhood.
- `gamma = 0.0` — uniform weight over all neighbors in the kernel support.

The implementation lives in [openTSNE/affinity.py](openTSNE/affinity.py)
(see the docstring and transform at
[affinity.py:480](openTSNE/affinity.py#L480) and
[affinity.py:512](openTSNE/affinity.py#L512-L518)). `gamma` is accepted by
`PerplexityBasedNN`, `StudentTNN`, `FixedSigmaNN`, `MultiscaleMixture`,
`Multiscale`, and `Uniform`.

> **Note:** `TSNE.__init__` does **not** take `gamma`. To use it, build the
> affinity object yourself and pass it to `TSNE.fit(..., affinities=aff)`. The
> helper `run_tsne()` in the experiment script does exactly this — see
> [experiments/smooth_tsne_opentsne_gamma.py:95](experiments/smooth_tsne_opentsne_gamma.py#L95).

Minimal usage:

```python
from openTSNE import TSNE
from openTSNE.affinity import PerplexityBasedNN

aff = PerplexityBasedNN(X, perplexity=30, gamma=0.7)   # smooth t-SNE
embedding = TSNE().fit(X, affinities=aff)
```

---

## 2. Repository layout

The experiment code is kept **completely separate** from the library: the
library lives in [openTSNE/](openTSNE/), and everything related to these
experiments (driver scripts, data loaders, datasets, and results) lives under
[experiments/](experiments/).

| Path | What it is |
|------|------------|
| [openTSNE/](openTSNE/) | The forked library. Only `affinity.py` was modified (gamma + `StudentTNN`/tt-SNE). |
| [experiments/](experiments/) | All experiment code + outputs, independent of the library. |
| [experiments/smooth_tsne_opentsne_gamma.py](experiments/smooth_tsne_opentsne_gamma.py) | **Driver A** — Experiments 1–10: standard vs. smooth vs. sharp t-SNE. |
| [experiments/compare_affinity_variants.py](experiments/compare_affinity_variants.py) | **Driver B** — compares 9 affinity kernels by Neighborhood Overlap. |
| [experiments/data/load_data.py](experiments/data/load_data.py) | Dataset loaders (MNIST, mouse cortex, adult census) + `rnaseqTools.py` and the dataset files. |
| [experiments/results/](experiments/results/) | Generated CSVs + PNGs. Driver A → `<dataset>_gamma_tsne/`, Driver B → `affinity_comparison_<dataset>/`. |
| [experiments/logs/](experiments/logs/) | Captured stdout from full runs. |
| `Appendix.pdf` | Write-up of the results. |
| `README.rst` | The **original upstream** openTSNE readme (kept for reference). |

> The scripts locate `data/` and write `results/` relative to their own
> location, so they can be launched from anywhere (e.g. `python
> experiments/smooth_tsne_opentsne_gamma.py ...` from the repo root).

---

## 3. Datasets

All experiments accept `--dataset {mnist, mouse, adult}`. Loaders are in
[experiments/data/load_data.py](experiments/data/load_data.py).

| Dataset | Source | Notes |
|---------|--------|-------|
| `mnist` | Fetched via `sklearn.datasets.fetch_openml` (downloads on first run) | 70k handwritten digits, reduced to 50 PCs. Use `--mnist_subsample N` to subsample. |
| `mouse` | `experiments/data/tasic2018.pickle` (+ `importantGenesTasic2018.npy`) | Mouse cortex single-cell transcriptomes. Loaded from `experiments/data/` by default. |
| `adult` | Fetched via OpenML | Census income tabular data; one-hot + scaled. `--adult_max_rows N`. |

> The pickle and `.h5ad` data files are large and are git-ignored — confirm the
> files under `experiments/data/` exist before running the `mouse` dataset.

---

## 4. Running the experiments

### Setup

```bash
# install the forked library in editable mode (compiles the Cython extension)
pip install -e .
pip install numpy pandas scipy scikit-learn matplotlib
```

### Driver A — gamma experiments (1–10)

```bash
# Full run on MNIST with 5 random restarts
python experiments/smooth_tsne_opentsne_gamma.py --dataset mnist --gamma_s 0.7 --n_runs 5

# Regenerate figures only, from the CSVs already in experiments/results/ (no t-SNE)
python experiments/smooth_tsne_opentsne_gamma.py --dataset mnist --plot_only

# Skip individual experiments
python experiments/smooth_tsne_opentsne_gamma.py --dataset mouse --skip_exp4 --skip_exp9
```

Output goes to `experiments/results/<dataset>_gamma_tsne/expN_*/`, with each
figure prefixed by the dataset name (e.g. `mnist_05_embedding.png`). Key knobs:
`--gamma_s` (smooth value, default 0.7), `--gamma_h` (sharp value, default 1.5),
`--perplexity`, `--n_runs`. See `parse_args()` at
[experiments/smooth_tsne_opentsne_gamma.py:1847](experiments/smooth_tsne_opentsne_gamma.py#L1847) for
the full list.

### Driver B — affinity-variant comparison

```bash
# Default output: experiments/results/affinity_comparison_<dataset>/
python experiments/compare_affinity_variants.py --dataset mnist --n_runs 1

python experiments/compare_affinity_variants.py --dataset mnist --plot_only
```

Compares 9 kernels — `γ=1.0` (standard), `γ=0.7` (smooth), `γ=0.0`, `γ=1.5`
(sharp), `FixedSigmaNN`, `MultiscaleMixture`, `Multiscale`, `Uniform`, and
`tt-SNE` (heavy-tailed Student-t HD kernel) — by Neighborhood Overlap @ k.

---

## 5. The experiments (Driver A)

Each experiment writes a CSV (raw numbers) and one or more PNGs. The
`--plot_only` flag rebuilds every figure from those CSVs without re-running
t-SNE, so plots can be tweaked cheaply.

| # | Name | Question it answers | Output folder |
|---|------|---------------------|---------------|
| 1 | Affinity row sharpness | How does gamma reshape one point's neighbor distribution? | `exp1_affinity_sharpness/` |
| 2 | Effective perplexity | How does gamma shift the per-point *effective* perplexity? (2b = heatmap over γ×perplexity) | `exp2_eff_perplexity/` |
| 3 | Per-point Δρ correlations | What point-level features predict how much smoothing changes a point? | `exp3_delta_perp/` |
| 4 | NH overlap vs γ sweep | At fixed perplexity, how does Neighborhood Overlap@k vary with gamma? | `exp4_neighborhood_overlap_gamma_sweep/` |
| 5 | Embedding comparison | Side-by-side scatter of standard vs. smooth embeddings. | `exp5_embedding/` |
| 6 | NH sensitivity heatmaps | NH-overlap AUC across the γ × perplexity grid. | `exp6_sensitivity/`, `sensitivity_grid/` |
| 7 | NH: standard/smooth/sharp | Mean NH@k curves for the three regimes. | `exp7_neighborhood_overlap_comparison/` |
| 8 | Global Spearman vs γ | Global structure preservation (HD vs LD distance rank correlation) as gamma varies. | `exp8_global_spearman/` |
| 9 | Smooth vs. perplexity envelope | Does one smooth run match the *best-over-all-perplexities* standard run? | `exp9_smooth_vs_envelope/` |
| 10 | NH overlap vs perplexity | NH@k as a function of perplexity, smooth vs. standard. | `exp10_no_vs_perplexity/` |

> Experiments 1–3 read the conditional `P` matrix from `build_affinity()`;
> 4–10 only need the final embedding. Metrics used throughout: **Neighborhood
> Overlap @ k** (fraction of HD k-NN that remain LD k-NN) and **global Spearman
> ρ** (rank correlation of pairwise distances).

---

## 6. Reproducing a result from scratch

```bash
pip install -e .
python experiments/smooth_tsne_opentsne_gamma.py --dataset mnist --n_runs 5
python experiments/compare_affinity_variants.py --dataset mnist --n_runs 3
```

Figures land under `experiments/results/`. Compare them against the committed
versions and the `Appendix.pdf` write-up.
