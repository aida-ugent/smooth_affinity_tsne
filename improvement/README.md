# Curriculum-γ t-SNE — study notes

*Last updated: 2026-06-25. Branch: `dev-improvements`.*

A self-contained record of the curriculum-γ experiment so we don't lose the
motivation, design, results, and conclusions. If you read nothing else, read
the **TL;DR** and the **Recommended recipe**.

---

## TL;DR

We added a `gamma` knob to openTSNE's affinities (a row-wise power transform on
the high-D neighbour distribution). Earlier work found a **trade-off**: sharp γ
(≈1.5–2) gives the best *local* neighbour preservation, smooth γ (≲1) gives the
best *global* structure, and standard t-SNE (γ=1) is dominated on both ends.

**Idea (PaCMAP-inspired):** instead of one fixed γ, *anneal γ across
optimization* — smooth first to lay out global structure, then sharpen to lock
in local neighbours. **It works.** A single γ schedule recovers most of the
sharp specialist's local quality *and* most of the smooth specialist's global
quality at once — escaping the trade-off any fixed γ is stuck on — and ties or
beats PaCMAP / UMAP / TriMap / a VAE on the fair (method-agnostic) metrics.

**Recommended recipe:** 3 stages, **γ = 0.5 → 1.0 → 2.0**, **front-heavy**
durations **50% / 30% / 20%**, early exaggeration during the first (global)
stage.

---

## 1. Background — the `gamma` knob

`gamma` lives on the affinity classes (not `TSNE.__init__`). It applies a
row-wise power transform to the conditional neighbour probabilities *before*
symmetrization:

```
p_{j|i}  ->  p_{j|i}^γ / Σ_m p_{m|i}^γ
```

- `γ = 1` (default) is the identity → byte-identical to upstream openTSNE.
- `γ > 1` **sharpens** (mass concentrates on the nearest neighbours).
- `0 < γ < 1` **smooths** (mass spreads over more neighbours).

Implementation: [../openTSNE/affinity.py](../openTSNE/affinity.py) (search
`gamma`). Established trade-off (from `../experiments/`): sharp → best local,
smooth → best global.

## 2. Hypothesis — curriculum over γ

Form each *scale* of structure with the γ that is best at it, in stages:

1. **GLOBAL** — smooth γ (≤1), *with early exaggeration*, lays out the coarse
   global arrangement of clusters.
2. **MID** — γ ≈ 1 settles coarse neighbourhoods.
3. **LOCAL** — sharp γ (>1) tightens the exact fine-grained nearest neighbours.

## 3. Mechanism — how a γ schedule is realised

γ is baked into the high-D joint affinity matrix `P`. We change γ mid-run by
**recomputing `P` with the new γ from the *same* kNN graph**, then continuing
gradient descent from the current embedding. So every stage and every fixed-γ
baseline shares one kNN graph — only the power transform changes. (See
`joint_P_for_gamma` / `run_curriculum` in [exp.py](exp.py).)

## 4. Files & how to run

| file | what it is |
|---|---|
| [exp.py](exp.py) | main experiment: baselines (fixed γ) vs a small set of curriculum schedules vs external DR, with the full metric suite and an automatic verdict. |
| [baselines.py](baselines.py) | external DR wrappers: PaCMAP, UMAP, TriMap, and a VAE (2-D latent, posterior mean = embedding). Auto-detects which libs are installed. |
| [tune_curriculum.py](tune_curriculum.py) | schedule-space *search* (stage count, per-stage γ, durations). Single seed by design — a search, not a significance test. |

```bash
# env: graph_tsne_py310  (numpy 1.26, openTSNE editable, pacmap/umap/trimap/torch)
python exp.py --dataset mnist --n_samples 10000 --n_seeds 3   # main comparison
python tune_curriculum.py --dataset mnist                      # schedule search
python exp.py --dataset mnist --plot_only                      # re-plot/verdict only
# --quick on either = fast smoke test
```

Outputs land in `results/<dataset>_curriculum/` and
`results/<dataset>_curriculum_tuning/` (csv + curve/scalar plots + a
`*_VERDICT.md` / `*_TUNING.md`).

## 5. Metrics — and the "home-field" caveat

All methods get the **same PCA-50 input**, **same seeds**, **same metrics**.

| metric | type | meaning | fair across method families? |
|---|---|---|---|
| **NO@k** | curve | Neighbourhood Overlap — fraction of true k-NN preserved | **No** — it scores preservation of the exact Euclidean kNN graph that t-SNE optimises, so it's *home-field* for the t-SNE family. Use it only to compare the **γ models against each other**. |
| **Trust@k** | curve | Trustworthiness — penalises false (intruding) LD neighbours | Yes (objective-independent) |
| **knn@10** | scalar | label kNN purity (supervised local quality) | Yes |
| **global_spearman** | scalar | Spearman ρ of HD vs LD pairwise distances | Yes |
| **triplet_acc** | scalar | random-triplet distance-ordering accuracy | Yes |

When comparing against PaCMAP/UMAP/TriMap/VAE, **NO@k is excluded** and only the
fair metrics are used.

## 6. Main result — MNIST, 10k points, 3 seeds

Fixed-γ baselines (γ schedule must beat these):

| baseline | NO@1-10 | trust@10 | knn@10 | spearman | triplet |
|---|---|---|---|---|---|
| sharp (γ=2) | 0.5017 | 0.9686 | 0.9210 | 0.3523 | 0.6266 |
| smooth (γ=0.5) | 0.3311 | 0.9789 | 0.9103 | **0.4309** | **0.6527** |
| standard (γ=1) | 0.4854 | 0.9795 | **0.9268** | 0.3739 | 0.6340 |

Curriculum models:

| curriculum | NO@1-10 | trust@10 | knn@10 | spearman | triplet |
|---|---|---|---|---|---|
| glob0.5_sharp2_no_mid | 0.4993 | 0.9630 | 0.9125 | 0.4208 | 0.6504 |
| glob0.5_sharp2_split5050 | 0.4960 | 0.9710 | 0.9190 | 0.4227 | 0.6482 |
| glob0.5_sharp2_localheavy | 0.4968 | 0.9707 | 0.9188 | 0.4199 | 0.6475 |
| glob1_sharp2_split5050 | 0.4960 | 0.9702 | 0.9186 | 0.3731 | 0.6339 |
| glob1_sharp2_localheavy | 0.4970 | 0.9706 | 0.9184 | 0.3705 | 0.6327 |

**Headroom capture** (vs standard, normalised so 0 = standard, 1 = the best
specialist baseline; local = NO@1-10, global = spearman):

| curriculum | local capture | global capture | min |
|---|---|---|---|
| **glob0.5_sharp2_no_mid** | **0.86** | **0.82** | **0.82** |
| glob0.5_sharp2_split5050 | 0.65 | 0.86 | 0.65 |
| glob0.5_sharp2_localheavy | 0.70 | 0.81 | 0.70 |
| glob1_sharp2_* | ~0.71 | ~0 | ~0 |

Two clear reads:
- The **smooth global stage is essential**: `glob0.5` configs capture ~80%+ of
  the global headroom; `glob1` configs capture ≈0 (no smoothing → no global
  gain), confirming global structure is set by the first stage.
- No config strictly *dominates* standard on **all** metrics at once (it trades
  a little mid-scale / trust), but the best one grabs ~86% local + ~82% global
  headroom simultaneously — the trade-off is escaped.

**vs external DR** (fair metrics only; best curriculum = `glob0.5_sharp2_no_mid`
at trust 0.9630 / knn 0.9125 / spearman 0.4208 / triplet 0.6504):

| method | trust@10 | knn@10 | spearman | triplet |
|---|---|---|---|---|
| pacmap | 0.9517 | 0.8976 | 0.3302 | 0.6200 |
| umap | 0.9624 | 0.9135 | 0.3326 | 0.6191 |
| trimap | 0.9444 | 0.8790 | 0.2576 | 0.6113 |
| vae | 0.9268 | 0.6579 | 0.3584 | 0.6251 |

The curriculum ties/beats **all four** on global (spearman + triplet) and on
trust/knn vs all except UMAP's knn (essentially tied).

## 7. Schedule search — what's best and why

`tune_curriculum.py` explored 23 schedules (MNIST, 5k points, single seed 42,
750-iter budget). A schedule has two independent knobs — the *γ trajectory* and
the *time split* — and each family freezes one and sweeps the other:

| family | sweeps | holds fixed | values tested |
|---|---|---|---|
| **ramp** | γ trajectory (# stages × range) | equal time/stage | ranges {0.5→2, 0.25→2.5, 0→3} × {2,3,4,5} stages |
| **fine** | γ trajectory, *many* small steps | equal time/stage | ranges {0→3, 0.5→2.5} × {8,12} steps |
| **duration** | the time split | γ fixed at [0.5,1,2], 3 stages | weights frontheavy [.5,.3,.2] / equal / backheavy [.2,.3,.5] / localheavy [.2,.2,.6] |

Ranked by a **balanced score** = geometric mean of min-max-normalised local
(NO@1-10) and global (spearman) — high only when near-best on *both*.

**Winner: `dur_3s_frontheavy`** (balanced 0.945; best global of the whole
search: spearman 0.4475, triplet 0.6536). Exact schedule:

| phase | role | γ | iters (of 750) | exaggeration | momentum |
|---|---|---|---|---|---|
| 1a | global (EE warm-up) | 0.5 | 200 | 12 | 0.5 |
| 1b | global (plain) | 0.5 | 175 | 1.0 | 0.8 |
| 2 | mid | 1.0 | 225 | 1.0 | 0.8 |
| 3 | local | 2.0 | 150 | 1.0 | 0.8 |

→ global 50% · mid 30% · local 20%.

Three patterns from the search:
1. **Duration is the real lever.** frontheavy 0.945 > equal 0.917 > backheavy
   0.889 > localheavy 0.873. Spend *more* time smooth/global early, *less* on
   sharp/local. (Global layout is fragile and set early; local sharpening
   converges fast, so over-investing in it erodes global structure.)
2. **Moderate γ range beats extreme.** 0.5→2 dominates; widening to 0→3 always
   hurt (γ≈0 over-smooths, global collapses).
3. **Stage count barely matters & local is robust.** 2–5 stages ≈ 0.89–0.90;
   near-continuous "fine" ramps no better. NO@1-10 is flat (0.49–0.51) across
   all schedules — the sharp final γ reliably restores local structure. The
   only thing worth tuning is *protecting the global score*.

## 8. Recommended recipe

> **3 stages · γ = 0.5 → 1.0 → 2.0 · durations 50% / 30% / 20% · early
> exaggeration during stage 1.** The iteration *counts* scale with the budget;
> the *proportions* and γ values are the transferable part.

## 9. Open questions / next steps

- [ ] Add the front-heavy schedule to `exp.py`'s model set and run it across
      3 seeds — the search winner is **single-seed**; its global score (0.4475)
      edges the multi-seed `no_mid` best (0.4208), so confirm it holds.
- [ ] Replicate on **mouse** (Tasic scRNA-seq) and **adult** — does the
      50/30/20 recipe transfer, or is the split dataset-dependent?
- [ ] The curriculum gives up a little **mid-scale / trust**; is that
      intrinsic, or recoverable with a slightly longer mid stage?
- [ ] Tune the **VAE** (it's deliberately vanilla) if a stronger deep baseline
      is wanted — currently weak on knn@10 (0.66).

## 10. Environment

Conda env `graph_tsne_py310` (Python 3.10). Key versions verified during the
study: numpy 1.26.4, openTSNE (editable, this fork), pacmap 0.9.1, umap 0.5.11,
trimap 1.1.5, torch 2.12.1+cpu.
