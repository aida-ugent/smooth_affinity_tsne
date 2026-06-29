# Schedule-γ t-SNE — verdict (mnist)

Metrics: **NO@1-10** = local neighbour overlap (t-SNE-family home-field — comparable *only* within the γ models); **trust@10 / knn@10** = fair local quality; **spearman / triplet** = fair global structure.

## Baselines

                   NO@1-10  trust@10  knn@10  spearman  triplet
baseline_sharp      0.5017    0.9686  0.9210    0.3523   0.6266
baseline_smooth     0.3311    0.9789  0.9103    0.4309   0.6527
baseline_standard   0.4854    0.9795  0.9268    0.3739   0.6340

## Schedule models

                                 NO@1-10  trust@10  knn@10  spearman  triplet
sched_glob0.5_sharp2_fh_502030    0.4863    0.9684  0.9100    0.4306   0.6501
sched_glob0.5_sharp2_fh_502525    0.4856    0.9684  0.9092    0.4314   0.6505
sched_glob0.5_sharp2_frontheavy   0.4844    0.9673  0.9098    0.4249   0.6488
sched_glob0.5_sharp2_localheavy   0.4968    0.9707  0.9188    0.4199   0.6475
sched_glob0.5_sharp2_no_mid       0.4993    0.9630  0.9125    0.4208   0.6504
sched_glob0.5_sharp2_split5050    0.4960    0.9710  0.9190    0.4227   0.6482
sched_glob1_sharp2_localheavy     0.4970    0.9706  0.9184    0.3705   0.6327
sched_glob1_sharp2_split5050      0.4960    0.9702  0.9186    0.3731   0.6339
sched_pacmap_blend                0.4974    0.9687  0.9170    0.3891   0.6377

## Headroom capture (local=NO@1-10, global=spearman; vs standard, normalised to best baseline)

                                 local_capture  global_capture  min_capture
sched_glob0.5_sharp2_fh_502030           0.057           0.994        0.057
sched_glob0.5_sharp2_fh_502525           0.014           1.008        0.014
sched_glob0.5_sharp2_frontheavy         -0.061           0.895       -0.061
sched_glob0.5_sharp2_localheavy          0.701           0.808        0.701
sched_glob0.5_sharp2_no_mid              0.856           0.822        0.822
sched_glob0.5_sharp2_split5050           0.649           0.857        0.649
sched_glob1_sharp2_localheavy            0.710          -0.059       -0.059
sched_glob1_sharp2_split5050             0.652          -0.013       -0.013
sched_pacmap_blend                       0.740           0.266        0.266

## Verdict

- No schedule config beats standard t-SNE on local, trust and global at once.
- Best trade-off config: **sched_glob0.5_sharp2_no_mid** — captures 86% of the local headroom and 82% of the global headroom that the specialist baselines offer.
- **CONCLUSION: the schedule idea WORKS.** A single γ schedule reaches most of the sharp specialist's local quality AND most of the smooth specialist's global quality — escaping the trade-off any single fixed γ is stuck on.

## vs external DR methods (FAIR metrics only)

NO@1-10 is *excluded* here — it scores preservation of the exact Euclidean kNN graph that t-SNE optimises directly, so it is not a neutral cross-family metric. trust@10 / knn@10 / spearman / triplet are method-agnostic.

        trust@10  knn@10  spearman  triplet
pacmap    0.9517  0.8976    0.3302   0.6200
trimap    0.9444  0.8790    0.2576   0.6113
umap      0.9624  0.9135    0.3326   0.6191
vae       0.9268  0.6579    0.3584   0.6251

Best schedule (**sched_glob0.5_sharp2_no_mid**): trust@10=0.9630, knn@10=0.9125, spearman=0.4208, triplet=0.6504.

- vs **pacmap**: schedule ties/beats on trust@10, knn@10, spearman, triplet.
- vs **trimap**: schedule ties/beats on trust@10, knn@10, spearman, triplet.
- vs **umap**: schedule ties/beats on trust@10, spearman, triplet.
- vs **vae**: schedule ties/beats on trust@10, knn@10, spearman, triplet.
