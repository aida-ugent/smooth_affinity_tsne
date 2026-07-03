# Experiment 4 — effective neighborhood size

```json
{
  "experiment": "no_k_effective_size",
  "dataset": "mnist",
  "n_points": 70000,
  "n_subsample": null,
  "rhos": [
    10,
    20,
    30,
    50,
    75,
    100,
    125,
    150,
    175,
    200
  ],
  "gammas_computed": [
    0.0,
    0.214286,
    0.428571,
    0.5,
    0.6,
    0.642857,
    0.7,
    0.8,
    0.857143,
    0.9,
    1.0,
    1.071429,
    1.1,
    1.2,
    1.285714,
    1.3,
    1.4,
    1.5,
    1.714286,
    1.928571,
    2.142857,
    2.357143,
    2.571429,
    2.785714,
    3.0
  ],
  "gamma_fine_for_merge": [
    0.0,
    0.21428571428571427,
    0.42857142857142855,
    0.6428571428571428,
    0.8571428571428571,
    1.0714285714285714,
    1.2857142857142856,
    1.5,
    1.7142857142857142,
    1.9285714285714284,
    2.142857142857143,
    2.357142857142857,
    2.571428571428571,
    2.7857142857142856,
    3.0
  ],
  "gamma_dir_fixed": [
    0.8,
    1.0,
    1.2
  ],
  "rho_dir_fixed": [
    30,
    100
  ],
  "gamma_dir_sweep": [
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5
  ],
  "target_ks": [
    10,
    30,
    100
  ],
  "measures": [
    "thr_c0.25",
    "thr_c0.5",
    "thr_c1.0",
    "participation_ratio",
    "two_pow_entropy"
  ],
  "thresh_cs": [
    0.25,
    0.5,
    1.0
  ],
  "knn_seed": 42,
  "data_seed": 42,
  "no_k_source": "exp2_gamma_rho_2d/fine (seed-aggregated)",
  "tsne_settings": {
    "init": "pca",
    "early_exaggeration": 12.0,
    "early_exaggeration_iter": 250,
    "n_iter": 750,
    "ee_momentum": 0.5,
    "main_momentum": 0.8,
    "metric": "euclidean",
    "pca_dims": 50
  }
}
```
