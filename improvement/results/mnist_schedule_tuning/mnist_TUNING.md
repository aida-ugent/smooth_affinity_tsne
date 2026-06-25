# Schedule-γ schedule search — mnist (single seed)

`balanced` = geometric mean of min-max-normalised local (NO@1-10) and global (spearman) scores across all configs. High only when a schedule is near-best on BOTH axes (escapes the trade-off). `local_n`/`global_n` are the normalised components.

## Reference points (single fixed γ)

              NO@1-10  trust@10  knn@10  spearman  triplet  local_n  global_n  balanced
ref_sharp       0.518     0.969   0.893     0.324    0.618    1.000     0.000     0.000
ref_smooth      0.359     0.977   0.879     0.450    0.656    0.000     1.000     0.000
ref_standard    0.506     0.978   0.897     0.413    0.640    0.925     0.706     0.808

## Schedules ranked by balanced score

                   NO@1-10  trust@10  knn@10  spearman  triplet  local_n  global_n  balanced
dur_3s_frontheavy    0.504     0.966   0.879     0.447    0.654    0.914     0.978     0.945
dur_3s_equal         0.509     0.967   0.878     0.436    0.652    0.943     0.891     0.917
ramp_g0.5-2_s5       0.510     0.969   0.882     0.433    0.650    0.949     0.862     0.904
ramp_g0.5-2_s3       0.511     0.966   0.882     0.431    0.651    0.953     0.846     0.898
ramp_g0.5-2_s2       0.510     0.964   0.881     0.430    0.651    0.950     0.843     0.895
ramp_g0.5-2_s4       0.514     0.971   0.886     0.427    0.648    0.973     0.815     0.891
dur_3s_backheavy     0.514     0.966   0.881     0.427    0.647    0.973     0.813     0.889
ramp_g0.25-2.5_s5    0.504     0.960   0.876     0.433    0.652    0.909     0.866     0.887
ramp_g0.25-2.5_s4    0.505     0.963   0.878     0.431    0.649    0.918     0.847     0.882
ramp_g0.25-2.5_s3    0.502     0.960   0.871     0.433    0.649    0.898     0.861     0.879
fine_g0.5-2.5_s8     0.503     0.962   0.876     0.431    0.651    0.908     0.847     0.877
dur_3s_localheavy    0.513     0.966   0.881     0.423    0.646    0.968     0.787     0.873
ramp_g0.25-2.5_s2    0.498     0.951   0.864     0.433    0.653    0.873     0.862     0.867
ramp_g0-3_s2         0.489     0.948   0.856     0.418    0.652    0.819     0.741     0.779
fine_g0.5-2.5_s12    0.503     0.962   0.875     0.408    0.646    0.902     0.662     0.773
ramp_g0-3_s4         0.497     0.957   0.868     0.399    0.639    0.867     0.591     0.716
ramp_g0-3_s3         0.492     0.951   0.859     0.398    0.642    0.834     0.588     0.701
fine_g0-3_s12        0.487     0.951   0.852     0.398    0.641    0.803     0.589     0.687
ramp_g0-3_s5         0.496     0.955   0.866     0.390    0.638    0.861     0.521     0.670
fine_g0-3_s8         0.489     0.954   0.859     0.372    0.630    0.818     0.382     0.559

## Winners

- **Best balanced schedule: `dur_3s_frontheavy`** (balanced=0.945; local_n=0.91, global_n=0.98).
- Best local (NO@1-10): `ramp_g0.5-2_s4` = 0.5139
- Best global (spearman): `dur_3s_frontheavy` = 0.4475
- Best fair-local (knn@10): `ramp_g0.5-2_s4` = 0.8863
- Best global (triplet): `dur_3s_frontheavy` = 0.6536
