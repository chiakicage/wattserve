# Two-GEMM / Norm Steady Window Sweep

- Generated at: `2026-04-28T17:11:35.021717Z`
- Output directory: `/home/cage/wattserve/results/two_gemm_norm_steady_window_sweep/20260428T170350Z`
- Successful cases: `13/13`
- Metrics are computed from the tail analysis window only.

| GEMM Steps | Norm Steps | Active (s) | Tail GEMM Phase (ms) | Tail GEMM Clock (MHz) | Tail TFLOPs/s | Tail GEMM Phases |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 34.83 | 29.63 | 1410.00 | 296.87 | 674 |
| 2 | 0 | 34.50 | 59.23 | 1410.00 | 297.01 | 337 |
| 4 | 0 | 34.49 | 118.44 | 1410.00 | 297.06 | 168 |
| 8 | 0 | 34.48 | 236.96 | 1409.41 | 296.96 | 84 |
| 12 | 0 | 34.61 | 355.66 | 1408.36 | 296.78 | 56 |
| 16 | 0 | 34.71 | 474.24 | 1408.33 | 296.77 | 42 |
| 18 | 0 | 34.23 | 533.56 | 1408.12 | 296.74 | 37 |
| 20 | 0 | 34.47 | 593.05 | 1407.93 | 296.64 | 33 |
| 22 | 0 | 33.98 | 652.08 | 1408.24 | 296.77 | 30 |
| 24 | 0 | 34.20 | 711.26 | 1408.50 | 296.81 | 28 |
| 32 | 0 | 34.16 | 947.72 | 1409.28 | 297.00 | 21 |
| 48 | 0 | 35.52 | 1420.97 | 1410.00 | 297.13 | 14 |
| 64 | 0 | 36.00 | 1894.55 | 1410.00 | 297.14 | 10 |

## Artifacts

- Summary CSV: `results/two_gemm_norm_steady_window_sweep/20260428T170350Z/summary.csv`
- Monitor traces: `results/two_gemm_norm_steady_window_sweep/20260428T170350Z/monitor`
- Phase event traces: `results/two_gemm_norm_steady_window_sweep/20260428T170350Z/phase_events`

## n0 Tail Baseline Interpretation

- Every case ran at least `33.98s` active CUDA time in this completed run.
- Metrics below use only the last `20s` tail window.
- After removing startup/transient behavior, `n0` is effectively flat across GEMM phase lengths.
- Tail GEMM clock range: `1407.9-1410.0 MHz`.
- Tail GEMM TFLOPs/s range: `296.64-297.14`.
- Plot: `n0_tail_baseline.png`.
