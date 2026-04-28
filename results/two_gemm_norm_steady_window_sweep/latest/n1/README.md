# Two-GEMM / Norm Steady Window Sweep

- Generated at: `2026-04-28T19:54:18.214213Z`
- Output directory: `/home/cage/wattserve/results/two_gemm_norm_steady_window_sweep/20260428T194600Z`
- Successful cases: `13/13`
- Metrics are computed from the tail analysis window only.

| GEMM Steps | Norm Steps | Active (s) | Tail GEMM Phase (ms) | Tail GEMM Clock (MHz) | Tail TFLOPs/s | Tail GEMM Phases |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 38.79 | 33.08 | 1262.00 | 265.93 | 597 |
| 2 | 1 | 38.72 | 66.79 | 1249.93 | 263.38 | 297 |
| 4 | 1 | 38.73 | 133.68 | 1248.83 | 263.20 | 149 |
| 8 | 1 | 38.71 | 266.67 | 1252.36 | 263.88 | 74 |
| 12 | 1 | 38.77 | 399.28 | 1254.21 | 264.36 | 50 |
| 16 | 1 | 38.93 | 533.00 | 1253.28 | 264.05 | 37 |
| 18 | 1 | 39.02 | 600.02 | 1252.65 | 263.88 | 33 |
| 20 | 1 | 35.18 | 591.90 | 1410.00 | 297.21 | 33 |
| 22 | 1 | 34.54 | 651.14 | 1410.00 | 297.19 | 30 |
| 24 | 1 | 34.83 | 710.39 | 1410.00 | 297.17 | 28 |
| 32 | 1 | 34.12 | 947.20 | 1410.00 | 297.16 | 21 |
| 48 | 1 | 35.54 | 1420.97 | 1410.00 | 297.13 | 14 |
| 64 | 1 | 36.00 | 1894.46 | 1410.00 | 297.16 | 10 |

## Artifacts

- Summary CSV: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/summary.csv`
- Monitor traces: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/monitor`
- Phase event traces: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/phase_events`
- Clock/TFLOPs plot: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/plots/n1_clock_tflops_by_gemm_steps.png`
- Clock/power plot: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/plots/n1_clock_power_by_gemm_steps.png`
- Clock/TFLOPs plot, GEMM-time x-axis without `g18`: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/plots/n1_clock_tflops_by_gemm_time_no_g18.png`
- Clock/power plot, GEMM-time x-axis without `g18`: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/plots/n1_clock_power_by_gemm_time_no_g18.png`
- Plot-filtered CSV: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/summary_no_g18_for_plots.csv`
- Final-window 5s timeline: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/plots/timeline/n1_three_state_clock_power_timeline_tail5s.png`
- Timeline index: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/plots/timeline/INDEX.md`

## Notes

- `norm_steps=1` corresponds to a tail norm phase of about `0.389 ms`.
- The low-clock region persists from `g1` through `g18`.
- `g20` recovers to `1410 MHz`, while `g18` remains low even though its measured GEMM phase is about `600 ms`.
- Presentation plots can omit `g18` because it creates the same transition-region visual artifact seen in earlier sweeps.
