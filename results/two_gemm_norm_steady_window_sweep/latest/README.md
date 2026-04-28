# Two-GEMM Norm Steady-Window Results

This snapshot publishes the curated artifacts for the two-GEMM + norm phase-cadence experiments.

## Main Finding

For `(M, N, K) = (8192, 8192, 32768)`, the observed GEMM clock behavior is better explained as a phase-cadence / DVFS-state effect than as a simple GEMM-to-norm time-ratio threshold.

The steady-window runs use state-chain execution, run each case for at least 30s, and compute reported metrics only from the final 20s window.

Key observations:

- `n0` no-norm baseline is flat after final-window filtering: about `1408-1410 MHz` and `296.6-297.1 TFLOPs/s`.
- Adding norm phases can put GEMM into a lower-clock, lower-throughput, high-power regime for shorter configured GEMM bursts.
- A longer configured GEMM burst around `g20` recovers full clock in these sweeps.
- The `n1` sweep shows that even a tiny norm phase, about `0.389 ms`, can trigger the low-clock regime for `g1-g18`.

See [GEMM_NORM_RATIO.md](GEMM_NORM_RATIO.md) for the consolidated analysis.

## Published Artifacts

- `merged/`: combined steady-window sweep across norm durations, with `g18`-filtered plotting CSVs and key plots.
- `n1/`: dedicated tiny-norm sweep and timeline.
- `n0_baseline/`: corrected no-norm baseline used to validate the steady-window method.
- `nonzero_norm_sweep/`: source nonzero-norm steady-window sweep summary.

Representative plots:

- `merged/plots/gemm_phase_clock_by_norm_duration_tail_time_no_g18_connected.png`
- `merged/plots/gemm_phase_tflops_by_norm_duration_tail_time_no_g18_connected.png`
- `merged/plots/gemm_norm_time_ratio_vs_clock_no_g18.png`
- `merged/plots/timeline_three_states/n128_three_state_clock_power_timeline_tail5s.png`
- `merged/plots/timeline_three_states/n512_three_state_clock_power_timeline_tail5s.png`
- `merged/plots/timeline_three_states/n1024_three_state_clock_power_timeline_tail5s.png`
- `n1/plots/n1_clock_power_by_gemm_time_no_g18.png`
- `n1/plots/timeline/n1_three_state_clock_power_timeline_tail5s.png`
