# GEMM / Norm Ratio Findings

This note summarizes the steady-window two-GEMM + norm experiments around GPU clock, power, and GEMM/norm phase timing.

## Method

- Shape: `(M, N, K) = (8192, 8192, 32768)`
- GPU: physical GPU 3
- Execution mode: state-chain
- Workload: repeated two-GEMM phase plus optional `fused_add_rmsnorm`
- Timing policy: every case runs at least 30s active CUDA time; metrics use only the final 20s window
- Presentation filtering: `g18` is usually omitted from plots because it is a transition point and makes the visual trend harder to read

The steady-window method matters. Earlier short-window results mixed startup DVFS behavior into the averages. After switching to final-20s metrics, the `n0` no-norm baseline becomes flat.

## Main Findings

The GEMM/norm time ratio is not the main explanatory variable.

The clearer pattern is phase-state driven:

- With no norm (`n0`), GEMM stays at full clock across all GEMM phase lengths.
- With norm inserted, GEMM can enter a lower-clock state in the middle part of the sweep.
- Around the transition from `g16` to `g20`, GEMM recovers to full clock in most plotted cases.
- The transition is not purely a function of measured GEMM phase duration: `g18` can measure around `600 ms` and still remain low, while `g20` around `592 ms` can recover.

So the useful observation is not a clean ratio threshold. It is that the phase cadence and DVFS state can push GEMM into a lower-clock regime, and a longer configured GEMM burst can recover full clock.

## Baseline: n0

The corrected `n0` baseline is flat after final-20s filtering:

- Tail GEMM clock range: `1407.9-1410.0 MHz`
- Tail GEMM TFLOPs/s range: `296.64-297.14`

This confirms that the earlier `n0` variation was a startup-window artifact, not a replay-vs-chain issue.

## Norm Sweep: n8 and Above

For norm phases around `3.2-51.0 ms`:

- The last low-clock point is consistently near `g16`, with GEMM phase duration around `529-532 ms`.
- The next plotted point `g20` is around `592 ms` and returns to about `1410 MHz`.
- Low-clock GEMM performance is roughly `262-266 TFLOPs/s`; recovered full-clock performance is about `297 TFLOPs/s`.

For longer norm phases (`n512+`), the first short-GEMM point can start high, then the middle of the sweep still falls into the lower-clock regime, and later points recover again. This is why the ratio plot does not give a simple monotonic story.

## n1 Sweep

The `n1` run is useful because the norm phase is tiny:

- `norm_steps=1` corresponds to only about `0.389 ms` of norm work.
- `g1-g18` all stay low, about `1249-1262 MHz`.
- `g20` recovers to `1410 MHz`.
- `g18` is the transition anomaly: it measures about `600 ms` but remains at `1252.6 MHz`, while `g20` measures about `591.9 ms` and recovers to full clock.

This strengthens the conclusion that measured GEMM phase duration alone is not enough. The configured burst/cadence matters.

## Power Observation

In the `n1` sweep, the low-clock region also runs at high power:

- `g1-g18`: roughly `397-399 W`, low clock, about `263-266 TFLOPs/s`
- `g20-g32`: roughly `352-353 W`, full clock, about `297 TFLOPs/s`
- `g48-g64`: full clock remains, while power rises again to about `397 W`

So power is not simply lower when clock is lower. The low-clock middle regime can be both slower and high-power.

## Artifacts

Steady-window merged sweep:

- Summary: `results/two_gemm_norm_steady_window_sweep/gemm_duration_by_norm_duration_steady/summary.csv`
- Main report: `results/two_gemm_norm_steady_window_sweep/gemm_duration_by_norm_duration_steady/README.md`
- GEMM duration vs clock: `results/two_gemm_norm_steady_window_sweep/gemm_duration_by_norm_duration_steady/plots/gemm_phase_clock_by_norm_duration_tail.png`
- GEMM/norm ratio vs clock: `results/two_gemm_norm_steady_window_sweep/gemm_duration_by_norm_duration_steady/plots/gemm_norm_time_ratio_vs_clock_no_g18.png`
- Three-state timelines: `results/two_gemm_norm_steady_window_sweep/gemm_duration_by_norm_duration_steady/plots/timeline_three_states/INDEX.md`

`n1` sweep:

- Report: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/README.md`
- Summary: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/summary.csv`
- GEMM-time x-axis plot without `g18`: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/plots/n1_clock_tflops_by_gemm_time_no_g18.png`
- Clock/power plot without `g18`: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/plots/n1_clock_power_by_gemm_time_no_g18.png`
- Final-window 5s timeline: `results/two_gemm_norm_steady_window_sweep/20260428T194600Z/plots/timeline/n1_three_state_clock_power_timeline_tail5s.png`

## Working Conclusion

For this workload and shape, GEMM clock behavior is best described as a phase-cadence / DVFS-state effect, not a GEMM-to-norm time-ratio effect.

The concise finding is:

> Adding even tiny norm phases can put GEMM into a lower-clock, lower-throughput, high-power regime for shorter configured GEMM bursts. A longer configured GEMM burst around `g20` recovers full clock, even though the measured phase duration is close to the low-clock `g18` transition point.
