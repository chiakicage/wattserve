# Two-GEMM + Norm Steady-Window Sweep

This is the corrected redo of the original GEMM-phase-duration vs norm-phase-duration experiment. Each case runs a state-chain workload for at least 30s of active CUDA time and all reported metrics are computed only over the final 20s tail window.

- Shape: `(M, N, K) = (8192, 8192, 32768)`
- GPU: physical GPU 3 via `CUDA_VISIBLE_DEVICES=3`, process device `cuda:0`, NVML index `3`
- State chain: `current -> middle -> next_current`; the output is kept as the next input
- GEMM phase sweep: `g1,g2,g4,g8,g12,g16,g18,g20,g22,g24,g32,g48,g64`
- Norm phase sweep: `n0,n8,n16,n32,n64,n128,n512,n1024,n2048,n4096`
- Active target used here: `32s`, with a `20s` analysis window

## Plots

- [GEMM phase clock by norm duration](plots/gemm_phase_clock_by_norm_duration_tail.png)
- [GEMM TFLOPs by norm duration](plots/gemm_phase_tflops_by_norm_duration_tail.png)

## Main Check

The corrected `n0` baseline is now flat across different GEMM phase lengths:

- tail GEMM clock range: `1407.9-1410.0 MHz`
- tail GEMM TFLOPs range: `296.64-297.14`

That confirms the earlier `n0` variation was a startup/transient-window artifact, not a replay-vs-chain difference. The workload here is still state-chain.

## Per-Series Tail Clock Summary

- norm 0 ms: min 1407.9 MHz at ~593.1 ms; recovers >=1400 MHz at ~29.6 ms.
- norm 3.2 ms: min 1245.3 MHz at ~134.1 ms; recovers >=1400 MHz at ~592.6 ms.
- norm 6.4 ms: min 1248.0 MHz at ~133.8 ms; recovers >=1400 MHz at ~592.6 ms.
- norm 12.7 ms: min 1256.9 MHz at ~531.4 ms; recovers >=1400 MHz at ~592.5 ms.
- norm 25.5 ms: min 1258.7 MHz at ~530.9 ms; recovers >=1400 MHz at ~592.4 ms.
- norm 51.0 ms: min 1266.0 MHz at ~595.3 ms; recovers >=1400 MHz at ~592.4 ms.
- norm 204.1 ms: min 1276.0 MHz at ~593.4 ms; recovers >=1400 MHz at ~30.0 ms.
- norm 408.3 ms: min 1280.3 MHz at ~593.3 ms; recovers >=1400 MHz at ~29.9 ms.
- norm 816.8 ms: min 1279.9 MHz at ~592.1 ms; recovers >=1400 MHz at ~29.9 ms.
- norm 1.63 s: min 1279.7 MHz at ~592.6 ms; recovers >=1400 MHz at ~30.0 ms.

## Source Runs

- n0 baseline: `/home/cage/wattserve/results/two_gemm_norm_steady_window_sweep/20260428T170350Z`
- nonzero norm sweep: `/home/cage/wattserve/results/two_gemm_norm_steady_window_sweep/20260428T172623Z`

Full merged rows are in [summary.csv](summary.csv).
