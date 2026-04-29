# GEMM Continuous Runtime-DVFS Results

This snapshot publishes the curated artifacts for the continuous two-GEMM
runtime-DVFS experiment, including unlocked and locked-clock follow-up runs.

## Summary

- GPU: physical GPU 2
- Shape: `(M, N, K) = (8192, 8192, 32768)`
- Workload: state-chain continuous two-GEMM
- Unlocked active CUDA time: `10755.18 ms`
- Unlocked transition time: `1.993 s` after active start

The run reproduced the behavior where continuous GEMM starts at lower clock and later jumps to full clock:

- Unlocked first 2s: `1236.28 MHz`, raw `243.04 TFLOPs/s`
- Unlocked first 2s excluding cold-start unit 0: `265.92 TFLOPs/s`
- Unlocked last 2s: `1410.00 MHz`, `296.07 TFLOPs/s`

Locked-clock follow-ups refined the result:

- `1320 MHz` lock: clock still starts below the cap and later reaches the cap;
  corrected GEMM throughput changes modestly from `268.06` to `278.13 TFLOPs/s`.
- `1110 MHz` lock: clock is flat at `1110 MHz`; the apparent first/last
  throughput gap is explained by cold-start unit 0, and post-unit-0 throughput
  is already steady around `234 TFLOPs/s`.

Nsight Systems did not observe a cuBLAS/CUDA kernel name switch across the transition. The active GEMM kernel stayed:

- `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn`

See [GEMM_CONTINUOUS.md](GEMM_CONTINUOUS.md) for the consolidated analysis.

## Published Artifacts

- `primary/`: original unlocked primary NVML run with monitor CSV, unit
  events, summary, and timeline plots.
- `nsys/`: original unlocked Nsight Systems active-range CSV summaries and
  kernel bucket analysis.
- `locked_1320/`: locked `1320 MHz` primary run artifacts.
- `nsys_locked_1320/`: locked `1320 MHz` Nsight Systems CSV summaries.
- `locked_1110/`: locked `1110 MHz` primary run artifacts.
- `nsys_locked_1110/`: locked `1110 MHz` Nsight Systems CSV summaries.

Raw `.nsys-rep` and `.sqlite` files are intentionally left out of git.
