# GEMM Continuous Runtime-DVFS Results

This snapshot publishes the curated artifacts for the continuous two-GEMM runtime-DVFS experiment.

## Summary

- GPU: physical GPU 2
- Shape: `(M, N, K) = (8192, 8192, 32768)`
- Workload: state-chain continuous two-GEMM
- Primary active CUDA time: `10755.18 ms`
- Transition time: `1.993 s` after active start

The run reproduced the behavior where continuous GEMM starts at lower clock and later jumps to full clock:

- First 2s: `1236.28 MHz`, `243.04 TFLOPs/s`
- Last 2s: `1410.00 MHz`, `296.07 TFLOPs/s`

Nsight Systems did not observe a cuBLAS/CUDA kernel name switch across the transition. The active GEMM kernel stayed:

- `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn`

See [GEMM_CONTINUOUS.md](GEMM_CONTINUOUS.md) for the consolidated analysis.

## Published Artifacts

- `primary/`: primary NVML run with monitor CSV, unit events, summary, and timeline plots.
- `nsys/`: Nsight Systems active-range CSV summaries and kernel bucket analysis.

Raw `.nsys-rep` and `.sqlite` files are intentionally left out of git.
