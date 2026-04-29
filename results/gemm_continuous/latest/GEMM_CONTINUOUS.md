# GEMM Continuous Runtime-DVFS Experiment

This note tracks experiments for the continuous two-GEMM runtime-DVFS effect:
the same GEMM workload can start in a lower-clock state and later jump to a
higher clock while the kernel stream remains continuous.

## Method

- GPU: physical GPU 2
- Process device: `CUDA_VISIBLE_DEVICES=2`, in-process `cuda:0`
- Monitor index: NVML GPU 2
- Shape: `(M, N, K) = (8192, 8192, 32768)`
- Workload: state-chain two-GEMM unit
  `(M,N) x (N,K) -> (M,K)`, then `(M,K) x (K,N) -> (M,N)`
- Active target: continuous GEMM for about 10s
- Nsight Systems path: `/usr/local/bin/nsys`

## Hypotheses

1. cuBLAS may switch to a different kernel implementation during the run.
2. GPU temperature may correlate with the clock transition.

## Current Results

Latest primary NVML run:

- `results/gemm_continuous/20260429T115714Z/`

Latest separate Nsight Systems run:

- `results/gemm_continuous/20260429T115741Z/`

## Primary Timeline Result

The continuous two-GEMM run reproduced the runtime clock transition.

![Power, clock, and temperature timeline](results/gemm_continuous/20260429T115714Z/plots/power_clock_temperature_timeline.png)

![Per-unit GEMM TFLOPs timeline](results/gemm_continuous/20260429T115714Z/plots/gemm_tflops_timeline.png)

| Metric | First 2s | Last 2s |
| --- | ---: | ---: |
| Avg GPU clock | `1236.28 MHz` | `1410.00 MHz` |
| Avg power | `329.63 W` | `371.76 W` |
| Avg temperature | `40.31 C` | `50.63 C` |
| GEMM throughput | `243.04 TFLOPs/s` | `296.07 TFLOPs/s` |

Detected transition:

- Active CUDA time: `10755.18 ms`
- Transition time: `1.993 s` after active start
- Transition pre-500ms: `1293.00 MHz`, `392.27 W`, `46.18 C`,
  `274.40 TFLOPs/s`
- Transition post-500ms: `1410.00 MHz`, `359.66 W`, `45.42 C`,
  `296.33 TFLOPs/s`

This means the local transition itself is not explained by a rising
temperature. Around the transition, temperature is slightly lower after the
clock jump (`46.18 C -> 45.42 C`), while clock rises to full speed and power
drops. The broader first-to-last 2s temperature increase is a consequence of
the active workload heating the GPU, but it does not align as the immediate
cause of the clock jump in this 10s run.

## Nsight Systems Kernel Result

Nsight Systems was run separately with:

```sh
/usr/local/bin/nsys profile --trace=cuda,nvtx,cublas,cublas-verbose
```

The active range was filtered by NVTX range `active_continuous_gemm`.

Kernel bucket summary:

- `results/gemm_continuous/20260429T115741Z/nsys/kernel_bucket_summary.csv`
- `results/gemm_continuous/20260429T115741Z/nsys/kernel_change_summary.md`

Observed active kernel names were unchanged across the run:

- `[CUDA memset]`
- `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn`

So this run did not observe a cuBLAS/CUDA kernel implementation name switch
between the low-clock early phase and the full-clock later phase. This does
not rule out internal library or driver state changes that are invisible in the
kernel name, but it argues against the simple hypothesis that cuBLAS swapped to
a different GEMM kernel.

## Artifacts

Primary run:

- Summary: `results/gemm_continuous/20260429T115714Z/summary.csv`
- Monitor CSV: `results/gemm_continuous/20260429T115714Z/monitor/continuous_gemm.csv`
- Unit events: `results/gemm_continuous/20260429T115714Z/unit_events/continuous_gemm.csv`
- Power/clock/temperature timeline:
  `results/gemm_continuous/20260429T115714Z/plots/power_clock_temperature_timeline.png`
- GEMM TFLOPs timeline:
  `results/gemm_continuous/20260429T115714Z/plots/gemm_tflops_timeline.png`

Nsight Systems run:

- Profile report: `results/gemm_continuous/20260429T115741Z/nsys/profile.nsys-rep`
- Active CUDA trace:
  `results/gemm_continuous/20260429T115741Z/nsys/cuda_gpu_trace_active_cuda_gpu_trace_nvtx=active_continuous_gemm.csv`
- Active kernel summary:
  `results/gemm_continuous/20260429T115741Z/nsys/cuda_gpu_kern_sum_active_cuda_gpu_kern_sum_nvtx=active_continuous_gemm.csv`
- Kernel bucket summary:
  `results/gemm_continuous/20260429T115741Z/nsys/kernel_bucket_summary.csv`

## Working Conclusion

For continuous state-chain two-GEMM on `(8192, 8192, 32768)`, GPU 2 starts in a
lower-clock state for about 2s, then jumps to full `1410 MHz`. Throughput rises
from about `243 TFLOPs/s` in the first 2s to about `296 TFLOPs/s` in the last
2s.

In the current data, the transition is not explained by an observable cuBLAS
kernel name change, and the local transition window does not support
temperature as the immediate cause.
