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

Unlocked primary NVML run:

- `results/gemm_continuous/20260429T115714Z/`

Unlocked separate Nsight Systems run:

- `results/gemm_continuous/20260429T115741Z/`

Locked-clock follow-up runs:

- `1320 MHz`: `results/gemm_continuous/20260429T130616Z/`
- `1110 MHz`: `results/gemm_continuous/20260429T131517Z/`

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

The raw first-2s throughput includes unit 0, which is a cold-start outlier:

- Unit 0: `207.52 ms`, `42.39 TFLOPs/s`
- First 2s excluding unit 0: `265.92 TFLOPs/s`
- 0.5s-2.5s window: `276.88 TFLOPs/s`
- Last 2s: `296.07 TFLOPs/s`

So the stable part of the unlocked run still improves after the clock
transition, but the size of the effect is smaller than the raw `243 -> 296`
first/last comparison suggests.

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
from about `266 TFLOPs/s` in the first 2s after excluding the cold-start unit
to about `296 TFLOPs/s` in the last 2s. A transition-local comparison shows
`274.40 TFLOPs/s` in the 500ms before the clock jump and `296.33 TFLOPs/s` in
the 500ms after the jump.

In the current data, the transition is not explained by an observable cuBLAS
kernel name change, and the local transition window does not support
temperature as the immediate cause.

## Locked 1320 MHz Run

I reran the same experiment after locking GPU 2 graphics clock to `1320 MHz`.

Primary NVML run:

- `results/gemm_continuous/20260429T130616Z/`

Separate Nsight Systems run:

- `results/gemm_continuous/20260429T130718Z/`

The lock capped the final steady clock at `1320 MHz`, but it did not remove the
early lower-clock region.

![Locked 1320 MHz power, clock, and temperature timeline](results/gemm_continuous/20260429T130616Z/plots/power_clock_temperature_timeline.png)

![Locked 1320 MHz per-unit GEMM TFLOPs timeline](results/gemm_continuous/20260429T130616Z/plots/gemm_tflops_timeline.png)

| Metric | First 2s | Last 2s |
| --- | ---: | ---: |
| Avg GPU clock | `1281.51 MHz` | `1320.00 MHz` |
| Avg power | `337.31 W` | `299.63 W` |
| Avg temperature | `43.97 C` | `49.03 C` |
| GEMM throughput | `239.02 TFLOPs/s` | `278.13 TFLOPs/s` |

Again, the raw first-2s throughput includes unit 0:

- Unit 0: `252.10 ms`, `34.89 TFLOPs/s`
- First 2s excluding unit 0: `268.06 TFLOPs/s`
- 0.5s-2.5s window: `270.72 TFLOPs/s`
- Last 2s: `278.13 TFLOPs/s`

The original transition detector reports `False` because it is hard-coded to
look for recovery to `>=1400 MHz`, which is impossible under a `1320 MHz` lock.
The timeline still shows a smaller version of the same effect: GEMM starts
below the locked ceiling and later reaches the locked ceiling. The corrected
throughput comparison is therefore a modest `268 -> 278 TFLOPs/s`, not the raw
`239 -> 278 TFLOPs/s` gap.

Nsight Systems again observed the same active kernel names across the run:

- `[CUDA memset]`
- `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn`

So lowering the graphics clock cap changes the final steady-state throughput
but does not support a cuBLAS kernel-switch explanation.

## Locked 1110 MHz Run

I reran the same experiment after locking GPU 2 graphics clock to `1110 MHz`.

Primary NVML run:

- `results/gemm_continuous/20260429T131517Z/`

Separate Nsight Systems run:

- `results/gemm_continuous/20260429T131601Z/`

At this lower lock, GPU clock is flat from the first 2s to the last 2s. The
raw first-2s throughput number is lower only because it includes unit 0, which
is a cold-start outlier. After unit 0, the per-unit GEMM throughput is already
at the same level as the last 2s.

![Locked 1110 MHz power, clock, and temperature timeline](results/gemm_continuous/20260429T131517Z/plots/power_clock_temperature_timeline.png)

![Locked 1110 MHz per-unit GEMM TFLOPs timeline](results/gemm_continuous/20260429T131517Z/plots/gemm_tflops_timeline.png)

| Metric | First 2s | Last 2s |
| --- | ---: | ---: |
| Avg GPU clock | `1110.00 MHz` | `1110.00 MHz` |
| Avg power | `255.06 W` | `211.00 W` |
| Avg temperature | `40.24 C` | `43.00 C` |
| GEMM throughput | `208.85 TFLOPs/s` | `234.11 TFLOPs/s` |

The unit-level event data shows the source of the apparent throughput change:

- Unit 0: `255.14 ms`, `34.48 TFLOPs/s`
- First 2s excluding unit 0: `234.04 TFLOPs/s`
- Last 2s: `234.11 TFLOPs/s`

So the 1110 MHz run does not show a sustained throughput ramp after the first
unit. It mainly shows that the earlier frequency transition is removed by the
lower clock lock.

Nsight Systems again observed the same active kernel names across the run:

- `[CUDA memset]`
- `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn`

This is a useful refinement: at a lower lock, the graphics clock transition is
gone. The remaining first-2s vs last-2s throughput gap is explained by the
initial cold-start unit, not by a sustained steady-state performance change at
the same reported clock.
