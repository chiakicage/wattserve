# GEMM Continuous Runtime-DVFS Experiment

- Generated at: `2026-04-29T11:57:29.367876Z`
- Shape: `M=8192, N=8192, K=32768`
- GEMM units: `350`
- Active CUDA time: `10755.182 ms`
- Transition time: `1.993 s`

## First vs Last 2s

| Window | Avg Clock (MHz) | Avg Power (W) | Avg Temp (C) | TFLOPs/s |
| --- | ---: | ---: | ---: | ---: |
| First 2s | 1236.28 | 329.63 | 40.31 | 243.04 |
| Last 2s | 1410.00 | 371.76 | 50.63 | 296.07 |

## Artifacts

- Summary: `results/gemm_continuous/20260429T115714Z/summary.csv`
- Monitor CSV: `results/gemm_continuous/20260429T115714Z/monitor/continuous_gemm.csv`
- Unit events: `results/gemm_continuous/20260429T115714Z/unit_events/continuous_gemm.csv`
- Power/clock/temperature timeline: `results/gemm_continuous/20260429T115714Z/plots/power_clock_temperature_timeline.png`
- TFLOPs timeline: `results/gemm_continuous/20260429T115714Z/plots/gemm_tflops_timeline.png`
