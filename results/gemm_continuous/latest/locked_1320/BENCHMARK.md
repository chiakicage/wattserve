# GEMM Continuous Runtime-DVFS Experiment

- Generated at: `2026-04-29T13:06:32.023091Z`
- Shape: `M=8192, N=8192, K=32768`
- GEMM units: `350`
- Active CUDA time: `11355.736 ms`
- Transition time: `not detected`

## First vs Last 2s

| Window | Avg Clock (MHz) | Avg Power (W) | Avg Temp (C) | TFLOPs/s |
| --- | ---: | ---: | ---: | ---: |
| First 2s | 1281.51 | 337.31 | 43.97 | 239.02 |
| Last 2s | 1320.00 | 299.63 | 49.03 | 278.13 |

## Artifacts

- Summary: `results/gemm_continuous/20260429T130616Z/summary.csv`
- Monitor CSV: `results/gemm_continuous/20260429T130616Z/monitor/continuous_gemm.csv`
- Unit events: `results/gemm_continuous/20260429T130616Z/unit_events/continuous_gemm.csv`
- Power/clock/temperature timeline: `results/gemm_continuous/20260429T130616Z/plots/power_clock_temperature_timeline.png`
- TFLOPs timeline: `results/gemm_continuous/20260429T130616Z/plots/gemm_tflops_timeline.png`
