# GEMM Continuous Runtime-DVFS Experiment

- Generated at: `2026-04-29T13:15:35.159619Z`
- Shape: `M=8192, N=8192, K=32768`
- GEMM units: `350`
- Active CUDA time: `13369.923 ms`
- Transition time: `not detected`

## First vs Last 2s

| Window | Avg Clock (MHz) | Avg Power (W) | Avg Temp (C) | TFLOPs/s |
| --- | ---: | ---: | ---: | ---: |
| First 2s | 1110.00 | 255.06 | 40.24 | 208.85 |
| Last 2s | 1110.00 | 211.00 | 43.00 | 234.11 |

## Artifacts

- Summary: `results/gemm_continuous/20260429T131517Z/summary.csv`
- Monitor CSV: `results/gemm_continuous/20260429T131517Z/monitor/continuous_gemm.csv`
- Unit events: `results/gemm_continuous/20260429T131517Z/unit_events/continuous_gemm.csv`
- Power/clock/temperature timeline: `results/gemm_continuous/20260429T131517Z/plots/power_clock_temperature_timeline.png`
- TFLOPs timeline: `results/gemm_continuous/20260429T131517Z/plots/gemm_tflops_timeline.png`
