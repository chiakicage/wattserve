# Cross-Shape n1 GEMM Clock Sweep

This is the curated publish snapshot for the cross-shape `norm_steps=1`
two-GEMM state-chain sweep.

## Method

- GPU: physical GPU 3
- Workload: repeated two-GEMM phase followed by one fused RMSNorm unit
- Execution mode: state-chain
- Timing policy: each case runs at least 30s active CUDA time; metrics use only the final 20s window
- `g` in the plots means two-GEMM units per GEMM phase; the actual GEMM kernel count is `2*g`

## Final Filtering

The plotted data keeps all stable points and removes only:

- `llama7b_s4096 / g88`, an off-trend around-200ms clock dip found by the dense local scan
- timing anomalies where a larger configured `g` produced a shorter measured GEMM phase time than the preceding retained point

The timing-anomaly exclusions are recorded in
`excluded_timing_anomaly_points.csv`.

## Final Plots

- `plots/cross_shape_n1_clock_by_gemm_steps_timing_anomaly_filtered.png`
- `plots/cross_shape_n1_clock_by_gemm_steps_zoom_timing_anomaly_filtered.png`

## CSV Artifacts

- `summary_timing_anomaly_filtered.csv`: data used by the final two plots
- `transition_summary_filtered.csv`: low-clock to full-clock transition table
- `excluded_points.csv`: manually filtered off-trend point
- `excluded_timing_anomaly_points.csv`: monotonic timing anomaly exclusions

## Transition Summary

| Shape | Last low | First full |
| --- | ---: | ---: |
| `llama7b_s4096` | `g30` / `87.04 ms @ 1270.49 MHz` | `g35` / `91.42 ms @ 1410.00 MHz` |
| `llama7b_s8192` | `g33` / `190.90 ms @ 1268.98 MHz` | `g38` / `197.81 ms @ 1410.00 MHz` |
| `llama13b_s8192` | `g27` / `237.86 ms @ 1260.00 MHz` | `g34` / `267.84 ms @ 1410.00 MHz` |
| `llama34b_s8192` | `g20` / `445.66 ms @ 1244.41 MHz` | `g24` / `471.81 ms @ 1410.00 MHz` |
| `llama70b_s8192` | `g18` / `529.00 ms @ 1249.74 MHz` | `g22` / `573.25 ms @ 1410.00 MHz` |
| `llama70b_s16384` | `g17` / `1006.08 ms @ 1236.30 MHz` | `g24` / `1245.35 ms @ 1410.00 MHz` |
