# Full-Block Norm Time Gap Plots

- Source result: `results/state_chain_block_sweep/latest`
- Included batch_size * sequence_length: `S > 64`
- Included models: `7B, 13B, 34B, 70B`
- Derived CSV: [`plots/norm_time_gap/norm_gap_summary.csv`](plots/norm_time_gap/norm_gap_summary.csv)

## Plots

- [norm_time_gap_pct.png](plots/norm_time_gap/norm_time_gap_pct.png)
- [power_delta_watts.png](plots/norm_time_gap/power_delta_watts.png)
- [baseline_power_watts.png](plots/norm_time_gap/baseline_power_watts.png)
- [clock_increase_without_norm_mhz.png](plots/norm_time_gap/clock_increase_without_norm_mhz.png)
- [clock_increase_without_norm_pct.png](plots/norm_time_gap/clock_increase_without_norm_pct.png)
- [norm_gap_power_clock.png](plots/norm_time_gap/norm_gap_power_clock.png)
- [70B_clock_with_without_norm.png](plots/norm_time_gap/70B_clock_with_without_norm.png)
- [70B_32768_time_difference.png](plots/norm_time_gap/70B_32768_time_difference.png)

## Derived Metrics

`norm_time_gap_pct = ((iter_time_with_norm - iter_time_without_norm) - norm_self_time) / iter_time_with_norm * 100`

`norm_time_gap_ms` is kept in the derived CSV as the absolute intermediate value.

| Metric | Case | Value |
| --- | --- | ---: |
| Largest norm time gap | 34B S=16384 | 9.17% |
| Largest power increase | 7B S=1024 | 110.16 W |
| Largest w/o-norm clock increase | 34B S=8192 | 132.00 MHz |
| Largest relative w/o-norm clock increase | 34B S=8192 | 10.33% |
