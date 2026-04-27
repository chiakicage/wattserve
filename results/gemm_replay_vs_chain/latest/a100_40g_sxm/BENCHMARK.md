# MLP Silu Replay vs State-Chain Latest Snapshot

This git-tracked snapshot keeps the minimal four-row result from the corrected GEMM-only replay-vs-chain microbenchmark.

- Device: `A100 40G SXM`
- Source run: `results/gemm_replay_vs_chain/20260427T100209Z`
- Shape: `Llama-13B`, `prompt_len=8192`, `hidden=5120`, `intermediate=13824`, `dtype=bfloat16`
- Workload: `gate + up -> silu_and_mul -> down`, with and without `fused_add_rmsnorm`
- Steps per run: `40`
- Metric rule: `GEMM TFLOPs/s` is gate/up/down GEMM FLOPs divided by CUDA event time around those `torch.mm` kernels only. `silu_and_mul`, activation copy, and `fused_add_rmsnorm` are excluded from the denominator.

## Results

| Mode | Workload | GEMM TFLOPs/s | GEMM Time / Iter (ms) | Total Iter Time (ms) | Avg Power (W) | Avg GPU Clock (MHz) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed_replay | mlp_silu | 268.38 | 518.502 | 578.158 | 400.97 | 1289.11 |
| fixed_replay | mlp_silu_fused_add_norm | 268.78 | 517.738 | 586.894 | 392.11 | 1290.00 |
| state_chain | mlp_silu | 292.98 | 474.964 | 532.024 | 316.99 | 1410.00 |
| state_chain | mlp_silu_fused_add_norm | 270.26 | 514.901 | 585.306 | 398.39 | 1295.46 |

## Key Deltas

| Comparison | GEMM Delta (%) | Power Delta (W) | Clock Delta (MHz) |
| --- | ---: | ---: | ---: |
| fixed replay, add norm | 0.15 | -8.86 | 0.89 |
| state chain, add norm | -7.76 | 81.41 | -114.54 |
| state chain vs fixed replay, without norm | 9.17 | -83.98 | 120.89 |
| state chain vs fixed replay, with norm | 0.55 | 6.28 | 5.46 |

## Finding

- Fixed replay barely changes when adding `fused_add_rmsnorm`: the GEMM-only delta is small and the clock stays near the same operating point.
- State chain without norm reaches the high-clock state (`1410 MHz`) and high GEMM-only throughput (`292.98 TFLOPs/s`).
- State chain with norm drops to `270.26 TFLOPs/s`, raises power to `398.39 W`, and lowers clock to `1295.46 MHz`.
- Because the denominator is GEMM-only, the throughput drop is the surrounding GEMM kernels running slower after the state-carrying norm/residual cadence, not direct inclusion of norm or silu time.
