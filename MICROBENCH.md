# Latest Llama Operator Microbenchmark

This file indexes the device-specific git-tracked latest operator microbenchmark snapshots.

- Git-tracked latest snapshots root: `results/llama_operator_microbench/latest`
- Latest-by-device index: [results/llama_operator_microbench/latest/BENCHMARK.md](results/llama_operator_microbench/latest/BENCHMARK.md)
- Canonical suite: `Llama-13B`, `prompt_len=8192`, `dtype=bfloat16`
- Canonical latest keeps the original operator combos and also includes the newer power-focused `steady_block` and `stack + final_norm + lm_head` workloads.

## Devices

| Device | Slug | Report | Summary CSV | Metadata | Source Run | Run Started At |
| --- | --- | --- | --- | --- | --- | --- |
| A100 40G SXM | `a100_40g_sxm` | [report](results/llama_operator_microbench/latest/a100_40g_sxm/BENCHMARK.md) | [summary](results/llama_operator_microbench/latest/a100_40g_sxm/summary.csv) | [metadata](results/llama_operator_microbench/latest/a100_40g_sxm/metadata.json) | `results/llama_operator_microbench/20260424T152146Z` | `2026-04-24T15:21:50.853175Z` |

## Current Findings

- End-to-end `replace_ln` on `13B / 8192` still shows a large gain after removing q/k norm from the model path: `261.74 -> 284.64 TFLOPs/s`, `397.60 -> 318.34 W`, `1312.87 -> 1402.74 MHz`. Reference: [results/llama_replace_ln_prefill/20260424T145723Z/BENCHMARK.md](results/llama_replace_ln_prefill/20260424T145723Z/BENCHMARK.md)
- The old short combo microbench does not reproduce that behavior. In the canonical latest suite, `qkv + attn + o + gate_up + down` vs `+ 2x fused_add_norm` is only `253.94 -> 254.76 GEMM TFLOPs/s`, with power `397.66 -> 397.62 W`. Reference: [results/llama_operator_microbench/latest/a100_40g_sxm/BENCHMARK.md](results/llama_operator_microbench/latest/a100_40g_sxm/BENCHMARK.md)
- A single faithful decoder block only reproduces part of the effect: `steady block` is `231.05 -> 241.23 TFLOPs/s(eq)`, `397.49 -> 397.05 W`, `1302.80 -> 1326.78 MHz`.
- A 40-layer faithful stack does reproduce the end-to-end effect. `stack + final_norm + lm_head` is `232.77 -> 255.39 TFLOPs/s(eq)`, `397.16 -> 327.11 W`, `1298.96 -> 1401.93 MHz`. Reference: [results/llama_operator_microbench/20260424T152146Z/BENCHMARK.md](results/llama_operator_microbench/20260424T152146Z/BENCHMARK.md)
- Removing `lm_head` does not remove the effect. `stack + final_norm` is still `233.14 -> 255.11 TFLOPs/s(eq)`, `398.10 -> 319.61 W`, `1302.05 -> 1403.62 MHz`, so `lm_head` is not the main driver. Reference: [results/llama_operator_microbench/20260424T153223Z/BENCHMARK.md](results/llama_operator_microbench/20260424T153223Z/BENCHMARK.md)

## Depth Sweep

The depth sweep below uses `stack + final_norm` without `lm_head`, timed with layer counts `1/2/4/8/16/40`. Reference: [results/llama_operator_microbench/20260424T154128Z/BENCHMARK.md](results/llama_operator_microbench/20260424T154128Z/BENCHMARK.md)

| Depth | TFLOPs Delta (%) | Power Delta (W) | Clock Delta (MHz) |
| ---: | ---: | ---: | ---: |
| 1 | 3.78 | -3.80 | 8.30 |
| 2 | 4.94 | 8.40 | 26.91 |
| 4 | 5.43 | -6.70 | 37.20 |
| 8 | 7.36 | -27.92 | 69.95 |
| 16 | 8.68 | -58.44 | 88.58 |
| 40 | 9.73 | -80.70 | 103.37 |

## Interpretation

- The mismatch between `replace_ln` and the old operator-combo microbench is mainly a workload-structure issue, not a kernel self-time issue.
- `fused_add_rmsnorm` itself only occupies a small share of the short combo runtime, but repeatedly inserting norm/residual boundaries across many layers pushes the long prefill chain into a higher-power, lower-frequency operating point.
- The effect accumulates with depth. It is weak at `1-4` layers, starts to become obvious around `8` layers, and is large by `16-40` layers.
- The main phenomenon is therefore not "`norm` is expensive by itself", but "`norm` changes the frequency state seen by the surrounding large attention/GEMM kernels when the full stack is long enough."

## Component Depth Reproduction

The component-incremental reproduction below was run on GPU 3 with
`Llama-13B`, `prompt_len=8192`, `dtype=bfloat16`, and depth
`1/2/4/8/16/40`. It reports `GEMM TFLOPs/s`: projection GEMMs plus
attention QK/PV matmuls, with causal prefill attention counted using the
lower-triangular `S*(S+1)/2` pairs. Norm, RoPE, activation, and copy kernels
are timed but excluded from FLOPs. Reference:
[results/component_depth_repro/20260427T072505Z/BENCHMARK.md](results/component_depth_repro/20260427T072505Z/BENCHMARK.md)

| Workload | First Clear Depth | Depth 40 GEMM Delta (%) | Depth 40 Power Delta (W) | Depth 40 Clock Delta (MHz) | Reproduces? |
| --- | ---: | ---: | ---: | ---: | --- |
| `o_chain` | n/a | 11.64 | -3.21 | -33.83 | no |
| `mlp_chain` | 8 | 9.73 | -53.16 | 111.59 | yes |
| `o_mlp_chain` | 16 | 10.77 | -51.55 | 108.60 | yes |
| `qkv_attn_o_chain` | n/a | 5.21 | -29.20 | 36.17 | weak/no |
| `full_block_no_final` | 4 | 10.27 | -71.01 | 111.96 | yes |

Current component-level finding:

- A single `o` GEMM chain is still a negative control: removing the norm makes
  GEMM throughput higher, but the deep-chain clock delta is negative rather
  than the end-to-end direction.
- `mlp_chain` is the first workload that reproduces the depth-dependent
  power/clock split. At depth 8/16/40, `replace_ln` reaches
  `+5.32/+7.66/+9.73%` GEMM TFLOPs/s, `-22.73/-43.02/-53.16 W`, and
  `+47.94/+80.68/+111.59 MHz`.
- Attention plus `o` alone is not enough in this run; its depth-40 clock delta
  is only `+36.17 MHz`.
- The full block without `final_norm` remains the positive control and matches
  the original depth trend, so `final_norm` is not required for the effect.

### MLP Split

The follow-up MLP split keeps the same shape/depth/metric setup and compares
large-matrix-only chains against the real gated MLP chain. Reference:
[results/component_depth_repro/20260427T074211Z/BENCHMARK.md](results/component_depth_repro/20260427T074211Z/BENCHMARK.md)

| Workload | Depth 40 GEMM Delta (%) | Depth 40 Power Delta (W) | Depth 40 Clock Delta (MHz) | Reproduces? |
| --- | ---: | ---: | ---: | --- |
| `mlp_gate_down_chain` | 2.71 | -1.59 | -2.27 | no |
| `mlp_gate_up_down_no_act` | 1.72 | 0.17 | -1.00 | no |
| `mlp_down_up_chain` | 1.70 | -0.89 | -11.40 | no |
| `mlp_chain` | 9.65 | -61.23 | 107.26 | yes |

MLP split finding:

- Large MLP-shaped matrices alone are not sufficient. `H->I + I->H`,
  `H->I + H->I + I->H` without activation, and `I->H + H->I` all stay near
  `1.7-2.7%` GEMM delta at depth 40 and do not produce the high-clock,
  low-power state.
- The effect appears only when the real gated MLP dataflow is present:
  `gate + up -> silu_and_mul -> down -> residual/norm boundary`.
  This points at the interaction between the gated activation/copy path, the
  following large `down` GEMM, and repeated residual/norm boundaries rather than
  matrix size by itself.

### No Reset-Copy Rerun

The component and MLP split experiments above still reset `hidden`/`residual`
once per timed `run_once()`. After the Nsight result showed that reset copies
change the repeated-single-block structure, the same component matrix was rerun
with `--no-reset-copy`, so state is initialized once and then carried across
timed repeats. Reference:
[results/component_depth_repro/20260427T082144Z/BENCHMARK.md](results/component_depth_repro/20260427T082144Z/BENCHMARK.md)

| Workload | Depth 40 GEMM Delta (%) | Depth 40 Power Delta (W) | Depth 40 Clock Delta (MHz) | Reproduces? |
| --- | ---: | ---: | ---: | --- |
| `o_chain` | 22.85 | -15.60 | 93.08 | yes |
| `mlp_gate_down_chain` | 13.67 | -4.82 | 135.71 | yes |
| `mlp_gate_up_down_no_act` | 12.46 | -6.93 | 132.97 | yes |
| `mlp_down_up_chain` | 13.72 | -4.22 | 136.44 | yes |
| `mlp_chain` | 10.48 | -66.49 | 120.87 | yes |
| `o_mlp_chain` | 11.83 | -67.29 | 119.23 | yes |
| `qkv_attn_o_chain` | 5.72 | -62.22 | 43.54 | weak/no |
| `full_block_no_final` | 10.32 | -84.03 | 112.35 | yes |

No-reset-copy finding:

- Removing the reset copy changes the component-level conclusion. `o_chain`
  becomes a positive reproducer, so the earlier `o_chain` negative result was
  benchmark-structure dependent.
- The earlier "`silu_and_mul` is necessary" conclusion is also reset-copy
  dependent. Without reset copies, `mlp_gate_down_chain`,
  `mlp_gate_up_down_no_act`, and `mlp_down_up_chain` all reproduce the
  high-clock, low-power state.
- The cleanest durable requirement is now a long contiguous state-carrying
  chain with residual/norm boundaries in the baseline. The real gated MLP path
  remains a strong positive reproducer, but it is not uniquely required once
  artificial per-iteration input resets are removed.
- Attention plus `o` still remains weaker than GEMM/MLP-like chains in this
  rerun: it has large power savings but only `+43.54 MHz` at depth 40.

## Minimal MLP Replay vs State Chain

The final focused replay-vs-chain check keeps only the real gated MLP path:
`gate + up -> silu_and_mul -> down`, with and without `fused_add_rmsnorm`.
It uses `steps=40` only. This depth is a presentation choice; the sweep showed
the same qualitative pattern at smaller step counts. Reference snapshot:
[results/gemm_replay_vs_chain/latest/a100_40g_sxm/BENCHMARK.md](results/gemm_replay_vs_chain/latest/a100_40g_sxm/BENCHMARK.md).
Source run:
[results/gemm_replay_vs_chain/20260427T100209Z/BENCHMARK.md](results/gemm_replay_vs_chain/20260427T100209Z/BENCHMARK.md).

Metric rule for this result is strict: `GEMM TFLOPs/s` is computed from
gate/up/down GEMM FLOPs divided by CUDA event time around those `torch.mm`
kernels only. `silu_and_mul`, activation copy, and `fused_add_rmsnorm` are
timed in total iteration time but excluded from the GEMM TFLOPs denominator.

| Mode | Norm | GEMM TFLOPs/s | Total Iter Time (ms) | Avg Power (W) | Avg GPU Clock (MHz) |
| --- | --- | ---: | ---: | ---: | ---: |
| `fixed_replay` | no | 268.38 | 578.158 | 400.97 | 1289.11 |
| `fixed_replay` | yes | 268.78 | 586.894 | 392.11 | 1290.00 |
| `state_chain` | no | 292.98 | 532.024 | 316.99 | 1410.00 |
| `state_chain` | yes | 270.26 | 585.306 | 398.39 | 1295.46 |

Focused finding:

- Fixed replay barely reacts to adding `fused_add_rmsnorm`: GEMM-only
  throughput is `+0.15%`, power is `-8.86 W`, and clock is `+0.89 MHz`.
- State chain reacts strongly to the same norm boundary: GEMM-only throughput
  is `-7.76%`, power is `+81.41 W`, and clock is `-114.54 MHz`.
- Without the norm boundary, state chain reaches the high-clock/low-power
  state: `292.98 TFLOPs/s`, `316.99 W`, and `1410.00 MHz`.
- With the norm boundary, state chain falls back near the fixed-replay GEMM
  throughput and clock point: `270.26 TFLOPs/s`, `398.39 W`, and
  `1295.46 MHz`.
- Because the TFLOPs denominator is GEMM-only, the drop is the surrounding
  GEMM kernels running slower under the state-carrying norm/residual cadence,
  not hidden inclusion of norm or activation time in the metric.

## Nsight Stack vs Repeated Single Block

Nsight Systems was run on GPU 3 with
`/opt/nvidia/nsight-systems-cli/2025.2.1/bin/nsys` to compare three variants of
`full_block_no_final`, `baseline`, depth 40:

- `stack`: one `run_once()` contains a depth-40 chain.
- `repeat-single`: call a depth-1 `run_once()` forty times.
- `repeat-single-no-reset`: call forty one-layer steps, but reset
  `hidden`/`residual` only once at the beginning of the chain.

Reports:

- `results/nsys_stack_vs_repeat/full_block_baseline_stack_depth40.nsys-rep`
- `results/nsys_stack_vs_repeat/full_block_baseline_repeat_single_depth40.nsys-rep`
- `results/nsys_stack_vs_repeat/full_block_baseline_repeat_single_no_reset_depth40.nsys-rep`

The profiling helper is
`scripts/benchmarks/profile_stack_vs_repeat.py`. Run with
`CUDA_VISIBLE_DEVICES=3`; inside the process `cuda:0` then maps to physical GPU
3.

Active NVTX-range D2D memcpy counts:

| Mode | D2D memcpy count | D2D memcpy time (ms) | D2D bytes |
| --- | ---: | ---: | ---: |
| `stack` | 2 | 0.236 | 167,772,160 |
| `repeat-single` | 80 | 9.450 | 6,710,886,400 |
| `repeat-single-no-reset` | 2 | 0.236 | 167,772,160 |

The extra D2D copies in `repeat-single` are exactly the artificial per-block
input reset:

- `state["hidden"].copy_(state["source"])`
- `state["residual"].copy_(state["hidden"])`

Each hidden tensor copy is `8192 * 5120 * 2 = 83,886,080` bytes. Calling a
depth-1 block forty times therefore produces `2 * 40 = 80` D2D copies. This is
not equivalent to a true stack and pollutes the repeated-single-block trace.

After removing the per-block reset copy, the repeated one-layer form still
reproduces the stack/end-to-end behavior:

| Variant | Iter Time (ms) | GEMM TFLOPs/s | Avg Power (W) | Avg GPU Clock (MHz) |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 1012.261 | 232.52 | 397.45 | 1291.45 |
| `replace_ln` | 920.296 | 255.75 | 331.40 | 1399.23 |

Nsight interpretation:

- The earlier difference between `stack` and naive repeated single block was
  partly caused by benchmark structure: repeated single block did a full input
  and residual reset before every layer.
- Once that reset is moved outside the forty-layer chain, the D2D copy count
  matches `stack` and the power/clock effect remains.
- The durable phenomenon is therefore still the long contiguous operator
  cadence, not the reset copy itself.

## Power/Clock Clues So Far

The current evidence points to a workload-state effect rather than a standalone
kernel-cost effect. The important pattern is that the baseline path with
repeated residual/norm boundaries enters a high-power, lower-clock state over a
long contiguous chain, while `replace_ln` avoids enough of that cadence to run
at higher clock and lower power once the chain is deep enough.

Known non-drivers:

- `lm_head` is not required. Removing it keeps the effect.
- `final_norm` is not required. `full_block_no_final` still reproduces the
  depth trend.
- Attention plus `o` alone is still comparatively weak. It does not reach the
  full-block clock delta at depth 40 in either component rerun.
- Naive repeated single-block benchmarking was polluted by artificial D2D input
  reset copies, but the D2D copies are not the durable cause. Removing the
  per-block reset makes repeated single-layer execution match the stack-like
  D2D count and still reproduce the power/clock effect.

Known positive clues:

- Depth matters. The effect is weak at `1-4` layers, starts around `8`, and is
  large by `16-40` in reset-copy runs. In no-reset-copy runs, several
  state-carrying chains enter the high-clock `replace_ln` state earlier.
- A long contiguous cadence matters. The reproducing cases are stack-like
  chains where outputs flow into subsequent work without per-layer input reset.
- The clean four-row MLP replay-vs-chain snapshot isolates this cadence:
  `fixed_replay + mlp_silu` is insensitive to adding norm, while
  `state_chain + mlp_silu` loses `7.76%` GEMM-only throughput, gains
  `81.41 W`, and loses `114.54 MHz` when the same norm boundary is inserted.
- With per-iteration reset copies still present, the first component-level
  reproducer is `mlp_chain`: `gate + up -> silu_and_mul -> down ->
  residual/norm boundary`.
- Without reset copies, the first component-level reproducer becomes
  `o_chain`; MLP large-matrix splits also reproduce. This means
  `silu_and_mul` is not a durable necessary condition, although the real gated
  MLP remains a strong positive case.
- The full block without `final_norm` remains the closest positive control:
  at depth 40 it reaches `+10.27%` GEMM TFLOPs/s, `-71.01 W`, and
  `+111.96 MHz` with reset copy, and `+10.32%`, `-84.03 W`,
  `+112.35 MHz` without reset copy.

Current working hypothesis:

- The trigger is not simply "norm is slow"; it is also not tied uniquely to
  `silu_and_mul`.
- The trigger is the repeated interaction of residual/norm boundaries with a
  long contiguous, state-carrying chain of surrounding GEMM-like work. The real
  gated MLP dataflow makes this easy to reproduce, but no-reset-copy tests show
  simpler GEMM chains can also enter the same operating point.
- The baseline path appears to sustain a higher-power operating point that
  depresses SM clock for the following large GEMM work. Removing those
  residual/norm boundaries in `replace_ln` changes the operating point, so the
  surrounding GEMMs run faster even though the saved norm kernel time alone is
  too small to explain the total delta.
