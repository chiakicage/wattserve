# Full-Block Experiment Notes

This note summarizes the full-block state-chain experiment in
`results/state_chain_block_sweep/20260427T145456Z/`. It is intended as an
editable writeup rather than a replacement for the raw benchmark report.

## Problem Statement

The issue under investigation is `Norm-Induced Power Throttling`: removing the
Norm path can reduce total block time by more than the measured execution time
of the Norm kernels themselves. We call this unexplained remainder, normalized
against the `with_norm` baseline iteration time, the `norm time gap`.

That means the effect is not simply "Norm is slow." In the larger block runs,
removing Norm also lowers sustained power, lets the GPU recover clock, and
improves the efficiency of the surrounding GEMM kernels. The working hypothesis
is therefore that Norm kernels inserted into a continuous GEMM-heavy sequence
raise the power state of the full workload enough to cause throttling.

## Experiment Goal

The goal is to test whether inserting RMSNorm kernels into an otherwise
GEMM-heavy transformer block changes GPU power/clock behavior enough to reduce
the measured GEMM throughput of the whole block.

This experiment focuses on a full Llama-like block instead of isolated GEMM or
RMSNorm microbenchmarks. The block is run as a state chain, so the output hidden
state of one iteration becomes the input of the next iteration. This avoids a
pure replay benchmark where every iteration reads the same tensors.

## Block Definition

Each benchmarked iteration executes one synthetic decoder block:

```text
hidden
  -> q/k/v GEMMs
  -> RoPE
  -> causal attention
  -> o GEMM
  -> optional fused_add_rmsnorm
  -> gate/up GEMMs
  -> silu_and_mul
  -> down GEMM
  -> optional fused_add_rmsnorm
  -> next hidden
```

The `without_norm` variant removes both `fused_add_rmsnorm` calls. The
`with_norm` variant keeps both norm calls, one after the attention output
projection and one after the MLP down projection.

The experiment uses Llama-family shapes for `7B`, `13B`, `34B`, and `70B`.
The benchmark's `batch_size` is also the causal sequence length `S`; the
attention call therefore uses a triangular causal workload over `S` token rows.

## Run Configuration

- Script: `scripts/benchmarks/run_state_chain_block_sweep.py`
- Output directory: `results/state_chain_block_sweep/20260427T145456Z/`
- Device selection: `CUDA_VISIBLE_DEVICES=3`
- NVML monitor GPU index: `3`
- Dtype: `bfloat16`
- Models: `7B`, `13B`, `34B`, `70B`
- Batch sizes / sequence lengths: `32` through `32768`, powers of two
- Variants: `without_norm`, `with_norm`
- Warmup iterations: `20`
- Timed region target: about `3s` per case, with repeat count calibrated per
  case
- NVML monitor interval: `10ms`
- Successful cases: `88/88`

Raw data and generated artifacts:

- Raw summary: `results/state_chain_block_sweep/20260427T145456Z/summary.csv`
- Existing benchmark report:
  `results/state_chain_block_sweep/20260427T145456Z/BENCHMARK.md`
- NVML monitor traces:
  `results/state_chain_block_sweep/20260427T145456Z/monitor/`
- Kernel profiles:
  `results/state_chain_block_sweep/20260427T145456Z/kernel_profile/`
- Power/clock timeline plots:
  `results/state_chain_block_sweep/20260427T145456Z/plots/`

## Metrics

`Raw GEMM TFLOPs/s` is computed only from CUDA-event time around the GEMM
kernels: q, k, v, o, gate, up, and down. It excludes RoPE, attention,
activation, copies, and RMSNorm time. This metric is useful for detecting
whether the surrounding block mix changes the speed of the GEMM kernels
themselves.

`Effective TFLOPs/s` is computed from the end-to-end block time. It includes
GEMM FLOPs plus an equivalent causal-attention FLOP estimate:

```text
attention_flops = 4 * q_size * S * (S + 1) / 2
effective_flops = gemm_flops + attention_flops
```

Power and clock values are averages from NVML samples collected during the
timed region.

`Norm time gap` measures how much of the no-norm speedup is not explained by
the direct Norm kernel self-time, normalized by the `with_norm` baseline time:

```text
norm_time_gap =
  ((iter_time_with_norm - iter_time_without_norm) - norm_self_time)
  / iter_time_with_norm
```

The absolute intermediate value is still useful for debugging:

```text
norm_time_gap_ms =
  (iter_time_with_norm - iter_time_without_norm) - norm_self_time
```

A positive gap means that removing Norm saved additional time beyond the time
spent inside the Norm kernels themselves. In this experiment, that additional
time is attributed to the power/clock recovery of the surrounding workload.

## Main Observations

The direct symptom appears in the timing decomposition: the total per-iteration
slowdown from enabling Norm is often much larger than the profiler self-time of
the two Norm kernels. The remaining difference is the `norm time gap`.

Representative large-shape examples. `Norm time gap` is reported relative to
the `with_norm` baseline iteration time.

| Model | S | Total extra time with Norm | Norm self-time | Norm time gap | Absolute gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13B | 8192 | 2.013 ms | 0.472 ms | 6.20% | 1.542 ms |
| 34B | 8192 | 5.348 ms | 0.768 ms | 9.00% | 4.580 ms |
| 70B | 8192 | 6.220 ms | 0.768 ms | 8.80% | 5.452 ms |
| 34B | 32768 | 26.346 ms | 3.117 ms | 8.81% | 23.229 ms |
| 70B | 32768 | 29.930 ms | 3.107 ms | 8.69% | 26.824 ms |

For small batch sizes, the norm kernels do not consistently reduce clock or raw
GEMM throughput. At `S=32,64,128`, most cases stay close to the maximum observed
clock of about `1410 MHz`, and the raw GEMM delta is small or noisy.

Starting around `S=256`, the `with_norm` variant generally runs at higher power
and lower GPU clock than `without_norm`. Averaged across all four model shapes,
the clock delta is about `-63 MHz` at `S=256`, about `-94 MHz` at `S=2048`, and
about `-107 MHz` to `-109 MHz` for `S=8192..32768`.

The throughput penalty becomes larger for larger model shapes. For `34B` and
`70B`, large-batch cases show a stable raw GEMM throughput drop of roughly
`7-9%` with norm enabled, while effective throughput drops roughly `9-10%`.
For `13B`, large-batch raw GEMM loss is roughly `5-6%`. For `7B`, the loss is
smaller, typically about `5%` at the largest batch sizes.

Representative large-batch cases:

| Model | S | Raw GEMM w/o norm | Raw GEMM w/ norm | Raw Delta | Clock w/o norm | Clock w/ norm | Power w/o norm | Power w/ norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7B | 8192 | 279.4 | 265.4 | -5.0% | 1410 MHz | 1334 MHz | 318 W | 398 W |
| 13B | 8192 | 291.0 | 272.6 | -6.3% | 1410 MHz | 1317 MHz | 327 W | 397 W |
| 34B | 8192 | 298.0 | 271.1 | -9.0% | 1410 MHz | 1278 MHz | 386 W | 398 W |
| 70B | 8192 | 294.9 | 268.9 | -8.8% | 1410 MHz | 1281 MHz | 389 W | 397 W |
| 34B | 32768 | 299.3 | 272.2 | -9.0% | 1410 MHz | 1281 MHz | 365 W | 396 W |
| 70B | 32768 | 296.0 | 269.7 | -8.9% | 1404 MHz | 1277 MHz | 366 W | 405 W |

## Latest Plots

The latest rendered full-block norm gap plots are published under
`results/state_chain_block_sweep/latest/`.

- Summary report: `results/state_chain_block_sweep/latest/NORM_TIME_GAP.md`
- Derived CSV:
  `results/state_chain_block_sweep/latest/plots/norm_time_gap/norm_gap_summary.csv`
- Combined plot:
  `results/state_chain_block_sweep/latest/plots/norm_time_gap/norm_gap_power_clock.png`
- Individual plots:
  `norm_time_gap_pct.png`, `baseline_power_watts.png`, `power_delta_watts.png`,
  `clock_increase_without_norm_mhz.png`, and
  `clock_increase_without_norm_pct.png`
  in `results/state_chain_block_sweep/latest/plots/norm_time_gap/`
- 70B timing difference bar chart:
  `results/state_chain_block_sweep/latest/plots/norm_time_gap/70B_32768_time_difference.png`
- 70B baseline vs w/o norm clock chart:
  `results/state_chain_block_sweep/latest/plots/norm_time_gap/70B_clock_with_without_norm.png`

## Interpretation

The full-block result supports the hypothesis that RMSNorm changes more than
just its own latency. The primary observation is a positive `norm time gap`:
removing Norm reduces total block time by more than the Norm kernel self-time.
The mechanism suggested by the monitor data is that removing Norm lowers
average power, which lets the GPU recover clock. Since `Raw GEMM TFLOPs/s` is
measured only inside GEMM CUDA events, the drop in that metric indicates that
the GEMM kernels themselves are running more slowly in the `with_norm` block
context.

The effect is shape dependent. Small shapes do not maintain enough pressure for
long enough to produce the same clock behavior. Larger shapes, especially
`34B` and `70B`, spend enough time near the power limit that adding the norm
kernels pushes the full block into a lower-clock operating point. In this
wording, `Norm-Induced Power Throttling` means that Norm inserted into a
continuous GEMM sequence raises sustained power and indirectly slows the
neighboring GEMMs through frequency reduction.

## Operator Phase and Replay-vs-Chain Follow-up

Follow-up profiling was added to separate two effects that should not be
conflated:

1. isolated fixed-replay behavior, where a single operator repeatedly consumes
   the same allocated tensors for about `10s`; and
2. state-chain behavior, where an output tensor becomes the next step's input.

The isolated per-operator profile for `70B`, `S=32768`, `bfloat16` is under:

- `results/fullblock_operator_phase_profile/20260429T141335Z/`
- Script: `scripts/benchmarks/run_fullblock_operator_phase_profile.py`

In that profile, the compute-heavy phases sit near the power wall, while the
memory-heavy phases do not:

| Op | Mode | Time / Iter | Throughput | Avg Power | Avg Clock |
| --- | --- | ---: | ---: | ---: | ---: |
| `q_gemm` | isolated fixed replay | 16.486 ms | 266.78 TFLOPs/s | 397.86 W | 1262 MHz |
| `k_gemm` | isolated fixed replay | 2.150 ms | 255.68 TFLOPs/s | 398.97 W | 1276 MHz |
| `v_gemm` | isolated fixed replay | 2.152 ms | 255.48 TFLOPs/s | 399.10 W | 1275 MHz |
| `rope` | isolated fixed replay | 0.910 ms | 1327.62 GB/s | 269.29 W | 1410 MHz |
| `causal_attention` | isolated fixed replay | 79.847 ms | 220.33 TFLOPs/s | 398.01 W | 1341 MHz |
| `o_gemm` | isolated fixed replay | 15.765 ms | 278.98 TFLOPs/s | 399.33 W | 1320 MHz |
| `post_attn_fused_add_rmsnorm` | isolated fixed replay | 1.574 ms | 1364.04 GB/s | 272.95 W | 1410 MHz |
| `gate_gemm` | isolated fixed replay | 57.706 ms | 266.75 TFLOPs/s | 398.16 W | 1261 MHz |
| `up_gemm` | isolated fixed replay | 57.669 ms | 266.92 TFLOPs/s | 397.68 W | 1261 MHz |
| `gate_copy_to_cat` | isolated fixed replay | 3.850 ms | 976.06 GB/s | 307.54 W | 1410 MHz |
| `up_copy_to_cat` | isolated fixed replay | 3.850 ms | 976.06 GB/s | 308.29 W | 1410 MHz |
| `silu_and_mul` | isolated fixed replay | 4.128 ms | 1365.75 GB/s | 303.57 W | 1410 MHz |
| `down_gemm` | isolated fixed replay | 60.827 ms | 253.07 TFLOPs/s | 398.03 W | 1210 MHz |
| `post_ffn_fused_add_rmsnorm` | isolated fixed replay | 1.576 ms | 1362.99 GB/s | 273.61 W | 1410 MHz |

This shows that `fused_add_rmsnorm` is not, by itself, a high-power operator in
isolation. Like RoPE and `silu_and_mul`, it runs at full `1410 MHz` when
repeated alone. The isolated profile therefore cannot explain the full-block
effect as "the norm kernel itself draws 400 W."

The missing distinction is fixed replay versus state chain. Follow-up
replay-vs-chain runs used the same `70B`, `S=32768` shapes:

- H-to-H GEMM:
  `results/gemm_replay_vs_chain/manual_70B_S32768_o_10s_serial/`
- MLP with `silu_and_mul`:
  `results/gemm_replay_vs_chain/manual_70B_S32768_mlp_silu_10s_serial/`

| Workload | Execution Mode | GEMM TFLOPs/s | Avg Power | Avg Clock |
| --- | --- | ---: | ---: | ---: |
| H-to-H GEMM | fixed replay | 266.31 | 397.58 W | 1265 MHz |
| H-to-H GEMM | state chain | 297.78 | 396.52 W | 1371 MHz |
| MLP + `silu_and_mul` | fixed replay | 264.84 | 395.82 W | 1259 MHz |
| MLP + `silu_and_mul` | state chain | 296.91 | 349.61 W | 1410 MHz |

This explains why the isolated GEMM phase profile showed lower clocks than the
`without_norm` full-block run. Repeating the same input/output buffers is a
fixed-replay stress case: it maintains a high tensor-core activity pattern,
stays near `~398 W`, and is frequency-limited. In the state-chain case, the
output distribution is fed forward. Without normalization, that evolving state
can reduce effective switching activity and sustained power, especially in the
MLP path. The lower-power state-chain run can then recover clock.

The `70B`, `S=32768` full-block numbers line up with that interpretation:

| Variant | GEMM TFLOPs/s | Avg Power | Avg Clock |
| --- | ---: | ---: | ---: |
| `without_norm` | 295.99 | 366.18 W | 1404 MHz |
| `with_norm` | 269.71 | 405.13 W | 1277 MHz |

The current interpretation is therefore:

- RoPE and `silu_and_mul` are memory-heavy, but they do not reset the hidden
  state distribution and do not individually approach the power wall.
- `fused_add_rmsnorm` is also not high-power in isolation, but it sits at
  residual/block boundaries and repeatedly restores the hidden-state scale.
- With Norm enabled, subsequent GEMMs keep a higher-activity input distribution,
  the full block stays close to the power wall, and clocks drop.
- Without Norm, the state chain drifts into a lower-power regime, which lets the
  GPU recover clock and makes the surrounding GEMMs faster.

## Caveats

This is a synthetic single-block state-chain benchmark. It captures the kernel
mix and data dependencies of a decoder block, but it is not a full model
serving benchmark.

The `without_norm` variant is intentionally not a numerically equivalent model
block; it removes the two fused residual-plus-RMSNorm kernels to isolate their
effect on power and clock behavior.

The NVML monitor interval is `10ms`, so power and clock are coarse compared
with individual kernels. The strongest evidence is therefore the combination of
end-to-end NVML averages and CUDA-event GEMM timings, not per-kernel NVML
attribution.

Kernel-profile CSVs were generated with a short profiler repeat and should be
used for kernel identification and rough composition checks, not as the primary
throughput source. The `Norm self-time` and `norm time gap` values above use
those profiler CSVs, so they should be read as evidence for scale, not as a
replacement for the main timed benchmark.

## Working Conclusion

In the full-block experiment, enabling the two fused add RMSNorm kernels causes
a repeatable power/clock shift for medium and large sequence lengths. The
positive `norm time gap` shows that removing those kernels saves more than
their own measured self-time because the no-norm run draws less power, recovers
clock, and executes the surrounding GEMMs more efficiently. The effect is most
visible for `34B` and `70B` shapes, where large-batch raw GEMM throughput falls
by about `7-9%` and average GPU clock falls by roughly `100-130 MHz` relative
to the no-norm variant.
