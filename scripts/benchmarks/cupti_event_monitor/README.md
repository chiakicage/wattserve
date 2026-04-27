# CUPTI Active Event Monitor

This is a small CUDA injection library for continuous CUPTI Event API sampling.
It is intended for replay-vs-chain experiments where the active workload must
run as a contiguous stream and must not be converted into per-kernel replay.

Build:

```sh
make -C scripts/benchmarks/cupti_event_monitor
```

Run one benchmark case with active-window start/stop:

```sh
CUDA_VISIBLE_DEVICES=3 \
CUDA_INJECTION64_PATH=$PWD/scripts/benchmarks/cupti_event_monitor/libcupti_active_event_monitor.so \
CUPTI_EVENT_MONITOR_CSV=results/cupti_events.csv \
CUPTI_EVENT_MONITOR_EVENTS=inst_executed \
uv run python scripts/benchmarks/run_gemm_replay_vs_chain_microbench.py \
  --case-kind mlp \
  --modes state_chain \
  --workloads mlp_silu \
  --steps 40 \
  --target-timed-seconds 10 \
  --monitor-gpu-index 3 \
  --profile-active-nvtx \
  --profile-cuda-profiler-api
```

Notes:

- `--profile-cuda-profiler-api` calls `cudaProfilerStart/Stop` only around the
  timed active loop. Warmup, calibration, and GEMM timing are outside the CUPTI
  sampling window.
- `CUPTI_EVENT_MONITOR_EVENTS` takes CUPTI legacy event names, not Nsight
  Compute PerfWorks metric names. Use CUPTI event query tooling to find A100
  L2/DRAM event names available on the host.
- This monitor samples counters in continuous mode. It is a lower-level probe
  than NCU metrics and is meant to avoid profiler-induced replay/cache-reset
  behavior.
