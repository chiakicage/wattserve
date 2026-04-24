# WattServe

WattServe is an experimental LLM serving and benchmarking repository focused on GPU inference behavior, especially prefill throughput, TTFT / TPOT, GPU power draw, clocks, and the runtime impact of normalization-heavy dataflows.

## Environment

This repository uses `uv` and keeps the active virtual environment in `.venv`.

Unless noted otherwise, run commands from the repository root. If you are
invoking `fish -lc` from elsewhere, replace `<repo_root>` with your local
`wattserve` checkout path.

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; uv sync'
```

If you want to change the index URL of pip in uv, you can modify `~/.config/uv/uv.toml`:

```toml
index-url = "https://mirrors.zju.edu.cn/pypi/web/simple"
```

## Development

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; uv sync --dev'
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; pre-commit install'
```

## Llama Benchmarking

### Single Run

Baseline:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096'
```

`replace_ln` ablation:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096 --replace_ln'
```

Other component ablations:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096 --replace_attention'
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096 --replace_rope'
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096 --replace_activation'
```

Optional controls:

- `--warmup`: warmup iterations before timing, default `5`
- `--repeat`: timed iterations averaged into TTFT, default `10`
- `--monitor_interval`: NVML sampling interval in seconds, default `0.01`
- `--monitor_csv_path`: optional path for exporting the raw GPU monitor trace
- `--replace_attention` / `--replace_rope` / `--replace_activation`: additional component ablation switches that can also be combined manually with `--replace_ln`

### Batch `replace_ln` Matrix

Run the fixed benchmark matrix:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py'
```

The batch runner benchmarks:

- models: `7B`, `13B`, `34B`, `70B`
- prompt lengths: `16`, `32`, `64`, `128`, `256`, `512`, `1024`, `2048`, `4096`, `8192`
- variants: `baseline`, `replace_ln`

By default, the matrix runner uses `warmup=5`, `repeat=10`, and `monitor_interval=0.01`.

The default output directory is:

```text
results/llama_replace_ln_prefill/<UTC_TIMESTAMP>/
```

Each run writes:

- `summary.csv`: one row per model / prompt length / variant run
- `metadata.json`: matrix, defaults, and runtime environment metadata
- `plots/*.png`: prefill TFLOPs/s, TFLOPs uplift vs baseline, average power, and average GPU clock figures
- `BENCHMARK.md`: the result-local Markdown report for that run
- `monitor/*.csv`: raw GPU power / clock traces for successful runs

The batch runner also refreshes the repo-root `BENCHMARK.md` index and republishes a git-tracked snapshot at `results/llama_replace_ln_prefill/latest/`.

### Batch Component Ablation Matrix

Run the fixed component ablation matrix:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_component_ablation_matrix.py'
```

The component ablation runner benchmarks:

- models: `7B`, `13B`, `34B`, `70B`
- prompt lengths: `16`, `32`, `64`, `128`, `256`, `512`, `1024`, `2048`, `4096`, `8192`
- variants: `baseline`, `replace_ln`, `replace_attention`, `replace_rope`, `replace_activation`

By default, the component ablation runner also uses `warmup=5`, `repeat=10`, and `monitor_interval=0.01`.

The default output directory is:

```text
results/llama_component_ablation_prefill/<UTC_TIMESTAMP>/
```

Each run writes the same result-local artifacts as the `replace_ln` matrix:

- `summary.csv`
- `metadata.json`
- `plots/*.png`
- `BENCHMARK.md`
- `monitor/*.csv`

The component ablation runner writes a timestamped result directory by default and does not modify the git-tracked `results/llama_component_ablation_prefill/latest/` snapshot unless you explicitly publish it.

### Re-render Report and Plots

You can regenerate plots and the result-local `BENCHMARK.md` from an existing result directory:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<UTC_TIMESTAMP>'
```

To also refresh the repo-root `BENCHMARK.md` index and republish the git-tracked latest snapshot:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<UTC_TIMESTAMP> --refresh_root_index'
```

For component ablation results:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_component_ablation_report.py --output_dir results/llama_component_ablation_prefill/<UTC_TIMESTAMP>'
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_component_ablation_report.py --output_dir results/llama_component_ablation_prefill/<UTC_TIMESTAMP> --refresh_root_index'
```

To run the component ablation matrix and explicitly republish the git-tracked `latest/` snapshot:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_component_ablation_matrix.py --publish_latest'
```

## Benchmark Notes

### Normalization-Induced Power Throttling

We use the name `Normalization-Induced Power Throttling` for the observation that a seemingly cheap normalization path can raise sustained power enough to trigger downclocking and therefore reduce end-to-end LLM serving throughput by much more than its standalone FLOPs share would suggest.

### Current Working Conclusion

As of the latest canonical matrix on `2026-04-14` ([results/llama_replace_ln_prefill/latest/BENCHMARK.md](results/llama_replace_ln_prefill/latest/BENCHMARK.md)), the current working onset region for `Normalization-Induced Power Throttling` is `prompt_len >= 256`.

We treat a case as a clearly throttled point when the baseline run simultaneously shows:

- average power near the A100 power wall, roughly `398-406 W` in the current dataset
- average GPU clock below the `1410 MHz` top clock
- a matching recovery under `--replace_ln` in power, clock, and TTFT / TFLOPs

In the latest matrix:

- `13B`, `34B`, and `70B` already satisfy that pattern at `prompt_len=256`, with baseline average clocks around `1335 MHz` and `--replace_ln` restoring `1410 MHz`
- `7B` at `prompt_len=256` is a near-threshold point (`365.89 W`, `1410 MHz` baseline), while its clearer sustained throttle appears from `prompt_len=512` onward (`400.28 W`, `1399.17 MHz` baseline)
- for follow-up work, `prompt_len >= 256` is the right first-pass search region, while `prompt_len >= 512` is the higher-confidence cross-model throttle region on the current power-of-two prompt grid
- follow-up studies should stay on alignment-aware prompt lengths instead of arbitrary dense samples, because the relevant CUDA kernel behavior is sensitive to size-alignment boundaries and powers of two are the more meaningful control points in the current setup

### Experimental Method

The recommended method for studying this effect is now:

1. Alignment-aware onset scan: use the canonical power-of-two prompt lengths `16/32/64/128/256/512/1024/2048/4096/8192` to locate where baseline power approaches the wall and baseline clocks start to drop. Avoid arbitrary dense prompt lengths unless there is a kernel-level reason to add them.
2. Component ablation in `python/models/llama.py`: add switches for other potentially relevant paths, especially attention, RoPE, activation, and normalization-related code paths, so the conclusion is not over-attributed to LayerNorm alone.
3. Differential comparison: for each switch setting, compare baseline and ablated runs using `ttft_ms`, `prefill_tflops_s`, `avg_power_watts`, and `avg_gpu_clock_mhz`.
4. Raw-trace confirmation: inspect `monitor/*.csv` and only call the effect obvious power throttling when the trace shows sustained or repeatedly recurring power-cap behavior together with reduced clocks, not just a transient spike.
5. Operator follow-up: if the evidence continues to isolate LayerNorm as the main cause, modify the LayerNorm / RMSNorm operator in vendored FlashInfer and benchmark that integrated path inside this repository.
6. Simplified reproduction: build smaller tests outside the full Llama stack, such as `GEMM + LayerNorm` and `GEMM + Attention + LayerNorm`, to see whether the same power / clock pattern appears in reduced settings.

### Research Plan

The current research plan is:

1. Add new control switches in `python/models/llama.py` so attention, RoPE, activation, and normalization-related paths can be toggled independently during prefill benchmarking.
2. Re-run the aligned power-of-two benchmark grid with those switches to determine whether the observed throttle signature is unique to LayerNorm or shared by other components.
3. If LayerNorm is the only component that consistently reproduces the throttle signature, modify the LayerNorm / RMSNorm operator in vendored FlashInfer and integrate that modified kernel into the project for end-to-end comparison.
4. Add reduced microbenchmarks, especially `GEMM + LayerNorm` and `GEMM + Attention + LayerNorm`, to check whether the same power-throttle behavior can be reproduced in a simpler setting.
5. Use raw monitor traces plus TTFT / TFLOPs data to separate direct compute-cost reduction from clock-recovery effects.
6. Keep reporting centered on the power-of-two prompt grid so the conclusions remain aligned with the CUDA kernel alignment boundaries that are most likely to matter.

### Important Caveats

`--replace_ln` is an ablation flag for the current implementation, not a numerically equivalent model variant.

In the current Llama path it bypasses the `RMSNorm` flow in `python/models/llama.py`, and it also skips the fused residual add-and-normalize behavior. It therefore measures the performance impact of removing a normalization-related source of sustained power pressure in the current dataflow, not just the isolated cost of RMSNorm itself.

`LLAMA2_MAX_POSITION_EMBEDDINGS = 16384` in `python/models/llama_config.py` is only a position-length limit. The current A100 40GB fitting logic trims layer count based on persistent weight memory and does not guarantee that every long-sequence benchmark combination will fit transient activations, workspaces, or all runtime allocations. For this reason, the canonical batch matrix is limited to `16/32/64/128/256/512/1024/2048/4096/8192`, while non-standard prompt lengths should be treated as ad hoc reference runs.

Latest canonical `replace_ln` batch results should be read from the repo-root `BENCHMARK.md` index and the linked git-tracked `results/llama_replace_ln_prefill/latest/BENCHMARK.md`.

Latest multi-component ablation batch results should be read from the repo-root `BENCHMARK_COMPONENT_ABLATION.md` index and the linked git-tracked `results/llama_component_ablation_prefill/latest/BENCHMARK.md`.
