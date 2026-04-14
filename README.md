# WattServe

WattServe is an experimental LLM serving and benchmarking repository focused on GPU inference behavior, especially prefill throughput, TTFT / TPOT, GPU power draw, clocks, and the runtime impact of normalization-heavy dataflows.

## Environment

This repository uses `uv` and keeps the active virtual environment in `.venv`.

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; uv sync'
```

Run the Qwen weight-loading path:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python python/main.py --model /share/models/Qwen3-4B'
```

If you want to change the index URL of pip in uv, you can modify `~/.config/uv/uv.toml`:

```toml
index-url = "https://mirrors.zju.edu.cn/pypi/web/simple"
```

## Development

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; uv sync --dev'
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; pre-commit install'
```

## Llama Benchmarking

### Single Run

Baseline:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096'
```

`replace_ln` ablation:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096 --replace_ln'
```

Optional controls:

- `--warmup`: warmup iterations before timing, default `3`
- `--repeat`: timed iterations averaged into TTFT, default `5`
- `--monitor_interval`: NVML sampling interval in seconds, default `0.01`
- `--monitor_csv_path`: optional path for exporting the raw GPU monitor trace

### Batch `replace_ln` Matrix

Run the fixed benchmark matrix:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py'
```

The batch runner benchmarks:

- models: `7B`, `13B`, `34B`, `70B`
- prompt lengths: `512`, `1024`, `2048`, `8192`
- variants: `baseline`, `replace_ln`

The default output directory is:

```text
results/llama_replace_ln_prefill/<UTC_TIMESTAMP>/
```

Each run writes:

- `summary.csv`: one row per model / prompt length / variant run
- `metadata.json`: matrix, defaults, and runtime environment metadata
- `plots/*.png`: TTFT, average power, and average GPU clock figures
- `BENCHMARK.md`: the result-local Markdown report for that run
- `monitor/*.csv`: raw GPU power / clock traces for successful runs

The batch runner also refreshes the repo-root `BENCHMARK.md` index so it points to the latest result directory and its local report.

### Re-render Report and Plots

You can regenerate plots and the result-local `BENCHMARK.md` from an existing result directory:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<UTC_TIMESTAMP>'
```

To also refresh the repo-root `BENCHMARK.md` index:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<UTC_TIMESTAMP> --refresh_root_index'
```

## Benchmark Notes

### Normalization-Induced Power Throttling

We use the name `Normalization-Induced Power Throttling` for the observation that a seemingly cheap normalization path can raise sustained power enough to trigger downclocking and therefore reduce end-to-end LLM serving throughput by much more than its standalone FLOPs share would suggest.

### Important Caveats

`--replace_ln` is an ablation flag for the current implementation, not a numerically equivalent model variant.

In the current Llama path it bypasses the `RMSNorm` flow in `python/models/llama.py`, and it also skips the fused residual add-and-normalize behavior. It therefore measures the performance impact of removing a normalization-related source of sustained power pressure in the current dataflow, not just the isolated cost of RMSNorm itself.

`LLAMA2_MAX_POSITION_EMBEDDINGS = 16384` in `python/models/llama_config.py` is only a position-length limit. The current A100 40GB fitting logic trims layer count based on persistent weight memory and does not guarantee that every long-sequence benchmark combination will fit transient activations, workspaces, or all runtime allocations. For this reason, `16384` is no longer part of the standard benchmark matrix.

Latest batch results should be read from the repo-root `BENCHMARK.md` index and the linked `results/llama_replace_ln_prefill/<timestamp>/BENCHMARK.md`.
