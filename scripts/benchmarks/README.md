# Benchmark Scripts

## Files

- `run_llama_replace_ln_matrix.py`: runs the fixed Llama prefill benchmark matrix for `7B/13B/34B/70B`, `16/32/64/128/256/512/1024/2048/4096/8192`, and `baseline/replace_ln`; writes `summary.csv`, `metadata.json`, `monitor/*.csv`, `plots/*.png`, and `BENCHMARK.md` inside `results/llama_replace_ln_prefill/<timestamp>/`; refreshes the repo-root `BENCHMARK.md` index by default.
- `render_llama_replace_ln_report.py`: renders `plots/*.png` and the result-local `BENCHMARK.md` from an existing result directory or `summary.csv`; optionally refreshes the repo-root `BENCHMARK.md` index.

## Run

Use the repo-local fish environment described in `CLAUDE.md`:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py'
```

The batch runner defaults to `warmup=5`, `repeat=10`, and `monitor_interval=0.01`.

Optional output directory override:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py --output_dir results/llama_replace_ln_prefill/manual-run'
```

Re-render plots and the local report for an existing result directory:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<timestamp>'
```

Re-render and refresh the repo-root index:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<timestamp> --refresh_root_index'
```
