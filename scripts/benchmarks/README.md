# Benchmark Scripts

## Files

- `run_llama_replace_ln_matrix.py`: runs the fixed Llama prefill benchmark matrix for `7B/13B/34B/70B`, `512/1024/2048/8192/16384`, and `baseline/replace_ln`; writes `summary.csv`, `metadata.json`, and per-run GPU monitor traces under `results/llama_replace_ln_prefill/<timestamp>/`; refreshes the repo-root `BENCHMARK.md`.

## Run

Use the repo-local fish environment described in `CLAUDE.md`:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py'
```

Optional output directory override:

```sh
fish -lc 'source /home/cage/wattserve/.venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py --output_dir results/llama_replace_ln_prefill/manual-run'
```
