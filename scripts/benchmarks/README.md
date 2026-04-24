# Benchmark Scripts

## Files

- `run_llama_replace_ln_matrix.py`: runs the fixed Llama prefill benchmark matrix for `7B/13B/34B/70B`, `16/32/64/128/256/512/1024/2048/4096/8192`, and `baseline/replace_ln`; writes `summary.csv`, `metadata.json`, `monitor/*.csv`, `plots/*.png`, and `BENCHMARK.md` inside `results/llama_replace_ln_prefill/<timestamp>/`; refreshes the repo-root `BENCHMARK.md` index and republishes the git-tracked `results/llama_replace_ln_prefill/latest/` snapshot by default.
- `render_llama_replace_ln_report.py`: renders `plots/*.png` and the result-local `BENCHMARK.md` from an existing result directory or `summary.csv`; when `--refresh_root_index` is set, it also republishes the git-tracked `results/llama_replace_ln_prefill/latest/` snapshot.
- `run_llama_component_ablation_matrix.py`: runs the fixed Llama component ablation matrix for `7B/13B/34B/70B`, `16/32/64/128/256/512/1024/2048/4096/8192`, and `baseline/replace_ln/replace_attention/replace_rope/replace_activation`; writes `summary.csv`, `metadata.json`, `monitor/*.csv`, `plots/*.png`, and `BENCHMARK.md` inside `results/llama_component_ablation_prefill/<timestamp>/`; does not modify the git-tracked `results/llama_component_ablation_prefill/latest/` snapshot unless `--publish_latest` is set.
- `render_llama_component_ablation_report.py`: renders `plots/*.png` and the result-local `BENCHMARK.md` from an existing component ablation result directory or `summary.csv`; when `--refresh_root_index` is set, it also republishes the git-tracked `results/llama_component_ablation_prefill/latest/` snapshot.

## Run

Use the repo-local fish environment described in `CLAUDE.md`:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py'
```

The batch runner defaults to `warmup=5`, `repeat=10`, and `monitor_interval=0.01`.

Run the component ablation matrix:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_component_ablation_matrix.py'
```

Publish the current component ablation result as the git-tracked `latest/` snapshot:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_component_ablation_matrix.py --publish_latest'
```

Optional output directory override:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py --output_dir results/llama_replace_ln_prefill/manual-run'
```

Optional output directory override for component ablation:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_component_ablation_matrix.py --output_dir results/llama_component_ablation_prefill/manual-run'
```

Re-render plots and the local report for an existing result directory:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<timestamp>'
```

Re-render, refresh the repo-root index, and republish the git-tracked latest snapshot:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<timestamp> --refresh_root_index'
```

Component ablation re-render:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_component_ablation_report.py --output_dir results/llama_component_ablation_prefill/<timestamp>'
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_component_ablation_report.py --output_dir results/llama_component_ablation_prefill/<timestamp> --refresh_root_index'
```
