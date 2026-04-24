# CLAUDE.md

This file is the repository-level overview and execution guide for agent work in this repo.

## Project Overview

WattServe is an experimental LLM serving and benchmarking repository focused on GPU inference behavior, especially:

- prefill throughput
- TTFT / TPOT
- GPU power draw and clocks
- the runtime impact of normalization and other memory-heavy operations

The current repository includes custom Llama inference paths built on top of FlashInfer kernels. The Llama path now also includes batch benchmark runners for the canonical `replace_ln` matrix and the broader component ablation matrix.

## Main Entry Points

- `README.md`: setup, benchmark workflow, and caveats
- `BENCHMARK.md`: repo-root index pointing to the latest canonical Llama benchmark result directory
- `python/bench_llama.py`: single-run Llama prefill benchmark with structured result output and optional monitor CSV export
- `scripts/benchmarks/run_llama_replace_ln_matrix.py`: batch runner for the fixed Llama `replace_ln` matrix
- `scripts/benchmarks/render_llama_replace_ln_report.py`: renders `plots/*.png` and the result-local `BENCHMARK.md` for an existing result directory
- `scripts/benchmarks/run_llama_component_ablation_matrix.py`: batch runner for the fixed Llama component ablation matrix
- `scripts/benchmarks/render_llama_component_ablation_report.py`: renders `plots/*.png` and the result-local `BENCHMARK.md` for an existing component ablation result directory
- `scripts/benchmarks/README.md`: explains what each benchmark script does and how to run it
- `python/models/llama.py`: inference-only Llama model using FlashInfer ops
- `python/models/llama_config.py`: Llama model specs plus parameter / FLOPs / memory estimators
- `python/monitor/gpu_monitor.py`: NVML-based power / clock sampling
- `results/llama_replace_ln_prefill/`: generated batch benchmark outputs, one timestamped directory per run
- `results/llama_component_ablation_prefill/`: generated component ablation outputs, one timestamped directory per run
- `BENCHMARK_COMPONENT_ABLATION.md`: repo-root index pointing to the latest component ablation result directory
- `3rdparty/`: vendored dependencies, notably `flashinfer` and `cutlass`

## Current Codebase State

- The repository is currently mid-refactor from `python/*.py` model files to `python/models/*.py`.
- Prefer the `python/models/` implementations when reading or editing model code.
- Do not restore deleted top-level model files unless explicitly asked.

## Environment And Command Execution

This repository uses `uv` for dependency management and keeps the active virtual environment in the repo-local `.venv`.

### Required shell convention

The user works in `fish`, and this repository should be run with the repo-local fish activation script:

```sh
source .venv/bin/activate.fish
```

For non-interactive agent commands, the reliable pattern is:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; <command>'
```

Use that pattern instead of assuming `python` is already on `PATH`. Replace
`<repo_root>` with the local `wattserve` checkout path when running from
outside the repository root.

### Verified environment

Verified in this workspace on 2026-04-14:

- repository root: the current `wattserve` checkout
- active Python: `<repo_root>/.venv/bin/python`
- Python version: `3.13.2`
- `sys.prefix`: `<repo_root>/.venv`
- `uv`: available on `PATH`
- `uv` version: `0.7.2`
- `torch`: `2.11.0+cu130`
- `flashinfer`: `0.6.7.post3`
- `transformers`: `5.5.3`

Validation that succeeded in this environment:

- `python python/bench_llama.py --help`
- `python scripts/benchmarks/run_llama_replace_ln_matrix.py --help`
- `python scripts/benchmarks/render_llama_replace_ln_report.py --help`
- `python scripts/benchmarks/run_llama_component_ablation_matrix.py --help`
- `python scripts/benchmarks/render_llama_component_ablation_report.py --help`
- `python -c "import torch, flashinfer, transformers"`

## Common Commands

Initialize dependencies:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; uv sync'
```

Initialize development dependencies:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; uv sync --dev'
```

Install git hooks:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; pre-commit install'
```

Run the Llama prefill benchmark:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096'
```

Run the Llama `replace_ln` ablation benchmark:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096 --replace_ln'
```

Run the Llama component ablation benchmarks:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096 --replace_attention'
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096 --replace_rope'
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python python/bench_llama.py --model 13B --prompt_len 4096 --replace_activation'
```

Run the batch Llama ablation matrix:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py'
```

Run the batch Llama component ablation matrix:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_component_ablation_matrix.py'
```

The canonical batch benchmark defaults are `warmup=5`, `repeat=10`, and `monitor_interval=0.01`.

Override the batch output directory:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_replace_ln_matrix.py --output_dir results/llama_replace_ln_prefill/manual-run'
```

Override the component ablation batch output directory:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/run_llama_component_ablation_matrix.py --output_dir results/llama_component_ablation_prefill/manual-run'
```

Re-render plots and the result-local report for an existing result directory:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<timestamp>'
```

Re-render and refresh the repo-root benchmark index:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_replace_ln_report.py --output_dir results/llama_replace_ln_prefill/<timestamp> --refresh_root_index'
```

Re-render the component ablation report for an existing result directory:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_component_ablation_report.py --output_dir results/llama_component_ablation_prefill/<timestamp>'
```

Re-render and refresh the repo-root component ablation index:

```sh
fish -lc 'cd <repo_root>; source .venv/bin/activate.fish; python scripts/benchmarks/render_llama_component_ablation_report.py --output_dir results/llama_component_ablation_prefill/<timestamp> --refresh_root_index'
```

## Runtime Assumptions

- Benchmarks assume CUDA is available and use `cuda:0`.
- Model weights and linear layers are created in `bfloat16`.
- `python/monitor/gpu_monitor.py` depends on NVML via `pynvml` for power and clock measurements.
- The Llama `34B` and `70B` configs preserve large-model tensor shapes but reduce layer count to fit an A100 40GB persistent-memory budget.
- `--replace_ln` is an ablation flag, not a numerically equivalent model variant.
- `--replace_attention`, `--replace_rope`, and `--replace_activation` are also ablation flags intended for component-level performance study, not numerically equivalent model variants.
- The canonical single-run and batch benchmark defaults are `warmup=5`, `repeat=10`, and `monitor_interval=0.01`.
- `LLAMA2_MAX_POSITION_EMBEDDINGS = 16384` is only a positional limit. It does not guarantee that every `prompt_len=16384` benchmark combination will fit or run stably on the current GPU.
- The standard batch benchmark matrix is now fixed to `16/32/64/128/256/512/1024/2048/4096/8192`. Prompt lengths outside that set should be treated as ad hoc reference runs.
- The current A100 40GB fitting logic only constrains persistent weight memory. It does not account for transient activations, intermediate tensors, workspaces, or all runtime allocations.
- Each canonical benchmark run writes its own `summary.csv`, `metadata.json`, `plots/*.png`, `monitor/*.csv`, and result-local `BENCHMARK.md` inside a timestamped result directory.
- The component ablation benchmark writes the same artifact set inside `results/llama_component_ablation_prefill/<timestamp>/` and refreshes `BENCHMARK_COMPONENT_ABLATION.md` when requested.
- The batch runner records individual run failures to `summary.csv` and continues with the remaining matrix entries.

## Working Notes For Future Edits

- Check imports before editing because the repo currently contains both moved and deleted paths in git status.
- Keep README benchmark claims aligned with the actual scripts and generated outputs.
- When a command fails because `python` is missing, treat that as an environment activation problem first, not a project packaging problem.
- For long benchmark runs, write partial progress to `summary.csv` as results arrive instead of buffering everything until the end.
