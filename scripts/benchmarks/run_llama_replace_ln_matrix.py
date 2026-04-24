#!/usr/bin/env python3

import argparse
import csv
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .render_llama_replace_ln_report import (
        ROOT_BENCHMARK_INDEX_PATH,
        render_result_report,
    )
except ImportError:
    from render_llama_replace_ln_report import (  # type: ignore[no-redef]
        ROOT_BENCHMARK_INDEX_PATH,
        render_result_report,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "llama_replace_ln_prefill"
DEFAULT_MODELS = ["7B", "13B", "34B", "70B"]
DEFAULT_PROMPT_LENGTHS = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 10
DEFAULT_MONITOR_INTERVAL = 0.01
DEFAULT_VARIANTS = [
    ("baseline", False),
    ("replace_ln", True),
]
SUMMARY_FIELDNAMES = [
    "run_timestamp_utc",
    "model",
    "prompt_len",
    "variant",
    "replace_ln",
    "status",
    "error_type",
    "error_message",
    "canonical_num_hidden_layers",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "parameter_count",
    "parameter_count_with_lm_head",
    "estimated_runtime_memory_gib",
    "warmup",
    "repeat",
    "ttft_ms",
    "prefill_tflops_s",
    "avg_power_watts",
    "max_power_watts",
    "avg_gpu_clock_mhz",
    "max_gpu_clock_mhz",
    "monitor_sample_count",
    "monitor_csv",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_python_dir_on_path() -> None:
    python_dir = str(PYTHON_DIR)
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)


def _load_bench_llama_module() -> Any:
    _ensure_python_dir_on_path()
    return importlib.import_module("bench_llama")


def _timestamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_order = {model: index for index, model in enumerate(DEFAULT_MODELS)}
    prompt_order = {
        prompt_len: index
        for index, prompt_len in enumerate(DEFAULT_PROMPT_LENGTHS)
    }
    variant_order = {
        variant: index for index, (variant, _) in enumerate(DEFAULT_VARIANTS)
    }
    return sorted(
        rows,
        key=lambda row: (
            model_order.get(str(row["model"]), len(model_order)),
            prompt_order.get(int(row["prompt_len"]), len(prompt_order)),
            int(row["prompt_len"]),
            variant_order.get(str(row["variant"]), len(variant_order)),
        ),
    )


def write_summary_csv(
    rows: list[dict[str, Any]],
    summary_csv_path: Path,
) -> None:
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in _sort_rows(rows):
            writer.writerow(
                {
                    fieldname: row.get(fieldname, "")
                    for fieldname in SUMMARY_FIELDNAMES
                }
            )


def _safe_package_version(name: str) -> str | None:
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return None


def collect_environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_available": None,
        "cuda_device_name": None,
        "cuda_device_count": None,
        "flashinfer_version": _safe_package_version("flashinfer"),
        "transformers_version": _safe_package_version("transformers"),
    }

    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            metadata["cuda_device_count"] = torch.cuda.device_count()
            metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        metadata["torch_error"] = f"{type(exc).__name__}: {exc}"

    return metadata


def write_metadata_json(metadata_path: Path, metadata: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def run_llama_replace_ln_matrix(
    output_dir: Path,
    warmup: int,
    repeat: int,
    monitor_interval: float,
    refresh_root_index: bool = False,
    root_index_path: Path = ROOT_BENCHMARK_INDEX_PATH,
) -> list[dict[str, Any]]:
    bench_llama = _load_bench_llama_module()
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "summary.csv"
    metadata_path = output_dir / "metadata.json"
    benchmark_md_path = output_dir / "BENCHMARK.md"
    plots_dir = output_dir / "plots"
    run_started_at_utc = _utc_now_iso()
    rows: list[dict[str, Any]] = []

    metadata: dict[str, Any] = {
        "run_started_at_utc": run_started_at_utc,
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv_path),
        "benchmark_markdown": str(benchmark_md_path),
        "plots_dir": str(plots_dir),
        "models": DEFAULT_MODELS,
        "prompt_lengths": DEFAULT_PROMPT_LENGTHS,
        "variants": [variant for variant, _ in DEFAULT_VARIANTS],
        "warmup": warmup,
        "repeat": repeat,
        "monitor_interval": monitor_interval,
        "environment": collect_environment_metadata(),
    }
    write_metadata_json(metadata_path, metadata)

    for model in DEFAULT_MODELS:
        for prompt_len in DEFAULT_PROMPT_LENGTHS:
            for variant, replace_ln in DEFAULT_VARIANTS:
                monitor_csv_path = (
                    monitor_dir / f"{model}_prompt{prompt_len}_{variant}.csv"
                )
                print(
                    "Running "
                    f"model={model} prompt_len={prompt_len} variant={variant}",
                    flush=True,
                )
                result = bench_llama.benchmark(
                    config_name=model,
                    prompt_len=prompt_len,
                    replace_ln=replace_ln,
                    warmup=warmup,
                    repeat=repeat,
                    monitor_interval=monitor_interval,
                    monitor_csv_path=str(monitor_csv_path),
                )
                if result.get("monitor_csv"):
                    result["monitor_csv"] = monitor_csv_path.relative_to(
                        output_dir
                    ).as_posix()
                rows.append(result)
                write_summary_csv(rows, summary_csv_path)

    metadata["run_completed_at_utc"] = _utc_now_iso()
    metadata["row_count"] = len(rows)
    metadata["success_count"] = sum(
        1 for row in rows if row.get("status") == "ok"
    )
    metadata["failure_count"] = len(rows) - metadata["success_count"]
    write_metadata_json(metadata_path, metadata)

    render_result_report(
        output_dir=output_dir,
        refresh_root_index=refresh_root_index,
        root_index_path=root_index_path,
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Llama replace_ln benchmark matrix."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to "
            "results/llama_replace_ln_prefill/<UTC_TIMESTAMP>."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Number of warmup iterations per benchmark.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help="Number of timed iterations per benchmark.",
    )
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=DEFAULT_MONITOR_INTERVAL,
        help="GPU monitor sampling interval in seconds.",
    )
    parser.add_argument(
        "--publish_latest",
        action="store_true",
        help=(
            "Refresh the repo-root BENCHMARK.md index and republish the "
            "git-tracked latest device snapshot under "
            "results/llama_replace_ln_prefill/latest/<device_slug>/."
        ),
    )
    parser.add_argument(
        "--skip_root_index",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.publish_latest and args.skip_root_index:
        parser.error(
            "--publish_latest and --skip_root_index cannot be used together"
        )
    output_dir = args.output_dir or (
        DEFAULT_RESULTS_ROOT / _timestamp_for_path()
    )

    run_llama_replace_ln_matrix(
        output_dir=output_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        monitor_interval=args.monitor_interval,
        refresh_root_index=args.publish_latest and not args.skip_root_index,
    )
    print(f"Summary CSV: {output_dir / 'summary.csv'}")
    print(f"Result BENCHMARK.md: {output_dir / 'BENCHMARK.md'}")
    if args.publish_latest and not args.skip_root_index:
        print(f"Root BENCHMARK index: {ROOT_BENCHMARK_INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
