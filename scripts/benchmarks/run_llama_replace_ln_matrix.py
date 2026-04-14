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


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "llama_replace_ln_prefill"
DEFAULT_BENCHMARK_MD_PATH = REPO_ROOT / "BENCHMARK.md"
DEFAULT_MODELS = ["7B", "13B", "34B", "70B"]
DEFAULT_PROMPT_LENGTHS = [512, 1024, 2048, 8192, 16384]
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
    variant_order = {
        variant: index for index, (variant, _) in enumerate(DEFAULT_VARIANTS)
    }
    return sorted(
        rows,
        key=lambda row: (
            model_order.get(row["model"], len(model_order)),
            int(row["prompt_len"]),
            variant_order.get(row["variant"], len(variant_order)),
        ),
    )


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


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


def _format_float(value: Any, precision: int = 2) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{precision}f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def _percentage_delta(
    baseline_value: Any,
    replace_value: Any,
) -> float | None:
    if baseline_value in (None, "") or replace_value in (None, ""):
        return None
    baseline = float(baseline_value)
    replace = float(replace_value)
    if baseline == 0:
        return None
    return ((replace - baseline) / baseline) * 100.0


def _pair_success_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    successful_rows = [row for row in rows if row.get("status") == "ok"]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in successful_rows:
        key = (str(row["model"]), int(row["prompt_len"]))
        grouped.setdefault(key, {})[str(row["variant"])] = row

    ordered_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in _sort_rows(successful_rows):
        key = (str(row["model"]), int(row["prompt_len"]))
        variants = grouped.get(key, {})
        if row["variant"] != "baseline":
            continue
        if "baseline" in variants and "replace_ln" in variants:
            ordered_pairs.append((variants["baseline"], variants["replace_ln"]))
    return ordered_pairs


def _collect_failed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _sort_rows([row for row in rows if row.get("status") != "ok"])


def _collect_unpaired_success_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    paired_keys = {
        (baseline["model"], int(baseline["prompt_len"]))
        for baseline, _ in _pair_success_rows(rows)
    }
    unpaired = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["model"], int(row["prompt_len"]))
        if key not in paired_keys:
            unpaired.append(row)
    return _sort_rows(unpaired)


def build_benchmark_markdown(
    rows: list[dict[str, Any]],
    run_started_at_utc: str,
    output_dir: Path,
    summary_csv_path: Path,
    metadata: dict[str, Any],
) -> str:
    paired_rows = _pair_success_rows(rows)
    failed_rows = _collect_failed_rows(rows)
    unpaired_success_rows = _collect_unpaired_success_rows(rows)
    output_dir_display = _display_path(output_dir)
    summary_csv_display = _display_path(summary_csv_path)
    environment = metadata.get("environment", {})
    device_name = environment.get("cuda_device_name", "n/a")

    lines = [
        "# Llama `replace_ln` Benchmark",
        "",
        f"Generated at `{run_started_at_utc}`.",
        "",
        "## Summary",
        "",
        "- Matrix: `7B/13B/34B/70B` x `512/1024/2048/8192/16384` x `baseline/replace_ln`",
        f"- Result directory: `{output_dir_display}`",
        f"- Summary CSV: `{summary_csv_display}`",
        "- `--replace_ln` is an ablation flag, not a numerically equivalent model variant.",
        "- `LLAMA2_MAX_POSITION_EMBEDDINGS = 16384` is only a position-length limit. The A100 40GB fitting logic only constrains persistent weight memory and does not guarantee that every long-sequence benchmark combination will fit transient activations and runtime workspaces.",
        "",
        "## Environment",
        "",
        f"- Python: `{environment.get('python_version', 'n/a')}`",
        f"- Torch: `{environment.get('torch_version', 'n/a')}`",
        f"- CUDA available: `{environment.get('cuda_available', 'n/a')}`",
        f"- CUDA device: `{device_name}`",
        f"- Warmup / repeat / monitor interval: `{metadata['warmup']}` / `{metadata['repeat']}` / `{metadata['monitor_interval']}`",
        "",
        "## Successful Pairs",
        "",
    ]

    if paired_rows:
        lines.extend(
            [
                "| Model | Prompt Len | baseline TFLOPs/s | replace_ln TFLOPs/s | delta TFLOPs/s | baseline TTFT (ms) | replace_ln TTFT (ms) | delta TTFT | baseline Avg Power (W) | replace_ln Avg Power (W) | delta Power | baseline Avg GPU Clock (MHz) | replace_ln Avg GPU Clock (MHz) | delta Clock |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for baseline_row, replace_row in paired_rows:
            lines.append(
                "| "
                f"{baseline_row['model']} | "
                f"{baseline_row['prompt_len']} | "
                f"{_format_float(baseline_row['prefill_tflops_s'])} | "
                f"{_format_float(replace_row['prefill_tflops_s'])} | "
                f"{_format_delta(_percentage_delta(baseline_row['prefill_tflops_s'], replace_row['prefill_tflops_s']))} | "
                f"{_format_float(baseline_row['ttft_ms'])} | "
                f"{_format_float(replace_row['ttft_ms'])} | "
                f"{_format_delta(_percentage_delta(baseline_row['ttft_ms'], replace_row['ttft_ms']))} | "
                f"{_format_float(baseline_row['avg_power_watts'])} | "
                f"{_format_float(replace_row['avg_power_watts'])} | "
                f"{_format_delta(_percentage_delta(baseline_row['avg_power_watts'], replace_row['avg_power_watts']))} | "
                f"{_format_float(baseline_row['avg_gpu_clock_mhz'])} | "
                f"{_format_float(replace_row['avg_gpu_clock_mhz'])} | "
                f"{_format_delta(_percentage_delta(baseline_row['avg_gpu_clock_mhz'], replace_row['avg_gpu_clock_mhz']))} |"
            )
    else:
        lines.append("No paired successful runs were recorded.")

    lines.extend(
        [
            "",
            "## Failed Runs",
            "",
        ]
    )

    if failed_rows:
        lines.extend(
            [
                "| Model | Prompt Len | Variant | Error Type |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for row in failed_rows:
            lines.append(
                "| "
                f"{row['model']} | {row['prompt_len']} | {row['variant']} | "
                f"{row.get('error_type') or 'unknown'} |"
            )
    else:
        lines.append("No failed runs were recorded.")

    if unpaired_success_rows:
        lines.extend(
            [
                "",
                "## Unpaired Successful Runs",
                "",
                "| Model | Prompt Len | Variant |",
                "| --- | ---: | --- |",
            ]
        )
        for row in unpaired_success_rows:
            lines.append(
                f"| {row['model']} | {row['prompt_len']} | {row['variant']} |"
            )

    return "\n".join(lines) + "\n"


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
    benchmark_md_path: Path,
    warmup: int,
    repeat: int,
    monitor_interval: float,
) -> list[dict[str, Any]]:
    bench_llama = _load_bench_llama_module()
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "summary.csv"
    metadata_path = output_dir / "metadata.json"
    run_started_at_utc = _utc_now_iso()
    rows: list[dict[str, Any]] = []

    metadata: dict[str, Any] = {
        "run_started_at_utc": run_started_at_utc,
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv_path),
        "benchmark_markdown": str(benchmark_md_path),
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

    benchmark_md = build_benchmark_markdown(
        rows=rows,
        run_started_at_utc=run_started_at_utc,
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        metadata=metadata,
    )
    benchmark_md_path.write_text(benchmark_md)
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
        "--benchmark_md_path",
        type=Path,
        default=DEFAULT_BENCHMARK_MD_PATH,
        help="Path for the generated Markdown summary.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations per benchmark.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of timed iterations per benchmark.",
    )
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=0.01,
        help="GPU monitor sampling interval in seconds.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir or (
        DEFAULT_RESULTS_ROOT / _timestamp_for_path()
    )

    run_llama_replace_ln_matrix(
        output_dir=output_dir,
        benchmark_md_path=args.benchmark_md_path,
        warmup=args.warmup,
        repeat=args.repeat,
        monitor_interval=args.monitor_interval,
    )
    print(f"Summary CSV: {output_dir / 'summary.csv'}")
    print(f"BENCHMARK.md: {args.benchmark_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
