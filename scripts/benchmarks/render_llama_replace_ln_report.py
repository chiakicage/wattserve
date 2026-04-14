#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_BENCHMARK_INDEX_PATH = REPO_ROOT / "BENCHMARK.md"
DEFAULT_MODELS = ["7B", "13B", "34B", "70B"]
DEFAULT_PROMPT_LENGTHS = [512, 1024, 2048, 8192]
DEFAULT_VARIANTS = ["baseline", "replace_ln"]
PLOT_METRICS = [
    ("ttft_ms", "TTFT (ms)"),
    ("avg_power_watts", "Avg Power (W)"),
    ("avg_gpu_clock_mhz", "Avg GPU Clock (MHz)"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_order = {model: index for index, model in enumerate(DEFAULT_MODELS)}
    prompt_order = {
        prompt_len: index
        for index, prompt_len in enumerate(DEFAULT_PROMPT_LENGTHS)
    }
    variant_order = {
        variant: index for index, variant in enumerate(DEFAULT_VARIANTS)
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


def load_summary_rows(summary_csv_path: Path) -> list[dict[str, Any]]:
    with summary_csv_path.open("r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def write_metadata_json(metadata_path: Path, metadata: dict[str, Any]) -> None:
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def _standard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row["prompt_len"]) in set(DEFAULT_PROMPT_LENGTHS)
    ]


def _excluded_prompt_lengths(rows: list[dict[str, Any]]) -> list[int]:
    standard_prompt_lengths = set(DEFAULT_PROMPT_LENGTHS)
    return sorted(
        {
            int(row["prompt_len"])
            for row in rows
            if int(row["prompt_len"]) not in standard_prompt_lengths
        }
    )


def _pair_success_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    successful_rows = [
        row for row in _standard_rows(rows) if row.get("status") == "ok"
    ]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in successful_rows:
        key = (str(row["model"]), int(row["prompt_len"]))
        grouped.setdefault(key, {})[str(row["variant"])] = row

    ordered_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for model in DEFAULT_MODELS:
        for prompt_len in DEFAULT_PROMPT_LENGTHS:
            variants = grouped.get((model, prompt_len), {})
            if "baseline" in variants and "replace_ln" in variants:
                ordered_pairs.append(
                    (variants["baseline"], variants["replace_ln"])
                )
    return ordered_pairs


def _collect_failed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    standard_prompt_lengths = set(DEFAULT_PROMPT_LENGTHS)
    return _sort_rows(
        [
            row
            for row in rows
            if int(row["prompt_len"]) in standard_prompt_lengths
            and row.get("status") != "ok"
        ]
    )


def _collect_unpaired_success_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    paired_keys = {
        (baseline["model"], int(baseline["prompt_len"]))
        for baseline, _ in _pair_success_rows(rows)
    }
    unpaired = []
    for row in _standard_rows(rows):
        if row.get("status") != "ok":
            continue
        key = (row["model"], int(row["prompt_len"]))
        if key not in paired_keys:
            unpaired.append(row)
    return _sort_rows(unpaired)


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


def _configure_prompt_len_axis(axis: Any) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(DEFAULT_PROMPT_LENGTHS)
    axis.set_xticklabels(
        [str(prompt_len) for prompt_len in DEFAULT_PROMPT_LENGTHS]
    )
    axis.minorticks_off()
    axis.set_xlim(
        DEFAULT_PROMPT_LENGTHS[0] * 0.85, DEFAULT_PROMPT_LENGTHS[-1] * 1.15
    )


def generate_metric_plots(
    rows: list[dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    paired_rows = _pair_success_rows(rows)
    plots_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    for metric_name, y_label in PLOT_METRICS:
        fig, axes = plt.subplots(
            2, 2, figsize=(14, 10), constrained_layout=True
        )
        any_data = False

        for axis, model in zip(axes.flat, DEFAULT_MODELS):
            model_pairs = [
                pair for pair in paired_rows if pair[0]["model"] == model
            ]
            x_values = [
                int(baseline["prompt_len"]) for baseline, _ in model_pairs
            ]
            baseline_values = [
                float(baseline[metric_name]) for baseline, _ in model_pairs
            ]
            replace_values = [
                float(replace[metric_name]) for _, replace in model_pairs
            ]

            if x_values:
                any_data = True
                axis.plot(
                    x_values,
                    baseline_values,
                    marker="o",
                    linewidth=2,
                    color="#1f77b4",
                    label="baseline",
                )
                axis.plot(
                    x_values,
                    replace_values,
                    marker="o",
                    linewidth=2,
                    color="#d62728",
                    label="replace_ln",
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No paired data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )

            axis.set_title(f"Llama-{model}")
            axis.set_xlabel("Prompt Len")
            axis.set_ylabel(y_label)
            _configure_prompt_len_axis(axis)
            axis.grid(True, alpha=0.3)

        if any_data:
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
            )
        fig.suptitle(f"Llama replace_ln: {y_label}", fontsize=16)

        output_path = plots_dir / f"{metric_name}.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        created_paths.append(output_path)

    return created_paths


def build_result_benchmark_markdown(
    rows: list[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any],
) -> str:
    paired_rows = _pair_success_rows(rows)
    failed_rows = _collect_failed_rows(rows)
    unpaired_success_rows = _collect_unpaired_success_rows(rows)
    excluded_prompt_lengths = metadata.get("excluded_prompt_lengths", [])
    plots_dir_display = _display_path(output_dir / "plots")
    output_dir_display = _display_path(output_dir)
    summary_csv_display = _display_path(output_dir / "summary.csv")
    metadata_display = _display_path(output_dir / "metadata.json")
    environment = metadata.get("environment", {})
    device_name = environment.get("cuda_device_name", "n/a")

    lines = [
        "# Llama `replace_ln` Benchmark",
        "",
        f"Generated at `{metadata.get('render_generated_at_utc', metadata.get('run_started_at_utc', 'n/a'))}`.",
        "",
        "## Summary",
        "",
        "- Standard matrix: `7B/13B/34B/70B` x `512/1024/2048/8192` x `baseline/replace_ln`",
        f"- Result directory: `{output_dir_display}`",
        f"- Summary CSV: `{summary_csv_display}`",
        f"- Metadata: `{metadata_display}`",
        f"- Plots directory: `{plots_dir_display}`",
        "- `--replace_ln` is an ablation flag, not a numerically equivalent model variant.",
        "- `16384` has been removed from the standard benchmark matrix because it is not reliable on the current setup.",
    ]

    if excluded_prompt_lengths:
        excluded_display = "/".join(
            str(length) for length in excluded_prompt_lengths
        )
        lines.append(
            f"- Historical prompt lengths excluded from this report and plots: `{excluded_display}`"
        )

    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: `{environment.get('python_version', 'n/a')}`",
            f"- Torch: `{environment.get('torch_version', 'n/a')}`",
            f"- CUDA available: `{environment.get('cuda_available', 'n/a')}`",
            f"- CUDA device: `{device_name}`",
            f"- Warmup / repeat / monitor interval: `{metadata.get('warmup', 'n/a')}` / `{metadata.get('repeat', 'n/a')}` / `{metadata.get('monitor_interval', 'n/a')}`",
            "",
            "## Plots",
            "",
            "### TTFT",
            "",
            "![TTFT](plots/ttft_ms.png)",
            "",
            "### Avg Power",
            "",
            "![Avg Power](plots/avg_power_watts.png)",
            "",
            "### Avg GPU Clock",
            "",
            "![Avg GPU Clock](plots/avg_gpu_clock_mhz.png)",
            "",
            "## Successful Pairs",
            "",
        ]
    )

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


def build_root_index_markdown(
    latest_output_dir: Path,
    metadata: dict[str, Any],
) -> str:
    latest_dir_display = _display_path(latest_output_dir)
    report_path = latest_output_dir / "BENCHMARK.md"
    summary_path = latest_output_dir / "summary.csv"
    metadata_path = latest_output_dir / "metadata.json"
    plots_dir = latest_output_dir / "plots"

    lines = [
        "# Latest Llama `replace_ln` Benchmark",
        "",
        "This file indexes the latest canonical Llama benchmark report.",
        "",
        f"- Latest result directory: `{latest_dir_display}`",
        f"- Standard prompt lengths: `{'/'.join(str(prompt_len) for prompt_len in DEFAULT_PROMPT_LENGTHS)}`",
        f"- Latest report: [{_display_path(report_path)}]({_display_path(report_path)})",
        f"- Latest summary CSV: [{_display_path(summary_path)}]({_display_path(summary_path)})",
        f"- Latest metadata: [{_display_path(metadata_path)}]({_display_path(metadata_path)})",
        f"- Latest plots directory: `{_display_path(plots_dir)}`",
        "- `16384` is no longer part of the standard benchmark matrix. Older result directories may still contain historical `16384` rows and should be treated as non-canonical reference data.",
    ]

    if metadata.get("run_started_at_utc"):
        lines.append(
            f"- Latest benchmark run started at: `{metadata['run_started_at_utc']}`"
        )

    return "\n".join(lines) + "\n"


def render_result_report(
    output_dir: Path | None = None,
    summary_csv_path: Path | None = None,
    refresh_root_index: bool = False,
    root_index_path: Path = ROOT_BENCHMARK_INDEX_PATH,
) -> dict[str, Path]:
    if output_dir is None and summary_csv_path is None:
        raise ValueError(
            "Either output_dir or summary_csv_path must be provided"
        )

    if summary_csv_path is None:
        assert output_dir is not None
        summary_csv_path = output_dir / "summary.csv"
    else:
        output_dir = summary_csv_path.parent

    assert output_dir is not None
    metadata_path = output_dir / "metadata.json"
    benchmark_md_path = output_dir / "BENCHMARK.md"
    plots_dir = output_dir / "plots"

    if not summary_csv_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv_path}")

    rows = load_summary_rows(summary_csv_path)
    metadata = load_metadata(metadata_path)
    plot_paths = generate_metric_plots(rows, plots_dir)

    metadata["benchmark_markdown"] = str(benchmark_md_path)
    metadata["plots_dir"] = str(plots_dir)
    metadata["plot_files"] = [str(path) for path in plot_paths]
    metadata["report_prompt_lengths"] = DEFAULT_PROMPT_LENGTHS
    metadata["excluded_prompt_lengths"] = _excluded_prompt_lengths(rows)
    metadata["render_generated_at_utc"] = _utc_now_iso()
    write_metadata_json(metadata_path, metadata)

    benchmark_md = build_result_benchmark_markdown(
        rows=rows,
        output_dir=output_dir,
        metadata=metadata,
    )
    benchmark_md_path.write_text(benchmark_md)

    if refresh_root_index:
        root_index = build_root_index_markdown(output_dir, metadata)
        root_index_path.write_text(root_index)

    return {
        "output_dir": output_dir,
        "summary_csv": summary_csv_path,
        "metadata_json": metadata_path,
        "benchmark_markdown": benchmark_md_path,
        "plots_dir": plots_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render plots and BENCHMARK.md for a Llama benchmark result directory."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output_dir",
        type=Path,
        help="Result directory containing summary.csv and metadata.json.",
    )
    group.add_argument(
        "--summary_csv",
        type=Path,
        help="Path to a summary.csv file. The parent directory is treated as the result directory.",
    )
    parser.add_argument(
        "--refresh_root_index",
        action="store_true",
        help="Refresh the repo-root BENCHMARK.md index to point at this result directory.",
    )
    parser.add_argument(
        "--root_index_path",
        type=Path,
        default=ROOT_BENCHMARK_INDEX_PATH,
        help="Path to the repo-root BENCHMARK.md index file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result_paths = render_result_report(
        output_dir=args.output_dir,
        summary_csv_path=args.summary_csv,
        refresh_root_index=args.refresh_root_index,
        root_index_path=args.root_index_path,
    )
    print(f"Result BENCHMARK.md: {result_paths['benchmark_markdown']}")
    print(f"Plots directory: {result_paths['plots_dir']}")
    if args.refresh_root_index:
        print(f"Root BENCHMARK index: {args.root_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
