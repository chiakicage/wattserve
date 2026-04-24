#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_COMPONENT_ABLATION_INDEX_PATH = (
    REPO_ROOT / "BENCHMARK_COMPONENT_ABLATION.md"
)
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
DEFAULT_VARIANTS = [
    "baseline",
    "replace_ln",
    "replace_attention",
    "replace_rope",
    "replace_activation",
]
PLOT_METRICS = [
    ("ttft_ms", "TTFT (ms)"),
    ("prefill_tflops_s", "Prefill TFLOPs/s"),
    ("avg_power_watts", "Avg Power (W)"),
    ("avg_gpu_clock_mhz", "Avg GPU Clock (MHz)"),
]
METRIC_AXIS_CONFIG = {
    "avg_power_watts": {
        "ylim": (-10, 410),
        "yticks": [0, 100, 200, 300, 400],
    },
    "avg_gpu_clock_mhz": {
        "ylim": (1050, 1450),
        "yticks": [1100, 1200, 1300, 1400],
    },
}
VARIANT_STYLES = {
    "baseline": {
        "color": "#1f77b4",
        "label": "baseline",
    },
    "replace_ln": {
        "color": "#d62728",
        "label": "replace_ln",
    },
    "replace_attention": {
        "color": "#2ca02c",
        "label": "replace_attention",
    },
    "replace_rope": {
        "color": "#ff7f0e",
        "label": "replace_rope",
    },
    "replace_activation": {
        "color": "#9467bd",
        "label": "replace_activation",
    },
}
PLOT_FIGSIZE = (16, 12)
PLOT_DPI = 240
PLOT_LINEWIDTH = 3.2
PLOT_MARKERSIZE = 8
PLOT_AXIS_TITLE_FONTSIZE = 18
PLOT_SUPTITLE_FONTSIZE = 22
PLOT_LABEL_FONTSIZE = 15
PLOT_TICK_LABELSIZE = 13
PLOT_LEGEND_FONTSIZE = 13
PLOT_GRID_LINEWIDTH = 1.2
PLOT_SPINE_LINEWIDTH = 1.8


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _prompt_lengths_display(prompt_lengths: list[int]) -> str:
    return "/".join(str(prompt_len) for prompt_len in prompt_lengths)


def _variants_display(variants: list[str]) -> str:
    return "/".join(variants)


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
            str(row["variant"]),
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
    standard_prompt_lengths = set(DEFAULT_PROMPT_LENGTHS)
    return [
        row
        for row in rows
        if int(row["prompt_len"]) in standard_prompt_lengths
    ]


def _successful_standard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in _standard_rows(rows) if row.get("status") == "ok"
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


def _group_success_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in _successful_standard_rows(rows):
        key = (str(row["model"]), int(row["prompt_len"]))
        grouped.setdefault(key, {})[str(row["variant"])] = row
    return grouped


def _pair_success_rows(
    rows: list[dict[str, Any]],
    variant: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped = _group_success_rows(rows)
    ordered_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for model in DEFAULT_MODELS:
        for prompt_len in DEFAULT_PROMPT_LENGTHS:
            variants = grouped.get((model, prompt_len), {})
            if "baseline" in variants and variant in variants:
                ordered_pairs.append((variants["baseline"], variants[variant]))
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
    grouped = _group_success_rows(rows)
    unpaired_rows = []
    for row in _successful_standard_rows(rows):
        key = (str(row["model"]), int(row["prompt_len"]))
        present_variants = set(grouped.get(key, {}))
        if row["variant"] == "baseline":
            if len(present_variants - {"baseline"}) == 0:
                unpaired_rows.append(row)
        elif "baseline" not in present_variants:
            unpaired_rows.append(row)
    return _sort_rows(unpaired_rows)


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
    variant_value: Any,
) -> float | None:
    if baseline_value in (None, "") or variant_value in (None, ""):
        return None
    baseline = float(baseline_value)
    variant = float(variant_value)
    if baseline == 0:
        return None
    return ((variant - baseline) / baseline) * 100.0


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


def configure_metric_axis(axis: Any, metric_name: str) -> None:
    config = METRIC_AXIS_CONFIG.get(metric_name, {})

    yscale = config.get("yscale")
    if yscale is not None:
        axis.set_yscale(yscale)

    ylim = config.get("ylim")
    if ylim is not None:
        axis.set_ylim(*ylim)

    yticks = config.get("yticks")
    if yticks is not None:
        axis.set_yticks(yticks)


def _style_axis_for_presentation(axis: Any) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(PLOT_SPINE_LINEWIDTH)
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=PLOT_TICK_LABELSIZE,
        width=PLOT_SPINE_LINEWIDTH,
        length=6,
    )


def generate_metric_plots(
    rows: list[dict[str, Any]],
    plots_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    successful_rows = _successful_standard_rows(rows)
    plots_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    for metric_name, y_label in PLOT_METRICS:
        fig, axes = plt.subplots(
            2, 2, figsize=PLOT_FIGSIZE, constrained_layout=True
        )
        any_data = False

        for axis, model in zip(axes.flat, DEFAULT_MODELS):
            plotted_variant = False
            for variant in DEFAULT_VARIANTS:
                model_rows = [
                    row
                    for row in successful_rows
                    if row["model"] == model and row["variant"] == variant
                ]
                model_rows = _sort_rows(model_rows)
                x_values = [int(row["prompt_len"]) for row in model_rows]
                y_values = [float(row[metric_name]) for row in model_rows]
                if not x_values:
                    continue
                plotted_variant = True
                any_data = True
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    markersize=PLOT_MARKERSIZE,
                    linewidth=PLOT_LINEWIDTH,
                    color=VARIANT_STYLES[variant]["color"],
                    label=VARIANT_STYLES[variant]["label"],
                )

            if not plotted_variant:
                axis.text(
                    0.5,
                    0.5,
                    "No successful data",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    fontsize=PLOT_LABEL_FONTSIZE,
                    fontweight="bold",
                )

            axis.set_title(
                f"Llama-{model}",
                fontsize=PLOT_AXIS_TITLE_FONTSIZE,
                fontweight="bold",
            )
            axis.set_xlabel(
                "Prompt Len",
                fontsize=PLOT_LABEL_FONTSIZE,
                fontweight="bold",
            )
            axis.set_ylabel(
                y_label,
                fontsize=PLOT_LABEL_FONTSIZE,
                fontweight="bold",
            )
            _configure_prompt_len_axis(axis)
            configure_metric_axis(axis, metric_name)
            _style_axis_for_presentation(axis)
            axis.grid(True, alpha=0.35, linewidth=PLOT_GRID_LINEWIDTH)

        legend_entries: dict[str, Any] = {}
        for axis in axes.flat:
            handles, labels = axis.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_entries[label] = handle
        if any_data and legend_entries:
            fig.legend(
                list(legend_entries.values()),
                list(legend_entries.keys()),
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=PLOT_LEGEND_FONTSIZE,
            )
        fig.suptitle(
            f"Llama component ablation: {y_label}",
            fontsize=PLOT_SUPTITLE_FONTSIZE,
            fontweight="bold",
        )

        output_path = plots_dir / f"{metric_name}.png"
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        created_paths.append(output_path)

    return created_paths


def build_result_benchmark_markdown(
    rows: list[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any],
) -> str:
    failed_rows = _collect_failed_rows(rows)
    unpaired_success_rows = _collect_unpaired_success_rows(rows)
    excluded_prompt_lengths = metadata.get("excluded_prompt_lengths", [])
    report_variants = metadata.get("report_variants", DEFAULT_VARIANTS)
    plots_dir_display = _display_path(output_dir / "plots")
    output_dir_display = _display_path(output_dir)
    summary_csv_display = _display_path(output_dir / "summary.csv")
    metadata_display = _display_path(output_dir / "metadata.json")
    environment = metadata.get("environment", {})
    device_name = environment.get("cuda_device_name", "n/a")

    lines = [
        "# Llama Component Ablation Benchmark",
        "",
        f"Generated at `{metadata.get('render_generated_at_utc', metadata.get('run_started_at_utc', 'n/a'))}`.",
        "",
        "## Summary",
        "",
        (
            "- Standard matrix: "
            f"`7B/13B/34B/70B` x "
            f"`{_prompt_lengths_display(DEFAULT_PROMPT_LENGTHS)}` x "
            f"`{_variants_display(report_variants)}`"
        ),
        f"- Result directory: `{output_dir_display}`",
        f"- Summary CSV: `{summary_csv_display}`",
        f"- Metadata: `{metadata_display}`",
        f"- Plots directory: `{plots_dir_display}`",
        "- These variants are component ablations for performance study, not numerically equivalent model variants.",
        "- Prompt lengths outside the standard matrix are excluded from the summary tables and plots in this report.",
    ]

    if excluded_prompt_lengths:
        excluded_display = "/".join(
            str(length) for length in excluded_prompt_lengths
        )
        lines.append(
            f"- Non-standard prompt lengths excluded from this report and plots: `{excluded_display}`"
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
            (
                "- Warmup / repeat / monitor interval: "
                f"`{metadata.get('warmup', 'n/a')}` / "
                f"`{metadata.get('repeat', 'n/a')}` / "
                f"`{metadata.get('monitor_interval', 'n/a')}`"
            ),
            "",
            "## Plots",
            "",
            "### TTFT",
            "",
            "![TTFT](plots/ttft_ms.png)",
            "",
            "### Prefill TFLOPs/s",
            "",
            "![Prefill TFLOPs/s](plots/prefill_tflops_s.png)",
            "",
            "### Avg Power",
            "",
            "![Avg Power](plots/avg_power_watts.png)",
            "",
            "### Avg GPU Clock",
            "",
            "![Avg GPU Clock](plots/avg_gpu_clock_mhz.png)",
        ]
    )

    for variant in DEFAULT_VARIANTS:
        if variant == "baseline":
            continue
        paired_rows = _pair_success_rows(rows, variant)
        lines.extend(
            [
                "",
                f"## Baseline vs {variant}",
                "",
            ]
        )
        if paired_rows:
            lines.extend(
                [
                    "| Model | Prompt Len | baseline TTFT (ms) | "
                    f"{variant} TTFT (ms) | delta TTFT | baseline TFLOPs/s | "
                    f"{variant} TFLOPs/s | delta TFLOPs/s | "
                    "baseline Avg Power (W) | "
                    f"{variant} Avg Power (W) | delta Power | "
                    "baseline Avg GPU Clock (MHz) | "
                    f"{variant} Avg GPU Clock (MHz) | delta Clock |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for baseline_row, variant_row in paired_rows:
                lines.append(
                    "| "
                    f"{baseline_row['model']} | "
                    f"{baseline_row['prompt_len']} | "
                    f"{_format_float(baseline_row['ttft_ms'])} | "
                    f"{_format_float(variant_row['ttft_ms'])} | "
                    f"{_format_delta(_percentage_delta(baseline_row['ttft_ms'], variant_row['ttft_ms']))} | "
                    f"{_format_float(baseline_row['prefill_tflops_s'])} | "
                    f"{_format_float(variant_row['prefill_tflops_s'])} | "
                    f"{_format_delta(_percentage_delta(baseline_row['prefill_tflops_s'], variant_row['prefill_tflops_s']))} | "
                    f"{_format_float(baseline_row['avg_power_watts'])} | "
                    f"{_format_float(variant_row['avg_power_watts'])} | "
                    f"{_format_delta(_percentage_delta(baseline_row['avg_power_watts'], variant_row['avg_power_watts']))} | "
                    f"{_format_float(baseline_row['avg_gpu_clock_mhz'])} | "
                    f"{_format_float(variant_row['avg_gpu_clock_mhz'])} | "
                    f"{_format_delta(_percentage_delta(baseline_row['avg_gpu_clock_mhz'], variant_row['avg_gpu_clock_mhz']))} |"
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
        "# Latest Llama Component Ablation Benchmark",
        "",
        "This file indexes the latest multi-component Llama ablation benchmark report.",
        "",
        f"- Latest result directory: `{latest_dir_display}`",
        (
            "- Standard prompt lengths: "
            f"`{_prompt_lengths_display(DEFAULT_PROMPT_LENGTHS)}`"
        ),
        (
            "- Variants: "
            f"`{_variants_display(metadata.get('report_variants', DEFAULT_VARIANTS))}`"
        ),
        f"- Latest report: [{_display_path(report_path)}]({_display_path(report_path)})",
        f"- Latest summary CSV: [{_display_path(summary_path)}]({_display_path(summary_path)})",
        f"- Latest metadata: [{_display_path(metadata_path)}]({_display_path(metadata_path)})",
        f"- Latest plots directory: `{_display_path(plots_dir)}`",
        "- Prompt lengths outside the standard matrix may still appear in older result directories and should be treated as non-canonical reference data.",
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
    root_index_path: Path = ROOT_COMPONENT_ABLATION_INDEX_PATH,
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
    metadata["report_variants"] = DEFAULT_VARIANTS
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
        description=(
            "Render plots and BENCHMARK.md for a Llama component ablation "
            "benchmark result directory."
        )
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
        help=(
            "Path to a summary.csv file. The parent directory is treated "
            "as the result directory."
        ),
    )
    parser.add_argument(
        "--refresh_root_index",
        action="store_true",
        help="Refresh the repo-root BENCHMARK_COMPONENT_ABLATION.md index.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result_paths = render_result_report(
        output_dir=args.output_dir,
        summary_csv_path=args.summary_csv,
        refresh_root_index=args.refresh_root_index,
    )
    print(f"BENCHMARK.md: {result_paths['benchmark_markdown']}")
    print(f"Plots: {result_paths['plots_dir']}")
    if args.refresh_root_index:
        print(
            "Root BENCHMARK index: "
            f"{ROOT_COMPONENT_ABLATION_INDEX_PATH}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
