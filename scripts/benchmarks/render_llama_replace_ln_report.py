#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .device_snapshot import (
        list_device_snapshots,
        resolve_device_label,
        resolve_device_slug,
        resolve_device_snapshot_dir,
    )
except ImportError:
    from device_snapshot import (  # type: ignore[no-redef]
        list_device_snapshots,
        resolve_device_label,
        resolve_device_slug,
        resolve_device_snapshot_dir,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_BENCHMARK_INDEX_PATH = REPO_ROOT / "BENCHMARK.md"
GIT_TRACKED_LATEST_RESULTS_ROOT = (
    REPO_ROOT / "results" / "llama_replace_ln_prefill" / "latest"
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
DEFAULT_VARIANTS = ["baseline", "replace_ln"]
PLOT_METRICS = [
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
        "ylim": (0, 1450),
        "yticks": [0, 250, 500, 750, 1000, 1250, 1450],
    },
}
PLOT_FIGSIZE = (18, 13)
PLOT_DPI = 240
PLOT_LINEWIDTH = 4.0
PLOT_MARKERSIZE = 9
PLOT_AXIS_TITLE_FONTSIZE = 22
PLOT_SUPTITLE_FONTSIZE = 28
PLOT_LABEL_FONTSIZE = 19
PLOT_TICK_LABELSIZE = 16
PLOT_LEGEND_FONTSIZE = 15
PLOT_GRID_LINEWIDTH = 1.4
PLOT_SPINE_LINEWIDTH = 2.4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _prompt_lengths_display(prompt_lengths: list[int]) -> str:
    return "/".join(str(prompt_len) for prompt_len in prompt_lengths)


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
        length=7,
    )
    for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        tick_label.set_fontweight("bold")


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
            2, 2, figsize=PLOT_FIGSIZE, constrained_layout=True
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
                    markersize=PLOT_MARKERSIZE,
                    linewidth=PLOT_LINEWIDTH,
                    color="#1f77b4",
                    label="baseline",
                )
                axis.plot(
                    x_values,
                    replace_values,
                    marker="o",
                    markersize=PLOT_MARKERSIZE,
                    linewidth=PLOT_LINEWIDTH,
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

        if any_data:
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=PLOT_LEGEND_FONTSIZE,
            )
        fig.suptitle(
            f"Llama replace_ln: {y_label}",
            fontsize=PLOT_SUPTITLE_FONTSIZE,
            fontweight="bold",
        )

        output_path = plots_dir / f"{metric_name}.png"
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        created_paths.append(output_path)

    return created_paths


def generate_tflops_uplift_plot(
    rows: list[dict[str, Any]],
    plots_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    paired_rows = _pair_success_rows(rows)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 2, figsize=PLOT_FIGSIZE, constrained_layout=True
    )

    for axis, model in zip(axes.flat, DEFAULT_MODELS):
        model_pairs = [
            pair for pair in paired_rows if pair[0]["model"] == model
        ]
        x_values = [int(baseline["prompt_len"]) for baseline, _ in model_pairs]
        uplift_values = [
            _percentage_delta(
                baseline["prefill_tflops_s"], replace["prefill_tflops_s"]
            )
            for baseline, replace in model_pairs
        ]

        if x_values and all(value is not None for value in uplift_values):
            axis.plot(
                x_values,
                [float(value) for value in uplift_values if value is not None],
                marker="o",
                markersize=PLOT_MARKERSIZE,
                linewidth=PLOT_LINEWIDTH,
                color="#2ca02c",
            )
            axis.axhline(
                0.0,
                color="#7f7f7f",
                linewidth=PLOT_GRID_LINEWIDTH + 0.3,
                linestyle="--",
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No paired data",
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
            "TFLOPs Uplift (%)",
            fontsize=PLOT_LABEL_FONTSIZE,
            fontweight="bold",
        )
        _configure_prompt_len_axis(axis)
        _style_axis_for_presentation(axis)
        axis.grid(True, alpha=0.35, linewidth=PLOT_GRID_LINEWIDTH)

    fig.suptitle(
        "Llama replace_ln: TFLOPs uplift vs baseline",
        fontsize=PLOT_SUPTITLE_FONTSIZE,
        fontweight="bold",
    )

    output_path = plots_dir / "prefill_tflops_uplift_pct.png"
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
    source_output_dir = metadata.get("source_output_dir")
    source_output_dir_display = None
    if source_output_dir:
        source_output_dir_display = _display_path(Path(source_output_dir))
    environment = metadata.get("environment", {})
    device_name = environment.get("cuda_device_name", "n/a")

    lines = [
        "# Llama `replace_ln` Benchmark",
        "",
        f"Generated at `{metadata.get('render_generated_at_utc', metadata.get('run_started_at_utc', 'n/a'))}`.",
        "",
        "## Summary",
        "",
        (
            "- Standard matrix: "
            f"`7B/13B/34B/70B` x `{_prompt_lengths_display(DEFAULT_PROMPT_LENGTHS)}` "
            "x `baseline/replace_ln`"
        ),
        f"- Result directory: `{output_dir_display}`",
        f"- Summary CSV: `{summary_csv_display}`",
        f"- Metadata: `{metadata_display}`",
        f"- Plots directory: `{plots_dir_display}`",
        "- `--replace_ln` is an ablation flag, not a numerically equivalent model variant.",
        "- Prompt lengths outside the standard matrix are excluded from the summary tables and plots in this report.",
    ]

    if (
        source_output_dir_display is not None
        and source_output_dir_display != output_dir_display
    ):
        lines.append(f"- Source run directory: `{source_output_dir_display}`")
        lines.append(
            "- This directory is the git-tracked latest snapshot of that source run."
        )

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
            f"- Warmup / repeat / monitor interval: `{metadata.get('warmup', 'n/a')}` / `{metadata.get('repeat', 'n/a')}` / `{metadata.get('monitor_interval', 'n/a')}`",
            "",
            "## Plots",
            "",
            "### Prefill TFLOPs/s",
            "",
            "![Prefill TFLOPs/s](plots/prefill_tflops_s.png)",
            "",
            "### TFLOPs Uplift vs Baseline",
            "",
            "![TFLOPs Uplift vs Baseline](plots/prefill_tflops_uplift_pct.png)",
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


def _device_snapshot_table_lines(
    snapshots: list[dict[str, Any]],
) -> list[str]:
    if not snapshots:
        return ["No device-specific latest snapshots are currently tracked."]

    lines = [
        "| Device | Slug | Report | Summary CSV | Metadata | Source Run | Run Started At |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for snapshot in snapshots:
        source_output_dir = snapshot.get("source_output_dir")
        source_display = (
            f"`{_display_path(Path(source_output_dir))}`"
            if source_output_dir
            else "n/a"
        )
        lines.append(
            "| "
            f"{snapshot['label']} | "
            f"`{snapshot['slug']}` | "
            f"[report]({_display_path(snapshot['benchmark_markdown'])}) | "
            f"[summary]({_display_path(snapshot['summary_csv'])}) | "
            f"[metadata]({_display_path(snapshot['metadata_json'])}) | "
            f"{source_display} | "
            f"`{snapshot.get('run_started_at_utc') or 'n/a'}` |"
        )
    return lines


def build_latest_root_benchmark_markdown(
    latest_root_dir: Path,
    snapshots: list[dict[str, Any]],
) -> str:
    latest_root_display = _display_path(latest_root_dir)
    lines = [
        "# Llama `replace_ln` Latest Snapshots",
        "",
        "This directory stores the git-tracked latest canonical Llama `replace_ln` benchmark snapshot for each device.",
        "",
        f"- Snapshots root: `{latest_root_display}`",
        f"- Standard prompt lengths: `{_prompt_lengths_display(DEFAULT_PROMPT_LENGTHS)}`",
        "- Publishing is device-scoped and explicit. Timestamped runs do not overwrite these snapshots unless requested.",
        "- Prompt lengths outside the standard matrix may still appear in older result directories and should be treated as non-canonical reference data.",
        "",
        "## Devices",
        "",
        *_device_snapshot_table_lines(snapshots),
    ]
    return "\n".join(lines) + "\n"


def build_root_index_markdown(
    latest_root_dir: Path,
    snapshots: list[dict[str, Any]],
) -> str:
    latest_root_display = _display_path(latest_root_dir)
    latest_root_report = latest_root_dir / "BENCHMARK.md"
    lines = [
        "# Latest Llama `replace_ln` Benchmark",
        "",
        "This file indexes the device-specific git-tracked latest canonical Llama benchmark snapshots.",
        "",
        f"- Git-tracked latest snapshots root: `{latest_root_display}`",
        f"- Latest-by-device index: [{_display_path(latest_root_report)}]({_display_path(latest_root_report)})",
        f"- Standard prompt lengths: `{_prompt_lengths_display(DEFAULT_PROMPT_LENGTHS)}`",
        "- Prompt lengths outside the standard matrix may still appear in older result directories and should be treated as non-canonical reference data.",
        "",
        "## Devices",
        "",
        *_device_snapshot_table_lines(snapshots),
    ]
    return "\n".join(lines) + "\n"


def refresh_latest_indices(
    latest_root_dir: Path,
    root_index_path: Path = ROOT_BENCHMARK_INDEX_PATH,
) -> None:
    latest_root_dir.mkdir(parents=True, exist_ok=True)
    snapshots = list_device_snapshots(latest_root_dir)
    (latest_root_dir / "BENCHMARK.md").write_text(
        build_latest_root_benchmark_markdown(latest_root_dir, snapshots)
    )
    root_index_path.write_text(
        build_root_index_markdown(latest_root_dir, snapshots)
    )


def publish_git_tracked_latest_snapshot(
    source_output_dir: Path,
    root_index_path: Path = ROOT_BENCHMARK_INDEX_PATH,
    git_snapshot_output_dir: Path = GIT_TRACKED_LATEST_RESULTS_ROOT,
) -> dict[str, Path]:
    source_output_dir = source_output_dir.resolve()
    git_snapshot_output_dir = git_snapshot_output_dir.resolve()

    source_summary_csv_path = source_output_dir / "summary.csv"
    if not source_summary_csv_path.exists():
        raise FileNotFoundError(
            f"summary.csv not found for git snapshot publish: {source_summary_csv_path}"
        )

    source_metadata_path = source_output_dir / "metadata.json"
    source_monitor_dir = source_output_dir / "monitor"
    snapshot_metadata = load_metadata(source_metadata_path)
    device_slug = resolve_device_slug(snapshot_metadata)
    device_label = resolve_device_label(snapshot_metadata)
    device_snapshot_output_dir = resolve_device_snapshot_dir(
        git_snapshot_output_dir, snapshot_metadata
    )

    if source_output_dir != device_snapshot_output_dir:
        device_snapshot_output_dir.parent.mkdir(parents=True, exist_ok=True)
        if device_snapshot_output_dir.exists():
            shutil.rmtree(device_snapshot_output_dir)
        device_snapshot_output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(
            source_summary_csv_path,
            device_snapshot_output_dir / "summary.csv",
        )
        if source_monitor_dir.exists():
            shutil.copytree(
                source_monitor_dir,
                device_snapshot_output_dir / "monitor",
            )

    snapshot_metadata["source_output_dir"] = str(source_output_dir)
    snapshot_metadata["source_summary_csv"] = str(source_summary_csv_path)
    snapshot_metadata["source_metadata_json"] = str(source_metadata_path)
    snapshot_metadata["published_snapshot_root_dir"] = str(
        git_snapshot_output_dir
    )
    snapshot_metadata["published_snapshot_dir"] = str(
        device_snapshot_output_dir
    )
    snapshot_metadata["published_snapshot_generated_at_utc"] = _utc_now_iso()
    snapshot_metadata["published_device_slug"] = device_slug
    snapshot_metadata["published_device_label"] = device_label
    snapshot_metadata["output_dir"] = str(device_snapshot_output_dir)
    snapshot_metadata["summary_csv"] = str(
        device_snapshot_output_dir / "summary.csv"
    )
    snapshot_metadata["benchmark_markdown"] = str(
        device_snapshot_output_dir / "BENCHMARK.md"
    )
    snapshot_metadata["plots_dir"] = str(device_snapshot_output_dir / "plots")
    write_metadata_json(
        device_snapshot_output_dir / "metadata.json", snapshot_metadata
    )

    if source_output_dir != device_snapshot_output_dir:
        snapshot_paths = render_result_report(
            output_dir=device_snapshot_output_dir,
            refresh_root_index=False,
            root_index_path=root_index_path,
            git_snapshot_output_dir=git_snapshot_output_dir,
        )
    else:
        snapshot_paths = {
            "output_dir": device_snapshot_output_dir,
            "summary_csv": device_snapshot_output_dir / "summary.csv",
            "metadata_json": device_snapshot_output_dir / "metadata.json",
            "benchmark_markdown": device_snapshot_output_dir / "BENCHMARK.md",
            "plots_dir": device_snapshot_output_dir / "plots",
        }

    refresh_latest_indices(
        latest_root_dir=git_snapshot_output_dir,
        root_index_path=root_index_path,
    )
    return snapshot_paths


def render_result_report(
    output_dir: Path | None = None,
    summary_csv_path: Path | None = None,
    refresh_root_index: bool = False,
    root_index_path: Path = ROOT_BENCHMARK_INDEX_PATH,
    git_snapshot_output_dir: Path = GIT_TRACKED_LATEST_RESULTS_ROOT,
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
    plot_paths.append(generate_tflops_uplift_plot(rows, plots_dir))

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
        publish_git_tracked_latest_snapshot(
            source_output_dir=output_dir,
            root_index_path=root_index_path,
            git_snapshot_output_dir=git_snapshot_output_dir,
        )

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
        help=(
            "Refresh the repo-root BENCHMARK.md index and republish the "
            "git-tracked latest device snapshot under "
            "results/llama_replace_ln_prefill/latest/<device_slug>/."
        ),
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
        print(
            "Git-tracked latest snapshots root: "
            f"{GIT_TRACKED_LATEST_RESULTS_ROOT}"
        )
        print(f"Root BENCHMARK index: {args.root_index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
