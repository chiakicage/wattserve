#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "state_chain_block_sweep"
DEFAULT_LATEST_DIR = DEFAULT_RESULTS_ROOT / "latest"
SINGLE_PLOT_FIGSIZE = (14, 8)
COMBINED_PLOT_FIGSIZE = (14, 16)
TITLE_FONTSIZE = 26
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 17
LEGEND_FONTSIZE = 17
LINE_WIDTH = 4.4
MARKER_SIZE = 11
AXIS_SPINE_WIDTH = 2.4

SUMMARY_FIELDNAMES = [
    "model",
    "batch_size",
    "iter_time_without_norm_ms",
    "iter_time_with_norm_ms",
    "total_extra_time_with_norm_ms",
    "norm_self_time_ms",
    "norm_time_gap_ms",
    "norm_time_gap_pct",
    "norm_time_ratio",
    "power_without_norm_watts",
    "power_with_norm_watts",
    "power_delta_watts",
    "clock_without_norm_mhz",
    "clock_with_norm_mhz",
    "clock_increase_without_norm_mhz",
    "clock_increase_without_norm_pct",
    "raw_gemm_without_norm_tflops_s",
    "raw_gemm_with_norm_tflops_s",
    "raw_gemm_delta_pct",
    "effective_without_norm_tflops_s",
    "effective_with_norm_tflops_s",
    "effective_delta_pct",
]


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _markdown_target(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return _display_path(path)


def _latest_timestamped_result(results_root: Path) -> Path:
    candidates = [
        path
        for path in results_root.iterdir()
        if path.is_dir()
        and path.name != "latest"
        and (path / "summary.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"no result directories under {results_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _model_sort_key(model: str) -> tuple[int, str]:
    try:
        return (int(model.rstrip("B")), model)
    except ValueError:
        return (10**9, model)


def _read_ok_rows(summary_csv: Path) -> list[dict[str, Any]]:
    with summary_csv.open(newline="") as csv_file:
        return [
            row for row in csv.DictReader(csv_file) if row.get("status") == "ok"
        ]


def _norm_self_time_ms(profile_csv: Path) -> float:
    if not profile_csv.exists():
        return 0.0
    total = 0.0
    with profile_csv.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            name = str(row.get("name", "")).lower()
            if "rmsnorm" in name or ("fused_add" in name and "norm" in name):
                total += float(row.get("self_cuda_time_ms") or 0.0)
    return total


def build_gap_rows(
    output_dir: Path, min_batch_size: int
) -> list[dict[str, Any]]:
    rows = _read_ok_rows(output_dir / "summary.csv")
    by_key = {
        (row["model"], int(row["batch_size"]), row["variant"]): row
        for row in rows
    }
    models = sorted({row["model"] for row in rows}, key=_model_sort_key)
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    gap_rows: list[dict[str, Any]] = []

    for model in models:
        for batch_size in batch_sizes:
            if batch_size <= min_batch_size:
                continue
            without = by_key.get((model, batch_size, "without_norm"))
            with_norm = by_key.get((model, batch_size, "with_norm"))
            if without is None or with_norm is None:
                continue

            total_extra_ms = _float(with_norm, "iter_time_ms") - _float(
                without, "iter_time_ms"
            )
            profile_csv = (
                output_dir
                / "kernel_profile"
                / f"{model}__batch_size={batch_size}__with_norm.csv"
            )
            norm_self_ms = _norm_self_time_ms(profile_csv)
            raw_without = _float(without, "gemm_tflops_s_raw")
            raw_with = _float(with_norm, "gemm_tflops_s_raw")
            effective_without = _float(without, "effective_tflops_s")
            effective_with = _float(with_norm, "effective_tflops_s")
            power_without = _float(without, "avg_power_watts")
            power_with = _float(with_norm, "avg_power_watts")
            clock_without = _float(without, "avg_gpu_clock_mhz")
            clock_with = _float(with_norm, "avg_gpu_clock_mhz")

            norm_time_gap_ms = total_extra_ms - norm_self_ms
            iter_time_with_norm_ms = _float(with_norm, "iter_time_ms")

            gap_rows.append(
                {
                    "model": model,
                    "batch_size": batch_size,
                    "iter_time_without_norm_ms": _float(
                        without, "iter_time_ms"
                    ),
                    "iter_time_with_norm_ms": iter_time_with_norm_ms,
                    "total_extra_time_with_norm_ms": total_extra_ms,
                    "norm_self_time_ms": norm_self_ms,
                    "norm_time_gap_ms": norm_time_gap_ms,
                    "norm_time_gap_pct": (
                        norm_time_gap_ms / iter_time_with_norm_ms * 100.0
                        if iter_time_with_norm_ms
                        else ""
                    ),
                    "norm_time_ratio": (
                        total_extra_ms / norm_self_ms
                        if norm_self_ms > 0
                        else ""
                    ),
                    "power_without_norm_watts": power_without,
                    "power_with_norm_watts": power_with,
                    "power_delta_watts": power_with - power_without,
                    "clock_without_norm_mhz": clock_without,
                    "clock_with_norm_mhz": clock_with,
                    "clock_increase_without_norm_mhz": clock_without
                    - clock_with,
                    "clock_increase_without_norm_pct": (
                        (clock_without - clock_with) / clock_with * 100.0
                        if clock_with
                        else ""
                    ),
                    "raw_gemm_without_norm_tflops_s": raw_without,
                    "raw_gemm_with_norm_tflops_s": raw_with,
                    "raw_gemm_delta_pct": (
                        (raw_with - raw_without) / raw_without * 100.0
                        if raw_without
                        else ""
                    ),
                    "effective_without_norm_tflops_s": effective_without,
                    "effective_with_norm_tflops_s": effective_with,
                    "effective_delta_pct": (
                        (effective_with - effective_without)
                        / effective_without
                        * 100.0
                        if effective_without
                        else ""
                    ),
                }
            )

    return gap_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in SUMMARY_FIELDNAMES}
            )
    return path


def _style_axis(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_SPINE_WIDTH)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONTSIZE,
        width=AXIS_SPINE_WIDTH,
        length=7,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=AXIS_SPINE_WIDTH * 0.75,
        length=4,
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def _save_png(fig: Any, output_path: Path, dpi: int = 220) -> None:
    fig.savefig(output_path, dpi=dpi)

    # Keep figures large/readable while satisfying the repository's 500 KiB
    # added-file hook for PNG artifacts.
    try:
        from PIL import Image

        image = Image.open(output_path)
        image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        image.save(output_path, optimize=True)
    except Exception:
        pass


def _plot_metric(
    rows: list[dict[str, Any]],
    output_path: Path,
    metric_key: str,
    ylabel: str,
    title: str,
    zero_line: bool = False,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in rows}, key=_model_sort_key)
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})

    fig, ax = plt.subplots(figsize=SINGLE_PLOT_FIGSIZE)
    for model in models:
        group = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: int(row["batch_size"]),
        )
        ax.plot(
            [int(row["batch_size"]) for row in group],
            [float(row[metric_key]) for row in group],
            marker="o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=1.4,
            label=model,
        )
    if zero_line:
        ax.axhline(0.0, color="black", linewidth=1.8, alpha=0.55)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value):d}")
    )
    ax.set_xlabel(
        "batch_size * sequence_length (S)",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=14)
    ax.grid(True, alpha=0.3, linewidth=1.1)
    _style_axis(ax)
    legend = ax.legend(
        loc="best",
        title="Model",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE + 1,
        frameon=True,
    )
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    _save_png(fig, output_path)
    plt.close(fig)
    return output_path


def write_plots(rows: list[dict[str, Any]], plots_dir: Path) -> list[Path]:
    paths = [
        _plot_metric(
            rows,
            plots_dir / "norm_time_gap_pct.png",
            "norm_time_gap_pct",
            "Norm time gap (% of w/ norm time)",
            "Relative norm time gap, batch size > 64",
            zero_line=True,
        ),
        _plot_metric(
            rows,
            plots_dir / "power_delta_watts.png",
            "power_delta_watts",
            "Power delta w/ norm - w/o norm (W)",
            "Power change from enabling Norm, batch size > 64",
            zero_line=True,
        ),
        _plot_metric(
            rows,
            plots_dir / "baseline_power_watts.png",
            "power_with_norm_watts",
            "Baseline power w/ norm (W)",
            "Baseline absolute power, batch size > 64",
            zero_line=False,
        ),
        _plot_metric(
            rows,
            plots_dir / "clock_increase_without_norm_mhz.png",
            "clock_increase_without_norm_mhz",
            "Clock increase w/o norm vs w/ norm (MHz)",
            "Clock increase from removing Norm, batch size > 64",
            zero_line=False,
        ),
        _plot_metric(
            rows,
            plots_dir / "clock_increase_without_norm_pct.png",
            "clock_increase_without_norm_pct",
            "Clock increase w/o norm vs w/ norm (%)",
            "Relative clock increase from removing Norm, batch size > 64",
            zero_line=False,
        ),
    ]
    paths.append(
        _write_combined_plot(rows, plots_dir / "norm_gap_power_clock.png")
    )
    clock_comparison_plot = _write_model_clock_comparison(
        rows=rows,
        output_path=plots_dir / "70B_clock_with_without_norm.png",
        model="70B",
    )
    if clock_comparison_plot is not None:
        paths.append(clock_comparison_plot)
    time_difference_plot = _write_time_difference_bar(
        rows=rows,
        output_path=plots_dir / "70B_32768_time_difference.png",
        model="70B",
        batch_size=32768,
    )
    if time_difference_plot is not None:
        paths.append(time_difference_plot)
    return paths


def _write_model_clock_comparison(
    rows: list[dict[str, Any]],
    output_path: Path,
    model: str,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    group = sorted(
        [row for row in rows if row["model"] == model],
        key=lambda row: int(row["batch_size"]),
    )
    if not group:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=SINGLE_PLOT_FIGSIZE)
    x_values = [int(row["batch_size"]) for row in group]
    ax.plot(
        x_values,
        [float(row["clock_with_norm_mhz"]) for row in group],
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markeredgewidth=1.4,
        label="w/ norm baseline",
        color="#E45756",
    )
    ax.plot(
        x_values,
        [float(row["clock_without_norm_mhz"]) for row in group],
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        markeredgewidth=1.4,
        label="w/o norm",
        color="#4C78A8",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_values)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value):d}")
    )
    ax.set_xlabel(
        "batch_size * sequence_length (S)",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax.set_ylabel("GPU clock (MHz)", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(
        f"{model}: baseline vs w/o Norm clock",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=14,
    )
    ax.grid(True, alpha=0.3, linewidth=1.1)
    _style_axis(ax)
    legend = ax.legend(
        loc="best",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE + 1,
        frameon=True,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    _save_png(fig, output_path)
    plt.close(fig)
    return output_path


def _write_time_difference_bar(
    rows: list[dict[str, Any]],
    output_path: Path,
    model: str,
    batch_size: int,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matches = [
        row
        for row in rows
        if row["model"] == model and int(row["batch_size"]) == batch_size
    ]
    if not matches:
        return None
    row = matches[0]

    without_ms = float(row["iter_time_without_norm_ms"])
    with_ms = float(row["iter_time_with_norm_ms"])
    total_extra_ms = float(row["total_extra_time_with_norm_ms"])
    norm_self_ms = float(row["norm_self_time_ms"])
    norm_gap_ms = float(row["norm_time_gap_ms"])
    norm_gap_pct = float(row["norm_time_gap_pct"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=SINGLE_PLOT_FIGSIZE)

    ax.bar(
        [0],
        [without_ms],
        width=0.55,
        color="#4C78A8",
        edgecolor="black",
        linewidth=AXIS_SPINE_WIDTH,
        label="w/o norm iteration time",
    )
    ax.bar(
        [1],
        [without_ms],
        width=0.55,
        color="#4C78A8",
        alpha=0.45,
        edgecolor="black",
        linewidth=AXIS_SPINE_WIDTH,
        label="shared baseline time",
    )
    ax.bar(
        [1],
        [norm_self_ms],
        width=0.55,
        bottom=[without_ms],
        color="#F58518",
        edgecolor="black",
        linewidth=AXIS_SPINE_WIDTH,
        label="Norm self-time",
    )
    ax.bar(
        [1],
        [norm_gap_ms],
        width=0.55,
        bottom=[without_ms + norm_self_ms],
        color="#E45756",
        edgecolor="black",
        linewidth=AXIS_SPINE_WIDTH,
        label="Norm time gap",
    )

    ax.text(
        0,
        without_ms + 6,
        f"{without_ms:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax.text(
        1,
        with_ms + 6,
        f"{with_ms:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
    )

    ax.annotate(
        (
            f"+{total_extra_ms:.1f} ms total\n"
            f"Norm self: {norm_self_ms:.1f} ms\n"
            f"Gap: {norm_gap_ms:.1f} ms ({norm_gap_pct:.2f}%)"
        ),
        xy=(1.28, without_ms + total_extra_ms / 2.0),
        xytext=(1.45, without_ms + total_extra_ms / 2.0),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": AXIS_SPINE_WIDTH,
            "color": "black",
        },
        fontsize=LABEL_FONTSIZE - 1,
        fontweight="bold",
        va="center",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": AXIS_SPINE_WIDTH,
        },
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["w/o norm", "w/ norm"], fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(
        "Iteration time (ms)", fontsize=LABEL_FONTSIZE, fontweight="bold"
    )
    ax.set_title(
        f"{model} S={batch_size}: w/ vs w/o Norm time",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylim(0, with_ms * 1.27)
    ax.set_xlim(-0.6, 2.25)
    ax.grid(True, axis="y", alpha=0.3, linewidth=1.1)
    _style_axis(ax)
    legend = ax.legend(
        loc="upper left",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE + 1,
        frameon=True,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    _save_png(fig, output_path)
    plt.close(fig)
    return output_path


def _write_combined_plot(rows: list[dict[str, Any]], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in rows}, key=_model_sort_key)
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    panels = [
        (
            "norm_time_gap_pct",
            "Norm time gap (%)",
            "Norm time gap / w/ norm time",
        ),
        (
            "power_delta_watts",
            "Power delta (W)",
            "Power: w/ norm - w/o norm",
        ),
        (
            "clock_increase_without_norm_mhz",
            "Clock increase (MHz)",
            "Clock: w/o norm - w/ norm",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=COMBINED_PLOT_FIGSIZE, sharex=True)
    for ax, (metric_key, ylabel, title) in zip(axes, panels):
        for model in models:
            group = sorted(
                [row for row in rows if row["model"] == model],
                key=lambda row: int(row["batch_size"]),
            )
            ax.plot(
                [int(row["batch_size"]) for row in group],
                [float(row[metric_key]) for row in group],
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                markeredgewidth=1.4,
                label=model,
            )
        ax.axhline(0.0, color="black", linewidth=1.8, alpha=0.55)
        ax.set_xscale("log", base=2)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight="bold")
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=12)
        ax.grid(True, alpha=0.3, linewidth=1.1)
        _style_axis(ax)
    axes[-1].set_xticks(batch_sizes)
    axes[-1].xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value):d}")
    )
    axes[-1].set_xlabel(
        "batch_size * sequence_length (S)",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
    )
    legend = axes[0].legend(
        loc="best",
        title="Model",
        ncols=4,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE + 1,
        frameon=True,
    )
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout()
    _save_png(fig, output_path)
    plt.close(fig)
    return output_path


def write_report(
    output_dir: Path,
    plots_dir: Path,
    csv_path: Path,
    plot_paths: list[Path],
    rows: list[dict[str, Any]],
    min_batch_size: int,
) -> Path:
    report_path = output_dir / "NORM_TIME_GAP.md"
    report_base_dir = report_path.parent
    max_gap = max(rows, key=lambda row: float(row["norm_time_gap_pct"]))
    max_clock_increase = max(
        rows, key=lambda row: float(row["clock_increase_without_norm_mhz"])
    )
    max_clock_increase_pct = max(
        rows, key=lambda row: float(row["clock_increase_without_norm_pct"])
    )
    max_power = max(rows, key=lambda row: float(row["power_delta_watts"]))
    lines = [
        "# Full-Block Norm Time Gap Plots",
        "",
        f"- Source result: `{_display_path(output_dir)}`",
        f"- Included batch_size * sequence_length: `S > {min_batch_size}`",
        f"- Included models: `{', '.join(sorted({str(row['model']) for row in rows}, key=_model_sort_key))}`",
        f"- Derived CSV: [`{_markdown_target(csv_path, report_base_dir)}`]({_markdown_target(csv_path, report_base_dir)})",
        "",
        "## Plots",
        "",
    ]
    for path in plot_paths:
        lines.append(
            f"- [{path.name}]({_markdown_target(path, report_base_dir)})"
        )
    lines.extend(
        [
            "",
            "## Derived Metrics",
            "",
            "`norm_time_gap_pct = ((iter_time_with_norm - iter_time_without_norm) - norm_self_time) / iter_time_with_norm * 100`",
            "",
            "`norm_time_gap_ms` is kept in the derived CSV as the absolute intermediate value.",
            "",
            "| Metric | Case | Value |",
            "| --- | --- | ---: |",
            (
                f"| Largest norm time gap | {max_gap['model']} S={max_gap['batch_size']} | "
                f"{float(max_gap['norm_time_gap_pct']):.2f}% |"
            ),
            (
                f"| Largest power increase | {max_power['model']} S={max_power['batch_size']} | "
                f"{float(max_power['power_delta_watts']):.2f} W |"
            ),
            (
                f"| Largest w/o-norm clock increase | {max_clock_increase['model']} S={max_clock_increase['batch_size']} | "
                f"{float(max_clock_increase['clock_increase_without_norm_mhz']):.2f} MHz |"
            ),
            (
                f"| Largest relative w/o-norm clock increase | {max_clock_increase_pct['model']} S={max_clock_increase_pct['batch_size']} | "
                f"{float(max_clock_increase_pct['clock_increase_without_norm_pct']):.2f}% |"
            ),
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
    return report_path


def publish_latest(source_dir: Path, latest_dir: Path) -> Path:
    if latest_dir.exists():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    shutil.copytree(source_dir, latest_dir)
    (latest_dir / "PUBLISHED_FROM.txt").write_text(
        f"{_display_path(source_dir)}\n"
    )
    return latest_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render full-block norm time gap, power delta, and clock delta plots."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="State-chain block sweep result directory. Defaults to newest timestamped run.",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=64,
        help="Only include batch_size * sequence_length values strictly greater than this value.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Plot output directory. Defaults to <output-dir>/plots/norm_time_gap.",
    )
    parser.add_argument(
        "--publish-latest",
        action="store_true",
        help="Copy the rendered result directory to results/state_chain_block_sweep/latest.",
    )
    parser.add_argument(
        "--latest-dir",
        type=Path,
        default=DEFAULT_LATEST_DIR,
        help="Destination for --publish-latest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or _latest_timestamped_result(
        DEFAULT_RESULTS_ROOT
    )
    output_dir = output_dir.resolve()
    plots_dir = args.plots_dir or (output_dir / "plots" / "norm_time_gap")
    plots_dir = plots_dir.resolve()

    rows = build_gap_rows(
        output_dir=output_dir, min_batch_size=args.min_batch_size
    )
    if not rows:
        raise RuntimeError("no matching with_norm / without_norm rows found")
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    csv_path = write_csv(plots_dir / "norm_gap_summary.csv", rows)
    plot_paths = write_plots(rows, plots_dir)
    report_path = write_report(
        output_dir=output_dir,
        plots_dir=plots_dir,
        csv_path=csv_path,
        plot_paths=plot_paths,
        rows=rows,
        min_batch_size=args.min_batch_size,
    )
    print(f"Wrote {csv_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    print(f"Wrote {report_path}")
    if args.publish_latest:
        latest_dir = publish_latest(output_dir, args.latest_dir.resolve())
        latest_plots_dir = latest_dir / "plots" / "norm_time_gap"
        latest_plot_paths = [
            latest_plots_dir / path.name for path in plot_paths
        ]
        write_report(
            output_dir=latest_dir,
            plots_dir=latest_plots_dir,
            csv_path=latest_plots_dir / csv_path.name,
            plot_paths=latest_plot_paths,
            rows=rows,
            min_batch_size=args.min_batch_size,
        )
        print(f"Published latest snapshot to {latest_dir}")


if __name__ == "__main__":
    main()
