#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int_from_label(label: str, marker: str) -> int:
    return int(label.split(marker, 1)[1])


def _apply_axis_style(axis: object, label_size: int, tick_size: int) -> None:
    axis.tick_params(axis="both", labelsize=tick_size, width=2.2, length=7)
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_linewidth(2.2)


def _add_phase_spans(
    axis: object,
    phases: list[dict[str, str]],
    gemm_color: str,
    norm_color: str,
    xmin: float | None = None,
    xmax: float | None = None,
) -> None:
    for phase in phases:
        start = _float(phase, "monitor_start_s")
        end = _float(phase, "monitor_end_s")
        if xmin is not None and end < xmin:
            continue
        if xmax is not None and start > xmax:
            continue
        color = gemm_color if phase["phase"] == "gemm" else norm_color
        axis.axvspan(start, end, color=color, alpha=0.18, linewidth=0)


def _first_n_cycle_xlim(
    phase_rows: list[dict[str, str]],
    cycles: int,
) -> tuple[float, float] | None:
    selected = [row for row in phase_rows if int(row["cycle"]) < cycles]
    if not selected:
        return None
    start = min(_float(row, "monitor_start_s") for row in selected)
    end = max(_float(row, "monitor_end_s") for row in selected)
    pad = max(0.02, (end - start) * 0.02)
    return max(0.0, start - pad), end + pad


def _padded_limits(
    values: list[float], pad_fraction: float = 0.06
) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(1.0, abs(low) * pad_fraction)
    else:
        pad = (high - low) * pad_fraction
    return low - pad, high + pad


def _plot_case(
    monitor_rows: list[dict[str, str]],
    phase_rows: list[dict[str, str]],
    out_path: Path,
    title: str,
    xlim: tuple[float, float] | None,
    style: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    times = [_float(row, "elapsed_seconds") for row in monitor_rows]
    power = [_float(row, "power_watts") for row in monitor_rows]
    clock = [_float(row, "gpu_clock_mhz") for row in monitor_rows]
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for axis in axes:
        _add_phase_spans(
            axis,
            phase_rows,
            str(style["gemm_color"]),
            str(style["norm_color"]),
            *(xlim or (None, None)),
        )
        axis.grid(True, alpha=0.24, linewidth=1.1)
        _apply_axis_style(
            axis,
            int(style["axis_label_size"]),
            int(style["tick_label_size"]),
        )

    axes[0].plot(
        times,
        power,
        color=str(style["power_color"]),
        linewidth=float(style["line_width"]),
    )
    axes[1].plot(
        times,
        clock,
        color=str(style["clock_color"]),
        linewidth=float(style["line_width"]),
    )
    axes[0].set_ylabel(
        "GPU Power (W)",
        fontsize=int(style["axis_label_size"]),
        fontweight="bold",
    )
    axes[1].set_ylabel(
        "GPU Clock (MHz)",
        fontsize=int(style["axis_label_size"]),
        fontweight="bold",
    )
    axes[1].set_xlabel(
        "Monitor Elapsed Time (s)",
        fontsize=int(style["axis_label_size"]),
        fontweight="bold",
    )
    if xlim is not None:
        axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*style["power_ylim"])
    axes[1].set_ylim(*style["clock_ylim"])
    axes[0].set_title(
        title,
        fontsize=int(style["title_size"]),
        fontweight="bold",
        pad=16,
    )
    legend = [
        Patch(
            facecolor=str(style["gemm_color"]),
            alpha=0.30,
            label="GEMM phase",
        ),
        Patch(
            facecolor=str(style["norm_color"]),
            alpha=0.30,
            label="Norm phase",
        ),
    ]
    axes[0].legend(
        handles=legend,
        loc="upper right",
        fontsize=int(style["legend_size"]),
        title="Phase",
        title_fontproperties={
            "weight": "bold",
            "size": int(style["legend_size"]),
        },
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(style["dpi"]))
    plt.close(fig)


def _plot_overview(
    root: Path,
    selected_rows: list[tuple[str, int, float, float, float, float]],
    cycles: int,
    output_path: Path,
    style: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    monitor_dir = root / "monitor"
    phase_dir = root / "phase_events"
    fig, axes = plt.subplots(
        len(selected_rows),
        1,
        figsize=(18, max(4, 3.2 * len(selected_rows))),
        sharex=False,
    )
    if len(selected_rows) == 1:
        axes = [axes]
    for axis, (
        label,
        _norm_steps,
        _gemm_ms,
        norm_ms,
        avg_clock,
        avg_power,
    ) in zip(
        axes,
        selected_rows,
        strict=True,
    ):
        monitor_rows = _read_csv(monitor_dir / f"{label}.csv")
        phase_rows = _read_csv(phase_dir / f"{label}.csv")
        xlim = _first_n_cycle_xlim(phase_rows, cycles)
        times = [_float(row, "elapsed_seconds") for row in monitor_rows]
        power = [_float(row, "power_watts") for row in monitor_rows]
        clock = [_float(row, "gpu_clock_mhz") for row in monitor_rows]

        _add_phase_spans(
            axis,
            phase_rows,
            str(style["gemm_color"]),
            str(style["norm_color"]),
            *(xlim or (None, None)),
        )
        axis.plot(
            times,
            power,
            color=str(style["power_color"]),
            linewidth=float(style["line_width"]),
            label="Power",
        )
        axis2 = axis.twinx()
        _add_phase_spans(
            axis2,
            phase_rows,
            str(style["gemm_color"]),
            str(style["norm_color"]),
            *(xlim or (None, None)),
        )
        axis2.plot(
            times,
            clock,
            color=str(style["clock_color"]),
            linewidth=float(style["line_width"]),
            label="Clock",
        )
        axis.set_ylabel(
            "Power (W)",
            fontsize=int(style["overview_axis_label_size"]),
            fontweight="bold",
        )
        axis2.set_ylabel(
            "Clock (MHz)",
            fontsize=int(style["overview_axis_label_size"]),
            fontweight="bold",
        )
        axis.grid(True, alpha=0.22, linewidth=1.1)
        _apply_axis_style(
            axis,
            int(style["overview_axis_label_size"]),
            int(style["overview_tick_label_size"]),
        )
        _apply_axis_style(
            axis2,
            int(style["overview_axis_label_size"]),
            int(style["overview_tick_label_size"]),
        )
        if xlim is not None:
            axis.set_xlim(*xlim)
            axis2.set_xlim(*xlim)
        axis.set_ylim(*style["power_ylim"])
        axis2.set_ylim(*style["clock_ylim"])
        axis.set_title(
            (
                f"{label}: norm {norm_ms:.1f} ms, "
                f"avg clock {avg_clock:.1f} MHz, avg power {avg_power:.1f} W"
            ),
            loc="left",
            fontsize=int(style["overview_title_size"]),
            fontweight="bold",
            pad=10,
        )
    axes[-1].set_xlabel(
        f"Monitor Elapsed Time (s), First {cycles} Cycles",
        fontsize=int(style["axis_label_size"]),
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(
                facecolor=str(style["gemm_color"]),
                alpha=0.30,
                label="GEMM phase",
            ),
            Patch(
                facecolor=str(style["norm_color"]),
                alpha=0.30,
                label="Norm phase",
            ),
        ],
        loc="upper center",
        ncol=2,
        fontsize=int(style["legend_size"]),
        title="Phase",
        title_fontproperties={
            "weight": "bold",
            "size": int(style["legend_size"]),
        },
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=int(style["dpi"]))
    plt.close(fig)


def render(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")

    root = args.output_dir
    monitor_dir = root / "monitor"
    phase_dir = root / "phase_events"
    plot_dir = root / "plots" / "timeline"
    full_dir = plot_dir / "full"
    zoom_dir = plot_dir / f"zoom_first{args.zoom_cycles}_cycles"
    full_dir.mkdir(parents=True, exist_ok=True)
    zoom_dir.mkdir(parents=True, exist_ok=True)

    style: dict[str, object] = {
        "gemm_color": "#4f83cc",
        "norm_color": "#f59e0b",
        "power_color": "#111827",
        "clock_color": "#b91c1c",
        "line_width": 2.8,
        "title_size": 24,
        "axis_label_size": 22,
        "tick_label_size": 18,
        "legend_size": 18,
        "overview_title_size": 17,
        "overview_axis_label_size": 16,
        "overview_tick_label_size": 14,
        "dpi": 180,
    }

    labels = sorted(
        [path.stem for path in phase_dir.glob(f"{args.case_prefix}_n*.csv")],
        key=lambda label: _int_from_label(label, "_n"),
    )
    all_power: list[float] = []
    all_clock: list[float] = []
    for label in labels:
        for row in _read_csv(monitor_dir / f"{label}.csv"):
            all_power.append(_float(row, "power_watts"))
            all_clock.append(_float(row, "gpu_clock_mhz"))

    rows_for_index: list[
        tuple[str, int, float, float, float, float, Path, Path]
    ] = []
    style["power_ylim"] = _padded_limits(all_power)
    style["clock_ylim"] = _padded_limits(all_clock)
    for label in labels:
        monitor_rows = _read_csv(monitor_dir / f"{label}.csv")
        phase_rows = _read_csv(phase_dir / f"{label}.csv")
        norm_steps = _int_from_label(label, "_n")
        cycles = max(int(row["cycle"]) for row in phase_rows) + 1
        gemm_ms = (
            sum(
                _float(row, "cuda_duration_ms")
                for row in phase_rows
                if row["phase"] == "gemm"
            )
            / cycles
        )
        norm_phases = [row for row in phase_rows if row["phase"] == "norm"]
        norm_ms = (
            sum(_float(row, "cuda_duration_ms") for row in norm_phases) / cycles
            if norm_phases
            else 0.0
        )
        avg_clock = sum(
            _float(row, "gpu_clock_mhz") for row in monitor_rows
        ) / len(monitor_rows)
        avg_power = sum(
            _float(row, "power_watts") for row in monitor_rows
        ) / len(monitor_rows)
        full_path = full_dir / f"{label}.png"
        zoom_path = zoom_dir / f"{label}.png"
        title = (
            f"{label}: M=8192 N=8192 K=32768, "
            f"GEMM {gemm_ms:.1f} ms, Norm {norm_ms:.1f} ms"
        )
        _plot_case(
            monitor_rows=monitor_rows,
            phase_rows=phase_rows,
            out_path=full_path,
            title=title,
            xlim=None,
            style=style,
        )
        _plot_case(
            monitor_rows=monitor_rows,
            phase_rows=phase_rows,
            out_path=zoom_path,
            title=f"{title} (First {args.zoom_cycles} Cycles)",
            xlim=_first_n_cycle_xlim(phase_rows, args.zoom_cycles),
            style=style,
        )
        rows_for_index.append(
            (
                label,
                norm_steps,
                gemm_ms,
                norm_ms,
                avg_clock,
                avg_power,
                full_path,
                zoom_path,
            )
        )

    selected_rows = [
        row[:6]
        for row in rows_for_index
        if row[1] in set(args.overview_norm_steps)
    ]
    overview_path = plot_dir / (
        f"norm_sweep_selected_overview_first{args.zoom_cycles}_cycles.png"
    )
    if selected_rows:
        _plot_overview(
            root, selected_rows, args.zoom_cycles, overview_path, style
        )

    index_path = plot_dir / "INDEX.md"
    with index_path.open("w") as index_file:
        index_file.write("# Norm Sweep Timeline Plots\n\n")
        index_file.write(
            "- Style: large bold titles/axis labels/ticks per `CLAUDE.md` plotting standards\n"
        )
        index_file.write(
            f"- Shared y-axis limits: power `{style['power_ylim'][0]:.1f}` to "
            f"`{style['power_ylim'][1]:.1f}` W, clock `{style['clock_ylim'][0]:.1f}` "
            f"to `{style['clock_ylim'][1]:.1f}` MHz\n"
        )
        index_file.write("- Full timelines: `full/*.png`\n")
        index_file.write(
            f"- First-{args.zoom_cycles}-cycle zooms: `{zoom_dir.name}/*.png`\n"
        )
        if selected_rows:
            index_file.write(f"- Selected overview: `{overview_path.name}`\n")
        index_file.write("\n")
        index_file.write(
            "| Case | Norm Steps | GEMM Phase (ms) | Norm Phase (ms) | "
            "Avg Clock (MHz) | Avg Power (W) | Full | Zoom |\n"
        )
        index_file.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
        )
        for (
            label,
            norm_steps,
            gemm_ms,
            norm_ms,
            avg_clock,
            avg_power,
            full_path,
            zoom_path,
        ) in rows_for_index:
            index_file.write(
                f"| `{label}` | {norm_steps} | {gemm_ms:.1f} | {norm_ms:.1f} | "
                f"{avg_clock:.1f} | {avg_power:.1f} | "
                f"[{full_path.name}]({full_path.relative_to(plot_dir).as_posix()}) | "
                f"[{zoom_path.name}]({zoom_path.relative_to(plot_dir).as_posix()}) |\n"
            )

    print(f"Wrote {index_path}")
    print(
        f"Wrote {len(rows_for_index)} full plots and {len(rows_for_index)} zoom plots"
    )
    if selected_rows:
        print(f"Wrote {overview_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render phase-colored power/clock timeline plots for two-GEMM/norm phase sweeps."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-prefix", default="g16")
    parser.add_argument("--zoom-cycles", type=int, default=3)
    parser.add_argument(
        "--overview-norm-steps",
        nargs="+",
        type=int,
        default=[0, 8, 64, 512, 2048, 16384],
    )
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
