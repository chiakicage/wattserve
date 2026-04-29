#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from monitor.gpu_monitor import GPUMonitor  # noqa: E402
from state_chain_utils import (  # noqa: E402
    average,
    make_scaled_tensor,
    utc_now_iso,
    utc_stamp,
    write_csv,
    write_json,
)


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "gemm_continuous"
DEFAULT_M = 8192
DEFAULT_N = 8192
DEFAULT_K = 32768
DEFAULT_DTYPE = "bfloat16"
DEFAULT_GEMM_UNITS = 350
DEFAULT_MONITOR_INTERVAL = 0.01
DEFAULT_MONITOR_GPU_INDEX = 2
DEFAULT_PRE_IDLE_S = 2.0
DEFAULT_POST_IDLE_S = 1.0
ACTIVE_NVTX_RANGE = "active_continuous_gemm"

SUMMARY_FIELDNAMES = [
    "status",
    "error",
    "run_timestamp_utc",
    "m",
    "n",
    "k",
    "dtype",
    "gemm_units",
    "active_cuda_ms",
    "active_wall_s",
    "gemm_flops_per_unit",
    "gemm_tflops_s",
    "first_2s_avg_power_watts",
    "first_2s_avg_gpu_clock_mhz",
    "first_2s_avg_temperature_c",
    "first_2s_gemm_tflops_s",
    "last_2s_avg_power_watts",
    "last_2s_avg_gpu_clock_mhz",
    "last_2s_avg_temperature_c",
    "last_2s_gemm_tflops_s",
    "transition_time_s",
    "transition_detected",
    "transition_pre_500ms_power_watts",
    "transition_pre_500ms_clock_mhz",
    "transition_pre_500ms_temperature_c",
    "transition_pre_500ms_tflops_s",
    "transition_post_500ms_power_watts",
    "transition_post_500ms_clock_mhz",
    "transition_post_500ms_temperature_c",
    "transition_post_500ms_tflops_s",
    "active_start_temperature_c",
    "transition_temperature_c",
    "active_end_temperature_c",
    "monitor_sample_count",
    "monitor_csv",
    "unit_events_csv",
    "power_clock_temperature_plot",
    "gemm_tflops_plot",
    "active_nvtx_range",
]

UNIT_FIELDNAMES = [
    "unit_index",
    "cuda_start_ms",
    "cuda_end_ms",
    "cuda_duration_ms",
    "monitor_start_s",
    "monitor_end_s",
    "tflops_s",
]


@dataclass
class UnitRecord:
    unit_index: int
    start_event: Any
    end_event: Any
    cuda_start_ms: float = 0.0
    cuda_end_ms: float = 0.0
    cuda_duration_ms: float = 0.0
    monitor_start_s: float = 0.0
    monitor_end_s: float = 0.0
    tflops_s: float = 0.0


class TwoGemmState:
    def __init__(
        self,
        torch: Any,
        m: int,
        n: int,
        k: int,
        device: str,
        dtype: Any,
    ) -> None:
        self.torch = torch
        self.current = make_scaled_tensor(torch, (m, n), device, dtype)
        self.next_current = torch.empty_like(self.current)
        self.middle = torch.empty((m, k), device=device, dtype=dtype)
        self.left_weight = make_scaled_tensor(torch, (n, k), device, dtype)
        self.right_weight = make_scaled_tensor(torch, (k, n), device, dtype)

    def run_unit(self) -> None:
        self.torch.mm(self.current, self.left_weight, out=self.middle)
        self.torch.mm(self.middle, self.right_weight, out=self.next_current)
        self.current, self.next_current = self.next_current, self.current


def _gemm_flops(m: int, n: int, k: int) -> float:
    return 4.0 * m * n * k


def _wait_for_monitor_start(
    monitor: GPUMonitor, timeout_s: float = 1.0
) -> float:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        start_time = getattr(monitor, "_start_time", None)
        if start_time is not None:
            return float(start_time)
        time.sleep(0.001)
    return time.time()


def _records_in_window(
    records: list[dict[str, Any]],
    start_s: float,
    end_s: float,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if start_s <= float(record["elapsed_seconds"]) <= end_s
    ]


def _units_in_window(
    units: list[UnitRecord],
    start_s: float,
    end_s: float,
) -> list[UnitRecord]:
    return [
        unit
        for unit in units
        if unit.monitor_start_s < end_s and unit.monitor_end_s > start_s
    ]


def _window_tflops(units: list[UnitRecord], flops_per_unit: float) -> float:
    if not units:
        return 0.0
    total_ms = sum(unit.cuda_duration_ms for unit in units)
    if total_ms <= 0.0:
        return 0.0
    return flops_per_unit * len(units) / (total_ms / 1000.0) / 1e12


def _window_summary(
    monitor_records: list[dict[str, Any]],
    unit_records: list[UnitRecord],
    flops_per_unit: float,
    start_s: float,
    end_s: float,
    prefix: str,
) -> dict[str, float]:
    records = _records_in_window(monitor_records, start_s, end_s)
    units = _units_in_window(unit_records, start_s, end_s)
    return {
        f"{prefix}_avg_power_watts": average(records, "power_watts"),
        f"{prefix}_avg_gpu_clock_mhz": average(records, "gpu_clock_mhz"),
        f"{prefix}_avg_temperature_c": average(records, "temperature_c"),
        f"{prefix}_gemm_tflops_s": _window_tflops(units, flops_per_unit),
    }


def _temperature_at(
    records: list[dict[str, Any]],
    target_s: float,
) -> float:
    if not records:
        return 0.0
    nearest = min(
        records,
        key=lambda record: abs(float(record["elapsed_seconds"]) - target_s),
    )
    return float(nearest.get("temperature_c", 0.0))


def _detect_transition(
    records: list[dict[str, Any]],
    active_start_s: float,
    active_end_s: float,
    window_s: float = 0.2,
    low_threshold_mhz: float = 1350.0,
    high_threshold_mhz: float = 1400.0,
) -> tuple[float, bool]:
    active_records = [
        record
        for record in records
        if active_start_s <= float(record["elapsed_seconds"]) <= active_end_s
    ]
    if not active_records:
        return 0.0, False

    saw_low = False
    for record in active_records:
        center_s = float(record["elapsed_seconds"])
        window = [
            float(other["gpu_clock_mhz"])
            for other in active_records
            if center_s - window_s / 2.0
            <= float(other["elapsed_seconds"])
            <= center_s + window_s / 2.0
        ]
        if not window:
            continue
        median_clock = statistics.median(window)
        if median_clock < low_threshold_mhz:
            saw_low = True
        if saw_low and median_clock >= high_threshold_mhz:
            return center_s - active_start_s, True
    return 0.0, False


def _write_unit_rows(
    path: Path,
    unit_records: list[UnitRecord],
) -> None:
    write_csv(
        path,
        UNIT_FIELDNAMES,
        [
            {
                "unit_index": unit.unit_index,
                "cuda_start_ms": unit.cuda_start_ms,
                "cuda_end_ms": unit.cuda_end_ms,
                "cuda_duration_ms": unit.cuda_duration_ms,
                "monitor_start_s": unit.monitor_start_s,
                "monitor_end_s": unit.monitor_end_s,
                "tflops_s": unit.tflops_s,
            }
            for unit in unit_records
        ],
    )


def _write_power_clock_temperature_plot(
    monitor_records: list[dict[str, Any]],
    active_start_s: float,
    active_end_s: float,
    transition_time_s: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [float(record["elapsed_seconds"]) for record in monitor_records]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(
        x_values,
        [float(record["power_watts"]) for record in monitor_records],
        color="#2563eb",
        linewidth=1.4,
    )
    axes[1].plot(
        x_values,
        [float(record["gpu_clock_mhz"]) for record in monitor_records],
        color="#111827",
        linewidth=1.4,
    )
    axes[2].plot(
        x_values,
        [float(record["temperature_c"]) for record in monitor_records],
        color="#dc2626",
        linewidth=1.4,
    )
    labels = ["Power (W)", "GPU Clock (MHz)", "Temperature (C)"]
    for axis, label in zip(axes, labels, strict=True):
        axis.axvspan(
            active_start_s,
            active_end_s,
            color="#10b981",
            alpha=0.08,
            linewidth=0,
            label="active",
        )
        if transition_time_s > 0.0:
            axis.axvline(
                active_start_s + transition_time_s,
                color="#7c3aed",
                linestyle="--",
                linewidth=1.4,
                label="transition",
            )
        axis.set_ylabel(label, fontsize=13, fontweight="bold")
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="both", labelsize=11)
    axes[0].set_title(
        "Continuous Two-GEMM Runtime Timeline",
        fontsize=15,
        fontweight="bold",
    )
    axes[-1].set_xlabel(
        "Monitor Elapsed Time (s)", fontsize=13, fontweight="bold"
    )
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _write_tflops_plot(
    unit_records: list[UnitRecord],
    active_start_s: float,
    transition_time_s: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [
        (unit.monitor_start_s + unit.monitor_end_s) / 2.0 - active_start_s
        for unit in unit_records
    ]
    fig, axis = plt.subplots(1, 1, figsize=(12, 5))
    axis.plot(
        x_values,
        [unit.tflops_s for unit in unit_records],
        marker="o",
        markersize=2.8,
        linewidth=1.2,
        color="#0f766e",
    )
    if transition_time_s > 0.0:
        axis.axvline(
            transition_time_s,
            color="#7c3aed",
            linestyle="--",
            linewidth=1.4,
            label="transition",
        )
        axis.legend(loc="best")
    axis.set_title(
        "Per-Unit Two-GEMM Throughput",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_xlabel("Active Time (s)", fontsize=13, fontweight="bold")
    axis.set_ylabel("TFLOPs/s", fontsize=13, fontweight="bold")
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _repo_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    try:
        return candidate.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _write_report(output_dir: Path, row: dict[str, Any]) -> None:
    transition = (
        f"{float(row['transition_time_s']):.3f} s"
        if row["transition_detected"]
        else "not detected"
    )
    lines = [
        "# GEMM Continuous Runtime-DVFS Experiment",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        f"- Shape: `M={row['m']}, N={row['n']}, K={row['k']}`",
        f"- GEMM units: `{row['gemm_units']}`",
        f"- Active CUDA time: `{float(row['active_cuda_ms']):.3f} ms`",
        f"- Transition time: `{transition}`",
        "",
        "## First vs Last 2s",
        "",
        "| Window | Avg Clock (MHz) | Avg Power (W) | Avg Temp (C) | TFLOPs/s |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| First 2s | {float(row['first_2s_avg_gpu_clock_mhz']):.2f} | "
            f"{float(row['first_2s_avg_power_watts']):.2f} | "
            f"{float(row['first_2s_avg_temperature_c']):.2f} | "
            f"{float(row['first_2s_gemm_tflops_s']):.2f} |"
        ),
        (
            f"| Last 2s | {float(row['last_2s_avg_gpu_clock_mhz']):.2f} | "
            f"{float(row['last_2s_avg_power_watts']):.2f} | "
            f"{float(row['last_2s_avg_temperature_c']):.2f} | "
            f"{float(row['last_2s_gemm_tflops_s']):.2f} |"
        ),
        "",
        "## Artifacts",
        "",
        f"- Summary: `{_repo_path(row['summary_csv'])}`",
        f"- Monitor CSV: `{_repo_path(row['monitor_csv'])}`",
        f"- Unit events: `{_repo_path(row['unit_events_csv'])}`",
        (
            "- Power/clock/temperature timeline: "
            f"`{_repo_path(row['power_clock_temperature_plot'])}`"
        ),
        f"- TFLOPs timeline: `{_repo_path(row['gemm_tflops_plot'])}`",
        "",
    ]
    (output_dir / "BENCHMARK.md").write_text("\n".join(lines) + "\n")


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(dtype)

    output_dir = args.output_dir or (DEFAULT_RESULTS_ROOT / utc_stamp())
    monitor_dir = output_dir / "monitor"
    unit_dir = output_dir / "unit_events"
    plots_dir = output_dir / "plots"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    unit_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    monitor_csv = monitor_dir / "continuous_gemm.csv"
    unit_events_csv = unit_dir / "continuous_gemm.csv"
    summary_csv = output_dir / "summary.csv"
    metadata_json = output_dir / "metadata.json"
    timeline_plot = plots_dir / "power_clock_temperature_timeline.png"
    tflops_plot = plots_dir / "gemm_tflops_timeline.png"

    row: dict[str, Any] = {
        "status": "error",
        "error": "",
        "run_timestamp_utc": utc_now_iso(),
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "dtype": args.dtype,
        "gemm_units": args.gemm_units,
        "active_nvtx_range": ACTIVE_NVTX_RANGE,
    }

    try:
        state = TwoGemmState(
            torch=torch,
            m=args.m,
            n=args.n,
            k=args.k,
            device="cuda:0",
            dtype=dtype,
        )
        flops_per_unit = _gemm_flops(args.m, args.n, args.k)

        with torch.inference_mode():
            torch.cuda.synchronize()
            monitor = GPUMonitor(
                gpu_index=args.monitor_gpu_index,
                interval=args.monitor_interval,
            )
            monitor.start()
            monitor_start_time = _wait_for_monitor_start(monitor)
            if args.pre_idle_s > 0:
                time.sleep(args.pre_idle_s)

            active_start_event = torch.cuda.Event(enable_timing=True)
            active_end_event = torch.cuda.Event(enable_timing=True)
            unit_records: list[UnitRecord] = []
            torch.cuda.synchronize()
            active_start_wall = time.time()
            active_start_event.record()
            with torch.cuda.nvtx.range(ACTIVE_NVTX_RANGE):
                for unit_index in range(args.gemm_units):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    state.run_unit()
                    end_event.record()
                    unit_records.append(
                        UnitRecord(
                            unit_index=unit_index,
                            start_event=start_event,
                            end_event=end_event,
                        )
                    )
                active_end_event.record()
                torch.cuda.synchronize()
                active_end_wall = time.time()
            if args.post_idle_s > 0:
                time.sleep(args.post_idle_s)
            monitor.stop()

        monitor_records = monitor.get_results()
        if monitor_records:
            monitor.export_csv(str(monitor_csv))

        active_cuda_ms = active_start_event.elapsed_time(active_end_event)
        active_start_monitor_s = active_start_wall - monitor_start_time
        active_end_monitor_s = active_end_wall - monitor_start_time
        for unit in unit_records:
            unit.cuda_start_ms = active_start_event.elapsed_time(
                unit.start_event
            )
            unit.cuda_end_ms = active_start_event.elapsed_time(unit.end_event)
            unit.cuda_duration_ms = unit.start_event.elapsed_time(
                unit.end_event
            )
            unit.monitor_start_s = (
                active_start_monitor_s + unit.cuda_start_ms / 1000.0
            )
            unit.monitor_end_s = (
                active_start_monitor_s + unit.cuda_end_ms / 1000.0
            )
            if unit.cuda_duration_ms > 0.0:
                unit.tflops_s = (
                    flops_per_unit / (unit.cuda_duration_ms / 1000.0) / 1e12
                )

        transition_time_s, transition_detected = _detect_transition(
            monitor_records,
            active_start_monitor_s,
            active_end_monitor_s,
        )
        first_2s = _window_summary(
            monitor_records,
            unit_records,
            flops_per_unit,
            active_start_monitor_s,
            min(active_end_monitor_s, active_start_monitor_s + 2.0),
            "first_2s",
        )
        last_2s = _window_summary(
            monitor_records,
            unit_records,
            flops_per_unit,
            max(active_start_monitor_s, active_end_monitor_s - 2.0),
            active_end_monitor_s,
            "last_2s",
        )
        if transition_detected:
            transition_abs_s = active_start_monitor_s + transition_time_s
            pre_transition_raw = _window_summary(
                monitor_records,
                unit_records,
                flops_per_unit,
                max(active_start_monitor_s, transition_abs_s - 0.5),
                transition_abs_s,
                "transition_pre_500ms",
            )
            post_transition_raw = _window_summary(
                monitor_records,
                unit_records,
                flops_per_unit,
                transition_abs_s,
                min(active_end_monitor_s, transition_abs_s + 0.5),
                "transition_post_500ms",
            )
            pre_transition = {
                "transition_pre_500ms_power_watts": pre_transition_raw[
                    "transition_pre_500ms_avg_power_watts"
                ],
                "transition_pre_500ms_clock_mhz": pre_transition_raw[
                    "transition_pre_500ms_avg_gpu_clock_mhz"
                ],
                "transition_pre_500ms_temperature_c": pre_transition_raw[
                    "transition_pre_500ms_avg_temperature_c"
                ],
                "transition_pre_500ms_tflops_s": pre_transition_raw[
                    "transition_pre_500ms_gemm_tflops_s"
                ],
            }
            post_transition = {
                "transition_post_500ms_power_watts": post_transition_raw[
                    "transition_post_500ms_avg_power_watts"
                ],
                "transition_post_500ms_clock_mhz": post_transition_raw[
                    "transition_post_500ms_avg_gpu_clock_mhz"
                ],
                "transition_post_500ms_temperature_c": post_transition_raw[
                    "transition_post_500ms_avg_temperature_c"
                ],
                "transition_post_500ms_tflops_s": post_transition_raw[
                    "transition_post_500ms_gemm_tflops_s"
                ],
            }
            transition_temp = _temperature_at(monitor_records, transition_abs_s)
        else:
            pre_transition = {
                "transition_pre_500ms_power_watts": 0.0,
                "transition_pre_500ms_clock_mhz": 0.0,
                "transition_pre_500ms_temperature_c": 0.0,
                "transition_pre_500ms_tflops_s": 0.0,
            }
            post_transition = {
                "transition_post_500ms_power_watts": 0.0,
                "transition_post_500ms_clock_mhz": 0.0,
                "transition_post_500ms_temperature_c": 0.0,
                "transition_post_500ms_tflops_s": 0.0,
            }
            transition_temp = 0.0

        row.update(
            {
                "status": "ok",
                "active_cuda_ms": active_cuda_ms,
                "active_wall_s": active_end_wall - active_start_wall,
                "gemm_flops_per_unit": flops_per_unit,
                "gemm_tflops_s": (
                    flops_per_unit
                    * len(unit_records)
                    / (active_cuda_ms / 1000.0)
                    / 1e12
                    if active_cuda_ms > 0.0
                    else 0.0
                ),
                **first_2s,
                **last_2s,
                "transition_time_s": transition_time_s,
                "transition_detected": transition_detected,
                **pre_transition,
                **post_transition,
                "active_start_temperature_c": _temperature_at(
                    monitor_records, active_start_monitor_s
                ),
                "transition_temperature_c": transition_temp,
                "active_end_temperature_c": _temperature_at(
                    monitor_records, active_end_monitor_s
                ),
                "monitor_sample_count": len(monitor_records),
                "monitor_csv": str(monitor_csv),
                "unit_events_csv": str(unit_events_csv),
                "power_clock_temperature_plot": str(timeline_plot),
                "gemm_tflops_plot": str(tflops_plot),
            }
        )
        _write_unit_rows(unit_events_csv, unit_records)
        _write_power_clock_temperature_plot(
            monitor_records,
            active_start_monitor_s,
            active_end_monitor_s,
            transition_time_s,
            timeline_plot,
        )
        _write_tflops_plot(
            unit_records,
            active_start_monitor_s,
            transition_time_s,
            tflops_plot,
        )
    except Exception as exc:
        row["error"] = repr(exc)

    write_csv(summary_csv, SUMMARY_FIELDNAMES, [row])
    metadata = {
        "generated_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "process_device": "cuda:0",
        "monitor_gpu_index": args.monitor_gpu_index,
        "active_nvtx_range": ACTIVE_NVTX_RANGE,
        "args": _jsonable_args(args),
    }
    write_json(metadata_json, metadata)
    row["summary_csv"] = str(summary_csv)
    if row["status"] == "ok":
        _write_report(output_dir, row)
    print(output_dir)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a continuous state-chain two-GEMM workload."
    )
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--gemm-units", type=int, default=DEFAULT_GEMM_UNITS)
    parser.add_argument(
        "--monitor-interval", type=float, default=DEFAULT_MONITOR_INTERVAL
    )
    parser.add_argument(
        "--monitor-gpu-index", type=int, default=DEFAULT_MONITOR_GPU_INDEX
    )
    parser.add_argument("--pre-idle-s", type=float, default=DEFAULT_PRE_IDLE_S)
    parser.add_argument(
        "--post-idle-s", type=float, default=DEFAULT_POST_IDLE_S
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
