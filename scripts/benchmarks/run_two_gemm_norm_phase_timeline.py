#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

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
    monitor_summary,
    utc_now_iso,
    utc_stamp,
    write_csv,
    write_json,
)


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "two_gemm_norm_phase_timeline"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_M = 2048
DEFAULT_N = 2048
DEFAULT_K = 65536
DEFAULT_CYCLES = 20
DEFAULT_TARGET_GEMM_PHASE_MS = 200.0
DEFAULT_TARGET_NORM_PHASE_MS = 200.0
DEFAULT_MAX_STEPS_PER_PHASE = 50000
DEFAULT_WARMUP = 10
DEFAULT_PROBE_REPEAT = 20
DEFAULT_MONITOR_INTERVAL = 0.01
DEFAULT_MONITOR_GPU_INDEX = 3
DEFAULT_EPS = 1e-6

SUMMARY_FIELDNAMES = [
    "run_timestamp_utc",
    "m",
    "n",
    "k",
    "dtype",
    "cycles",
    "gemm_steps_per_phase",
    "norm_steps_per_phase",
    "target_gemm_phase_ms",
    "target_norm_phase_ms",
    "calibrated_gemm_step_ms",
    "calibrated_norm_step_ms",
    "gemm_steps_limited_by_max",
    "norm_steps_limited_by_max",
    "active_cuda_ms",
    "active_wall_s",
    "gemm_phase_cuda_total_ms",
    "norm_phase_cuda_total_ms",
    "avg_gemm_phase_ms",
    "avg_norm_phase_ms",
    "gemm_units_total",
    "norm_units_total",
    "gemm_flops_per_unit",
    "gemm_tflops_s_during_gemm_phases",
    "avg_power_watts",
    "max_power_watts",
    "avg_gpu_clock_mhz",
    "max_gpu_clock_mhz",
    "monitor_sample_count",
    "avg_gemm_phase_power_watts",
    "avg_norm_phase_power_watts",
    "avg_gemm_phase_clock_mhz",
    "avg_norm_phase_clock_mhz",
    "phase_power_delta_watts_norm_minus_gemm",
    "phase_clock_delta_mhz_norm_minus_gemm",
    "monitor_csv",
    "phase_events_csv",
    "phase_summary_csv",
    "timeline_plot",
    "eps",
]

PHASE_FIELDNAMES = [
    "cycle",
    "phase_index",
    "phase",
    "steps",
    "cuda_start_ms",
    "cuda_end_ms",
    "cuda_duration_ms",
    "monitor_start_s",
    "monitor_end_s",
    "avg_power_watts",
    "avg_gpu_clock_mhz",
    "sample_count",
]


@dataclass
class PhaseRecord:
    cycle: int
    phase_index: int
    phase: str
    steps: int
    start_event: Any
    end_event: Any
    cuda_start_ms: float = 0.0
    cuda_end_ms: float = 0.0
    cuda_duration_ms: float = 0.0
    monitor_start_s: float = 0.0
    monitor_end_s: float = 0.0
    avg_power_watts: float = 0.0
    avg_gpu_clock_mhz: float = 0.0
    sample_count: int = 0


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


class TwoGemmNormState:
    def __init__(
        self,
        torch: Any,
        flashinfer: Any,
        m: int,
        n: int,
        k: int,
        device: str,
        dtype: Any,
        eps: float,
    ) -> None:
        self.torch = torch
        self.flashinfer = flashinfer
        self.m = m
        self.n = n
        self.k = k
        self.eps = eps
        self.current = make_scaled_tensor(torch, (m, n), device, dtype)
        self.next_current = torch.empty_like(self.current)
        self.middle = torch.empty((m, k), device=device, dtype=dtype)
        self.left_weight = make_scaled_tensor(torch, (n, k), device, dtype)
        self.right_weight = make_scaled_tensor(torch, (k, n), device, dtype)
        self.residual = make_scaled_tensor(torch, (m, n), device, dtype)
        self.norm_weight = torch.ones((n,), device=device, dtype=dtype)

    def run_gemm_unit(self) -> None:
        self.torch.mm(self.current, self.left_weight, out=self.middle)
        self.torch.mm(self.middle, self.right_weight, out=self.next_current)
        self.current, self.next_current = self.next_current, self.current

    def run_norm_unit(self) -> None:
        self.flashinfer.fused_add_rmsnorm(
            self.current,
            self.residual,
            self.norm_weight,
            eps=self.eps,
        )

    def run_gemm_phase(self, steps: int) -> None:
        for _ in range(steps):
            self.run_gemm_unit()

    def run_norm_phase(self, steps: int) -> None:
        for _ in range(steps):
            self.run_norm_unit()


def _measure_unit_ms(
    torch: Any,
    fn: Any,
    repeat: int,
) -> float:
    repeat = max(1, repeat)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(repeat):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / repeat


def _measure_phase_ms(
    torch: Any,
    fn: Any,
    steps: int,
) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    fn(steps)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


def _choose_steps(
    target_ms: float,
    unit_ms: float,
    max_steps: int,
    explicit_steps: int | None,
) -> tuple[int, bool]:
    if explicit_steps is not None:
        return max(1, explicit_steps), False
    if unit_ms <= 0.0:
        return 1, False
    requested = max(1, math.ceil(target_ms / unit_ms))
    return min(requested, max_steps), requested > max_steps


def _refine_steps(
    target_ms: float,
    measured_phase_ms: float,
    steps: int,
    max_steps: int,
    explicit_steps: int | None,
) -> tuple[int, bool]:
    if explicit_steps is not None or measured_phase_ms <= 0.0:
        return steps, False
    if measured_phase_ms >= target_ms * 0.85:
        return steps, False
    requested = max(1, math.ceil(steps * target_ms / measured_phase_ms))
    return min(requested, max_steps), requested > max_steps


def _samples_for_phase(
    records: list[dict[str, Any]],
    phase: PhaseRecord,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if phase.monitor_start_s
        <= float(record["elapsed_seconds"])
        <= phase.monitor_end_s
    ]


def _write_phase_rows(
    path: Path,
    phase_records: list[PhaseRecord],
) -> None:
    rows = [
        {
            "cycle": phase.cycle,
            "phase_index": phase.phase_index,
            "phase": phase.phase,
            "steps": phase.steps,
            "cuda_start_ms": phase.cuda_start_ms,
            "cuda_end_ms": phase.cuda_end_ms,
            "cuda_duration_ms": phase.cuda_duration_ms,
            "monitor_start_s": phase.monitor_start_s,
            "monitor_end_s": phase.monitor_end_s,
            "avg_power_watts": phase.avg_power_watts,
            "avg_gpu_clock_mhz": phase.avg_gpu_clock_mhz,
            "sample_count": phase.sample_count,
        }
        for phase in phase_records
    ]
    write_csv(path, PHASE_FIELDNAMES, rows)


def _write_timeline_plot(
    monitor_records: list[dict[str, Any]],
    phase_records: list[PhaseRecord],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    x_values = [float(record["elapsed_seconds"]) for record in monitor_records]
    axes[0].plot(
        x_values,
        [float(record["power_watts"]) for record in monitor_records],
        color="#1f2937",
        linewidth=1.2,
    )
    axes[1].plot(
        x_values,
        [float(record["gpu_clock_mhz"]) for record in monitor_records],
        color="#111827",
        linewidth=1.2,
    )

    colors = {"gemm": "#60a5fa", "norm": "#f59e0b"}
    labeled: set[str] = set()
    for phase in phase_records:
        label = phase.phase.upper() if phase.phase not in labeled else None
        labeled.add(phase.phase)
        for axis in axes:
            axis.axvspan(
                phase.monitor_start_s,
                phase.monitor_end_s,
                color=colors[phase.phase],
                alpha=0.18,
                label=label,
                linewidth=0,
            )

    axes[0].set_ylabel("Power (W)")
    axes[1].set_ylabel("GPU Clock (MHz)")
    axes[1].set_xlabel("Monitor Elapsed Time (s)")
    axes[0].set_title(title)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles, strict=False))
            axis.legend(by_label.values(), by_label.keys(), loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _write_report(
    output_dir: Path,
    summary_row: dict[str, Any],
) -> None:
    lines = [
        "# Two-GEMM / Norm Phase Timeline",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        f"- Shape: `M={summary_row['m']}, N={summary_row['n']}, K={summary_row['k']}`",
        f"- Cycles: `{summary_row['cycles']}`",
        f"- GEMM steps / phase: `{summary_row['gemm_steps_per_phase']}`",
        f"- Norm steps / phase: `{summary_row['norm_steps_per_phase']}`",
        f"- Active CUDA time: `{float(summary_row['active_cuda_ms']):.3f} ms`",
        f"- Timeline plot: `{Path(summary_row['timeline_plot']).relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## Phase Averages",
        "",
        "| Phase | Avg Duration (ms) | Avg Power (W) | Avg GPU Clock (MHz) |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| GEMM | {float(summary_row['avg_gemm_phase_ms']):.3f} | "
            f"{float(summary_row['avg_gemm_phase_power_watts']):.2f} | "
            f"{float(summary_row['avg_gemm_phase_clock_mhz']):.2f} |"
        ),
        (
            f"| Norm | {float(summary_row['avg_norm_phase_ms']):.3f} | "
            f"{float(summary_row['avg_norm_phase_power_watts']):.2f} | "
            f"{float(summary_row['avg_norm_phase_clock_mhz']):.2f} |"
        ),
        "",
        "## Artifacts",
        "",
        f"- Summary: `{Path(summary_row['summary_csv']).relative_to(REPO_ROOT).as_posix()}`",
        f"- Monitor CSV: `{Path(summary_row['monitor_csv']).relative_to(REPO_ROOT).as_posix()}`",
        f"- Phase events: `{Path(summary_row['phase_events_csv']).relative_to(REPO_ROOT).as_posix()}`",
        f"- Phase summary: `{Path(summary_row['phase_summary_csv']).relative_to(REPO_ROOT).as_posix()}`",
        "",
    ]
    (output_dir / "BENCHMARK.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(dtype)

    output_dir = args.output_dir or (DEFAULT_RESULTS_ROOT / utc_stamp())
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_csv = output_dir / "monitor.csv"
    phase_events_csv = output_dir / "phase_events.csv"
    phase_summary_csv = output_dir / "phase_summary.csv"
    summary_csv = output_dir / "summary.csv"
    timeline_plot = output_dir / "plots" / "phase_timeline.png"
    metadata_json = output_dir / "metadata.json"

    state = TwoGemmNormState(
        torch=torch,
        flashinfer=flashinfer,
        m=args.m,
        n=args.n,
        k=args.k,
        device="cuda:0",
        dtype=dtype,
        eps=args.eps,
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            state.run_gemm_unit()
            state.run_norm_unit()
        torch.cuda.synchronize()

        gemm_unit_ms = _measure_unit_ms(
            torch, state.run_gemm_unit, args.probe_repeat
        )
        norm_unit_ms = _measure_unit_ms(
            torch, state.run_norm_unit, args.probe_repeat
        )
        gemm_steps, gemm_limited = _choose_steps(
            args.target_gemm_phase_ms,
            gemm_unit_ms,
            args.max_steps_per_phase,
            args.gemm_steps_per_phase,
        )
        norm_steps, norm_limited = _choose_steps(
            args.target_norm_phase_ms,
            norm_unit_ms,
            args.max_steps_per_phase,
            args.norm_steps_per_phase,
        )
        measured_gemm_phase_ms = _measure_phase_ms(
            torch, state.run_gemm_phase, gemm_steps
        )
        measured_norm_phase_ms = _measure_phase_ms(
            torch, state.run_norm_phase, norm_steps
        )
        refined_gemm_steps, refined_gemm_limited = _refine_steps(
            args.target_gemm_phase_ms,
            measured_gemm_phase_ms,
            gemm_steps,
            args.max_steps_per_phase,
            args.gemm_steps_per_phase,
        )
        refined_norm_steps, refined_norm_limited = _refine_steps(
            args.target_norm_phase_ms,
            measured_norm_phase_ms,
            norm_steps,
            args.max_steps_per_phase,
            args.norm_steps_per_phase,
        )
        gemm_steps = refined_gemm_steps
        norm_steps = refined_norm_steps
        gemm_limited = gemm_limited or refined_gemm_limited
        norm_limited = norm_limited or refined_norm_limited

        monitor = GPUMonitor(
            gpu_index=args.monitor_gpu_index,
            interval=args.monitor_interval,
        )
        monitor.start()
        monitor_start_time = _wait_for_monitor_start(monitor)
        phase_records: list[PhaseRecord] = []
        active_start_event = torch.cuda.Event(enable_timing=True)
        active_end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        active_start_wall = time.time()
        active_start_event.record()
        phase_index = 0
        for cycle in range(args.cycles):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.cuda.nvtx.range(f"cycle_{cycle:03d}_gemm"):
                state.run_gemm_phase(gemm_steps)
            end_event.record()
            phase_records.append(
                PhaseRecord(
                    cycle=cycle,
                    phase_index=phase_index,
                    phase="gemm",
                    steps=gemm_steps,
                    start_event=start_event,
                    end_event=end_event,
                )
            )
            phase_index += 1

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.cuda.nvtx.range(f"cycle_{cycle:03d}_norm"):
                state.run_norm_phase(norm_steps)
            end_event.record()
            phase_records.append(
                PhaseRecord(
                    cycle=cycle,
                    phase_index=phase_index,
                    phase="norm",
                    steps=norm_steps,
                    start_event=start_event,
                    end_event=end_event,
                )
            )
            phase_index += 1
        active_end_event.record()
        torch.cuda.synchronize()
        active_end_wall = time.time()
        monitor.stop()

    monitor_records = monitor.get_results()
    if monitor_records:
        monitor.export_csv(str(monitor_csv))

    active_start_monitor_elapsed = active_start_wall - monitor_start_time
    for phase in phase_records:
        phase.cuda_start_ms = active_start_event.elapsed_time(phase.start_event)
        phase.cuda_end_ms = active_start_event.elapsed_time(phase.end_event)
        phase.cuda_duration_ms = phase.start_event.elapsed_time(phase.end_event)
        phase.monitor_start_s = (
            active_start_monitor_elapsed + phase.cuda_start_ms / 1000.0
        )
        phase.monitor_end_s = (
            active_start_monitor_elapsed + phase.cuda_end_ms / 1000.0
        )
        samples = _samples_for_phase(monitor_records, phase)
        phase.sample_count = len(samples)
        phase.avg_power_watts = average(samples, "power_watts")
        phase.avg_gpu_clock_mhz = average(samples, "gpu_clock_mhz")

    _write_phase_rows(phase_events_csv, phase_records)
    _write_phase_rows(phase_summary_csv, phase_records)
    _write_timeline_plot(
        monitor_records=monitor_records,
        phase_records=phase_records,
        output_path=timeline_plot,
        title=(
            f"Two-GEMM / Norm phases: M={args.m}, N={args.n}, K={args.k}, "
            f"cycles={args.cycles}"
        ),
    )

    gemm_phases = [phase for phase in phase_records if phase.phase == "gemm"]
    norm_phases = [phase for phase in phase_records if phase.phase == "norm"]
    gemm_phase_cuda_total_ms = sum(
        phase.cuda_duration_ms for phase in gemm_phases
    )
    norm_phase_cuda_total_ms = sum(
        phase.cuda_duration_ms for phase in norm_phases
    )
    gemm_units_total = args.cycles * gemm_steps
    norm_units_total = args.cycles * norm_steps
    gemm_flops_per_unit = _gemm_flops(args.m, args.n, args.k)
    monitor_stats = monitor_summary(monitor_records)
    avg_gemm_power = average(
        [phase.__dict__ for phase in gemm_phases if phase.sample_count],
        "avg_power_watts",
    )
    avg_norm_power = average(
        [phase.__dict__ for phase in norm_phases if phase.sample_count],
        "avg_power_watts",
    )
    avg_gemm_clock = average(
        [phase.__dict__ for phase in gemm_phases if phase.sample_count],
        "avg_gpu_clock_mhz",
    )
    avg_norm_clock = average(
        [phase.__dict__ for phase in norm_phases if phase.sample_count],
        "avg_gpu_clock_mhz",
    )

    summary_row: dict[str, Any] = {
        "run_timestamp_utc": utc_now_iso(),
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "dtype": args.dtype,
        "cycles": args.cycles,
        "gemm_steps_per_phase": gemm_steps,
        "norm_steps_per_phase": norm_steps,
        "target_gemm_phase_ms": args.target_gemm_phase_ms,
        "target_norm_phase_ms": args.target_norm_phase_ms,
        "calibrated_gemm_step_ms": gemm_unit_ms,
        "calibrated_norm_step_ms": norm_unit_ms,
        "gemm_steps_limited_by_max": gemm_limited,
        "norm_steps_limited_by_max": norm_limited,
        "active_cuda_ms": active_start_event.elapsed_time(active_end_event),
        "active_wall_s": active_end_wall - active_start_wall,
        "gemm_phase_cuda_total_ms": gemm_phase_cuda_total_ms,
        "norm_phase_cuda_total_ms": norm_phase_cuda_total_ms,
        "avg_gemm_phase_ms": gemm_phase_cuda_total_ms / args.cycles,
        "avg_norm_phase_ms": norm_phase_cuda_total_ms / args.cycles,
        "gemm_units_total": gemm_units_total,
        "norm_units_total": norm_units_total,
        "gemm_flops_per_unit": gemm_flops_per_unit,
        "gemm_tflops_s_during_gemm_phases": (
            gemm_flops_per_unit
            * gemm_units_total
            / 1e12
            / (gemm_phase_cuda_total_ms / 1000.0)
            if gemm_phase_cuda_total_ms > 0
            else 0.0
        ),
        **monitor_stats,
        "avg_gemm_phase_power_watts": avg_gemm_power,
        "avg_norm_phase_power_watts": avg_norm_power,
        "avg_gemm_phase_clock_mhz": avg_gemm_clock,
        "avg_norm_phase_clock_mhz": avg_norm_clock,
        "phase_power_delta_watts_norm_minus_gemm": avg_norm_power
        - avg_gemm_power,
        "phase_clock_delta_mhz_norm_minus_gemm": avg_norm_clock
        - avg_gemm_clock,
        "monitor_csv": str(monitor_csv),
        "phase_events_csv": str(phase_events_csv),
        "phase_summary_csv": str(phase_summary_csv),
        "timeline_plot": str(timeline_plot),
        "summary_csv": str(summary_csv),
        "eps": args.eps,
    }

    write_csv(summary_csv, SUMMARY_FIELDNAMES, [summary_row])
    write_json(
        metadata_json,
        {
            "run_started_at_utc": summary_row["run_timestamp_utc"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "monitor_gpu_index": args.monitor_gpu_index,
            "monitor_interval": args.monitor_interval,
            "summary": summary_row,
        },
    )
    _write_report(output_dir, summary_row)
    print(f"Wrote {output_dir / 'BENCHMARK.md'}", flush=True)
    print(
        "Phase averages: "
        f"GEMM {summary_row['avg_gemm_phase_ms']:.2f} ms, "
        f"{summary_row['avg_gemm_phase_power_watts']:.2f} W, "
        f"{summary_row['avg_gemm_phase_clock_mhz']:.2f} MHz; "
        f"Norm {summary_row['avg_norm_phase_ms']:.2f} ms, "
        f"{summary_row['avg_norm_phase_power_watts']:.2f} W, "
        f"{summary_row['avg_norm_phase_clock_mhz']:.2f} MHz",
        flush=True,
    )
    return summary_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run alternating two-GEMM and norm phases for NVML timeline inspection."
    )
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--gemm-steps-per-phase", type=int)
    parser.add_argument("--norm-steps-per-phase", type=int)
    parser.add_argument(
        "--target-gemm-phase-ms",
        type=float,
        default=DEFAULT_TARGET_GEMM_PHASE_MS,
    )
    parser.add_argument(
        "--target-norm-phase-ms",
        type=float,
        default=DEFAULT_TARGET_NORM_PHASE_MS,
    )
    parser.add_argument(
        "--max-steps-per-phase",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_PHASE,
    )
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--probe-repeat", type=int, default=DEFAULT_PROBE_REPEAT
    )
    parser.add_argument(
        "--monitor-interval", type=float, default=DEFAULT_MONITOR_INTERVAL
    )
    parser.add_argument(
        "--monitor-gpu-index", type=int, default=DEFAULT_MONITOR_GPU_INDEX
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
