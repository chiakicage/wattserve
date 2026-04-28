#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
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
from run_two_gemm_norm_phase_timeline import (  # noqa: E402
    PhaseRecord,
    TwoGemmNormState,
    _gemm_flops,
    _samples_for_phase,
    _wait_for_monitor_start,
    _write_phase_rows,
)
from state_chain_utils import (  # noqa: E402
    average,
    monitor_summary,
    utc_now_iso,
    utc_stamp,
    write_csv,
    write_json,
)


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "two_gemm_norm_phase_sweep"
DEFAULT_M = 16384
DEFAULT_N = 16384
DEFAULT_K = 65536
DEFAULT_GEMM_STEPS = [1, 2, 4]
DEFAULT_NORM_STEPS = [0, 1, 4, 16, 64, 256, 631]
DEFAULT_CYCLES = 20
DEFAULT_DTYPE = "bfloat16"
DEFAULT_WARMUP = 10
DEFAULT_MONITOR_INTERVAL = 0.01
DEFAULT_MONITOR_GPU_INDEX = 3
DEFAULT_EPS = 1e-6

SUMMARY_FIELDNAMES = [
    "status",
    "error",
    "run_timestamp_utc",
    "m",
    "n",
    "k",
    "dtype",
    "cycles",
    "gemm_steps_per_phase",
    "norm_steps_per_phase",
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
    "min_power_watts",
    "avg_gpu_clock_mhz",
    "max_gpu_clock_mhz",
    "min_gpu_clock_mhz",
    "monitor_sample_count",
    "avg_gemm_phase_power_watts",
    "avg_norm_phase_power_watts",
    "avg_gemm_phase_clock_mhz",
    "avg_norm_phase_clock_mhz",
    "phase_power_delta_watts_norm_minus_gemm",
    "phase_clock_delta_mhz_norm_minus_gemm",
    "downclock_avg_below_mhz",
    "downclock_min_below_mhz",
    "monitor_csv",
    "phase_events_csv",
    "eps",
]


def _monitor_extrema(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "min_power_watts": min(
            (float(record["power_watts"]) for record in records), default=0.0
        ),
        "min_gpu_clock_mhz": min(
            (float(record["gpu_clock_mhz"]) for record in records), default=0.0
        ),
    }


def _run_case(
    args: argparse.Namespace,
    torch: Any,
    flashinfer: Any,
    output_dir: Path,
    gemm_steps: int,
    norm_steps: int,
) -> dict[str, Any]:
    label = f"g{gemm_steps}_n{norm_steps}"
    monitor_dir = output_dir / "monitor"
    phase_dir = output_dir / "phase_events"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {label}", flush=True)

    row: dict[str, Any] = {
        "status": "error",
        "error": "",
        "run_timestamp_utc": utc_now_iso(),
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "dtype": args.dtype,
        "cycles": args.cycles,
        "gemm_steps_per_phase": gemm_steps,
        "norm_steps_per_phase": norm_steps,
        "eps": args.eps,
    }
    try:
        state = TwoGemmNormState(
            torch=torch,
            flashinfer=flashinfer,
            m=args.m,
            n=args.n,
            k=args.k,
            device="cuda:0",
            dtype=getattr(torch, args.dtype),
            eps=args.eps,
        )

        with torch.inference_mode():
            for _ in range(args.warmup):
                state.run_gemm_unit()
                if norm_steps > 0:
                    state.run_norm_unit()
            torch.cuda.synchronize()

            monitor = GPUMonitor(
                gpu_index=args.monitor_gpu_index,
                interval=args.monitor_interval,
            )
            monitor.start()
            monitor_start_time = _wait_for_monitor_start(monitor)
            active_start_event = torch.cuda.Event(enable_timing=True)
            active_end_event = torch.cuda.Event(enable_timing=True)
            phase_records: list[PhaseRecord] = []

            torch.cuda.synchronize()
            active_start_wall = time.time()
            active_start_event.record()
            phase_index = 0
            try:
                for cycle in range(args.cycles):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with torch.cuda.nvtx.range(
                        f"{label}_cycle_{cycle:03d}_gemm"
                    ):
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

                    if norm_steps > 0:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        with torch.cuda.nvtx.range(
                            f"{label}_cycle_{cycle:03d}_norm"
                        ):
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
            finally:
                monitor.stop()

        monitor_records = monitor.get_results()
        monitor_csv = monitor_dir / f"{label}.csv"
        if monitor_records:
            monitor.export_csv(str(monitor_csv))
        active_start_monitor_elapsed = active_start_wall - monitor_start_time

        for phase in phase_records:
            phase.cuda_start_ms = active_start_event.elapsed_time(
                phase.start_event
            )
            phase.cuda_end_ms = active_start_event.elapsed_time(phase.end_event)
            phase.cuda_duration_ms = phase.start_event.elapsed_time(
                phase.end_event
            )
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

        phase_events_csv = phase_dir / f"{label}.csv"
        _write_phase_rows(phase_events_csv, phase_records)

        gemm_phases = [
            phase for phase in phase_records if phase.phase == "gemm"
        ]
        norm_phases = [
            phase for phase in phase_records if phase.phase == "norm"
        ]
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

        row.update(
            {
                "status": "ok",
                "active_cuda_ms": active_start_event.elapsed_time(
                    active_end_event
                ),
                "active_wall_s": active_end_wall - active_start_wall,
                "gemm_phase_cuda_total_ms": gemm_phase_cuda_total_ms,
                "norm_phase_cuda_total_ms": norm_phase_cuda_total_ms,
                "avg_gemm_phase_ms": (
                    gemm_phase_cuda_total_ms / args.cycles
                    if args.cycles
                    else 0.0
                ),
                "avg_norm_phase_ms": (
                    norm_phase_cuda_total_ms / args.cycles
                    if args.cycles and norm_steps > 0
                    else 0.0
                ),
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
                **_monitor_extrema(monitor_records),
                "avg_gemm_phase_power_watts": avg_gemm_power,
                "avg_norm_phase_power_watts": avg_norm_power,
                "avg_gemm_phase_clock_mhz": avg_gemm_clock,
                "avg_norm_phase_clock_mhz": avg_norm_clock,
                "phase_power_delta_watts_norm_minus_gemm": avg_norm_power
                - avg_gemm_power,
                "phase_clock_delta_mhz_norm_minus_gemm": avg_norm_clock
                - avg_gemm_clock,
                "downclock_avg_below_mhz": float(
                    monitor_stats["avg_gpu_clock_mhz"]
                )
                < args.downclock_threshold_mhz,
                "downclock_min_below_mhz": float(
                    _monitor_extrema(monitor_records)["min_gpu_clock_mhz"]
                )
                < args.downclock_threshold_mhz,
                "monitor_csv": str(monitor_csv),
                "phase_events_csv": str(phase_events_csv),
            }
        )
        print(
            f"{label}: active={row['active_cuda_ms']:.1f} ms, "
            f"gemm_phase={row['avg_gemm_phase_ms']:.1f} ms, "
            f"norm_phase={row['avg_norm_phase_ms']:.1f} ms, "
            f"avg_clock={row['avg_gpu_clock_mhz']:.1f} MHz, "
            f"min_clock={row['min_gpu_clock_mhz']:.1f} MHz, "
            f"gemm_tflops={row['gemm_tflops_s_during_gemm_phases']:.2f}",
            flush=True,
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED {label}: {row['error']}", flush=True)
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if args.cooldown_s > 0:
            time.sleep(args.cooldown_s)
    return row


def _write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    downclock_rows = [
        row
        for row in ok_rows
        if row.get("downclock_avg_below_mhz") == "True"
        or row.get("downclock_avg_below_mhz") is True
        or row.get("downclock_min_below_mhz") == "True"
        or row.get("downclock_min_below_mhz") is True
    ]
    ranked = sorted(
        ok_rows,
        key=lambda row: (
            float(row.get("avg_gpu_clock_mhz") or 0.0),
            float(row.get("min_gpu_clock_mhz") or 0.0),
        ),
    )
    lines = [
        "# Two-GEMM / Norm Phase Sweep",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        f"- Output directory: `{output_dir}`",
        f"- Successful cases: `{len(ok_rows)}/{len(rows)}`",
        f"- Downclock cases: `{len(downclock_rows)}`",
        "",
        "## Lowest-Clock Cases",
        "",
        "| GEMM Steps | Norm Steps | GEMM Phase (ms) | Norm Phase (ms) | Avg Clock (MHz) | Min Clock (MHz) | Avg Power (W) | GEMM TFLOPs/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked[:12]:
        lines.append(
            f"| {row['gemm_steps_per_phase']} | {row['norm_steps_per_phase']} | "
            f"{float(row['avg_gemm_phase_ms']):.2f} | "
            f"{float(row['avg_norm_phase_ms']):.2f} | "
            f"{float(row['avg_gpu_clock_mhz']):.2f} | "
            f"{float(row['min_gpu_clock_mhz']):.2f} | "
            f"{float(row['avg_power_watts']):.2f} | "
            f"{float(row['gemm_tflops_s_during_gemm_phases']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{(output_dir / 'summary.csv').relative_to(REPO_ROOT).as_posix()}`",
            f"- Monitor traces: `{(output_dir / 'monitor').relative_to(REPO_ROOT).as_posix()}`",
            f"- Phase event traces: `{(output_dir / 'phase_events').relative_to(REPO_ROOT).as_posix()}`",
            "",
        ]
    )
    (output_dir / "BENCHMARK.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(getattr(torch, args.dtype))

    output_dir = args.output_dir or (DEFAULT_RESULTS_ROOT / utc_stamp())
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_started_at_utc": utc_now_iso(),
        "output_dir": str(output_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "gemm_steps": args.gemm_steps,
        "norm_steps": args.norm_steps,
        "cycles": args.cycles,
        "monitor_interval": args.monitor_interval,
        "monitor_gpu_index": args.monitor_gpu_index,
        "downclock_threshold_mhz": args.downclock_threshold_mhz,
    }
    write_json(output_dir / "metadata.json", metadata)

    rows: list[dict[str, Any]] = []
    for gemm_steps in args.gemm_steps:
        for norm_steps in args.norm_steps:
            row = _run_case(
                args=args,
                torch=torch,
                flashinfer=flashinfer,
                output_dir=output_dir,
                gemm_steps=gemm_steps,
                norm_steps=norm_steps,
            )
            rows.append(row)
            write_csv(output_dir / "summary.csv", SUMMARY_FIELDNAMES, rows)
    _write_report(output_dir, rows)
    print(f"Wrote {output_dir / 'BENCHMARK.md'}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep alternating two-GEMM and norm burst lengths without plotting."
    )
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--gemm-steps", nargs="+", type=int, default=list(DEFAULT_GEMM_STEPS)
    )
    parser.add_argument(
        "--norm-steps", nargs="+", type=int, default=list(DEFAULT_NORM_STEPS)
    )
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--monitor-interval", type=float, default=DEFAULT_MONITOR_INTERVAL
    )
    parser.add_argument(
        "--monitor-gpu-index", type=int, default=DEFAULT_MONITOR_GPU_INDEX
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--downclock-threshold-mhz", type=float, default=1350.0)
    parser.add_argument("--cooldown-s", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
