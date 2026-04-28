#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
    utc_now_iso,
    utc_stamp,
    write_csv,
    write_json,
)


DEFAULT_RESULTS_ROOT = (
    REPO_ROOT / "results" / "two_gemm_norm_steady_window_sweep"
)
DEFAULT_M = 8192
DEFAULT_N = 8192
DEFAULT_K = 32768
DEFAULT_GEMM_STEPS = [1, 2, 4, 8, 12, 16, 18, 20, 22, 24, 32, 48, 64]
DEFAULT_NORM_STEPS = [0]
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
    "gemm_steps_per_phase",
    "norm_steps_per_phase",
    "cycles",
    "target_active_s",
    "analysis_window_s",
    "probe_cycle_ms",
    "active_cuda_ms",
    "active_wall_s",
    "tail_start_cuda_ms",
    "tail_end_cuda_ms",
    "tail_window_cuda_ms",
    "gemm_phase_count_total",
    "norm_phase_count_total",
    "gemm_phase_count_tail",
    "norm_phase_count_tail",
    "gemm_phase_cuda_total_ms_tail",
    "norm_phase_cuda_total_ms_tail",
    "avg_gemm_phase_ms_tail",
    "avg_norm_phase_ms_tail",
    "gemm_units_tail",
    "norm_units_tail",
    "gemm_flops_per_unit",
    "gemm_tflops_s_tail",
    "avg_power_watts_tail",
    "max_power_watts_tail",
    "min_power_watts_tail",
    "avg_gpu_clock_mhz_tail",
    "max_gpu_clock_mhz_tail",
    "min_gpu_clock_mhz_tail",
    "monitor_sample_count_tail",
    "avg_gemm_phase_power_watts_tail",
    "avg_norm_phase_power_watts_tail",
    "avg_gemm_phase_clock_mhz_tail",
    "avg_norm_phase_clock_mhz_tail",
    "monitor_csv",
    "phase_events_csv",
    "eps",
]


def _monitor_window_stats(
    records: list[dict[str, Any]],
) -> dict[str, float | int]:
    return {
        "avg_power_watts_tail": average(records, "power_watts"),
        "max_power_watts_tail": max(
            (float(record["power_watts"]) for record in records), default=0.0
        ),
        "min_power_watts_tail": min(
            (float(record["power_watts"]) for record in records), default=0.0
        ),
        "avg_gpu_clock_mhz_tail": average(records, "gpu_clock_mhz"),
        "max_gpu_clock_mhz_tail": max(
            (float(record["gpu_clock_mhz"]) for record in records), default=0.0
        ),
        "min_gpu_clock_mhz_tail": min(
            (float(record["gpu_clock_mhz"]) for record in records), default=0.0
        ),
        "monitor_sample_count_tail": len(records),
    }


def _measure_cycle_ms(
    torch: Any,
    state: TwoGemmNormState,
    gemm_steps: int,
    norm_steps: int,
) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    state.run_gemm_phase(gemm_steps)
    if norm_steps > 0:
        state.run_norm_phase(norm_steps)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


def _phase_average(phases: list[PhaseRecord], key: str) -> float:
    return average(
        [phase.__dict__ for phase in phases if phase.sample_count],
        key,
    )


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
        "gemm_steps_per_phase": gemm_steps,
        "norm_steps_per_phase": norm_steps,
        "target_active_s": args.target_active_s,
        "analysis_window_s": args.analysis_window_s,
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

            probe_cycle_ms = _measure_cycle_ms(
                torch,
                state,
                gemm_steps=gemm_steps,
                norm_steps=norm_steps,
            )
            target_ms = args.target_active_s * 1000.0
            requested_cycles = math.ceil(
                target_ms * args.cycle_margin / max(probe_cycle_ms, 1e-6)
            )
            cycles = max(1, requested_cycles)

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
                for cycle in range(cycles):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with torch.cuda.nvtx.range(
                        f"{label}_cycle_{cycle:04d}_gemm"
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
                            f"{label}_cycle_{cycle:04d}_norm"
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

        active_cuda_ms = active_start_event.elapsed_time(active_end_event)
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

        tail_window_ms = min(
            args.analysis_window_s * 1000.0,
            active_cuda_ms,
        )
        tail_start_cuda_ms = max(0.0, active_cuda_ms - tail_window_ms)
        tail_end_cuda_ms = active_cuda_ms
        tail_monitor_start_s = (
            active_start_monitor_elapsed + tail_start_cuda_ms / 1000.0
        )
        tail_monitor_end_s = (
            active_start_monitor_elapsed + active_cuda_ms / 1000.0
        )
        tail_monitor_records = [
            record
            for record in monitor_records
            if tail_monitor_start_s
            <= float(record["elapsed_seconds"])
            <= tail_monitor_end_s
        ]

        gemm_phases_total = [
            phase for phase in phase_records if phase.phase == "gemm"
        ]
        norm_phases_total = [
            phase for phase in phase_records if phase.phase == "norm"
        ]
        gemm_phases_tail = [
            phase
            for phase in gemm_phases_total
            if phase.cuda_start_ms >= tail_start_cuda_ms
        ]
        norm_phases_tail = [
            phase
            for phase in norm_phases_total
            if phase.cuda_start_ms >= tail_start_cuda_ms
        ]
        gemm_phase_cuda_total_ms_tail = sum(
            phase.cuda_duration_ms for phase in gemm_phases_tail
        )
        norm_phase_cuda_total_ms_tail = sum(
            phase.cuda_duration_ms for phase in norm_phases_tail
        )
        gemm_units_tail = sum(phase.steps for phase in gemm_phases_tail)
        norm_units_tail = sum(phase.steps for phase in norm_phases_tail)
        gemm_flops_per_unit = _gemm_flops(args.m, args.n, args.k)

        row.update(
            {
                "status": "ok",
                "cycles": cycles,
                "probe_cycle_ms": probe_cycle_ms,
                "active_cuda_ms": active_cuda_ms,
                "active_wall_s": active_end_wall - active_start_wall,
                "tail_start_cuda_ms": tail_start_cuda_ms,
                "tail_end_cuda_ms": tail_end_cuda_ms,
                "tail_window_cuda_ms": tail_window_ms,
                "gemm_phase_count_total": len(gemm_phases_total),
                "norm_phase_count_total": len(norm_phases_total),
                "gemm_phase_count_tail": len(gemm_phases_tail),
                "norm_phase_count_tail": len(norm_phases_tail),
                "gemm_phase_cuda_total_ms_tail": gemm_phase_cuda_total_ms_tail,
                "norm_phase_cuda_total_ms_tail": norm_phase_cuda_total_ms_tail,
                "avg_gemm_phase_ms_tail": (
                    gemm_phase_cuda_total_ms_tail / len(gemm_phases_tail)
                    if gemm_phases_tail
                    else 0.0
                ),
                "avg_norm_phase_ms_tail": (
                    norm_phase_cuda_total_ms_tail / len(norm_phases_tail)
                    if norm_phases_tail
                    else 0.0
                ),
                "gemm_units_tail": gemm_units_tail,
                "norm_units_tail": norm_units_tail,
                "gemm_flops_per_unit": gemm_flops_per_unit,
                "gemm_tflops_s_tail": (
                    gemm_flops_per_unit
                    * gemm_units_tail
                    / 1e12
                    / (gemm_phase_cuda_total_ms_tail / 1000.0)
                    if gemm_phase_cuda_total_ms_tail > 0
                    else 0.0
                ),
                **_monitor_window_stats(tail_monitor_records),
                "avg_gemm_phase_power_watts_tail": _phase_average(
                    gemm_phases_tail,
                    "avg_power_watts",
                ),
                "avg_norm_phase_power_watts_tail": _phase_average(
                    norm_phases_tail,
                    "avg_power_watts",
                ),
                "avg_gemm_phase_clock_mhz_tail": _phase_average(
                    gemm_phases_tail,
                    "avg_gpu_clock_mhz",
                ),
                "avg_norm_phase_clock_mhz_tail": _phase_average(
                    norm_phases_tail,
                    "avg_gpu_clock_mhz",
                ),
                "monitor_csv": str(monitor_csv),
                "phase_events_csv": str(phase_events_csv),
            }
        )
        print(
            f"{label}: active={active_cuda_ms/1000.0:.2f}s, "
            f"tail_gemm_phase={row['avg_gemm_phase_ms_tail']:.1f} ms, "
            f"tail_clock={row['avg_gemm_phase_clock_mhz_tail']:.1f} MHz, "
            f"tail_tflops={row['gemm_tflops_s_tail']:.2f}, "
            f"tail_phases={row['gemm_phase_count_tail']}",
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
    lines = [
        "# Two-GEMM / Norm Steady Window Sweep",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        f"- Output directory: `{output_dir}`",
        f"- Successful cases: `{len(ok_rows)}/{len(rows)}`",
        "- Metrics are computed from the tail analysis window only.",
        "",
        "| GEMM Steps | Norm Steps | Active (s) | Tail GEMM Phase (ms) | Tail GEMM Clock (MHz) | Tail TFLOPs/s | Tail GEMM Phases |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ok_rows:
        lines.append(
            f"| {row['gemm_steps_per_phase']} | {row['norm_steps_per_phase']} | "
            f"{float(row['active_cuda_ms']) / 1000.0:.2f} | "
            f"{float(row['avg_gemm_phase_ms_tail']):.2f} | "
            f"{float(row['avg_gemm_phase_clock_mhz_tail']):.2f} | "
            f"{float(row['gemm_tflops_s_tail']):.2f} | "
            f"{row['gemm_phase_count_tail']} |"
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
    (output_dir / "README.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(getattr(torch, args.dtype))

    output_dir = args.output_dir or (DEFAULT_RESULTS_ROOT / utc_stamp())
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "metadata.json",
        {
            "run_started_at_utc": utc_now_iso(),
            "output_dir": str(output_dir),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "gemm_steps": args.gemm_steps,
            "norm_steps": args.norm_steps,
            "target_active_s": args.target_active_s,
            "analysis_window_s": args.analysis_window_s,
            "cycle_margin": args.cycle_margin,
            "monitor_interval": args.monitor_interval,
            "monitor_gpu_index": args.monitor_gpu_index,
        },
    )

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
    print(f"Wrote {output_dir / 'README.md'}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-GEMM/norm sweeps and report only the steady tail window."
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
    parser.add_argument("--target-active-s", type=float, default=30.0)
    parser.add_argument("--analysis-window-s", type=float, default=20.0)
    parser.add_argument("--cycle-margin", type=float, default=1.08)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--monitor-interval", type=float, default=DEFAULT_MONITOR_INTERVAL
    )
    parser.add_argument(
        "--monitor-gpu-index", type=int, default=DEFAULT_MONITOR_GPU_INDEX
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--cooldown-s", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
