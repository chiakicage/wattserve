#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "run_gemm_replay_vs_chain_microbench.py"
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_nsys_path() -> str:
    preferred = Path("/opt/nvidia/nsight-systems-cli/2025.2.1/bin/nsys")
    if preferred.exists():
        return str(preferred)
    found = shutil.which("nsys")
    return found or "nsys"


def _default_ncu_path() -> str:
    found = shutil.which("ncu")
    return found or "ncu"


def _active_nvtx_label(mode: str, workload: str, steps: int) -> str:
    return f"gemm_replay_vs_chain__{mode}__{workload}__steps_{steps}__active"


def _case_slug(mode: str, workload: str, steps: int) -> str:
    return f"{mode}__{workload}__steps={steps}"


def _benchmark_cmd(
    args: argparse.Namespace,
    mode: str,
    workload: str,
    steps: int,
    output_dir: Path,
) -> list[str]:
    cmd = shlex.split(args.python_launcher)
    cmd.extend(
        [
            str(BENCHMARK_SCRIPT),
            "--case-kind",
            args.case_kind,
            "--modes",
            mode,
            "--workloads",
            workload,
            "--steps",
            str(steps),
            "--warmup",
            str(args.warmup),
            "--probe-repeat",
            str(args.probe_repeat),
            "--target-timed-seconds",
            str(args.target_timed_seconds),
            "--monitor-gpu-index",
            str(args.monitor_gpu_index),
            "--monitor-interval",
            str(args.monitor_interval),
            "--profile-active-nvtx",
            "--output-dir",
            str(output_dir),
        ]
    )
    if args.profile_cuda_profiler_api:
        cmd.append("--profile-cuda-profiler-api")
    return cmd


def _nsys_cmd(
    args: argparse.Namespace,
    case_dir: Path,
    bench_cmd: list[str],
) -> list[str]:
    output_prefix = case_dir / "nsys" / "profile"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    return [
        args.nsys_path,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        f"--gpu-metrics-devices={args.nsys_gpu_metrics_devices}",
        f"--gpu-metrics-set={args.nsys_gpu_metrics_set}",
        f"--gpu-metrics-frequency={args.nsys_gpu_metrics_frequency}",
        "--output",
        str(output_prefix),
        *bench_cmd,
    ]


def _ncu_cmd(
    args: argparse.Namespace,
    case_dir: Path,
    active_nvtx_label: str,
    bench_cmd: list[str],
) -> list[str]:
    output_prefix = case_dir / "ncu" / "profile"
    log_file = case_dir / "ncu" / "profile.csv"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.ncu_path,
        "--target-processes",
        "all",
        "--nvtx",
        "--nvtx-include",
        active_nvtx_label,
        "--replay-mode",
        args.ncu_replay_mode,
        "--cache-control",
        "none",
        "--clock-control",
        "none",
        "--pipeline-boost-state",
        "dynamic",
        "--devices",
        args.ncu_devices,
        "--pm-sampling-interval",
        str(args.ncu_pm_sampling_interval),
        "--csv",
        "--page",
        "raw",
        "--print-summary",
        "per-nvtx",
        "--log-file",
        str(log_file),
        "--export",
        str(output_prefix),
        "--force-overwrite",
    ]
    if args.ncu_metrics:
        cmd.extend(["--metrics", args.ncu_metrics])
    else:
        cmd.extend(["--set", args.ncu_set])
    cmd.extend(bench_cmd)
    return cmd


def _cupti_cmd(bench_cmd: list[str]) -> list[str]:
    if "--profile-cuda-profiler-api" in bench_cmd:
        return list(bench_cmd)
    return [*bench_cmd, "--profile-cuda-profiler-api"]


def _write_case_command(path: Path, command: list[str]) -> None:
    path.write_text(" ".join(shlex.quote(part) for part in command) + "\n")


def _run_command(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> int:
    _write_case_command(log_path.with_suffix(".cmd"), command)
    if dry_run:
        return 0
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return proc.returncode


def run(args: argparse.Namespace) -> int:
    output_root = args.output_dir or (
        REPO_ROOT
        / "results"
        / "gemm_replay_vs_chain_memory_profile"
        / _utc_stamp()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cases: list[dict[str, Any]] = []
    failures = 0
    for mode in args.modes:
        for workload in args.workloads:
            for steps in args.steps:
                slug = _case_slug(mode, workload, steps)
                case_dir = output_root / slug
                bench_output_dir = case_dir / "benchmark"
                case_dir.mkdir(parents=True, exist_ok=True)
                active_label = _active_nvtx_label(mode, workload, steps)
                bench_cmd = _benchmark_cmd(
                    args=args,
                    mode=mode,
                    workload=workload,
                    steps=steps,
                    output_dir=bench_output_dir,
                )

                case_record: dict[str, Any] = {
                    "mode": mode,
                    "workload": workload,
                    "steps": steps,
                    "active_nvtx_label": active_label,
                    "benchmark_output_dir": str(bench_output_dir),
                    "commands": {},
                    "returncodes": {},
                }

                if "nsys" in args.tools:
                    command = _nsys_cmd(args, case_dir, bench_cmd)
                    log_path = case_dir / "nsys.log"
                    print(f"Running nsys {slug}", flush=True)
                    returncode = _run_command(
                        command=command,
                        env=env,
                        log_path=log_path,
                        dry_run=args.dry_run,
                    )
                    case_record["commands"]["nsys"] = str(
                        log_path.with_suffix(".cmd")
                    )
                    case_record["returncodes"]["nsys"] = returncode
                    failures += int(returncode != 0)

                if "ncu" in args.tools:
                    command = _ncu_cmd(args, case_dir, active_label, bench_cmd)
                    log_path = case_dir / "ncu.log"
                    print(f"Running ncu {slug}", flush=True)
                    returncode = _run_command(
                        command=command,
                        env=env,
                        log_path=log_path,
                        dry_run=args.dry_run,
                    )
                    case_record["commands"]["ncu"] = str(
                        log_path.with_suffix(".cmd")
                    )
                    case_record["returncodes"]["ncu"] = returncode
                    failures += int(returncode != 0)

                if "cupti" in args.tools:
                    cupti_dir = case_dir / "cupti"
                    cupti_dir.mkdir(parents=True, exist_ok=True)
                    command = _cupti_cmd(bench_cmd)
                    cupti_env = env.copy()
                    cupti_env["CUDA_INJECTION64_PATH"] = str(args.cupti_library)
                    cupti_env["CUPTI_EVENT_MONITOR_CSV"] = str(
                        cupti_dir / "events.csv"
                    )
                    cupti_env["CUPTI_EVENT_MONITOR_EVENTS"] = args.cupti_events
                    cupti_env["CUPTI_EVENT_MONITOR_INTERVAL_MS"] = str(
                        args.cupti_interval_ms
                    )
                    log_path = cupti_dir / "cupti.log"
                    print(f"Running cupti {slug}", flush=True)
                    returncode = _run_command(
                        command=command,
                        env=cupti_env,
                        log_path=log_path,
                        dry_run=args.dry_run,
                    )
                    case_record["commands"]["cupti"] = str(
                        log_path.with_suffix(".cmd")
                    )
                    case_record["returncodes"]["cupti"] = returncode
                    case_record["cupti_events_csv"] = str(
                        cupti_dir / "events.csv"
                    )
                    failures += int(returncode != 0)

                cases.append(case_record)

    metadata = {
        "output_dir": str(output_root),
        "tools": args.tools,
        "cuda_visible_devices": args.cuda_visible_devices,
        "monitor_gpu_index": args.monitor_gpu_index,
        "nsys_gpu_metrics_devices": args.nsys_gpu_metrics_devices,
        "nsys_gpu_metrics_set": args.nsys_gpu_metrics_set,
        "nsys_gpu_metrics_frequency": args.nsys_gpu_metrics_frequency,
        "ncu_replay_mode": args.ncu_replay_mode,
        "ncu_cache_control": "none",
        "ncu_clock_control": "none",
        "ncu_pipeline_boost_state": "dynamic",
        "ncu_set": args.ncu_set,
        "ncu_metrics": args.ncu_metrics,
        "cupti_library": str(args.cupti_library),
        "cupti_events": args.cupti_events,
        "cupti_interval_ms": args.cupti_interval_ms,
        "target_timed_seconds": args.target_timed_seconds,
        "profile_cuda_profiler_api": args.profile_cuda_profiler_api,
        "dry_run": args.dry_run,
        "cases": cases,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# GEMM Replay vs Chain Memory Profiling",
                "",
                "This directory contains profiler runs for the continuous active",
                "window of `run_gemm_replay_vs_chain_microbench.py`.",
                "",
                "- The benchmark active loop is wrapped in an NVTX range.",
                "- Warmup, calibration, and GEMM event timing are outside that range.",
                "- NCU runs use `cache-control none`, `clock-control none`, and",
                "  `pipeline-boost-state dynamic` to avoid profiler defaults that",
                "  would distort the power/clock phenomenon.",
                "- NCU reports are auxiliary unless the run is confirmed not to need",
                "  destructive replay or cache reset for the selected metrics.",
                "- CUPTI runs use CUDA injection plus cudaProfilerStart/Stop around",
                "  the same active loop, so the event monitor samples only the",
                "  contiguous workload window.",
                "",
                "See `metadata.json` and per-case `*.cmd` files for exact commands.",
                "",
            ]
        )
    )
    print(f"Wrote {output_root}", flush=True)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run nsys/ncu memory profiling for GEMM replay-vs-chain active ranges."
        )
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=("nsys", "ncu", "cupti"),
        default=["nsys"],
        help="Profiler tools to run. Default is nsys only.",
    )
    parser.add_argument("--case-kind", choices=("o", "mlp"), default="mlp")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("fixed_replay", "state_chain"),
        default=["fixed_replay", "state_chain"],
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["mlp_silu", "mlp_silu_fused_add_norm"],
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[40])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--probe-repeat", type=int, default=10)
    parser.add_argument("--target-timed-seconds", type=float, default=10.0)
    parser.add_argument("--monitor-interval", type=float, default=0.01)
    parser.add_argument("--monitor-gpu-index", type=int, default=3)
    parser.add_argument("--cuda-visible-devices", default="3")
    parser.add_argument(
        "--python-launcher",
        default="uv run python",
        help="Launcher used for the Python benchmark process.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nsys-path", default=_default_nsys_path())
    parser.add_argument("--nsys-gpu-metrics-devices", default="3")
    parser.add_argument("--nsys-gpu-metrics-set", default="ga100")
    parser.add_argument("--nsys-gpu-metrics-frequency", type=int, default=10000)
    parser.add_argument("--ncu-path", default=_default_ncu_path())
    parser.add_argument(
        "--ncu-devices",
        default="0",
        help=(
            "NCU target device index. With CUDA_VISIBLE_DEVICES=3, the "
            "benchmark sees the physical GPU as logical device 0."
        ),
    )
    parser.add_argument(
        "--ncu-replay-mode",
        default="app-range",
        choices=("range", "app-range", "application"),
    )
    parser.add_argument("--ncu-set", default="pmsampling")
    parser.add_argument("--ncu-metrics")
    parser.add_argument("--ncu-pm-sampling-interval", default="0")
    parser.add_argument(
        "--profile-cuda-profiler-api",
        action="store_true",
        help="Also call cudaProfilerStart/Stop around the active loop.",
    )
    parser.add_argument(
        "--cupti-library",
        type=Path,
        default=(
            REPO_ROOT
            / "scripts"
            / "benchmarks"
            / "cupti_event_monitor"
            / "libcupti_active_event_monitor.so"
        ),
    )
    parser.add_argument(
        "--cupti-events",
        default="inst_executed",
        help="Comma-separated CUPTI Event API legacy event names.",
    )
    parser.add_argument("--cupti-interval-ms", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
