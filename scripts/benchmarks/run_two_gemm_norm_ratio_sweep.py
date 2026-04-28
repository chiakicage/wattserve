#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


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
    make_scaled_tensor,
    monitor_summary,
    utc_now_iso,
    utc_stamp,
    write_csv,
    write_json,
    write_kernel_profile_csv,
    write_timeline_plot,
)


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "two_gemm_norm_ratio_sweep"
DEFAULT_DIM_VALUES = [2048, 4096, 8192, 16384]
DEFAULT_K_VALUES = [2048, 4096, 8192, 16384, 32768, 65536]
VARIANTS = ("two_gemm", "two_gemm_fused_add_norm")
DEFAULT_DTYPE = "bfloat16"
DEFAULT_WARMUP = 20
DEFAULT_PROBE_REPEAT = 10
DEFAULT_TARGET_TIMED_SECONDS = 3.0
DEFAULT_MAX_REPEAT = 50000
DEFAULT_MONITOR_INTERVAL = 0.01
DEFAULT_MONITOR_GPU_INDEX = 3
DEFAULT_TARGET_COMPONENT_TIME_MS = 100.0
DEFAULT_EPS = 1e-6

SUMMARY_FIELDNAMES = [
    "status",
    "error",
    "run_timestamp_utc",
    "m",
    "n",
    "k",
    "variant",
    "dtype",
    "repeat",
    "warmup",
    "probe_repeat",
    "target_timed_seconds",
    "target_gemm_time_ms",
    "target_norm_time_ms",
    "repeat_limited_by_max",
    "gemm_time_target_met",
    "norm_time_target_met",
    "total_time_s",
    "iter_time_ms",
    "probe_iter_ms",
    "probe_gemm_iter_time_ms",
    "probe_norm_iter_time_ms",
    "gemm_total_ms",
    "gemm_iter_time_ms",
    "norm_total_ms",
    "norm_iter_time_ms",
    "gemm_norm_time_ratio",
    "norm_time_share_pct",
    "gemm_tflops_s_raw",
    "effective_tflops_s",
    "gemm_flops_per_step",
    "avg_power_watts",
    "max_power_watts",
    "avg_gpu_clock_mhz",
    "max_gpu_clock_mhz",
    "monitor_sample_count",
    "monitor_csv",
    "timeline_plot",
    "kernel_profile_csv",
    "kernel_profile_error",
    "eps",
    "description",
]


@dataclass(frozen=True)
class RatioCase:
    m: int
    n: int
    k: int
    variant: str
    run_once: Callable[[], None]
    measure_component_times_ms: Callable[[int], tuple[float, float]]
    gemm_flops_per_step: float
    description: str

    @property
    def with_norm(self) -> bool:
        return self.variant == "two_gemm_fused_add_norm"


def _gemm_flops(m: int, n: int, k: int) -> float:
    return 4.0 * m * n * k


def _recorded_mm(
    torch: Any,
    events: list[tuple[Any, Any]],
    lhs: Any,
    rhs: Any,
    out: Any,
) -> None:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    torch.mm(lhs, rhs, out=out)
    end_event.record()
    events.append((start_event, end_event))


def _recorded_call(
    torch: Any,
    events: list[tuple[Any, Any]],
    fn: Callable[[], None],
) -> None:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    fn()
    end_event.record()
    events.append((start_event, end_event))


def _build_case(
    torch: Any,
    flashinfer: Any,
    m: int,
    n: int,
    k: int,
    variant: str,
    device: str,
    dtype: Any,
    eps: float,
) -> RatioCase:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    with_norm = variant == "two_gemm_fused_add_norm"

    current = make_scaled_tensor(torch, (m, n), device, dtype)
    next_current = torch.empty_like(current)
    middle = torch.empty((m, k), device=device, dtype=dtype)
    left_weight = make_scaled_tensor(torch, (n, k), device, dtype)
    right_weight = make_scaled_tensor(torch, (k, n), device, dtype)
    residual = make_scaled_tensor(torch, (m, n), device, dtype)
    norm_weight = torch.ones((n,), device=device, dtype=dtype)

    def run_norm() -> None:
        flashinfer.fused_add_rmsnorm(
            next_current,
            residual,
            norm_weight,
            eps=eps,
        )

    def run_step(
        gemm_events: list[tuple[Any, Any]] | None = None,
        norm_events: list[tuple[Any, Any]] | None = None,
    ) -> None:
        nonlocal current, next_current
        if gemm_events is None:
            torch.mm(current, left_weight, out=middle)
            torch.mm(middle, right_weight, out=next_current)
        else:
            _recorded_mm(torch, gemm_events, current, left_weight, middle)
            _recorded_mm(torch, gemm_events, middle, right_weight, next_current)
        if with_norm:
            if norm_events is None:
                run_norm()
            else:
                _recorded_call(torch, norm_events, run_norm)
        current, next_current = next_current, current

    def run_once() -> None:
        run_step()

    def measure_component_times_ms(repeat: int) -> tuple[float, float]:
        gemm_events: list[tuple[Any, Any]] = []
        norm_events: list[tuple[Any, Any]] = []
        torch.cuda.synchronize()
        for _ in range(repeat):
            run_step(gemm_events=gemm_events, norm_events=norm_events)
        torch.cuda.synchronize()
        gemm_total_ms = sum(
            start.elapsed_time(end) for start, end in gemm_events
        )
        norm_total_ms = sum(
            start.elapsed_time(end) for start, end in norm_events
        )
        return gemm_total_ms, norm_total_ms

    return RatioCase(
        m=m,
        n=n,
        k=k,
        variant=variant,
        run_once=run_once,
        measure_component_times_ms=measure_component_times_ms,
        gemm_flops_per_step=_gemm_flops(m, n, k),
        description=(
            "state-chain two-GEMM ratio step: (M,N)x(N,K)->(M,K), "
            "(M,K)x(K,N)->(M,N), optional fused_add_rmsnorm(M,N)"
        ),
    )


def _calibrate_ratio_repeat(
    torch: Any,
    case: RatioCase,
    target_timed_seconds: float,
    probe_repeat: int,
    max_repeat: int,
    target_gemm_time_ms: float,
    target_norm_time_ms: float,
) -> dict[str, Any]:
    probe_repeat = max(1, probe_repeat)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(probe_repeat):
        case.run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    probe_iter_seconds = elapsed / probe_repeat if probe_repeat else 0.0

    probe_gemm_ms, probe_norm_ms = case.measure_component_times_ms(probe_repeat)
    probe_gemm_iter_ms = probe_gemm_ms / probe_repeat
    probe_norm_iter_ms = probe_norm_ms / probe_repeat

    repeat_requirements = [1]
    if probe_iter_seconds > 0:
        repeat_requirements.append(
            math.ceil(target_timed_seconds / probe_iter_seconds)
        )
    if probe_gemm_iter_ms > 0:
        repeat_requirements.append(
            math.ceil(target_gemm_time_ms / probe_gemm_iter_ms)
        )
    if case.with_norm and probe_norm_iter_ms > 0:
        repeat_requirements.append(
            math.ceil(target_norm_time_ms / probe_norm_iter_ms)
        )

    requested_repeat = max(repeat_requirements)
    repeat_limited = requested_repeat > max_repeat
    repeat = max(1, min(max_repeat, requested_repeat))
    return {
        "repeat": repeat,
        "repeat_limited_by_max": repeat_limited,
        "probe_iter_ms": probe_iter_seconds * 1000.0,
        "probe_gemm_iter_time_ms": probe_gemm_iter_ms,
        "probe_norm_iter_time_ms": probe_norm_iter_ms,
    }


def _format_float(value: Any, precision: int = 2) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{precision}f}"


def _base_row(
    args: argparse.Namespace, m: int, n: int, k: int, variant: str
) -> dict[str, Any]:
    return {
        "status": "error",
        "error": "",
        "run_timestamp_utc": utc_now_iso(),
        "m": m,
        "n": n,
        "k": k,
        "variant": variant,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "probe_repeat": args.probe_repeat,
        "target_timed_seconds": args.target_timed_seconds,
        "target_gemm_time_ms": args.target_gemm_time_ms,
        "target_norm_time_ms": args.target_norm_time_ms,
        "eps": args.eps,
    }


def _run_case(
    args: argparse.Namespace,
    torch: Any,
    flashinfer: Any,
    output_dir: Path,
    m: int,
    n: int,
    k: int,
    variant: str,
) -> dict[str, Any]:
    row = _base_row(args, m, n, k, variant)
    monitor_dir = output_dir / "monitor"
    profile_dir = output_dir / "kernel_profile"
    label = f"M={m}/N={n}/K={k}/{variant}"
    print(f"Running {label}", flush=True)

    try:
        case = _build_case(
            torch=torch,
            flashinfer=flashinfer,
            m=m,
            n=n,
            k=k,
            variant=variant,
            device="cuda:0",
            dtype=getattr(torch, args.dtype),
            eps=args.eps,
        )
        with torch.inference_mode():
            for _ in range(args.warmup):
                case.run_once()
            torch.cuda.synchronize()
            calibration = _calibrate_ratio_repeat(
                torch=torch,
                case=case,
                target_timed_seconds=args.target_timed_seconds,
                probe_repeat=args.probe_repeat,
                max_repeat=args.max_repeat,
                target_gemm_time_ms=args.target_gemm_time_ms,
                target_norm_time_ms=args.target_norm_time_ms,
            )
            repeat = int(calibration["repeat"])

            monitor = GPUMonitor(
                gpu_index=args.monitor_gpu_index,
                interval=args.monitor_interval,
            )
            monitor.start()
            try:
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(repeat):
                    case.run_once()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            finally:
                monitor.stop()

            gemm_total_ms, norm_total_ms = case.measure_component_times_ms(
                repeat
            )

            kernel_profile_csv = ""
            kernel_profile_error = ""
            if not args.skip_kernel_profile:
                profile_path = (
                    profile_dir / f"M={m}__N={n}__K={k}__{variant}.csv"
                )
                try:
                    write_kernel_profile_csv(
                        torch=torch,
                        run_once=case.run_once,
                        output_path=profile_path,
                        repeat=args.profile_repeat,
                    )
                    kernel_profile_csv = str(profile_path)
                except Exception as exc:
                    kernel_profile_error = f"{type(exc).__name__}: {exc}"

        records = monitor.get_results()
        monitor_csv = ""
        if records:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            monitor_path = monitor_dir / f"M={m}__N={n}__K={k}__{variant}.csv"
            monitor.export_csv(str(monitor_path))
            monitor_csv = str(monitor_path)

        iter_time_s = elapsed / repeat
        gemm_time_s = gemm_total_ms / 1000.0
        norm_iter_time_ms = norm_total_ms / repeat if repeat else 0.0
        gemm_norm_time_ratio = (
            gemm_total_ms / norm_total_ms if norm_total_ms > 0.0 else ""
        )
        norm_time_share_pct = (
            norm_total_ms / (gemm_total_ms + norm_total_ms) * 100.0
            if norm_total_ms > 0.0
            else ""
        )
        row.update(
            {
                "status": "ok",
                "repeat": repeat,
                **calibration,
                "gemm_time_target_met": gemm_total_ms
                >= args.target_gemm_time_ms,
                "norm_time_target_met": (
                    norm_total_ms >= args.target_norm_time_ms
                    if case.with_norm
                    else ""
                ),
                "total_time_s": elapsed,
                "iter_time_ms": iter_time_s * 1000.0,
                "gemm_total_ms": gemm_total_ms,
                "gemm_iter_time_ms": gemm_total_ms / repeat,
                "norm_total_ms": norm_total_ms,
                "norm_iter_time_ms": norm_iter_time_ms,
                "gemm_norm_time_ratio": gemm_norm_time_ratio,
                "norm_time_share_pct": norm_time_share_pct,
                "gemm_tflops_s_raw": (
                    case.gemm_flops_per_step * repeat / 1e12 / gemm_time_s
                    if gemm_time_s > 0.0
                    else 0.0
                ),
                "effective_tflops_s": (
                    case.gemm_flops_per_step * repeat / 1e12 / elapsed
                    if elapsed > 0.0
                    else 0.0
                ),
                "gemm_flops_per_step": case.gemm_flops_per_step,
                "monitor_csv": monitor_csv,
                "kernel_profile_csv": kernel_profile_csv,
                "kernel_profile_error": kernel_profile_error,
                "description": case.description,
                **monitor_summary(records),
            }
        )
        print(
            f"{label}: raw={row['gemm_tflops_s_raw']:.2f} GEMM TFLOPs/s, "
            f"effective={row['effective_tflops_s']:.2f} TFLOPs/s, "
            f"gemm_ms={row['gemm_total_ms']:.2f}, "
            f"norm_ms={row['norm_total_ms']:.2f}, "
            f"power={row['avg_power_watts']:.2f} W, "
            f"clock={row['avg_gpu_clock_mhz']:.2f} MHz",
            flush=True,
        )
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["description"] = traceback.format_exc(limit=5)
        print(f"FAILED {label}: {row['error']}", flush=True)
        return row
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _generate_timeline_plots(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    plots_dir = output_dir / "plots" / "timeline"
    rows_by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or not row.get("monitor_csv"):
            continue
        key = (int(row["m"]), int(row["n"]), int(row["k"]))
        rows_by_key.setdefault(key, []).append(row)

    for (m, n, k), group in rows_by_key.items():
        ordered = sorted(group, key=lambda row: str(row["variant"]))
        output_path = plots_dir / f"M={m}__N={n}__K={k}.png"
        plot_path = write_timeline_plot(
            [
                (str(row["variant"]), Path(str(row["monitor_csv"])))
                for row in ordered
            ],
            output_path=output_path,
            title=f"M={m} N={n} K={k}: power and clock",
        )
        if plot_path is not None:
            for row in group:
                row["timeline_plot"] = str(plot_path)


def output_dir_display(path: str) -> str:
    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path


def _build_report(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    lines = [
        "# Two-GEMM + Norm Ratio Sweep",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        f"- Output directory: `{metadata['output_dir']}`",
        f"- CUDA_VISIBLE_DEVICES: `{metadata.get('cuda_visible_devices')}`",
        f"- Monitor GPU index: `{metadata['monitor_gpu_index']}`",
        f"- Dtype: `{metadata['dtype']}`",
        "- State chain: `(M,N)x(N,K)->(M,K)`, `(M,K)x(K,N)->(M,N)`, optional `fused_add_rmsnorm(M,N)`.",
        "- GEMM FLOPs per step: `4*M*N*K`.",
        "- Raw GEMM TFLOPs/s uses only CUDA-event time around the two GEMMs.",
        "",
        "## Results",
        "",
        "| M | N | K | Variant | Raw GEMM TFLOPs/s | Effective TFLOPs/s | GEMM Total (ms) | Norm Total (ms) | GEMM/Norm Time | Avg Power (W) | Avg GPU Clock (MHz) |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ok_rows:
        lines.append(
            f"| {row['m']} | {row['n']} | {row['k']} | {row['variant']} | "
            f"{_format_float(row['gemm_tflops_s_raw'])} | "
            f"{_format_float(row['effective_tflops_s'])} | "
            f"{_format_float(row['gemm_total_ms'])} | "
            f"{_format_float(row['norm_total_ms'])} | "
            f"{_format_float(row['gemm_norm_time_ratio'])} | "
            f"{_format_float(row['avg_power_watts'])} | "
            f"{_format_float(row['avg_gpu_clock_mhz'])} |"
        )

    by_key = {
        (int(row["m"]), int(row["n"]), int(row["k"]), row["variant"]): row
        for row in ok_rows
    }
    lines.extend(
        [
            "",
            "## Norm Delta",
            "",
            "| M | N | K | GEMM/Norm Time | w/o Norm Raw GEMM | w/ Norm Raw GEMM | Raw Delta (%) | Effective Delta (%) | Power Delta (W) | Clock Delta (MHz) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for m in metadata["m_values"]:
        for n in metadata["n_values"]:
            for k in metadata["k_values"]:
                without = by_key.get((int(m), int(n), int(k), "two_gemm"))
                with_norm = by_key.get(
                    (int(m), int(n), int(k), "two_gemm_fused_add_norm")
                )
                if without is None or with_norm is None:
                    continue
                raw_delta = (
                    (
                        with_norm["gemm_tflops_s_raw"]
                        - without["gemm_tflops_s_raw"]
                    )
                    / without["gemm_tflops_s_raw"]
                    * 100.0
                )
                effective_delta = (
                    (
                        with_norm["effective_tflops_s"]
                        - without["effective_tflops_s"]
                    )
                    / without["effective_tflops_s"]
                    * 100.0
                )
                lines.append(
                    f"| {m} | {n} | {k} | "
                    f"{_format_float(with_norm['gemm_norm_time_ratio'])} | "
                    f"{_format_float(without['gemm_tflops_s_raw'])} | "
                    f"{_format_float(with_norm['gemm_tflops_s_raw'])} | "
                    f"{_format_float(raw_delta)} | "
                    f"{_format_float(effective_delta)} | "
                    f"{_format_float(with_norm['avg_power_watts'] - without['avg_power_watts'])} | "
                    f"{_format_float(with_norm['avg_gpu_clock_mhz'] - without['avg_gpu_clock_mhz'])} |"
                )

    if failed_rows:
        lines.extend(
            [
                "",
                "## Failures",
                "",
                "| M | N | K | Variant | Error |",
                "| ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in failed_rows:
            lines.append(
                f"| {row.get('m')} | {row.get('n')} | {row.get('k')} | "
                f"{row.get('variant')} | `{row.get('error')}` |"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{metadata['summary_csv']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            f"- Timeline plots: `{output_dir_display(metadata['plots_dir'])}`",
            f"- Kernel profiles: `{output_dir_display(metadata['kernel_profile_dir'])}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError("--dtype must be float16, bfloat16, or float32")

    dtype = getattr(torch, args.dtype)
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(dtype)

    output_dir = args.output_dir or (DEFAULT_RESULTS_ROOT / utc_stamp())
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    summary_path = output_dir / "summary.csv"
    benchmark_path = output_dir / "BENCHMARK.md"

    metadata: dict[str, Any] = {
        "run_started_at_utc": utc_now_iso(),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_path),
        "metadata_json": str(metadata_path),
        "benchmark_markdown": str(benchmark_path),
        "plots_dir": str(output_dir / "plots"),
        "kernel_profile_dir": str(output_dir / "kernel_profile"),
        "monitor_dir": str(output_dir / "monitor"),
        "m_values": args.m_values,
        "n_values": args.n_values,
        "k_values": args.k_values,
        "variants": args.variants,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "probe_repeat": args.probe_repeat,
        "target_timed_seconds": args.target_timed_seconds,
        "max_repeat": args.max_repeat,
        "target_gemm_time_ms": args.target_gemm_time_ms,
        "target_norm_time_ms": args.target_norm_time_ms,
        "monitor_interval": args.monitor_interval,
        "monitor_gpu_index": args.monitor_gpu_index,
        "profile_repeat": args.profile_repeat,
        "skip_kernel_profile": args.skip_kernel_profile,
        "eps": args.eps,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gemm_flops_per_step": "4*M*N*K",
    }
    write_json(metadata_path, metadata)

    rows: list[dict[str, Any]] = []
    for m in args.m_values:
        for n in args.n_values:
            for k in args.k_values:
                for variant in args.variants:
                    row = _run_case(
                        args=args,
                        torch=torch,
                        flashinfer=flashinfer,
                        output_dir=output_dir,
                        m=m,
                        n=n,
                        k=k,
                        variant=variant,
                    )
                    rows.append(row)
                    metadata["completed_cases"] = len(rows)
                    metadata["failed_cases"] = sum(
                        1 for item in rows if item.get("status") != "ok"
                    )
                    write_csv(summary_path, SUMMARY_FIELDNAMES, rows)
                    write_json(metadata_path, metadata)

    _generate_timeline_plots(rows, output_dir)
    write_csv(summary_path, SUMMARY_FIELDNAMES, rows)
    benchmark_path.write_text(_build_report(rows, metadata) + "\n")
    print(f"Wrote {benchmark_path}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="State-chain two-GEMM plus norm time-ratio sweep."
    )
    parser.add_argument(
        "--m-values", nargs="+", type=int, default=list(DEFAULT_DIM_VALUES)
    )
    parser.add_argument(
        "--n-values", nargs="+", type=int, default=list(DEFAULT_DIM_VALUES)
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES)
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--probe-repeat", type=int, default=DEFAULT_PROBE_REPEAT
    )
    parser.add_argument(
        "--target-timed-seconds",
        type=float,
        default=DEFAULT_TARGET_TIMED_SECONDS,
    )
    parser.add_argument("--max-repeat", type=int, default=DEFAULT_MAX_REPEAT)
    parser.add_argument(
        "--target-gemm-time-ms",
        type=float,
        default=DEFAULT_TARGET_COMPONENT_TIME_MS,
    )
    parser.add_argument(
        "--target-norm-time-ms",
        type=float,
        default=DEFAULT_TARGET_COMPONENT_TIME_MS,
    )
    parser.add_argument(
        "--monitor-interval", type=float, default=DEFAULT_MONITOR_INTERVAL
    )
    parser.add_argument(
        "--monitor-gpu-index", type=int, default=DEFAULT_MONITOR_GPU_INDEX
    )
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--skip-kernel-profile", action="store_true")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
