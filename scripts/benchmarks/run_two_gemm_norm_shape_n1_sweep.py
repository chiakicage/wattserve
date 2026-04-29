#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_two_gemm_norm_phase_timeline import (  # noqa: E402
    TwoGemmNormState,
)
from run_two_gemm_norm_steady_window_sweep import (  # noqa: E402
    _run_case,
    SUMMARY_FIELDNAMES,
)
from state_chain_utils import (  # noqa: E402
    utc_now_iso,
    utc_stamp,
    write_csv,
    write_json,
)


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "two_gemm_norm_shape_n1_sweep"
DEFAULT_TARGET_PHASE_MS = [30, 120, 300, 450, 540, 600, 660, 800, 1200]
DEFAULT_TARGET_ACTIVE_S = 30.0
DEFAULT_ANALYSIS_WINDOW_S = 20.0
DEFAULT_DTYPE = "bfloat16"
DEFAULT_WARMUP = 8
DEFAULT_PROBE_REPEAT = 3
DEFAULT_MONITOR_INTERVAL = 0.01
DEFAULT_MONITOR_GPU_INDEX = 3
DEFAULT_EPS = 1e-6

SHAPES = [
    ("llama7b_s4096", 4096, 4096, 11008),
    ("llama7b_s8192", 8192, 4096, 11008),
    ("llama13b_s8192", 8192, 5120, 13824),
    ("llama34b_s8192", 8192, 8192, 22016),
    ("llama70b_s8192", 8192, 8192, 28672),
    ("llama70b_s16384", 16384, 8192, 28672),
]

COMBINED_FIELDNAMES = [
    "shape_name",
    "target_gemm_phase_ms",
    "estimated_gemm_unit_ms",
    "requested_gemm_steps",
    *SUMMARY_FIELDNAMES,
]

TRANSITION_FIELDNAMES = [
    "shape_name",
    "m",
    "n",
    "k",
    "norm_steps_per_phase",
    "full_clock_threshold_mhz",
    "last_low_gemm_steps",
    "last_low_gemm_phase_ms",
    "last_low_clock_mhz",
    "last_low_tflops_s",
    "first_full_gemm_steps",
    "first_full_gemm_phase_ms",
    "first_full_clock_mhz",
    "first_full_tflops_s",
    "transition_gap_ms",
]


@dataclass(frozen=True)
class ShapeConfig:
    name: str
    m: int
    n: int
    k: int


def _parse_shape(value: str) -> ShapeConfig:
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "shape must be name:M:N:K, for example llama7b_s8192:8192:4096:11008"
        )
    name, m, n, k = parts
    return ShapeConfig(name=name, m=int(m), n=int(n), k=int(k))


def _default_shapes() -> list[ShapeConfig]:
    return [ShapeConfig(name, m, n, k) for name, m, n, k in SHAPES]


def _measure_gemm_unit_ms(
    torch: Any,
    flashinfer: Any,
    shape: ShapeConfig,
    dtype: str,
    warmup: int,
    probe_repeat: int,
    eps: float,
) -> float:
    state = TwoGemmNormState(
        torch=torch,
        flashinfer=flashinfer,
        m=shape.m,
        n=shape.n,
        k=shape.k,
        device="cuda:0",
        dtype=getattr(torch, dtype),
        eps=eps,
    )
    with torch.inference_mode():
        for _ in range(max(1, warmup)):
            state.run_gemm_unit()
            state.run_norm_unit()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(max(1, probe_repeat)):
            state.run_gemm_unit()
        end_event.record()
        torch.cuda.synchronize()
    unit_ms = start_event.elapsed_time(end_event) / max(1, probe_repeat)
    del state
    torch.cuda.empty_cache()
    return unit_ms


def _choose_gemm_steps(
    target_ms_values: list[float],
    unit_ms: float,
) -> list[int]:
    steps = {
        max(1, int(round(target_ms / max(unit_ms, 1e-6))))
        for target_ms in target_ms_values
    }
    return sorted(steps)


def _target_for_steps(
    steps: int,
    target_ms_values: list[float],
    unit_ms: float,
) -> float:
    estimated_ms = steps * unit_ms
    return min(
        target_ms_values, key=lambda target_ms: abs(target_ms - estimated_ms)
    )


def _shape_args(
    args: argparse.Namespace,
    shape: ShapeConfig,
    gemm_steps: list[int],
    output_dir: Path,
) -> argparse.Namespace:
    return SimpleNamespace(
        m=shape.m,
        n=shape.n,
        k=shape.k,
        gemm_steps=gemm_steps,
        norm_steps=[1],
        target_active_s=args.target_active_s,
        analysis_window_s=args.analysis_window_s,
        cycle_margin=args.cycle_margin,
        dtype=args.dtype,
        warmup=args.warmup,
        monitor_interval=args.monitor_interval,
        monitor_gpu_index=args.monitor_gpu_index,
        eps=args.eps,
        cooldown_s=args.cooldown_s,
        output_dir=output_dir,
    )


def _write_shape_report(
    shape_dir: Path,
    shape: ShapeConfig,
    rows: list[dict[str, Any]],
) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    lines = [
        f"# {shape.name} n1 Steady Sweep",
        "",
        f"- Shape: `(M, N, K) = ({shape.m}, {shape.n}, {shape.k})`",
        "- Norm steps per phase: `1`",
        "- Metrics use the final steady analysis window.",
        "",
        "| Target GEMM Phase (ms) | GEMM Steps | Tail GEMM Phase (ms) | Tail GEMM Clock (MHz) | Tail TFLOPs/s | Tail Power (W) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ok_rows:
        lines.append(
            f"| {float(row['target_gemm_phase_ms']):.0f} | "
            f"{int(row['gemm_steps_per_phase'])} | "
            f"{float(row['avg_gemm_phase_ms_tail']):.2f} | "
            f"{float(row['avg_gemm_phase_clock_mhz_tail']):.2f} | "
            f"{float(row['gemm_tflops_s_tail']):.2f} | "
            f"{float(row['avg_gemm_phase_power_watts_tail']):.2f} |"
        )
    shape_dir.joinpath("README.md").write_text("\n".join(lines) + "\n")


def _transition_summary(
    rows: list[dict[str, Any]],
    full_clock_threshold_mhz: float,
) -> list[dict[str, Any]]:
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_shape.setdefault(str(row["shape_name"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for shape_name, shape_rows in sorted(by_shape.items()):
        shape_rows.sort(key=lambda row: float(row["avg_gemm_phase_ms_tail"]))
        low_rows = [
            row
            for row in shape_rows
            if float(row["avg_gemm_phase_clock_mhz_tail"])
            < full_clock_threshold_mhz
        ]
        full_rows = [
            row
            for row in shape_rows
            if float(row["avg_gemm_phase_clock_mhz_tail"])
            >= full_clock_threshold_mhz
        ]
        last_low = low_rows[-1] if low_rows else None
        first_full_after_low = None
        if last_low is not None:
            last_low_ms = float(last_low["avg_gemm_phase_ms_tail"])
            for row in full_rows:
                if float(row["avg_gemm_phase_ms_tail"]) >= last_low_ms:
                    first_full_after_low = row
                    break
        if first_full_after_low is None and full_rows:
            first_full_after_low = full_rows[0]

        first = shape_rows[0]
        summaries.append(
            {
                "shape_name": shape_name,
                "m": first["m"],
                "n": first["n"],
                "k": first["k"],
                "norm_steps_per_phase": first["norm_steps_per_phase"],
                "full_clock_threshold_mhz": full_clock_threshold_mhz,
                "last_low_gemm_steps": (
                    last_low.get("gemm_steps_per_phase", "") if last_low else ""
                ),
                "last_low_gemm_phase_ms": (
                    last_low.get("avg_gemm_phase_ms_tail", "")
                    if last_low
                    else ""
                ),
                "last_low_clock_mhz": (
                    last_low.get("avg_gemm_phase_clock_mhz_tail", "")
                    if last_low
                    else ""
                ),
                "last_low_tflops_s": (
                    last_low.get("gemm_tflops_s_tail", "") if last_low else ""
                ),
                "first_full_gemm_steps": (
                    first_full_after_low.get("gemm_steps_per_phase", "")
                    if first_full_after_low
                    else ""
                ),
                "first_full_gemm_phase_ms": (
                    first_full_after_low.get("avg_gemm_phase_ms_tail", "")
                    if first_full_after_low
                    else ""
                ),
                "first_full_clock_mhz": (
                    first_full_after_low.get(
                        "avg_gemm_phase_clock_mhz_tail", ""
                    )
                    if first_full_after_low
                    else ""
                ),
                "first_full_tflops_s": (
                    first_full_after_low.get("gemm_tflops_s_tail", "")
                    if first_full_after_low
                    else ""
                ),
                "transition_gap_ms": (
                    float(first_full_after_low["avg_gemm_phase_ms_tail"])
                    - float(last_low["avg_gemm_phase_ms_tail"])
                    if last_low and first_full_after_low
                    else ""
                ),
            }
        )
    return summaries


def _plot_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for row in ok_rows:
        by_shape.setdefault(str(row["shape_name"]), []).append(row)

    fig, axis = plt.subplots(figsize=(10, 6))
    for shape_name, shape_rows in sorted(by_shape.items()):
        shape_rows.sort(key=lambda row: float(row["avg_gemm_phase_ms_tail"]))
        axis.plot(
            [float(row["avg_gemm_phase_ms_tail"]) for row in shape_rows],
            [float(row["avg_gemm_phase_clock_mhz_tail"]) for row in shape_rows],
            marker="o",
            linewidth=2.0,
            label=shape_name,
        )
    axis.axvline(600.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    axis.set_xlabel("GEMM Phase Time (ms)", fontsize=14, fontweight="bold")
    axis.set_ylabel("GEMM Phase Clock (MHz)", fontsize=14, fontweight="bold")
    axis.set_title(
        "n1 Shape Sweep: GEMM Phase Clock", fontsize=16, fontweight="bold"
    )
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "clock_vs_gemm_phase_ms.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    for shape_name, shape_rows in sorted(by_shape.items()):
        shape_rows.sort(key=lambda row: float(row["avg_gemm_phase_ms_tail"]))
        axis.plot(
            [float(row["avg_gemm_phase_ms_tail"]) for row in shape_rows],
            [float(row["gemm_tflops_s_tail"]) for row in shape_rows],
            marker="o",
            linewidth=2.0,
            label=shape_name,
        )
    axis.axvline(600.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    axis.set_xlabel("GEMM Phase Time (ms)", fontsize=14, fontweight="bold")
    axis.set_ylabel("GEMM TFLOPs/s", fontsize=14, fontweight="bold")
    axis.set_title(
        "n1 Shape Sweep: GEMM Throughput", fontsize=16, fontweight="bold"
    )
    axis.tick_params(axis="both", labelsize=12)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "tflops_vs_gemm_phase_ms.png", dpi=180)
    plt.close(fig)

    if transition_rows:
        fig, axis = plt.subplots(figsize=(10, 5))
        labels = [str(row["shape_name"]) for row in transition_rows]
        low_ms = [
            float(row["last_low_gemm_phase_ms"])
            if row["last_low_gemm_phase_ms"] != ""
            else math.nan
            for row in transition_rows
        ]
        full_ms = [
            float(row["first_full_gemm_phase_ms"])
            if row["first_full_gemm_phase_ms"] != ""
            else math.nan
            for row in transition_rows
        ]
        x_values = list(range(len(labels)))
        axis.scatter(x_values, low_ms, label="last low-clock point", s=70)
        axis.scatter(x_values, full_ms, label="first full-clock point", s=70)
        axis.axhline(
            600.0, color="black", linestyle="--", linewidth=1.2, alpha=0.6
        )
        axis.set_xticks(x_values)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel("GEMM Phase Time (ms)", fontsize=14, fontweight="bold")
        axis.set_title(
            "n1 Shape Sweep: Transition Bracket", fontsize=16, fontweight="bold"
        )
        axis.tick_params(axis="both", labelsize=11)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(plots_dir / "transition_bracket_ms.png", dpi=180)
        plt.close(fig)


def _write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Two-GEMM n1 Shape Sweep",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        "- Norm steps per phase: `1`",
        "- Each case runs at least 30s active CUDA time by default.",
        "- Metrics below use the final steady analysis window.",
        "",
        "## Transition Bracket",
        "",
        "| Shape | M | N | K | Last Low Phase (ms) | Last Low Clock (MHz) | First Full Phase (ms) | First Full Clock (MHz) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in transition_rows:
        lines.append(
            f"| {row['shape_name']} | {row['m']} | {row['n']} | {row['k']} | "
            f"{_fmt(row['last_low_gemm_phase_ms'])} | "
            f"{_fmt(row['last_low_clock_mhz'])} | "
            f"{_fmt(row['first_full_gemm_phase_ms'])} | "
            f"{_fmt(row['first_full_clock_mhz'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Summary CSV: `summary.csv`",
            "- Transition summary: `transition_summary.csv`",
            "- Clock plot: `plots/clock_vs_gemm_phase_ms.png`",
            "- Throughput plot: `plots/tflops_vs_gemm_phase_ms.png`",
            "- Transition bracket plot: `plots/transition_bracket_ms.png`",
            "",
        ]
    )
    output_dir.joinpath("README.md").write_text("\n".join(lines))


def _fmt(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.2f}"


def run(args: argparse.Namespace) -> None:
    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(getattr(torch, args.dtype))

    shapes = args.shape or _default_shapes()
    output_dir = args.output_dir or (DEFAULT_RESULTS_ROOT / utc_stamp())
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "metadata.json",
        {
            "run_started_at_utc": utc_now_iso(),
            "output_dir": str(output_dir),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "shapes": [shape.__dict__ for shape in shapes],
            "target_gemm_phase_ms": args.target_gemm_phase_ms,
            "norm_steps": 1,
            "target_active_s": args.target_active_s,
            "analysis_window_s": args.analysis_window_s,
            "full_clock_threshold_mhz": args.full_clock_threshold_mhz,
            "monitor_interval": args.monitor_interval,
            "monitor_gpu_index": args.monitor_gpu_index,
        },
    )

    all_rows: list[dict[str, Any]] = []
    for shape in shapes:
        print(
            f"Probing {shape.name}: M={shape.m} N={shape.n} K={shape.k}",
            flush=True,
        )
        unit_ms = _measure_gemm_unit_ms(
            torch=torch,
            flashinfer=flashinfer,
            shape=shape,
            dtype=args.dtype,
            warmup=args.warmup,
            probe_repeat=args.probe_repeat,
            eps=args.eps,
        )
        gemm_steps = _choose_gemm_steps(args.target_gemm_phase_ms, unit_ms)
        print(
            f"{shape.name}: unit={unit_ms:.3f} ms, steps={gemm_steps}",
            flush=True,
        )
        shape_dir = output_dir / shape.name
        shape_dir.mkdir(parents=True, exist_ok=True)
        shape_args = _shape_args(args, shape, gemm_steps, shape_dir)
        shape_rows: list[dict[str, Any]] = []
        for steps in gemm_steps:
            target_ms = _target_for_steps(
                steps,
                args.target_gemm_phase_ms,
                unit_ms,
            )
            row = _run_case(
                args=shape_args,
                torch=torch,
                flashinfer=flashinfer,
                output_dir=shape_dir,
                gemm_steps=steps,
                norm_steps=1,
            )
            row.update(
                {
                    "shape_name": shape.name,
                    "target_gemm_phase_ms": target_ms,
                    "estimated_gemm_unit_ms": unit_ms,
                    "requested_gemm_steps": steps,
                }
            )
            shape_rows.append(row)
            all_rows.append(row)
            write_csv(output_dir / "summary.csv", COMBINED_FIELDNAMES, all_rows)
        write_csv(shape_dir / "summary.csv", COMBINED_FIELDNAMES, shape_rows)
        _write_shape_report(shape_dir, shape, shape_rows)

    transition_rows = _transition_summary(
        all_rows,
        full_clock_threshold_mhz=args.full_clock_threshold_mhz,
    )
    write_csv(
        output_dir / "transition_summary.csv",
        TRANSITION_FIELDNAMES,
        transition_rows,
    )
    _plot_summary(output_dir, all_rows, transition_rows)
    _write_report(output_dir, all_rows, transition_rows)
    print(f"Wrote {output_dir / 'README.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run norm_steps=1 steady-window two-GEMM sweeps across GEMM shapes."
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="Shape as name:M:N:K. May be repeated. Defaults to Llama-like shapes.",
    )
    parser.add_argument(
        "--target-gemm-phase-ms",
        nargs="+",
        type=float,
        default=list(DEFAULT_TARGET_PHASE_MS),
    )
    parser.add_argument(
        "--target-active-s", type=float, default=DEFAULT_TARGET_ACTIVE_S
    )
    parser.add_argument(
        "--analysis-window-s",
        type=float,
        default=DEFAULT_ANALYSIS_WINDOW_S,
    )
    parser.add_argument("--cycle-margin", type=float, default=1.08)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--probe-repeat", type=int, default=DEFAULT_PROBE_REPEAT
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=DEFAULT_MONITOR_INTERVAL,
    )
    parser.add_argument(
        "--monitor-gpu-index",
        type=int,
        default=DEFAULT_MONITOR_GPU_INDEX,
    )
    parser.add_argument(
        "--full-clock-threshold-mhz",
        type=float,
        default=1400.0,
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--cooldown-s", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
