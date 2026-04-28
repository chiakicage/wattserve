#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    calibrate_repeat,
    make_scaled_tensor,
    monitor_summary,
    utc_now_iso,
    utc_stamp,
    write_csv,
    write_json,
    write_kernel_profile_csv,
    write_timeline_plot,
)


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "state_chain_block_sweep"
DEFAULT_MODELS = ["7B", "13B", "34B", "70B"]
DEFAULT_BATCH_SIZES = [2**power for power in range(5, 16)]
VARIANTS = ("without_norm", "with_norm")
DEFAULT_DTYPE = "bfloat16"
DEFAULT_WARMUP = 20
DEFAULT_PROBE_REPEAT = 10
DEFAULT_TARGET_TIMED_SECONDS = 3.0
DEFAULT_MAX_REPEAT = 50000
DEFAULT_MONITOR_INTERVAL = 0.01
DEFAULT_MONITOR_GPU_INDEX = 3
DEFAULT_EPS = 1e-6

SUMMARY_FIELDNAMES = [
    "status",
    "error",
    "run_timestamp_utc",
    "model",
    "batch_size",
    "variant",
    "dtype",
    "steps_per_run",
    "repeat",
    "warmup",
    "probe_repeat",
    "target_timed_seconds",
    "total_time_s",
    "iter_time_ms",
    "probe_iter_ms",
    "gemm_total_ms",
    "gemm_iter_time_ms",
    "gemm_tflops_s_raw",
    "effective_tflops_s",
    "gemm_flops_per_step",
    "attention_flops_per_step",
    "effective_flops_per_step",
    "avg_power_watts",
    "max_power_watts",
    "avg_gpu_clock_mhz",
    "max_gpu_clock_mhz",
    "monitor_sample_count",
    "monitor_csv",
    "timeline_plot",
    "kernel_profile_csv",
    "kernel_profile_error",
    "hidden_size",
    "intermediate_size",
    "num_layers",
    "canonical_num_layers",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "q_size",
    "kv_size",
    "rope_theta",
    "eps",
    "description",
]


@dataclass(frozen=True)
class BlockShape:
    model: str
    batch_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    canonical_num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_size: int
    kv_size: int
    rope_theta: float


@dataclass(frozen=True)
class BlockCase:
    shape: BlockShape
    variant: str
    steps_per_run: int
    run_once: Callable[[], None]
    measure_gemm_time_ms: Callable[[int], float]
    gemm_flops_per_step: float
    attention_flops_per_step: float
    description: str

    @property
    def effective_flops_per_step(self) -> float:
        return self.gemm_flops_per_step + self.attention_flops_per_step


def _load_shape(model: str, batch_size: int) -> BlockShape:
    from models import llama_config

    loaders = {
        "7B": llama_config.get_llama_config_7B,
        "13B": llama_config.get_llama_config_13B,
        "34B": llama_config.get_llama_config_34B,
        "70B": llama_config.get_llama_config_70B,
    }
    config = loaders[model]()
    hidden_size = int(config.hidden_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
    return BlockShape(
        model=model,
        batch_size=batch_size,
        hidden_size=hidden_size,
        intermediate_size=int(config.intermediate_size),
        num_layers=int(config.num_hidden_layers),
        canonical_num_layers=int(
            getattr(
                config, "canonical_num_hidden_layers", config.num_hidden_layers
            )
        ),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_size=num_heads * head_dim,
        kv_size=num_kv_heads * head_dim,
        rope_theta=float(getattr(config, "rope_theta", 10000.0)),
    )


def _gemm_flops(m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n


def _attention_flops(shape: BlockShape) -> float:
    causal_pairs = shape.batch_size * (shape.batch_size + 1) / 2.0
    return 4.0 * causal_pairs * shape.q_size


def _block_gemm_flops(shape: BlockShape) -> float:
    return (
        _gemm_flops(shape.batch_size, shape.hidden_size, shape.q_size)
        + _gemm_flops(shape.batch_size, shape.hidden_size, shape.kv_size)
        + _gemm_flops(shape.batch_size, shape.hidden_size, shape.kv_size)
        + _gemm_flops(shape.batch_size, shape.q_size, shape.hidden_size)
        + _gemm_flops(
            shape.batch_size, shape.hidden_size, shape.intermediate_size
        )
        + _gemm_flops(
            shape.batch_size, shape.hidden_size, shape.intermediate_size
        )
        + _gemm_flops(
            shape.batch_size, shape.intermediate_size, shape.hidden_size
        )
    )


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


def _build_case(
    torch: Any,
    flashinfer: Any,
    shape: BlockShape,
    variant: str,
    steps_per_run: int,
    device: str,
    dtype: Any,
    eps: float,
) -> BlockCase:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    with_norm = variant == "with_norm"

    current = make_scaled_tensor(
        torch, (shape.batch_size, shape.hidden_size), device, dtype
    )
    next_hidden = torch.empty_like(current)
    residual_1 = make_scaled_tensor(
        torch, (shape.batch_size, shape.hidden_size), device, dtype
    )
    residual_2 = make_scaled_tensor(
        torch, (shape.batch_size, shape.hidden_size), device, dtype
    )
    norm_weight_1 = torch.ones((shape.hidden_size,), device=device, dtype=dtype)
    norm_weight_2 = torch.ones_like(norm_weight_1)

    q_weight = make_scaled_tensor(
        torch, (shape.hidden_size, shape.q_size), device, dtype
    )
    k_weight = make_scaled_tensor(
        torch, (shape.hidden_size, shape.kv_size), device, dtype
    )
    v_weight = make_scaled_tensor(
        torch, (shape.hidden_size, shape.kv_size), device, dtype
    )
    o_weight = make_scaled_tensor(
        torch, (shape.q_size, shape.hidden_size), device, dtype
    )
    gate_weight = make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )
    up_weight = make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )
    down_weight = make_scaled_tensor(
        torch, (shape.intermediate_size, shape.hidden_size), device, dtype
    )

    q_out = torch.empty(
        (shape.batch_size, shape.q_size), device=device, dtype=dtype
    )
    k_out = torch.empty(
        (shape.batch_size, shape.kv_size), device=device, dtype=dtype
    )
    v_out = torch.empty_like(k_out)
    o_out = torch.empty_like(current)
    gate_out = torch.empty(
        (shape.batch_size, shape.intermediate_size), device=device, dtype=dtype
    )
    up_out = torch.empty_like(gate_out)
    cat_out = torch.empty(
        (shape.batch_size, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    act_out = torch.empty_like(gate_out)

    positions = torch.arange(shape.batch_size, device=device)
    q_view = q_out.view(shape.batch_size, shape.num_heads, shape.head_dim)
    k_view = k_out.view(shape.batch_size, shape.num_kv_heads, shape.head_dim)
    v_view = v_out.view(shape.batch_size, shape.num_kv_heads, shape.head_dim)

    def mm(
        events: list[tuple[Any, Any]] | None, lhs: Any, rhs: Any, out: Any
    ) -> None:
        if events is None:
            torch.mm(lhs, rhs, out=out)
        else:
            _recorded_mm(torch, events, lhs, rhs, out)

    def run_step(events: list[tuple[Any, Any]] | None = None) -> None:
        nonlocal current, next_hidden

        mm(events, current, q_weight, q_out)
        mm(events, current, k_weight, k_out)
        mm(events, current, v_weight, v_out)
        q_rope, k_rope = flashinfer.apply_rope_pos_ids(
            q_view,
            k_view,
            positions,
            rotary_dim=shape.head_dim,
            rope_theta=shape.rope_theta,
        )
        attn_out = flashinfer.single_prefill_with_kv_cache(
            q_rope,
            k_rope,
            v_view,
            causal=True,
            sm_scale=shape.head_dim**-0.5,
        )
        mm(
            events,
            attn_out.reshape(shape.batch_size, shape.q_size),
            o_weight,
            o_out,
        )
        if with_norm:
            flashinfer.fused_add_rmsnorm(
                o_out, residual_1, norm_weight_1, eps=eps
            )

        mm(events, o_out, gate_weight, gate_out)
        mm(events, o_out, up_weight, up_out)
        cat_out[:, : shape.intermediate_size].copy_(gate_out)
        cat_out[:, shape.intermediate_size :].copy_(up_out)
        flashinfer.silu_and_mul(cat_out, out=act_out)
        mm(events, act_out, down_weight, next_hidden)
        if with_norm:
            flashinfer.fused_add_rmsnorm(
                next_hidden, residual_2, norm_weight_2, eps=eps
            )
        current, next_hidden = next_hidden, current

    def run_once() -> None:
        for _ in range(steps_per_run):
            run_step()

    def measure_gemm_time_ms(repeat: int) -> float:
        events: list[tuple[Any, Any]] = []
        torch.cuda.synchronize()
        for _ in range(repeat):
            for _ in range(steps_per_run):
                run_step(events)
        torch.cuda.synchronize()
        return sum(start.elapsed_time(end) for start, end in events)

    description = (
        "state-chain full block: q/k/v -> RoPE -> causal attention -> o -> "
        "optional fused_add_norm_1 -> gate/up -> silu_and_mul -> down -> "
        "optional fused_add_norm_2"
    )
    return BlockCase(
        shape=shape,
        variant=variant,
        steps_per_run=steps_per_run,
        run_once=run_once,
        measure_gemm_time_ms=measure_gemm_time_ms,
        gemm_flops_per_step=_block_gemm_flops(shape),
        attention_flops_per_step=_attention_flops(shape),
        description=description,
    )


def _format_float(value: Any, precision: int = 2) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{precision}f}"


def _base_row(
    args: argparse.Namespace,
    shape: BlockShape,
    variant: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "error": "",
        "run_timestamp_utc": utc_now_iso(),
        "model": shape.model,
        "batch_size": shape.batch_size,
        "variant": variant,
        "dtype": args.dtype,
        "steps_per_run": args.steps_per_run,
        "warmup": args.warmup,
        "probe_repeat": args.probe_repeat,
        "target_timed_seconds": args.target_timed_seconds,
        "hidden_size": shape.hidden_size,
        "intermediate_size": shape.intermediate_size,
        "num_layers": shape.num_layers,
        "canonical_num_layers": shape.canonical_num_layers,
        "num_heads": shape.num_heads,
        "num_kv_heads": shape.num_kv_heads,
        "head_dim": shape.head_dim,
        "q_size": shape.q_size,
        "kv_size": shape.kv_size,
        "rope_theta": shape.rope_theta,
        "eps": args.eps,
    }


def _run_case(
    args: argparse.Namespace,
    torch: Any,
    flashinfer: Any,
    output_dir: Path,
    model: str,
    batch_size: int,
    variant: str,
) -> dict[str, Any]:
    shape = _load_shape(model, batch_size)
    row = _base_row(args, shape, variant)
    monitor_dir = output_dir / "monitor"
    profile_dir = output_dir / "kernel_profile"
    label = f"{model}/batch_size={batch_size}/{variant}"
    print(f"Running {label}", flush=True)

    try:
        case = _build_case(
            torch=torch,
            flashinfer=flashinfer,
            shape=shape,
            variant=variant,
            steps_per_run=args.steps_per_run,
            device="cuda:0",
            dtype=getattr(torch, args.dtype),
            eps=args.eps,
        )
        with torch.inference_mode():
            for _ in range(args.warmup):
                case.run_once()
            torch.cuda.synchronize()
            repeat, probe_iter_seconds = calibrate_repeat(
                torch=torch,
                run_once=case.run_once,
                target_timed_seconds=args.target_timed_seconds,
                probe_repeat=args.probe_repeat,
                max_repeat=args.max_repeat,
            )

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

            gemm_total_ms = case.measure_gemm_time_ms(repeat)

            kernel_profile_csv = ""
            kernel_profile_error = ""
            if not args.skip_kernel_profile:
                profile_path = (
                    profile_dir
                    / f"{model}__batch_size={batch_size}__{variant}.csv"
                )
                try:
                    write_kernel_profile_csv(
                        torch=torch,
                        run_once=case.run_once,
                        output_path=profile_path,
                        repeat=args.profile_repeat,
                    )
                    kernel_profile_csv = str(profile_path)
                except (
                    Exception
                ) as exc:  # profiler should not hide primary data
                    kernel_profile_error = f"{type(exc).__name__}: {exc}"

        records = monitor.get_results()
        monitor_csv = ""
        if records:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            monitor_path = (
                monitor_dir / f"{model}__batch_size={batch_size}__{variant}.csv"
            )
            monitor.export_csv(str(monitor_path))
            monitor_csv = str(monitor_path)

        iter_time_s = elapsed / repeat
        gemm_time_s = gemm_total_ms / 1000.0
        gemm_flops_per_iter = case.gemm_flops_per_step * case.steps_per_run
        effective_flops_per_iter = (
            case.effective_flops_per_step * case.steps_per_run
        )
        row.update(
            {
                "status": "ok",
                "repeat": repeat,
                "total_time_s": elapsed,
                "iter_time_ms": iter_time_s * 1000.0,
                "probe_iter_ms": probe_iter_seconds * 1000.0,
                "gemm_total_ms": gemm_total_ms,
                "gemm_iter_time_ms": gemm_total_ms / repeat,
                "gemm_tflops_s_raw": (
                    gemm_flops_per_iter * repeat / 1e12 / gemm_time_s
                    if gemm_time_s > 0
                    else 0.0
                ),
                "effective_tflops_s": (
                    effective_flops_per_iter * repeat / 1e12 / elapsed
                    if elapsed > 0
                    else 0.0
                ),
                "gemm_flops_per_step": case.gemm_flops_per_step,
                "attention_flops_per_step": case.attention_flops_per_step,
                "effective_flops_per_step": case.effective_flops_per_step,
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
    rows_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or not row.get("monitor_csv"):
            continue
        key = (str(row["model"]), int(row["batch_size"]))
        rows_by_key.setdefault(key, []).append(row)

    for (model, batch_size), group in rows_by_key.items():
        ordered = sorted(group, key=lambda row: str(row["variant"]))
        output_path = plots_dir / f"{model}__batch_size={batch_size}.png"
        plot_path = write_timeline_plot(
            [
                (str(row["variant"]), Path(str(row["monitor_csv"])))
                for row in ordered
            ],
            output_path=output_path,
            title=f"{model} batch_size={batch_size}: power and clock",
        )
        if plot_path is not None:
            for row in group:
                row["timeline_plot"] = str(plot_path)


def _build_report(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    lines = [
        "# State-Chain Full-Block Sweep",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        f"- Output directory: `{metadata['output_dir']}`",
        f"- CUDA_VISIBLE_DEVICES: `{metadata.get('cuda_visible_devices')}`",
        f"- Monitor GPU index: `{metadata['monitor_gpu_index']}`",
        f"- Dtype: `{metadata['dtype']}`",
        "- Batch size means token rows / causal sequence length `S`.",
        "- Attention FLOPs use `4 * q_size * S * (S + 1) / 2`.",
        "- Raw GEMM TFLOPs/s uses CUDA-event time for q/k/v/o/gate/up/down only.",
        "- Effective TFLOPs/s uses GEMM plus causal-attention equivalent FLOPs over end-to-end time.",
        "",
        "## Results",
        "",
        "| Model | Batch Size | Variant | Raw GEMM TFLOPs/s | Effective TFLOPs/s | Iter Time (ms) | Avg Power (W) | Avg GPU Clock (MHz) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ok_rows:
        lines.append(
            f"| {row['model']} | {row['batch_size']} | {row['variant']} | "
            f"{_format_float(row['gemm_tflops_s_raw'])} | "
            f"{_format_float(row['effective_tflops_s'])} | "
            f"{_format_float(row['iter_time_ms'], 3)} | "
            f"{_format_float(row['avg_power_watts'])} | "
            f"{_format_float(row['avg_gpu_clock_mhz'])} |"
        )

    by_key = {
        (row["model"], int(row["batch_size"]), row["variant"]): row
        for row in ok_rows
    }
    lines.extend(
        [
            "",
            "## Norm Delta",
            "",
            "| Model | Batch Size | w/o Norm Raw GEMM | w/ Norm Raw GEMM | Raw Delta (%) | w/o Effective | w/ Effective | Effective Delta (%) | Power Delta (W) | Clock Delta (MHz) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in metadata["models"]:
        for batch_size in metadata["batch_sizes"]:
            without = by_key.get((model, int(batch_size), "without_norm"))
            with_norm = by_key.get((model, int(batch_size), "with_norm"))
            if without is None or with_norm is None:
                continue
            raw_delta = (
                (with_norm["gemm_tflops_s_raw"] - without["gemm_tflops_s_raw"])
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
                f"| {model} | {batch_size} | "
                f"{_format_float(without['gemm_tflops_s_raw'])} | "
                f"{_format_float(with_norm['gemm_tflops_s_raw'])} | "
                f"{_format_float(raw_delta)} | "
                f"{_format_float(without['effective_tflops_s'])} | "
                f"{_format_float(with_norm['effective_tflops_s'])} | "
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
                "| Model | Batch Size | Variant | Error |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for row in failed_rows:
            lines.append(
                f"| {row.get('model')} | {row.get('batch_size')} | "
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


def output_dir_display(path: str) -> str:
    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path


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
        "models": args.models,
        "batch_sizes": args.batch_sizes,
        "variants": args.variants,
        "steps_per_run": args.steps_per_run,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "probe_repeat": args.probe_repeat,
        "target_timed_seconds": args.target_timed_seconds,
        "max_repeat": args.max_repeat,
        "monitor_interval": args.monitor_interval,
        "monitor_gpu_index": args.monitor_gpu_index,
        "profile_repeat": args.profile_repeat,
        "skip_kernel_profile": args.skip_kernel_profile,
        "eps": args.eps,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attention_flops": "4 * q_size * S * (S + 1) / 2",
    }
    write_json(metadata_path, metadata)

    rows: list[dict[str, Any]] = []
    for model in args.models:
        for batch_size in args.batch_sizes:
            for variant in args.variants:
                row = _run_case(
                    args=args,
                    torch=torch,
                    flashinfer=flashinfer,
                    output_dir=output_dir,
                    model=model,
                    batch_size=batch_size,
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
        description="State-chain full Llama-block norm power sweep."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
        help="Token-row / causal sequence lengths S.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--steps-per-run", type=int, default=1)
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
