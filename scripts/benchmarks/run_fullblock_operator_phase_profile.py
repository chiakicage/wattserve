#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from monitor.gpu_monitor import GPUMonitor  # noqa: E402
from models import llama_config  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "fullblock_operator_phase_profile"
DEFAULT_MODEL = "70B"
DEFAULT_BATCH_SIZE = 32768
DEFAULT_DTYPE = "bfloat16"
DEFAULT_TARGET_SECONDS = 10.0
DEFAULT_WARMUP_SECONDS = 2.0
DEFAULT_PROBE_REPEAT = 3
DEFAULT_MONITOR_INTERVAL = 0.01
DEFAULT_MONITOR_GPU_INDEX = 3
DEFAULT_EPS = 1e-6

SUMMARY_FIELDS = [
    "run_timestamp_utc",
    "model",
    "batch_size",
    "dtype",
    "op",
    "category",
    "throughput_value",
    "throughput_unit",
    "iter_time_ms",
    "gpu_time_s",
    "wall_time_s",
    "repeat",
    "target_seconds",
    "warmup_seconds",
    "probe_iter_ms",
    "work_per_iter",
    "work_unit",
    "avg_power_watts",
    "max_power_watts",
    "avg_gpu_clock_mhz",
    "max_gpu_clock_mhz",
    "first_2s_power_watts",
    "first_2s_gpu_clock_mhz",
    "last_2s_power_watts",
    "last_2s_gpu_clock_mhz",
    "monitor_sample_count",
    "monitor_csv",
    "description",
]


@dataclass(frozen=True)
class BlockShape:
    model: str
    batch_size: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_size: int
    kv_size: int
    rope_theta: float


@dataclass(frozen=True)
class OperatorPhase:
    name: str
    category: str
    run_once: Callable[[], None]
    work_per_iter: float
    work_unit: str
    throughput_unit: str
    description: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in SUMMARY_FIELDS}
            )


def load_shape(model: str, batch_size: int) -> BlockShape:
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
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_size=num_heads * head_dim,
        kv_size=num_kv_heads * head_dim,
        rope_theta=float(getattr(config, "rope_theta", 10000.0)),
    )


def resolve_dtype(torch: Any, dtype_name: str) -> Any:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def make_scaled_tensor(
    torch: Any,
    tensor_shape: tuple[int, ...],
    device: str,
    dtype: Any,
    scale: float = 0.02,
) -> Any:
    tensor = torch.randn(tensor_shape, device=device, dtype=dtype)
    tensor.mul_(scale)
    return tensor


def gemm_flops(m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n


def attention_flops(shape: BlockShape) -> float:
    causal_pairs = shape.batch_size * (shape.batch_size + 1) / 2.0
    return 4.0 * causal_pairs * shape.q_size


def tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def monitor_average(
    records: list[dict[str, Any]],
    key: str,
    start_s: float | None = None,
    end_s: float | None = None,
) -> float:
    selected = []
    for record in records:
        elapsed = float(record["elapsed_seconds"])
        if start_s is not None and elapsed < start_s:
            continue
        if end_s is not None and elapsed > end_s:
            continue
        selected.append(float(record[key]))
    if not selected:
        return 0.0
    return sum(selected) / len(selected)


def monitor_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {
            "avg_power_watts": 0.0,
            "max_power_watts": 0.0,
            "avg_gpu_clock_mhz": 0.0,
            "max_gpu_clock_mhz": 0.0,
            "first_2s_power_watts": 0.0,
            "first_2s_gpu_clock_mhz": 0.0,
            "last_2s_power_watts": 0.0,
            "last_2s_gpu_clock_mhz": 0.0,
            "monitor_sample_count": 0,
        }
    last_elapsed = max(float(record["elapsed_seconds"]) for record in records)
    last_start = max(0.0, last_elapsed - 2.0)
    return {
        "avg_power_watts": monitor_average(records, "power_watts"),
        "max_power_watts": max(
            float(record["power_watts"]) for record in records
        ),
        "avg_gpu_clock_mhz": monitor_average(records, "gpu_clock_mhz"),
        "max_gpu_clock_mhz": max(
            float(record["gpu_clock_mhz"]) for record in records
        ),
        "first_2s_power_watts": monitor_average(
            records, "power_watts", 0.0, 2.0
        ),
        "first_2s_gpu_clock_mhz": monitor_average(
            records, "gpu_clock_mhz", 0.0, 2.0
        ),
        "last_2s_power_watts": monitor_average(
            records, "power_watts", last_start, None
        ),
        "last_2s_gpu_clock_mhz": monitor_average(
            records, "gpu_clock_mhz", last_start, None
        ),
        "monitor_sample_count": len(records),
    }


def calibrate_repeat(
    torch: Any,
    run_once: Callable[[], None],
    target_seconds: float,
    probe_repeat: int,
) -> tuple[int, float]:
    probe_repeat = max(1, probe_repeat)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(probe_repeat):
        run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    iter_seconds = elapsed / probe_repeat if elapsed > 0 else 0.0
    repeat = (
        max(1, int(math.ceil(target_seconds / iter_seconds)))
        if iter_seconds
        else 1
    )
    return repeat, iter_seconds


def run_for_repeats(
    torch: Any, run_once: Callable[[], None], repeat: int
) -> tuple[float, float]:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    for _ in range(repeat):
        run_once()
    end_event.record()
    torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - wall_start
    gpu_elapsed_s = start_event.elapsed_time(end_event) / 1000.0
    return gpu_elapsed_s, wall_elapsed


def run_for_min_gpu_seconds(
    torch: Any,
    run_once: Callable[[], None],
    initial_repeat: int,
    min_gpu_seconds: float,
) -> tuple[float, float, int]:
    total_gpu_time_s = 0.0
    total_wall_time_s = 0.0
    total_repeat = 0
    next_repeat = max(1, initial_repeat)
    while total_gpu_time_s < min_gpu_seconds:
        gpu_time_s, wall_time_s = run_for_repeats(torch, run_once, next_repeat)
        total_gpu_time_s += gpu_time_s
        total_wall_time_s += wall_time_s
        total_repeat += next_repeat
        if total_gpu_time_s <= 0.0:
            next_repeat = max(1, next_repeat)
            continue
        remaining_s = min_gpu_seconds - total_gpu_time_s
        if remaining_s <= 0.0:
            break
        iter_time_s = total_gpu_time_s / total_repeat
        next_repeat = max(1, int(math.ceil(remaining_s / iter_time_s)))
    return total_gpu_time_s, total_wall_time_s, total_repeat


def build_phases(
    torch: Any,
    flashinfer: Any,
    shape: BlockShape,
    dtype: Any,
    device: str,
    eps: float,
) -> list[OperatorPhase]:
    hidden = make_scaled_tensor(
        torch, (shape.batch_size, shape.hidden_size), device, dtype
    )
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
    o_out = torch.empty_like(hidden)
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
    down_out = torch.empty_like(hidden)
    positions = torch.arange(shape.batch_size, device=device)

    q_view = q_out.view(shape.batch_size, shape.num_heads, shape.head_dim)
    k_view = k_out.view(shape.batch_size, shape.num_kv_heads, shape.head_dim)
    v_view = v_out.view(shape.batch_size, shape.num_kv_heads, shape.head_dim)
    state: dict[str, Any] = {}

    def run_q_gemm() -> None:
        torch.mm(hidden, q_weight, out=q_out)

    def run_k_gemm() -> None:
        torch.mm(hidden, k_weight, out=k_out)

    def run_v_gemm() -> None:
        torch.mm(hidden, v_weight, out=v_out)

    def run_rope() -> None:
        state["q_rope"], state["k_rope"] = flashinfer.apply_rope_pos_ids(
            q_view,
            k_view,
            positions,
            rotary_dim=shape.head_dim,
            rope_theta=shape.rope_theta,
        )

    def run_attention() -> None:
        state["attn_out"] = flashinfer.single_prefill_with_kv_cache(
            state["q_rope"],
            state["k_rope"],
            v_view,
            causal=True,
            sm_scale=shape.head_dim**-0.5,
        )

    def run_o_gemm() -> None:
        torch.mm(
            state["attn_out"].reshape(shape.batch_size, shape.q_size),
            o_weight,
            out=o_out,
        )

    def run_post_attn_norm() -> None:
        flashinfer.fused_add_rmsnorm(o_out, residual_1, norm_weight_1, eps=eps)

    def run_gate_gemm() -> None:
        torch.mm(o_out, gate_weight, out=gate_out)

    def run_up_gemm() -> None:
        torch.mm(o_out, up_weight, out=up_out)

    def run_gate_copy() -> None:
        cat_out[:, : shape.intermediate_size].copy_(gate_out)

    def run_up_copy() -> None:
        cat_out[:, shape.intermediate_size :].copy_(up_out)

    def run_silu_and_mul() -> None:
        flashinfer.silu_and_mul(cat_out, out=act_out)

    def run_down_gemm() -> None:
        torch.mm(act_out, down_weight, out=down_out)

    def run_post_ffn_norm() -> None:
        flashinfer.fused_add_rmsnorm(
            down_out, residual_2, norm_weight_2, eps=eps
        )

    # Populate dependent tensors once before isolated phase profiling.
    with torch.inference_mode():
        run_q_gemm()
        run_k_gemm()
        run_v_gemm()
        run_rope()
        run_attention()
        run_o_gemm()
        run_post_attn_norm()
        run_gate_gemm()
        run_up_gemm()
        run_gate_copy()
        run_up_copy()
        run_silu_and_mul()
        run_down_gemm()
        run_post_ffn_norm()
        torch.cuda.synchronize()

    element_size = int(hidden.element_size())
    rope_bytes = 2 * (
        q_out.numel() + k_out.numel()
    ) * element_size + tensor_bytes(positions)
    norm_hidden_bytes = (
        4 * shape.batch_size * shape.hidden_size * element_size
        + shape.hidden_size * element_size
    )
    activation_bytes = (
        (2 * shape.batch_size * shape.intermediate_size)
        + (shape.batch_size * shape.intermediate_size)
    ) * element_size
    copy_bytes = 2 * shape.batch_size * shape.intermediate_size * element_size

    return [
        OperatorPhase(
            "q_gemm",
            "compute",
            run_q_gemm,
            gemm_flops(shape.batch_size, shape.hidden_size, shape.q_size),
            "FLOPs",
            "TFLOPs/s",
            "Q projection GEMM",
        ),
        OperatorPhase(
            "k_gemm",
            "compute",
            run_k_gemm,
            gemm_flops(shape.batch_size, shape.hidden_size, shape.kv_size),
            "FLOPs",
            "TFLOPs/s",
            "K projection GEMM",
        ),
        OperatorPhase(
            "v_gemm",
            "compute",
            run_v_gemm,
            gemm_flops(shape.batch_size, shape.hidden_size, shape.kv_size),
            "FLOPs",
            "TFLOPs/s",
            "V projection GEMM",
        ),
        OperatorPhase(
            "rope",
            "memory",
            run_rope,
            float(rope_bytes),
            "bytes",
            "GB/s",
            "FlashInfer RoPE over Q and K",
        ),
        OperatorPhase(
            "causal_attention",
            "compute",
            run_attention,
            attention_flops(shape),
            "FLOPs",
            "TFLOPs/s",
            "FlashInfer single_prefill_with_kv_cache causal attention",
        ),
        OperatorPhase(
            "o_gemm",
            "compute",
            run_o_gemm,
            gemm_flops(shape.batch_size, shape.q_size, shape.hidden_size),
            "FLOPs",
            "TFLOPs/s",
            "Attention output projection GEMM",
        ),
        OperatorPhase(
            "post_attn_fused_add_rmsnorm",
            "memory",
            run_post_attn_norm,
            float(norm_hidden_bytes),
            "bytes",
            "GB/s",
            "Fused residual add plus RMSNorm after attention",
        ),
        OperatorPhase(
            "gate_gemm",
            "compute",
            run_gate_gemm,
            gemm_flops(
                shape.batch_size, shape.hidden_size, shape.intermediate_size
            ),
            "FLOPs",
            "TFLOPs/s",
            "MLP gate projection GEMM",
        ),
        OperatorPhase(
            "up_gemm",
            "compute",
            run_up_gemm,
            gemm_flops(
                shape.batch_size, shape.hidden_size, shape.intermediate_size
            ),
            "FLOPs",
            "TFLOPs/s",
            "MLP up projection GEMM",
        ),
        OperatorPhase(
            "gate_copy_to_cat",
            "memory",
            run_gate_copy,
            float(copy_bytes),
            "bytes",
            "GB/s",
            "Copy gate projection output into silu_and_mul input buffer",
        ),
        OperatorPhase(
            "up_copy_to_cat",
            "memory",
            run_up_copy,
            float(copy_bytes),
            "bytes",
            "GB/s",
            "Copy up projection output into silu_and_mul input buffer",
        ),
        OperatorPhase(
            "silu_and_mul",
            "memory",
            run_silu_and_mul,
            float(activation_bytes),
            "bytes",
            "GB/s",
            "FlashInfer silu_and_mul activation",
        ),
        OperatorPhase(
            "down_gemm",
            "compute",
            run_down_gemm,
            gemm_flops(
                shape.batch_size, shape.intermediate_size, shape.hidden_size
            ),
            "FLOPs",
            "TFLOPs/s",
            "MLP down projection GEMM",
        ),
        OperatorPhase(
            "post_ffn_fused_add_rmsnorm",
            "memory",
            run_post_ffn_norm,
            float(norm_hidden_bytes),
            "bytes",
            "GB/s",
            "Fused residual add plus RMSNorm after FFN",
        ),
    ]


def export_monitor_csv(path: Path, monitor: GPUMonitor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    monitor.export_csv(str(path))


def benchmark_phase(
    torch: Any,
    phase: OperatorPhase,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"Profiling {phase.name} for ~{args.target_seconds:.1f}s", flush=True)

    _, probe_iter_seconds = calibrate_repeat(
        torch,
        phase.run_once,
        target_seconds=args.target_seconds,
        probe_repeat=args.probe_repeat,
    )
    warmup_repeat = max(
        0,
        int(round(args.warmup_seconds / probe_iter_seconds))
        if probe_iter_seconds > 0
        else 0,
    )
    if warmup_repeat:
        run_for_repeats(torch, phase.run_once, warmup_repeat)

    repeat, probe_iter_seconds = calibrate_repeat(
        torch,
        phase.run_once,
        target_seconds=args.target_seconds,
        probe_repeat=max(args.probe_repeat, 5),
    )

    monitor = GPUMonitor(
        gpu_index=args.monitor_gpu_index,
        interval=args.monitor_interval,
    )
    monitor.start()
    gpu_time_s, wall_time_s, repeat = run_for_min_gpu_seconds(
        torch,
        phase.run_once,
        repeat,
        min_gpu_seconds=args.target_seconds,
    )
    monitor.stop()
    records = monitor.get_results()
    summary = monitor_summary(records)

    monitor_csv = output_dir / "monitor" / f"{phase.name}.csv"
    if records:
        export_monitor_csv(monitor_csv, monitor)
        monitor_csv_value = str(monitor_csv)
    else:
        monitor_csv_value = ""

    if phase.throughput_unit == "TFLOPs/s":
        throughput_value = phase.work_per_iter * repeat / gpu_time_s / 1e12
    elif phase.throughput_unit == "GB/s":
        throughput_value = phase.work_per_iter * repeat / gpu_time_s / 1e9
    else:
        raise ValueError(
            f"unsupported throughput unit: {phase.throughput_unit}"
        )

    row = {
        "run_timestamp_utc": utc_now_iso(),
        "model": args.model,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "op": phase.name,
        "category": phase.category,
        "throughput_value": throughput_value,
        "throughput_unit": phase.throughput_unit,
        "iter_time_ms": gpu_time_s * 1000.0 / repeat,
        "gpu_time_s": gpu_time_s,
        "wall_time_s": wall_time_s,
        "repeat": repeat,
        "target_seconds": args.target_seconds,
        "warmup_seconds": args.warmup_seconds,
        "probe_iter_ms": probe_iter_seconds * 1000.0,
        "work_per_iter": phase.work_per_iter,
        "work_unit": phase.work_unit,
        "monitor_csv": monitor_csv_value,
        "description": phase.description,
        **summary,
    }
    print(
        f"{phase.name}: {throughput_value:.2f} {phase.throughput_unit}, "
        f"{row['iter_time_ms']:.3f} ms/iter, "
        f"{row['avg_power_watts']:.2f} W, "
        f"{row['avg_gpu_clock_mhz']:.2f} MHz",
        flush=True,
    )
    return row


def format_float(value: Any, precision: int = 2) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{precision}f}"


def build_report(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    lines = [
        "# Full-Block Operator Phase Profile",
        "",
        f"- Generated at: `{utc_now_iso()}`",
        f"- Model: `{metadata['model']}`",
        f"- Batch size / sequence length: `{metadata['batch_size']}`",
        f"- Dtype: `{metadata['dtype']}`",
        f"- Target timed window per op: `{metadata['target_seconds']}s`",
        f"- Warmup per op: about `{metadata['warmup_seconds']}s`",
        f"- CUDA_VISIBLE_DEVICES: `{metadata.get('cuda_visible_devices')}`",
        f"- Monitor GPU index: `{metadata['monitor_gpu_index']}`",
        "",
        "Throughput for compute phases is reported as TFLOPs/s. Throughput for memory phases is reported as decimal GB/s using estimated logical bytes, not hardware DRAM counter bytes.",
        "Each row is an isolated fixed-replay phase: the same operator is repeated against the same allocated tensors for the timed window. It is not a state-chain replay of the full block, so GEMM clock/power values should not be directly compared with state-chain `without_norm` full-block clocks.",
        "",
        "## Results",
        "",
        "| Op | Kind | Iter (ms) | Throughput | Avg Power (W) | Avg Clock (MHz) | First 2s Power (W) | Last 2s Power (W) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['op']} | {row['category']} | "
            f"{format_float(row['iter_time_ms'], 3)} | "
            f"{format_float(row['throughput_value'])} {row['throughput_unit']} | "
            f"{format_float(row['avg_power_watts'])} | "
            f"{format_float(row['avg_gpu_clock_mhz'])} | "
            f"{format_float(row['first_2s_power_watts'])} | "
            f"{format_float(row['last_2s_power_watts'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{metadata['summary_csv']}`",
            f"- Metadata: `{metadata['metadata_json']}`",
            f"- Monitor traces: `{metadata['monitor_dir']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile Llama full-block operator phases with per-phase NVML power and clock."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, choices=["7B", "13B", "34B", "70B"]
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--target_seconds", type=float, default=DEFAULT_TARGET_SECONDS
    )
    parser.add_argument(
        "--warmup_seconds", type=float, default=DEFAULT_WARMUP_SECONDS
    )
    parser.add_argument(
        "--probe_repeat", type=int, default=DEFAULT_PROBE_REPEAT
    )
    parser.add_argument(
        "--monitor_interval", type=float, default=DEFAULT_MONITOR_INTERVAL
    )
    parser.add_argument(
        "--monitor_gpu_index", type=int, default=DEFAULT_MONITOR_GPU_INDEX
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / utc_stamp())
    output_dir.mkdir(parents=True, exist_ok=True)

    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = resolve_dtype(torch, args.dtype)
    device = "cuda:0"
    shape = load_shape(args.model, args.batch_size)
    torch.set_default_device(torch.device(device))
    torch.set_default_dtype(dtype)

    metadata = {
        "run_started_at_utc": utc_now_iso(),
        "model": args.model,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "target_seconds": args.target_seconds,
        "warmup_seconds": args.warmup_seconds,
        "probe_repeat": args.probe_repeat,
        "monitor_interval": args.monitor_interval,
        "monitor_gpu_index": args.monitor_gpu_index,
        "eps": args.eps,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": getattr(torch, "__version__", ""),
        "flashinfer_version": getattr(flashinfer, "__version__", ""),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "shape": shape.__dict__,
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "summary.csv"),
        "metadata_json": str(output_dir / "metadata.json"),
        "benchmark_md": str(output_dir / "BENCHMARK.md"),
        "monitor_dir": str(output_dir / "monitor"),
    }
    write_json(output_dir / "metadata.json", metadata)

    with torch.inference_mode():
        phases = build_phases(torch, flashinfer, shape, dtype, device, args.eps)
        rows = []
        for phase in phases:
            row = benchmark_phase(torch, phase, output_dir, args)
            rows.append(row)
            write_csv(output_dir / "summary.csv", rows)
            metadata["completed_ops"] = [completed["op"] for completed in rows]
            write_json(output_dir / "metadata.json", metadata)

    (output_dir / "BENCHMARK.md").write_text(build_report(rows, metadata))
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Summary CSV: {output_dir / 'summary.csv'}", flush=True)
    print(f"Report: {output_dir / 'BENCHMARK.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
