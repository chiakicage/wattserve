#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "llama_operator_microbench"
DEFAULT_OPS = [
    "o",
    "attn_o",
    "qkv_attn_o",
    "gate_up_down",
    "fused_add_norm",
    "o_fused_add_norm",
    "gate_up_down_fused_add_norm",
    "attn_o_fused_add_norm",
    "qkv_attn_o_fused_add_norm",
    "qkv_attn_o_gate_up_down",
    "qkv_attn_o_fused_add_norm_gate_up_down_fused_add_norm",
    "steady_block",
    "steady_block_replace_ln",
    "stack_final_norm",
    "stack_final_norm_replace_ln",
    "stack_lm_head",
    "stack_lm_head_replace_ln",
]
DEFAULT_MODEL = "13B"
DEFAULT_PROMPT_LEN = 8192
DEFAULT_DTYPE = "bfloat16"
DEFAULT_WARMUP = 20
DEFAULT_PROBE_REPEAT = 10
DEFAULT_TARGET_TIMED_SECONDS = 2.0
DEFAULT_MAX_REPEAT = 50000
DEFAULT_MONITOR_INTERVAL = 0.01
DEFAULT_EPS = 1e-5

SUPPORTED_OPS = [
    *DEFAULT_OPS,
    "gate_up",
]

OP_ALIASES = {
    "gemm": "o",
    "prefill_attn_o_gemm": "attn_o",
    "qkv_gemm_prefill_attn_o_gemm": "qkv_attn_o",
    "gate_up_gemm": "gate_up",
    "gate_up_gemm_down_gemm": "gate_up_down",
    "gemm_fused_add_norm": "o_fused_add_norm",
    "gate_up_gemm_down_gemm_fused_add_norm": "gate_up_down_fused_add_norm",
    "prefill_attn_o_gemm_fused_add_norm": "attn_o_fused_add_norm",
    "qkv_gemm_prefill_attn_o_gemm_fused_add_norm": "qkv_attn_o_fused_add_norm",
    "qkv_gemm_prefill_attn_o_gemm_gate_up_gemm_down_gemm": "qkv_attn_o_gate_up_down",
    "qkv_gemm_prefill_attn_o_gemm_fused_add_norm_gate_up_gemm_down_gemm_fused_add_norm": (
        "qkv_attn_o_fused_add_norm_gate_up_down_fused_add_norm"
    ),
}

ALL_OP_CHOICES = sorted(set(SUPPORTED_OPS) | set(OP_ALIASES))

OP_DESCRIPTIONS = {
    "o": "`o` is the hidden-state projection GEMM `[8192, 5120] x [5120, 5120]` for Llama-13B.",
    "attn_o": "`attn_o` runs FlashInfer causal prefill attention, then the `o` projection.",
    "qkv_attn_o": "`qkv_attn_o` runs fused QKV projection, then FlashInfer causal prefill attention, then the `o` projection.",
    "gate_up": "`gate_up` runs the concatenated gate+up projection `[hidden, 2 * intermediate]`.",
    "gate_up_down": "`gate_up_down` runs `gate_up`, then `down` on the up half of the concatenated output.",
    "fused_add_norm": "`fused_add_norm` maps to `flashinfer.fused_add_rmsnorm`.",
    "o_fused_add_norm": "`o_fused_add_norm` runs `o` immediately followed by `flashinfer.fused_add_rmsnorm`.",
    "gate_up_down_fused_add_norm": "`gate_up_down_fused_add_norm` runs `gate_up`, then `down`, then `flashinfer.fused_add_rmsnorm`.",
    "attn_o_fused_add_norm": "`attn_o_fused_add_norm` runs FlashInfer prefill attention, then `o`, then `flashinfer.fused_add_rmsnorm`.",
    "qkv_attn_o_fused_add_norm": "`qkv_attn_o_fused_add_norm` runs `qkv`, then FlashInfer prefill attention, then `o`, then `flashinfer.fused_add_rmsnorm`.",
    "qkv_attn_o_gate_up_down": "`qkv_attn_o_gate_up_down` appends the FFN projections after the attention block.",
    "qkv_attn_o_fused_add_norm_gate_up_down_fused_add_norm": "`qkv_attn_o_fused_add_norm_gate_up_down_fused_add_norm` adds one fused_add_norm between attention and FFN and one after FFN.",
    "steady_block": "`steady_block` is a single steady-state Llama decoder block with input fused_add_norm, RoPE, attention, post-attention fused_add_norm, separate gate/up projections, `silu_and_mul`, and `down`.",
    "steady_block_replace_ln": "`steady_block_replace_ln` mirrors the current `replace_ln` semantics for the same steady-state block: it skips residual add + norm before attention and before FFN.",
    "stack_final_norm": "`stack_final_norm` chains the fitted Llama layer count for the selected model and includes the final norm, but excludes `lm_head`.",
    "stack_final_norm_replace_ln": "`stack_final_norm_replace_ln` mirrors the current end-to-end `replace_ln` semantics for the same stacked workload without `lm_head`.",
    "stack_lm_head": "`stack_lm_head` chains the fitted Llama layer count for the selected model, includes the final norm, and ends with `lm_head`.",
    "stack_lm_head_replace_ln": "`stack_lm_head_replace_ln` mirrors the current end-to-end `replace_ln` semantics for the same stacked workload.",
}

NORM_FREE_OPS = [
    "o",
    "attn_o",
    "qkv_attn_o",
    "gate_up_down",
    "qkv_attn_o_gate_up_down",
]

WITH_NORM_OPS = [
    "o_fused_add_norm",
    "attn_o_fused_add_norm",
    "qkv_attn_o_fused_add_norm",
    "gate_up_down_fused_add_norm",
    "qkv_attn_o_fused_add_norm_gate_up_down_fused_add_norm",
]

PAIRINGS = [
    ("o", "o_fused_add_norm"),
    ("attn_o", "attn_o_fused_add_norm"),
    ("qkv_attn_o", "qkv_attn_o_fused_add_norm"),
    ("gate_up_down", "gate_up_down_fused_add_norm"),
    (
        "qkv_attn_o_gate_up_down",
        "qkv_attn_o_fused_add_norm_gate_up_down_fused_add_norm",
    ),
]

DISPLAY_LABELS = {
    "o": "o",
    "attn_o": "attn + o",
    "qkv_attn_o": "qkv + attn + o",
    "gate_up": "gate_up",
    "gate_up_down": "gate_up + down",
    "qkv_attn_o_gate_up_down": "qkv + attn + o + gate_up + down",
    "o_fused_add_norm": "o + fused_add_norm",
    "attn_o_fused_add_norm": "attn + o + fused_add_norm",
    "qkv_attn_o_fused_add_norm": "qkv + attn + o + fused_add_norm",
    "gate_up_down_fused_add_norm": "gate_up + down + fused_add_norm",
    "qkv_attn_o_fused_add_norm_gate_up_down_fused_add_norm": "qkv + attn + o + fused_add_norm + gate_up + down + fused_add_norm",
    "steady_block": "steady block",
    "steady_block_replace_ln": "steady block (replace_ln)",
    "stack_final_norm": "stack + final_norm",
    "stack_final_norm_replace_ln": "stack (replace_ln)",
    "stack_lm_head": "stack + final_norm + lm_head",
    "stack_lm_head_replace_ln": "stack + lm_head (replace_ln)",
    "fused_add_norm": "fused_add_norm",
}

COMPONENT_DISPLAY_LABELS = {
    "gemm": "o",
    "o": "o",
    "prefill_attn": "attn",
    "attn": "attn",
    "qkv_gemm": "qkv",
    "qkv": "qkv",
    "q": "q",
    "k": "k",
    "v": "v",
    "o_gemm": "o",
    "gate_up_gemm": "gate_up",
    "gate_up": "gate_up",
    "gate": "gate",
    "up": "up",
    "down_gemm": "down",
    "down": "down",
    "rope": "rope",
    "activation": "silu_and_mul",
    "lm_head": "lm_head",
    "input_norm": "input_norm",
    "post_attn_norm": "post_attn_norm",
    "final_norm": "final_norm",
    "fused_add_norm": "fused_add_norm",
}

POWER_FOCUSED_PAIRS = [
    ("steady_block", "steady_block_replace_ln"),
    ("stack_final_norm", "stack_final_norm_replace_ln"),
    ("stack_lm_head", "stack_lm_head_replace_ln"),
]

SUMMARY_FIELDNAMES = [
    "run_timestamp_utc",
    "model",
    "prompt_len",
    "dtype",
    "op",
    "op_base",
    "stack_depth",
    "shape_source",
    "input_shape",
    "weight_shape",
    "output_shape",
    "hidden_size",
    "intermediate_size",
    "head_dim",
    "warmup",
    "probe_repeat",
    "repeat",
    "target_timed_seconds",
    "total_time_s",
    "iter_time_ms",
    "probe_iter_ms",
    "throughput_value",
    "throughput_unit",
    "avg_power_watts",
    "max_power_watts",
    "avg_gpu_clock_mhz",
    "max_gpu_clock_mhz",
    "monitor_sample_count",
    "monitor_csv",
    "component_breakdown_json",
    "combo_gemm_components",
    "combo_non_gemm_components",
    "combo_norm_component",
    "combo_gemm_time_ms",
    "combo_norm_time_ms",
    "combo_gemm_iter_time_ms",
    "combo_norm_iter_time_ms",
    "combo_gemm_tflops_s",
    "combo_non_gemm_time_ms",
    "combo_non_gemm_iter_time_ms",
]


@dataclass(frozen=True)
class OperatorShape:
    model: str
    prompt_len: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_size: int
    kv_size: int
    rope_theta: float
    gemm_m: int
    gemm_k: int
    gemm_n: int
    shape_source: str


@dataclass
class OperatorBenchmark:
    op: str
    input_shape: tuple[int, ...] | str
    weight_shape: tuple[int, ...] | str
    output_shape: tuple[int, ...] | str
    run_once: Callable[[], None]
    throughput_per_iter: float
    throughput_unit: str
    components: list[tuple[str, Callable[[], None]]] | None = None
    gemm_flops_per_iter: float | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _ensure_python_dir_on_path() -> None:
    python_dir = str(PYTHON_DIR)
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)


def _load_module(name: str) -> Any:
    _ensure_python_dir_on_path()
    return importlib.import_module(name)


def _safe_package_version(name: str) -> str | None:
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return None


def _resolve_dtype(dtype_name: str) -> Any:
    torch = importlib.import_module("torch")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported dtype {dtype_name!r}; choose from {sorted(dtype_map)}"
        )
    return dtype_map[dtype_name]


def _resolve_monitor_gpu_index(explicit_index: int | None) -> int:
    if explicit_index is not None:
        return explicit_index
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first_token = visible.split(",")[0].strip()
        if first_token.isdigit():
            return int(first_token)
    return 0


def collect_environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": None,
        "cuda_available": None,
        "cuda_device_name": None,
        "cuda_device_count": None,
        "flashinfer_version": _safe_package_version("flashinfer"),
        "transformers_version": _safe_package_version("transformers"),
    }

    try:
        torch = importlib.import_module("torch")
        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            metadata["cuda_device_count"] = torch.cuda.device_count()
            metadata["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        metadata["torch_error"] = f"{type(exc).__name__}: {exc}"

    return metadata


def _load_llama_shape(model: str, prompt_len: int) -> OperatorShape:
    llama_config = _load_module("models.llama_config")
    config_loaders = {
        "7B": llama_config.get_llama_config_7B,
        "13B": llama_config.get_llama_config_13B,
        "34B": llama_config.get_llama_config_34B,
        "70B": llama_config.get_llama_config_70B,
    }
    if model not in config_loaders:
        raise ValueError(
            f"Unsupported model {model!r}; choose from {sorted(config_loaders)}"
        )

    config = config_loaders[model]()
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_hidden_layers = int(config.num_hidden_layers)
    vocab_size = int(config.vocab_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    rope_theta = float(getattr(config, "rope_theta", 10000.0))

    # Use the [seq_len, hidden] x [hidden, hidden] projection shape because it
    # matches Llama hidden-state projection kernels and keeps GEMM output on the
    # same tensor shape used by the block RMSNorm.
    return OperatorShape(
        model=model,
        prompt_len=prompt_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_size=q_size,
        kv_size=kv_size,
        rope_theta=rope_theta,
        gemm_m=prompt_len,
        gemm_k=hidden_size,
        gemm_n=hidden_size,
        shape_source=(
            f"Llama-{model} attention/block shape: hidden=[{prompt_len}, {hidden_size}], "
            f"q=[{prompt_len}, {num_heads}, {head_dim}], "
            f"kv=[{prompt_len}, {num_kv_heads}, {head_dim}], "
            f"o_proj=[{hidden_size}, {hidden_size}], "
            f"gate/up=[{hidden_size}, {intermediate_size}], "
            f"down=[{intermediate_size}, {hidden_size}]"
        ),
    )


def _make_gemm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    x = torch.randn((shape.gemm_m, shape.gemm_k), device=device, dtype=dtype)
    weight = torch.randn(
        (shape.gemm_k, shape.gemm_n), device=device, dtype=dtype
    )
    out = torch.empty((shape.gemm_m, shape.gemm_n), device=device, dtype=dtype)

    def run_once() -> None:
        torch.mm(x, weight, out=out)

    gemm_flops = _calculate_gemm_flops(shape.gemm_m, shape.gemm_k, shape.gemm_n)
    return OperatorBenchmark(
        op="o",
        input_shape=tuple(x.shape),
        weight_shape=tuple(weight.shape),
        output_shape=tuple(out.shape),
        run_once=run_once,
        throughput_per_iter=gemm_flops,
        throughput_unit="TFLOPs/s",
        gemm_flops_per_iter=gemm_flops,
    )


def _calculate_gemm_flops(m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n


def _calculate_prefill_attention_flops(shape: OperatorShape) -> float:
    # The benchmark uses causal prefill attention, so each query only touches
    # the lower-triangular prefix instead of the full SxS matrix.
    causal_pairs = shape.prompt_len * (shape.prompt_len + 1) / 2.0
    qk_flops = 2.0 * causal_pairs * shape.q_size
    av_flops = 2.0 * causal_pairs * shape.q_size
    return qk_flops + av_flops


def _is_gemm_like_component(component_name: str) -> bool:
    return component_name in {
        "gemm",
        "o",
        "prefill_attn",
        "attn",
        "qkv_gemm",
        "qkv",
        "q",
        "k",
        "v",
        "o_gemm",
        "gate_up_gemm",
        "gate_up",
        "gate",
        "up",
        "down_gemm",
        "down",
        "lm_head",
    }


def _row_total_tflops(row: dict[str, Any]) -> float | None:
    if str(row.get("throughput_unit", "")).startswith("TFLOPs/s"):
        return float(row["throughput_value"])
    return None


def _row_base_op(row: dict[str, Any]) -> str:
    base_op = row.get("op_base")
    if base_op not in (None, ""):
        return str(base_op)
    return str(row["op"])


def _row_stack_depth(row: dict[str, Any]) -> int | None:
    value = row.get("stack_depth")
    if value in (None, ""):
        return None
    return int(value)


def _rows_by_base(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_row_base_op(row), []).append(row)
    return grouped


def _single_row_by_base(
    rows_by_base: dict[str, list[dict[str, Any]]],
    op_base: str,
) -> dict[str, Any] | None:
    matches = rows_by_base.get(op_base, [])
    if len(matches) == 1:
        return matches[0]
    return None


def _is_stack_op(op: str) -> bool:
    return op in {
        "stack_final_norm",
        "stack_final_norm_replace_ln",
        "stack_lm_head",
        "stack_lm_head_replace_ln",
    }


def _normalize_ops(ops: list[str]) -> list[str]:
    normalized_ops: list[str] = []
    seen: set[str] = set()
    for op in ops:
        canonical_op = OP_ALIASES.get(op, op)
        if canonical_op not in SUPPORTED_OPS:
            raise ValueError(
                f"Unsupported op {op!r}; choose from {sorted(ALL_OP_CHOICES)}"
            )
        if canonical_op not in seen:
            normalized_ops.append(canonical_op)
            seen.add(canonical_op)
    return normalized_ops


def _make_fused_add_norm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
    eps: float,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    x = torch.randn(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    residual = torch.randn_like(x)
    weight = torch.ones((shape.hidden_size,), device=device, dtype=dtype)

    def run_once() -> None:
        flashinfer.fused_add_rmsnorm(x, residual, weight, eps=eps)

    bytes_per_iter = (
        x.numel() * x.element_size() * 4
        + weight.numel() * weight.element_size()
    )
    return OperatorBenchmark(
        op="fused_add_norm",
        input_shape=tuple(x.shape),
        weight_shape=tuple(weight.shape),
        output_shape=tuple(x.shape),
        run_once=run_once,
        throughput_per_iter=float(bytes_per_iter),
        throughput_unit="GiB/s",
    )


def _make_prefill_attn_o_gemm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    q = torch.randn(
        (shape.prompt_len, shape.num_heads, shape.head_dim),
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        (shape.prompt_len, shape.num_kv_heads, shape.head_dim),
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        (shape.prompt_len, shape.num_kv_heads, shape.head_dim),
        device=device,
        dtype=dtype,
    )
    o_weight = torch.randn(
        (shape.q_size, shape.hidden_size), device=device, dtype=dtype
    )
    o_out = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    state: dict[str, torch.Tensor] = {}

    def run_prefill_attn() -> None:
        state["attn_out"] = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=True,
            sm_scale=shape.head_dim**-0.5,
        )

    def run_o_gemm() -> None:
        torch.mm(
            state["attn_out"].reshape(shape.prompt_len, shape.q_size),
            o_weight,
            out=o_out,
        )

    def run_once() -> None:
        run_prefill_attn()
        run_o_gemm()

    attn_flops = _calculate_prefill_attention_flops(shape)
    o_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len, shape.q_size, shape.hidden_size
    )
    return OperatorBenchmark(
        op="attn_o",
        input_shape=(
            f"q/k/v=({shape.prompt_len}, {shape.num_heads}, {shape.head_dim}) / "
            f"({shape.prompt_len}, {shape.num_kv_heads}, {shape.head_dim}) / "
            f"({shape.prompt_len}, {shape.num_kv_heads}, {shape.head_dim})"
        ),
        weight_shape=f"o_proj=({shape.q_size}, {shape.hidden_size})",
        output_shape=tuple(o_out.shape),
        run_once=run_once,
        throughput_per_iter=attn_flops + o_gemm_flops,
        throughput_unit="TFLOPs/s(eq_attn_o)",
        components=[
            ("attn", run_prefill_attn),
            ("o", run_o_gemm),
        ],
        gemm_flops_per_iter=attn_flops + o_gemm_flops,
    )


def _make_qkv_gemm_prefill_attn_o_gemm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    hidden_states = torch.randn(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    qkv_weight = torch.randn(
        (shape.hidden_size, shape.q_size + 2 * shape.kv_size),
        device=device,
        dtype=dtype,
    )
    qkv_out = torch.empty(
        (shape.prompt_len, shape.q_size + 2 * shape.kv_size),
        device=device,
        dtype=dtype,
    )
    o_weight = torch.randn(
        (shape.q_size, shape.hidden_size), device=device, dtype=dtype
    )
    o_out = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    state: dict[str, torch.Tensor] = {}

    q_view = qkv_out[:, : shape.q_size].view(
        shape.prompt_len, shape.num_heads, shape.head_dim
    )
    k_start = shape.q_size
    v_start = shape.q_size + shape.kv_size
    k_view = qkv_out[:, k_start:v_start].view(
        shape.prompt_len,
        shape.num_kv_heads,
        shape.head_dim,
    )
    v_view = qkv_out[:, v_start:].view(
        shape.prompt_len,
        shape.num_kv_heads,
        shape.head_dim,
    )

    def run_qkv_gemm() -> None:
        torch.mm(hidden_states, qkv_weight, out=qkv_out)

    def run_prefill_attn() -> None:
        state["attn_out"] = flashinfer.single_prefill_with_kv_cache(
            q_view,
            k_view,
            v_view,
            causal=True,
            sm_scale=shape.head_dim**-0.5,
        )

    def run_o_gemm() -> None:
        torch.mm(
            state["attn_out"].reshape(shape.prompt_len, shape.q_size),
            o_weight,
            out=o_out,
        )

    def run_once() -> None:
        run_qkv_gemm()
        run_prefill_attn()
        run_o_gemm()

    qkv_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.q_size + 2 * shape.kv_size,
    )
    attn_flops = _calculate_prefill_attention_flops(shape)
    o_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len, shape.q_size, shape.hidden_size
    )
    return OperatorBenchmark(
        op="qkv_attn_o",
        input_shape=tuple(hidden_states.shape),
        weight_shape=(
            f"qkv=({shape.hidden_size}, {shape.q_size + 2 * shape.kv_size}) + "
            f"o_proj=({shape.q_size}, {shape.hidden_size})"
        ),
        output_shape=tuple(o_out.shape),
        run_once=run_once,
        throughput_per_iter=qkv_gemm_flops + attn_flops + o_gemm_flops,
        throughput_unit="TFLOPs/s(eq_qkv_attn_o)",
        components=[
            ("qkv", run_qkv_gemm),
            ("attn", run_prefill_attn),
            ("o", run_o_gemm),
        ],
        gemm_flops_per_iter=qkv_gemm_flops + attn_flops + o_gemm_flops,
    )


def _make_gate_up_gemm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    x = torch.randn(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    gate_up_weight = torch.randn(
        (shape.hidden_size, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    gate_up_out = torch.empty(
        (shape.prompt_len, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )

    def run_gate_up_gemm() -> None:
        torch.mm(x, gate_up_weight, out=gate_up_out)

    gate_up_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        2 * shape.intermediate_size,
    )
    return OperatorBenchmark(
        op="gate_up",
        input_shape=tuple(x.shape),
        weight_shape=tuple(gate_up_weight.shape),
        output_shape=tuple(gate_up_out.shape),
        run_once=run_gate_up_gemm,
        throughput_per_iter=gate_up_gemm_flops,
        throughput_unit="TFLOPs/s(eq_gate_up)",
        components=[("gate_up", run_gate_up_gemm)],
        gemm_flops_per_iter=gate_up_gemm_flops,
    )


def _make_gate_up_gemm_down_gemm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    x = torch.randn(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    gate_up_weight = torch.randn(
        (shape.hidden_size, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    gate_up_out = torch.empty(
        (shape.prompt_len, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    down_input = gate_up_out[:, shape.intermediate_size :]
    down_weight = torch.randn(
        (shape.intermediate_size, shape.hidden_size),
        device=device,
        dtype=dtype,
    )
    down_out = torch.empty(
        (shape.prompt_len, shape.hidden_size),
        device=device,
        dtype=dtype,
    )

    def run_gate_up_gemm() -> None:
        torch.mm(x, gate_up_weight, out=gate_up_out)

    def run_down_gemm() -> None:
        torch.mm(down_input, down_weight, out=down_out)

    def run_once() -> None:
        run_gate_up_gemm()
        run_down_gemm()

    gate_up_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        2 * shape.intermediate_size,
    )
    down_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.intermediate_size,
        shape.hidden_size,
    )
    return OperatorBenchmark(
        op="gate_up_down",
        input_shape=tuple(x.shape),
        weight_shape=(
            f"gate_up=({shape.hidden_size}, {2 * shape.intermediate_size}) + "
            f"down=({shape.intermediate_size}, {shape.hidden_size})"
        ),
        output_shape=tuple(down_out.shape),
        run_once=run_once,
        throughput_per_iter=gate_up_gemm_flops + down_gemm_flops,
        throughput_unit="TFLOPs/s(eq_gate_up_down)",
        components=[
            ("gate_up", run_gate_up_gemm),
            ("down", run_down_gemm),
        ],
        gemm_flops_per_iter=gate_up_gemm_flops + down_gemm_flops,
    )


def _make_gate_up_gemm_down_gemm_fused_add_norm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
    eps: float,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    x = torch.randn(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    gate_up_weight = torch.randn(
        (shape.hidden_size, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    gate_up_out = torch.empty(
        (shape.prompt_len, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    down_input = gate_up_out[:, shape.intermediate_size :]
    down_weight = torch.randn(
        (shape.intermediate_size, shape.hidden_size),
        device=device,
        dtype=dtype,
    )
    down_out = torch.empty(
        (shape.prompt_len, shape.hidden_size),
        device=device,
        dtype=dtype,
    )
    residual = torch.randn_like(down_out)
    norm_weight = torch.ones((shape.hidden_size,), device=device, dtype=dtype)

    def run_gate_up_gemm() -> None:
        torch.mm(x, gate_up_weight, out=gate_up_out)

    def run_down_gemm() -> None:
        torch.mm(down_input, down_weight, out=down_out)

    def run_norm() -> None:
        flashinfer.fused_add_rmsnorm(down_out, residual, norm_weight, eps=eps)

    def run_once() -> None:
        run_gate_up_gemm()
        run_down_gemm()
        run_norm()

    gate_up_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        2 * shape.intermediate_size,
    )
    down_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.intermediate_size,
        shape.hidden_size,
    )
    return OperatorBenchmark(
        op="gate_up_down_fused_add_norm",
        input_shape=tuple(x.shape),
        weight_shape=(
            f"gate_up=({shape.hidden_size}, {2 * shape.intermediate_size}) + "
            f"down=({shape.intermediate_size}, {shape.hidden_size}) + "
            f"norm=({shape.hidden_size},)"
        ),
        output_shape=tuple(down_out.shape),
        run_once=run_once,
        throughput_per_iter=gate_up_gemm_flops + down_gemm_flops,
        throughput_unit="TFLOPs/s(eq_gate_up_down)",
        components=[
            ("gate_up", run_gate_up_gemm),
            ("down", run_down_gemm),
            ("fused_add_norm", run_norm),
        ],
        gemm_flops_per_iter=gate_up_gemm_flops + down_gemm_flops,
    )


def _make_gemm_fused_add_norm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
    eps: float,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    x = torch.randn((shape.gemm_m, shape.gemm_k), device=device, dtype=dtype)
    weight = torch.randn(
        (shape.gemm_k, shape.gemm_n), device=device, dtype=dtype
    )
    gemm_out = torch.empty(
        (shape.gemm_m, shape.gemm_n), device=device, dtype=dtype
    )
    residual = torch.randn_like(gemm_out)
    norm_weight = torch.ones((shape.hidden_size,), device=device, dtype=dtype)

    def run_gemm() -> None:
        torch.mm(x, weight, out=gemm_out)

    def run_norm() -> None:
        flashinfer.fused_add_rmsnorm(gemm_out, residual, norm_weight, eps=eps)

    def run_once() -> None:
        run_gemm()
        run_norm()

    gemm_flops = _calculate_gemm_flops(shape.gemm_m, shape.gemm_k, shape.gemm_n)
    return OperatorBenchmark(
        op="o_fused_add_norm",
        input_shape=tuple(x.shape),
        weight_shape=f"({shape.gemm_k}, {shape.gemm_n}) + ({shape.hidden_size},)",
        output_shape=tuple(gemm_out.shape),
        run_once=run_once,
        throughput_per_iter=gemm_flops,
        throughput_unit="TFLOPs/s(eq_o)",
        components=[("o", run_gemm), ("fused_add_norm", run_norm)],
        gemm_flops_per_iter=gemm_flops,
    )


def _make_prefill_attn_o_gemm_fused_add_norm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
    eps: float,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    q = torch.randn(
        (shape.prompt_len, shape.num_heads, shape.head_dim),
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        (shape.prompt_len, shape.num_kv_heads, shape.head_dim),
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        (shape.prompt_len, shape.num_kv_heads, shape.head_dim),
        device=device,
        dtype=dtype,
    )
    o_weight = torch.randn(
        (shape.q_size, shape.hidden_size), device=device, dtype=dtype
    )
    o_out = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    residual = torch.randn_like(o_out)
    norm_weight = torch.ones((shape.hidden_size,), device=device, dtype=dtype)
    state: dict[str, torch.Tensor] = {}

    def run_prefill_attn() -> None:
        state["attn_out"] = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=True,
            sm_scale=shape.head_dim**-0.5,
        )

    def run_o_gemm() -> None:
        torch.mm(
            state["attn_out"].reshape(shape.prompt_len, shape.q_size),
            o_weight,
            out=o_out,
        )

    def run_norm() -> None:
        flashinfer.fused_add_rmsnorm(o_out, residual, norm_weight, eps=eps)

    def run_once() -> None:
        run_prefill_attn()
        run_o_gemm()
        run_norm()

    attn_flops = _calculate_prefill_attention_flops(shape)
    o_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len, shape.q_size, shape.hidden_size
    )
    return OperatorBenchmark(
        op="attn_o_fused_add_norm",
        input_shape=(
            f"q/k/v=({shape.prompt_len}, {shape.num_heads}, {shape.head_dim}) / "
            f"({shape.prompt_len}, {shape.num_kv_heads}, {shape.head_dim}) / "
            f"({shape.prompt_len}, {shape.num_kv_heads}, {shape.head_dim})"
        ),
        weight_shape=f"o_proj=({shape.q_size}, {shape.hidden_size}) + norm=({shape.hidden_size},)",
        output_shape=tuple(o_out.shape),
        run_once=run_once,
        throughput_per_iter=attn_flops + o_gemm_flops,
        throughput_unit="TFLOPs/s(eq_attn_o)",
        components=[
            ("attn", run_prefill_attn),
            ("o", run_o_gemm),
            ("fused_add_norm", run_norm),
        ],
        gemm_flops_per_iter=attn_flops + o_gemm_flops,
    )


def _make_qkv_gemm_prefill_attn_o_gemm_fused_add_norm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
    eps: float,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    hidden_states = torch.randn(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    qkv_weight = torch.randn(
        (shape.hidden_size, shape.q_size + 2 * shape.kv_size),
        device=device,
        dtype=dtype,
    )
    qkv_out = torch.empty(
        (shape.prompt_len, shape.q_size + 2 * shape.kv_size),
        device=device,
        dtype=dtype,
    )
    o_weight = torch.randn(
        (shape.q_size, shape.hidden_size), device=device, dtype=dtype
    )
    o_out = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    residual = torch.randn_like(o_out)
    norm_weight = torch.ones((shape.hidden_size,), device=device, dtype=dtype)
    state: dict[str, torch.Tensor] = {}

    q_view = qkv_out[:, : shape.q_size].view(
        shape.prompt_len, shape.num_heads, shape.head_dim
    )
    k_start = shape.q_size
    v_start = shape.q_size + shape.kv_size
    k_view = qkv_out[:, k_start:v_start].view(
        shape.prompt_len,
        shape.num_kv_heads,
        shape.head_dim,
    )
    v_view = qkv_out[:, v_start:].view(
        shape.prompt_len,
        shape.num_kv_heads,
        shape.head_dim,
    )

    def run_qkv_gemm() -> None:
        torch.mm(hidden_states, qkv_weight, out=qkv_out)

    def run_prefill_attn() -> None:
        state["attn_out"] = flashinfer.single_prefill_with_kv_cache(
            q_view,
            k_view,
            v_view,
            causal=True,
            sm_scale=shape.head_dim**-0.5,
        )

    def run_o_gemm() -> None:
        torch.mm(
            state["attn_out"].reshape(shape.prompt_len, shape.q_size),
            o_weight,
            out=o_out,
        )

    def run_norm() -> None:
        flashinfer.fused_add_rmsnorm(o_out, residual, norm_weight, eps=eps)

    def run_once() -> None:
        run_qkv_gemm()
        run_prefill_attn()
        run_o_gemm()
        run_norm()

    qkv_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.q_size + 2 * shape.kv_size,
    )
    attn_flops = _calculate_prefill_attention_flops(shape)
    o_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len, shape.q_size, shape.hidden_size
    )
    return OperatorBenchmark(
        op="qkv_attn_o_fused_add_norm",
        input_shape=tuple(hidden_states.shape),
        weight_shape=(
            f"qkv=({shape.hidden_size}, {shape.q_size + 2 * shape.kv_size}) + "
            f"o_proj=({shape.q_size}, {shape.hidden_size}) + "
            f"norm=({shape.hidden_size},)"
        ),
        output_shape=tuple(o_out.shape),
        run_once=run_once,
        throughput_per_iter=qkv_gemm_flops + attn_flops + o_gemm_flops,
        throughput_unit="TFLOPs/s(eq_qkv_attn_o)",
        components=[
            ("qkv", run_qkv_gemm),
            ("attn", run_prefill_attn),
            ("o", run_o_gemm),
            ("fused_add_norm", run_norm),
        ],
        gemm_flops_per_iter=qkv_gemm_flops + attn_flops + o_gemm_flops,
    )


def _make_qkv_gemm_prefill_attn_o_gemm_gate_up_gemm_down_gemm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    hidden_states = torch.randn(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    qkv_weight = torch.randn(
        (shape.hidden_size, shape.q_size + 2 * shape.kv_size),
        device=device,
        dtype=dtype,
    )
    qkv_out = torch.empty(
        (shape.prompt_len, shape.q_size + 2 * shape.kv_size),
        device=device,
        dtype=dtype,
    )
    o_weight = torch.randn(
        (shape.q_size, shape.hidden_size), device=device, dtype=dtype
    )
    o_out = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    gate_up_weight = torch.randn(
        (shape.hidden_size, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    gate_up_out = torch.empty(
        (shape.prompt_len, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    down_input = gate_up_out[:, shape.intermediate_size :]
    down_weight = torch.randn(
        (shape.intermediate_size, shape.hidden_size),
        device=device,
        dtype=dtype,
    )
    down_out = torch.empty(
        (shape.prompt_len, shape.hidden_size),
        device=device,
        dtype=dtype,
    )
    state: dict[str, torch.Tensor] = {}

    q_view = qkv_out[:, : shape.q_size].view(
        shape.prompt_len, shape.num_heads, shape.head_dim
    )
    k_start = shape.q_size
    v_start = shape.q_size + shape.kv_size
    k_view = qkv_out[:, k_start:v_start].view(
        shape.prompt_len,
        shape.num_kv_heads,
        shape.head_dim,
    )
    v_view = qkv_out[:, v_start:].view(
        shape.prompt_len,
        shape.num_kv_heads,
        shape.head_dim,
    )

    def run_qkv_gemm() -> None:
        torch.mm(hidden_states, qkv_weight, out=qkv_out)

    def run_prefill_attn() -> None:
        state["attn_out"] = flashinfer.single_prefill_with_kv_cache(
            q_view,
            k_view,
            v_view,
            causal=True,
            sm_scale=shape.head_dim**-0.5,
        )

    def run_o_gemm() -> None:
        torch.mm(
            state["attn_out"].reshape(shape.prompt_len, shape.q_size),
            o_weight,
            out=o_out,
        )

    def run_gate_up_gemm() -> None:
        torch.mm(o_out, gate_up_weight, out=gate_up_out)

    def run_down_gemm() -> None:
        torch.mm(down_input, down_weight, out=down_out)

    def run_once() -> None:
        run_qkv_gemm()
        run_prefill_attn()
        run_o_gemm()
        run_gate_up_gemm()
        run_down_gemm()

    qkv_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.q_size + 2 * shape.kv_size,
    )
    attn_flops = _calculate_prefill_attention_flops(shape)
    o_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len, shape.q_size, shape.hidden_size
    )
    gate_up_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        2 * shape.intermediate_size,
    )
    down_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.intermediate_size,
        shape.hidden_size,
    )
    total_flops = (
        qkv_gemm_flops
        + attn_flops
        + o_gemm_flops
        + gate_up_gemm_flops
        + down_gemm_flops
    )
    return OperatorBenchmark(
        op="qkv_attn_o_gate_up_down",
        input_shape=tuple(hidden_states.shape),
        weight_shape=(
            f"qkv=({shape.hidden_size}, {shape.q_size + 2 * shape.kv_size}) + "
            f"o_proj=({shape.q_size}, {shape.hidden_size}) + "
            f"gate_up=({shape.hidden_size}, {2 * shape.intermediate_size}) + "
            f"down=({shape.intermediate_size}, {shape.hidden_size})"
        ),
        output_shape=tuple(down_out.shape),
        run_once=run_once,
        throughput_per_iter=total_flops,
        throughput_unit="TFLOPs/s(eq_qkv_attn_o_up_down)",
        components=[
            ("qkv", run_qkv_gemm),
            ("attn", run_prefill_attn),
            ("o", run_o_gemm),
            ("gate_up", run_gate_up_gemm),
            ("down", run_down_gemm),
        ],
        gemm_flops_per_iter=total_flops,
    )


def _make_qkv_gemm_prefill_attn_o_gemm_fused_add_norm_gate_up_gemm_down_gemm_fused_add_norm_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
    eps: float,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    hidden_states = torch.randn(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    qkv_weight = torch.randn(
        (shape.hidden_size, shape.q_size + 2 * shape.kv_size),
        device=device,
        dtype=dtype,
    )
    qkv_out = torch.empty(
        (shape.prompt_len, shape.q_size + 2 * shape.kv_size),
        device=device,
        dtype=dtype,
    )
    o_weight = torch.randn(
        (shape.q_size, shape.hidden_size), device=device, dtype=dtype
    )
    o_out = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    residual = torch.randn_like(o_out)
    post_attn_norm_weight = torch.ones(
        (shape.hidden_size,), device=device, dtype=dtype
    )
    gate_up_weight = torch.randn(
        (shape.hidden_size, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    gate_up_out = torch.empty(
        (shape.prompt_len, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    down_input = gate_up_out[:, shape.intermediate_size :]
    down_weight = torch.randn(
        (shape.intermediate_size, shape.hidden_size),
        device=device,
        dtype=dtype,
    )
    down_out = torch.empty(
        (shape.prompt_len, shape.hidden_size),
        device=device,
        dtype=dtype,
    )
    post_ffn_norm_weight = torch.ones(
        (shape.hidden_size,), device=device, dtype=dtype
    )
    state: dict[str, torch.Tensor] = {}

    q_view = qkv_out[:, : shape.q_size].view(
        shape.prompt_len, shape.num_heads, shape.head_dim
    )
    k_start = shape.q_size
    v_start = shape.q_size + shape.kv_size
    k_view = qkv_out[:, k_start:v_start].view(
        shape.prompt_len,
        shape.num_kv_heads,
        shape.head_dim,
    )
    v_view = qkv_out[:, v_start:].view(
        shape.prompt_len,
        shape.num_kv_heads,
        shape.head_dim,
    )

    def run_qkv_gemm() -> None:
        torch.mm(hidden_states, qkv_weight, out=qkv_out)

    def run_prefill_attn() -> None:
        state["attn_out"] = flashinfer.single_prefill_with_kv_cache(
            q_view,
            k_view,
            v_view,
            causal=True,
            sm_scale=shape.head_dim**-0.5,
        )

    def run_o_gemm() -> None:
        torch.mm(
            state["attn_out"].reshape(shape.prompt_len, shape.q_size),
            o_weight,
            out=o_out,
        )

    def run_post_attn_fused_add_norm() -> None:
        flashinfer.fused_add_rmsnorm(
            o_out,
            residual,
            post_attn_norm_weight,
            eps=eps,
        )

    def run_gate_up_gemm() -> None:
        torch.mm(o_out, gate_up_weight, out=gate_up_out)

    def run_down_gemm() -> None:
        torch.mm(down_input, down_weight, out=down_out)

    def run_post_ffn_fused_add_norm() -> None:
        flashinfer.fused_add_rmsnorm(
            down_out,
            residual,
            post_ffn_norm_weight,
            eps=eps,
        )

    def run_once() -> None:
        run_qkv_gemm()
        run_prefill_attn()
        run_o_gemm()
        run_post_attn_fused_add_norm()
        run_gate_up_gemm()
        run_down_gemm()
        run_post_ffn_fused_add_norm()

    qkv_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.q_size + 2 * shape.kv_size,
    )
    attn_flops = _calculate_prefill_attention_flops(shape)
    o_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len, shape.q_size, shape.hidden_size
    )
    gate_up_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        2 * shape.intermediate_size,
    )
    down_gemm_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.intermediate_size,
        shape.hidden_size,
    )
    total_flops = (
        qkv_gemm_flops
        + attn_flops
        + o_gemm_flops
        + gate_up_gemm_flops
        + down_gemm_flops
    )
    return OperatorBenchmark(
        op="qkv_attn_o_fused_add_norm_gate_up_down_fused_add_norm",
        input_shape=tuple(hidden_states.shape),
        weight_shape=(
            f"qkv=({shape.hidden_size}, {shape.q_size + 2 * shape.kv_size}) + "
            f"o_proj=({shape.q_size}, {shape.hidden_size}) + "
            f"norm1=({shape.hidden_size},) + "
            f"gate_up=({shape.hidden_size}, {2 * shape.intermediate_size}) + "
            f"down=({shape.intermediate_size}, {shape.hidden_size}) + "
            f"norm2=({shape.hidden_size},)"
        ),
        output_shape=tuple(down_out.shape),
        run_once=run_once,
        throughput_per_iter=total_flops,
        throughput_unit="TFLOPs/s(eq_qkv_attn_o_up_down)",
        components=[
            ("qkv", run_qkv_gemm),
            ("attn", run_prefill_attn),
            ("o", run_o_gemm),
            ("post_attn_fused_add_norm", run_post_attn_fused_add_norm),
            ("gate_up", run_gate_up_gemm),
            ("down", run_down_gemm),
            ("post_ffn_fused_add_norm", run_post_ffn_fused_add_norm),
        ],
        gemm_flops_per_iter=total_flops,
    )


def _make_scaled_tensor(
    torch: Any,
    tensor_shape: tuple[int, ...],
    device: str,
    dtype: Any,
    scale: float = 0.02,
) -> Any:
    tensor = torch.randn(tensor_shape, device=device, dtype=dtype)
    tensor.mul_(scale)
    return tensor


def _calculate_block_compute_flops(shape: OperatorShape) -> float:
    q_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.q_size,
    )
    k_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.kv_size,
    )
    v_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.kv_size,
    )
    attn_flops = _calculate_prefill_attention_flops(shape)
    o_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.q_size,
        shape.hidden_size,
    )
    gate_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.intermediate_size,
    )
    up_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.intermediate_size,
    )
    down_flops = _calculate_gemm_flops(
        shape.prompt_len,
        shape.intermediate_size,
        shape.hidden_size,
    )
    return (
        q_flops
        + k_flops
        + v_flops
        + attn_flops
        + o_flops
        + gate_flops
        + up_flops
        + down_flops
    )


def _calculate_lm_head_flops(shape: OperatorShape) -> float:
    return _calculate_gemm_flops(
        shape.prompt_len,
        shape.hidden_size,
        shape.vocab_size,
    )


def _make_steady_block_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
    eps: float,
    replace_ln: bool,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")

    hidden_source = _make_scaled_tensor(
        torch,
        (shape.prompt_len, shape.hidden_size),
        device,
        dtype,
    )
    residual_source = _make_scaled_tensor(
        torch,
        (shape.prompt_len, shape.hidden_size),
        device,
        dtype,
    )
    hidden = torch.empty_like(hidden_source)
    residual = torch.empty_like(residual_source)

    input_norm_weight = torch.ones(
        (shape.hidden_size,), device=device, dtype=dtype
    )
    post_attn_norm_weight = torch.ones(
        (shape.hidden_size,), device=device, dtype=dtype
    )
    q_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.q_size), device, dtype
    )
    k_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.kv_size), device, dtype
    )
    v_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.kv_size), device, dtype
    )
    o_weight = _make_scaled_tensor(
        torch, (shape.q_size, shape.hidden_size), device, dtype
    )
    gate_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )
    up_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )
    down_weight = _make_scaled_tensor(
        torch, (shape.intermediate_size, shape.hidden_size), device, dtype
    )

    q_out = torch.empty(
        (shape.prompt_len, shape.q_size), device=device, dtype=dtype
    )
    k_out = torch.empty(
        (shape.prompt_len, shape.kv_size), device=device, dtype=dtype
    )
    v_out = torch.empty(
        (shape.prompt_len, shape.kv_size), device=device, dtype=dtype
    )
    o_out = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    gate_out = torch.empty(
        (shape.prompt_len, shape.intermediate_size), device=device, dtype=dtype
    )
    up_out = torch.empty_like(gate_out)
    cat_out = torch.empty(
        (shape.prompt_len, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    act_out = torch.empty_like(gate_out)
    down_out = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )

    positions = torch.arange(shape.prompt_len, device=device)
    q_view = q_out.view(shape.prompt_len, shape.num_heads, shape.head_dim)
    k_view = k_out.view(shape.prompt_len, shape.num_kv_heads, shape.head_dim)
    v_view = v_out.view(shape.prompt_len, shape.num_kv_heads, shape.head_dim)

    def run_once() -> None:
        hidden.copy_(hidden_source)
        if not replace_ln:
            residual.copy_(residual_source)
            flashinfer.fused_add_rmsnorm(
                hidden, residual, input_norm_weight, eps=eps
            )

        torch.mm(hidden, q_weight, out=q_out)
        torch.mm(hidden, k_weight, out=k_out)
        torch.mm(hidden, v_weight, out=v_out)
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
        torch.mm(
            attn_out.reshape(shape.prompt_len, shape.q_size),
            o_weight,
            out=o_out,
        )

        mlp_input = o_out
        if not replace_ln:
            flashinfer.fused_add_rmsnorm(
                o_out, residual, post_attn_norm_weight, eps=eps
            )

        torch.mm(mlp_input, gate_weight, out=gate_out)
        torch.mm(mlp_input, up_weight, out=up_out)
        cat_out[:, : shape.intermediate_size].copy_(gate_out)
        cat_out[:, shape.intermediate_size :].copy_(up_out)
        flashinfer.silu_and_mul(cat_out, out=act_out)
        torch.mm(act_out, down_weight, out=down_out)

    block_flops = _calculate_block_compute_flops(shape)
    weight_shape = (
        f"q=({shape.hidden_size}, {shape.q_size}) + "
        f"k=({shape.hidden_size}, {shape.kv_size}) + "
        f"v=({shape.hidden_size}, {shape.kv_size}) + "
        f"o=({shape.q_size}, {shape.hidden_size}) + "
        f"gate=({shape.hidden_size}, {shape.intermediate_size}) + "
        f"up=({shape.hidden_size}, {shape.intermediate_size}) + "
        f"down=({shape.intermediate_size}, {shape.hidden_size})"
    )
    if not replace_ln:
        weight_shape += f" + norm=2x({shape.hidden_size},)"
    return OperatorBenchmark(
        op="steady_block_replace_ln" if replace_ln else "steady_block",
        input_shape=tuple(hidden_source.shape),
        weight_shape=weight_shape,
        output_shape=tuple(down_out.shape),
        run_once=run_once,
        throughput_per_iter=block_flops,
        throughput_unit="TFLOPs/s(eq_steady_block)",
        gemm_flops_per_iter=block_flops,
    )


def _make_stack_benchmark(
    shape: OperatorShape,
    dtype: Any,
    device: str,
    eps: float,
    replace_ln: bool,
    include_lm_head: bool,
    num_layers: int,
) -> OperatorBenchmark:
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")

    hidden_source = _make_scaled_tensor(
        torch,
        (shape.prompt_len, shape.hidden_size),
        device,
        dtype,
    )
    hidden = torch.empty_like(hidden_source)
    hidden_norm = torch.empty_like(hidden_source)
    residual = torch.empty_like(hidden_source)

    input_norm_weight = torch.ones(
        (shape.hidden_size,), device=device, dtype=dtype
    )
    post_attn_norm_weight = torch.ones(
        (shape.hidden_size,), device=device, dtype=dtype
    )
    final_norm_weight = torch.ones(
        (shape.hidden_size,), device=device, dtype=dtype
    )

    q_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.q_size), device, dtype
    )
    k_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.kv_size), device, dtype
    )
    v_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.kv_size), device, dtype
    )
    o_weight = _make_scaled_tensor(
        torch, (shape.q_size, shape.hidden_size), device, dtype
    )
    gate_weight = _make_scaled_tensor(
        torch,
        (shape.hidden_size, shape.intermediate_size),
        device,
        dtype,
    )
    up_weight = _make_scaled_tensor(
        torch,
        (shape.hidden_size, shape.intermediate_size),
        device,
        dtype,
    )
    down_weight = _make_scaled_tensor(
        torch,
        (shape.intermediate_size, shape.hidden_size),
        device,
        dtype,
    )
    lm_head_weight = None
    if include_lm_head:
        lm_head_weight = _make_scaled_tensor(
            torch, (shape.hidden_size, shape.vocab_size), device, dtype
        )

    q_out = torch.empty(
        (shape.prompt_len, shape.q_size), device=device, dtype=dtype
    )
    k_out = torch.empty(
        (shape.prompt_len, shape.kv_size), device=device, dtype=dtype
    )
    v_out = torch.empty(
        (shape.prompt_len, shape.kv_size), device=device, dtype=dtype
    )
    o_out = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    gate_out = torch.empty(
        (shape.prompt_len, shape.intermediate_size), device=device, dtype=dtype
    )
    up_out = torch.empty_like(gate_out)
    cat_out = torch.empty(
        (shape.prompt_len, 2 * shape.intermediate_size),
        device=device,
        dtype=dtype,
    )
    act_out = torch.empty_like(gate_out)
    lm_head_out = None
    if include_lm_head:
        lm_head_out = torch.empty(
            (shape.prompt_len, shape.vocab_size), device=device, dtype=dtype
        )

    positions = torch.arange(shape.prompt_len, device=device)
    q_view = q_out.view(shape.prompt_len, shape.num_heads, shape.head_dim)
    k_view = k_out.view(shape.prompt_len, shape.num_kv_heads, shape.head_dim)
    v_view = v_out.view(shape.prompt_len, shape.num_kv_heads, shape.head_dim)

    def run_attention_and_ffn(layer_input: Any) -> None:
        torch.mm(layer_input, q_weight, out=q_out)
        torch.mm(layer_input, k_weight, out=k_out)
        torch.mm(layer_input, v_weight, out=v_out)
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
        torch.mm(
            attn_out.reshape(shape.prompt_len, shape.q_size),
            o_weight,
            out=o_out,
        )
        mlp_input = o_out
        if not replace_ln:
            flashinfer.fused_add_rmsnorm(
                o_out,
                residual,
                post_attn_norm_weight,
                eps=eps,
            )
        torch.mm(mlp_input, gate_weight, out=gate_out)
        torch.mm(mlp_input, up_weight, out=up_out)
        cat_out[:, : shape.intermediate_size].copy_(gate_out)
        cat_out[:, shape.intermediate_size :].copy_(up_out)
        flashinfer.silu_and_mul(cat_out, out=act_out)
        torch.mm(act_out, down_weight, out=hidden)

    def run_once() -> None:
        hidden.copy_(hidden_source)
        if replace_ln:
            for _ in range(num_layers):
                run_attention_and_ffn(hidden)
            if include_lm_head:
                assert lm_head_weight is not None
                assert lm_head_out is not None
                torch.mm(hidden, lm_head_weight, out=lm_head_out)
            return

        flashinfer.rmsnorm(
            hidden,
            input_norm_weight,
            eps=eps,
            out=hidden_norm,
        )
        residual.copy_(hidden)
        run_attention_and_ffn(hidden_norm)

        for _ in range(1, num_layers):
            flashinfer.fused_add_rmsnorm(
                hidden,
                residual,
                input_norm_weight,
                eps=eps,
            )
            run_attention_and_ffn(hidden)

        flashinfer.fused_add_rmsnorm(
            hidden, residual, final_norm_weight, eps=eps
        )
        if include_lm_head:
            assert lm_head_weight is not None
            assert lm_head_out is not None
            torch.mm(hidden, lm_head_weight, out=lm_head_out)

    total_flops = num_layers * _calculate_block_compute_flops(shape)
    if include_lm_head:
        total_flops += _calculate_lm_head_flops(shape)
    weight_shape = (
        f"shared layer weights repeated x{num_layers}: "
        f"[q=({shape.hidden_size}, {shape.q_size}), "
        f"k=({shape.hidden_size}, {shape.kv_size}), "
        f"v=({shape.hidden_size}, {shape.kv_size}), "
        f"o=({shape.q_size}, {shape.hidden_size}), "
        f"gate=({shape.hidden_size}, {shape.intermediate_size}), "
        f"up=({shape.hidden_size}, {shape.intermediate_size}), "
        f"down=({shape.intermediate_size}, {shape.hidden_size})]"
    )
    if include_lm_head:
        weight_shape += f" + lm_head=({shape.hidden_size}, {shape.vocab_size})"
    if not replace_ln:
        weight_shape += f" + shared norms repeated over {num_layers} layers + final_norm=({shape.hidden_size},)"
    if include_lm_head:
        op = "stack_lm_head_replace_ln" if replace_ln else "stack_lm_head"
        throughput_unit = "TFLOPs/s(eq_stack_lm_head)"
        output_shape: tuple[int, ...] | str = tuple(lm_head_out.shape)
    else:
        op = "stack_final_norm_replace_ln" if replace_ln else "stack_final_norm"
        throughput_unit = "TFLOPs/s(eq_stack_final_norm)"
        output_shape = tuple(hidden.shape)
    return OperatorBenchmark(
        op=op,
        input_shape=tuple(hidden_source.shape),
        weight_shape=weight_shape,
        output_shape=output_shape,
        run_once=run_once,
        throughput_per_iter=total_flops,
        throughput_unit=throughput_unit,
        gemm_flops_per_iter=total_flops,
    )


def _build_operator_benchmark(
    op: str,
    shape: OperatorShape,
    dtype: Any,
    device: str,
    eps: float,
    stack_depth: int | None = None,
) -> OperatorBenchmark:
    effective_stack_depth = stack_depth or shape.num_hidden_layers
    if op == "o":
        return _make_gemm_benchmark(shape, dtype, device)
    if op == "attn_o":
        return _make_prefill_attn_o_gemm_benchmark(shape, dtype, device)
    if op == "qkv_attn_o":
        return _make_qkv_gemm_prefill_attn_o_gemm_benchmark(
            shape, dtype, device
        )
    if op == "gate_up":
        return _make_gate_up_gemm_benchmark(shape, dtype, device)
    if op == "gate_up_down":
        return _make_gate_up_gemm_down_gemm_benchmark(shape, dtype, device)
    if op == "fused_add_norm":
        return _make_fused_add_norm_benchmark(shape, dtype, device, eps)
    if op == "o_fused_add_norm":
        return _make_gemm_fused_add_norm_benchmark(shape, dtype, device, eps)
    if op == "gate_up_down_fused_add_norm":
        return _make_gate_up_gemm_down_gemm_fused_add_norm_benchmark(
            shape, dtype, device, eps
        )
    if op == "attn_o_fused_add_norm":
        return _make_prefill_attn_o_gemm_fused_add_norm_benchmark(
            shape, dtype, device, eps
        )
    if op == "qkv_attn_o_fused_add_norm":
        return _make_qkv_gemm_prefill_attn_o_gemm_fused_add_norm_benchmark(
            shape, dtype, device, eps
        )
    if op == "qkv_attn_o_gate_up_down":
        return (
            _make_qkv_gemm_prefill_attn_o_gemm_gate_up_gemm_down_gemm_benchmark(
                shape, dtype, device
            )
        )
    if op == "qkv_attn_o_fused_add_norm_gate_up_down_fused_add_norm":
        return _make_qkv_gemm_prefill_attn_o_gemm_fused_add_norm_gate_up_gemm_down_gemm_fused_add_norm_benchmark(
            shape,
            dtype,
            device,
            eps,
        )
    if op == "steady_block":
        return _make_steady_block_benchmark(
            shape, dtype, device, eps, replace_ln=False
        )
    if op == "steady_block_replace_ln":
        return _make_steady_block_benchmark(
            shape, dtype, device, eps, replace_ln=True
        )
    if op == "stack_final_norm":
        return _make_stack_benchmark(
            shape,
            dtype,
            device,
            eps,
            replace_ln=False,
            include_lm_head=False,
            num_layers=effective_stack_depth,
        )
    if op == "stack_final_norm_replace_ln":
        return _make_stack_benchmark(
            shape,
            dtype,
            device,
            eps,
            replace_ln=True,
            include_lm_head=False,
            num_layers=effective_stack_depth,
        )
    if op == "stack_lm_head":
        return _make_stack_benchmark(
            shape,
            dtype,
            device,
            eps,
            replace_ln=False,
            include_lm_head=True,
            num_layers=effective_stack_depth,
        )
    if op == "stack_lm_head_replace_ln":
        return _make_stack_benchmark(
            shape,
            dtype,
            device,
            eps,
            replace_ln=True,
            include_lm_head=True,
            num_layers=effective_stack_depth,
        )
    raise ValueError(f"Unsupported op {op!r}")


def _calibrate_repeat(
    run_once: Callable[[], None],
    target_timed_seconds: float,
    probe_repeat: int,
    max_repeat: int,
) -> tuple[int, float]:
    torch = importlib.import_module("torch")
    probe_repeat = max(1, probe_repeat)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(probe_repeat):
        run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    iter_seconds = elapsed / probe_repeat
    if iter_seconds <= 0:
        return 1, 0.0
    repeat = max(
        1, min(max_repeat, int(round(target_timed_seconds / iter_seconds)))
    )
    return repeat, iter_seconds


def _measure_component_times_ms(
    components: list[tuple[str, Callable[[], None]]],
    repeat: int,
) -> dict[str, float]:
    torch = importlib.import_module("torch")
    timings_ms = {name: 0.0 for name, _ in components}
    event_groups: list[list[Any]] = []

    torch.cuda.synchronize()
    for _ in range(repeat):
        events = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(len(components) + 1)
        ]
        events[0].record()
        for index, (_, fn) in enumerate(components):
            fn()
            events[index + 1].record()
        event_groups.append(events)
    torch.cuda.synchronize()

    for events in event_groups:
        for index, (name, _) in enumerate(components):
            timings_ms[name] += events[index].elapsed_time(events[index + 1])
    return timings_ms


def _summarize_monitor_results(
    monitor_results: list[dict[str, Any]],
) -> dict[str, float | int]:
    if not monitor_results:
        return {
            "avg_power_watts": 0.0,
            "max_power_watts": 0.0,
            "avg_gpu_clock_mhz": 0.0,
            "max_gpu_clock_mhz": 0.0,
            "monitor_sample_count": 0,
        }

    return {
        "avg_power_watts": sum(
            record["power_watts"] for record in monitor_results
        )
        / len(monitor_results),
        "max_power_watts": max(
            record["power_watts"] for record in monitor_results
        ),
        "avg_gpu_clock_mhz": sum(
            record["gpu_clock_mhz"] for record in monitor_results
        )
        / len(monitor_results),
        "max_gpu_clock_mhz": max(
            record["gpu_clock_mhz"] for record in monitor_results
        ),
        "monitor_sample_count": len(monitor_results),
    }


def _format_float(value: float | int | None, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _row_gemm_tflops(row: dict[str, Any]) -> float | None:
    combo_value = row.get("combo_gemm_tflops_s")
    if combo_value not in (None, ""):
        return float(combo_value)
    if str(row.get("throughput_unit", "")).startswith("TFLOPs/s"):
        return float(row["throughput_value"])
    return None


def _row_norm_iter_ms(row: dict[str, Any]) -> float | None:
    value = row.get("combo_non_gemm_iter_time_ms")
    if value in (None, ""):
        return None
    return float(value)


def _row_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["op"]): row for row in rows}


def write_summary_csv(
    rows: list[dict[str, Any]],
    summary_csv_path: Path,
) -> None:
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    fieldname: row.get(fieldname, "")
                    for fieldname in SUMMARY_FIELDNAMES
                }
            )


def write_metadata_json(metadata_path: Path, metadata: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def build_benchmark_markdown(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    row_by_op = _row_lookup(rows)
    rows_by_base = _rows_by_base(rows)
    output_dir = metadata.get("output_dir")
    summary_csv = metadata.get("summary_csv")
    metadata_json = metadata.get("metadata_json")
    benchmark_markdown = metadata.get("benchmark_markdown")
    source_output_dir = metadata.get("source_output_dir")
    output_dir_display = (
        _display_path(Path(output_dir)) if output_dir else "n/a"
    )
    summary_csv_display = (
        _display_path(Path(summary_csv)) if summary_csv else "n/a"
    )
    metadata_json_display = (
        _display_path(Path(metadata_json)) if metadata_json else "n/a"
    )
    benchmark_markdown_display = (
        _display_path(Path(benchmark_markdown)) if benchmark_markdown else "n/a"
    )
    source_output_dir_display = (
        _display_path(Path(source_output_dir)) if source_output_dir else None
    )
    lines = [
        "# Llama FlashInfer Operator Microbenchmark",
        "",
        f"Generated at `{_utc_now_iso()}`.",
        "",
        "## Summary",
        "",
        f"- Model: `{metadata['model']}`",
        f"- Prompt length: `{metadata['prompt_len']}`",
        f"- Dtype: `{metadata['dtype']}`",
        f"- Ops: `{', '.join(metadata['ops'])}`",
        f"- Result directory: `{output_dir_display}`",
        f"- Summary CSV: `{summary_csv_display}`",
        f"- Metadata: `{metadata_json_display}`",
        f"- Report: `{benchmark_markdown_display}`",
        f"- Shape source: `{metadata['shape_source']}`",
    ]
    if metadata.get("stack_depths"):
        lines.append(
            f"- Stack depths: `{', '.join(str(depth) for depth in metadata['stack_depths'])}`"
        )
    if (
        source_output_dir_display is not None
        and source_output_dir_display != output_dir_display
    ):
        lines.append(f"- Source run directory: `{source_output_dir_display}`")
        lines.append(
            "- This directory is the git-tracked latest snapshot of that source run."
        )
    for op in metadata["ops"]:
        description = OP_DESCRIPTIONS.get(op)
        if description is not None:
            lines.append(f"- {description}")
    lines.extend(
        [
            "- Attention combo workloads intentionally exclude RoPE and q/k norm so the benchmark isolates the requested operators.",
            "- Attention FLOPs use the causal lower-triangular prefill mask, not dense `S x S` attention.",
            "- Gate-up/down workloads omit `silu_and_mul`; `down` directly consumes the up half of the concatenated gate-up output.",
            "- The new power-focused faithful workloads below include RoPE, separate q/k/v and gate/up projections, `silu_and_mul`, and the current end-to-end residual-norm ordering where applicable.",
            "- The main comparison metrics below are `GEMM TFLOPs/s`, `Avg Power`, and `Avg GPU Clock`.",
            f"- CUDA_VISIBLE_DEVICES: `{metadata['environment'].get('cuda_visible_devices')}`",
            f"- Monitor GPU index: `{metadata['monitor_gpu_index']}`",
            f"- Warmup / probe / target timed seconds: "
            f"`{metadata['warmup']}` / `{metadata['probe_repeat']}` / "
            f"`{metadata['target_timed_seconds']}`",
            "",
            "## w/o Norm",
            "",
            "| Workload | GEMM TFLOPs/s | Avg Power (W) | Avg GPU Clock (MHz) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for op in NORM_FREE_OPS:
        row = _single_row_by_base(rows_by_base, op)
        if row is None:
            continue
        lines.append(
            f"| {DISPLAY_LABELS.get(op, op)} | "
            f"{_format_float(_row_gemm_tflops(row))} | "
            f"{_format_float(row['avg_power_watts'])} | "
            f"{_format_float(row['avg_gpu_clock_mhz'])} |"
        )

    lines.extend(
        [
            "",
            "## w/ Norm",
            "",
            "| Workload | Norm Time / Iter (ms) | GEMM TFLOPs/s | Avg Power (W) | Avg GPU Clock (MHz) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for op in WITH_NORM_OPS:
        row = _single_row_by_base(rows_by_base, op)
        if row is None:
            continue
        lines.append(
            f"| {DISPLAY_LABELS.get(op, op)} | "
            f"{_format_float(_row_norm_iter_ms(row))} | "
            f"{_format_float(_row_gemm_tflops(row))} | "
            f"{_format_float(row['avg_power_watts'])} | "
            f"{_format_float(row['avg_gpu_clock_mhz'])} |"
        )

    norm_only_row = _single_row_by_base(rows_by_base, "fused_add_norm")
    if norm_only_row is not None:
        lines.extend(
            [
                "",
                "## Norm Only",
                "",
                "| Op | Iter Time (ms) | Throughput | Avg Power (W) | Avg GPU Clock (MHz) |",
                "| --- | ---: | ---: | ---: | ---: |",
                (
                    f"| {DISPLAY_LABELS.get('fused_add_norm', 'fused_add_norm')} | "
                    f"{_format_float(norm_only_row['iter_time_ms'])} | "
                    f"{_format_float(norm_only_row['throughput_value'])} {norm_only_row['throughput_unit']} | "
                    f"{_format_float(norm_only_row['avg_power_watts'])} | "
                    f"{_format_float(norm_only_row['avg_gpu_clock_mhz'])} |"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## w/o vs w/ Norm",
            "",
            "| Workload | w/o Norm GEMM TFLOPs/s | w/ Norm GEMM TFLOPs/s | GEMM Delta (%) | w/o Avg Power (W) | w/ Avg Power (W) | Power Delta (W) | w/o Avg GPU Clock (MHz) | w/ Avg GPU Clock (MHz) | Clock Delta (MHz) | Norm Time / Iter (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for without_norm_op, with_norm_op in PAIRINGS:
        without_norm_row = _single_row_by_base(rows_by_base, without_norm_op)
        with_norm_row = _single_row_by_base(rows_by_base, with_norm_op)
        if without_norm_row is None or with_norm_row is None:
            continue
        without_norm_tflops = _row_gemm_tflops(without_norm_row)
        with_norm_tflops = _row_gemm_tflops(with_norm_row)
        gemm_delta_percent = None
        if without_norm_tflops not in (None, 0):
            gemm_delta_percent = (
                (with_norm_tflops - without_norm_tflops)
                / without_norm_tflops
                * 100.0
            )
        power_delta_watts = float(with_norm_row["avg_power_watts"]) - float(
            without_norm_row["avg_power_watts"]
        )
        clock_delta_mhz = float(with_norm_row["avg_gpu_clock_mhz"]) - float(
            without_norm_row["avg_gpu_clock_mhz"]
        )
        lines.append(
            f"| {DISPLAY_LABELS.get(without_norm_op, without_norm_op)} | "
            f"{_format_float(without_norm_tflops)} | "
            f"{_format_float(with_norm_tflops)} | "
            f"{_format_float(gemm_delta_percent)} | "
            f"{_format_float(without_norm_row['avg_power_watts'])} | "
            f"{_format_float(with_norm_row['avg_power_watts'])} | "
            f"{_format_float(power_delta_watts)} | "
            f"{_format_float(without_norm_row['avg_gpu_clock_mhz'])} | "
            f"{_format_float(with_norm_row['avg_gpu_clock_mhz'])} | "
            f"{_format_float(clock_delta_mhz)} | "
            f"{_format_float(_row_norm_iter_ms(with_norm_row))} |"
        )

    power_focused_rows = [
        row_by_op[row["op"]]
        for baseline_op, replace_op in POWER_FOCUSED_PAIRS
        for row in (
            _single_row_by_base(rows_by_base, baseline_op),
            _single_row_by_base(rows_by_base, replace_op),
        )
        if row is not None
    ]
    if power_focused_rows:
        lines.extend(
            [
                "",
                "## Power-Focused Workloads",
                "",
                "- `steady block` uses one decoder block with the same operator ordering as the current Llama prefill path.",
                "- `stack + ...` repeats a shared layer-weight set for the model layer count to preserve operator cadence without requiring full model weights in memory.",
                "",
                "| Workload | Variant | Total TFLOPs/s(eq) | Iter Time (ms) | Avg Power (W) | Avg GPU Clock (MHz) |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for baseline_op, replace_op in POWER_FOCUSED_PAIRS:
            workload_label = DISPLAY_LABELS.get(baseline_op, baseline_op)
            baseline_row = _single_row_by_base(rows_by_base, baseline_op)
            replace_row = _single_row_by_base(rows_by_base, replace_op)
            if baseline_row is not None:
                lines.append(
                    f"| {workload_label} | baseline | "
                    f"{_format_float(_row_total_tflops(baseline_row))} | "
                    f"{_format_float(baseline_row['iter_time_ms'])} | "
                    f"{_format_float(baseline_row['avg_power_watts'])} | "
                    f"{_format_float(baseline_row['avg_gpu_clock_mhz'])} |"
                )
            if replace_row is not None:
                lines.append(
                    f"| {workload_label} | replace_ln | "
                    f"{_format_float(_row_total_tflops(replace_row))} | "
                    f"{_format_float(replace_row['iter_time_ms'])} | "
                    f"{_format_float(replace_row['avg_power_watts'])} | "
                    f"{_format_float(replace_row['avg_gpu_clock_mhz'])} |"
                )

        lines.extend(
            [
                "",
                "## Power-Focused Deltas",
                "",
                "| Workload | baseline TFLOPs/s(eq) | replace_ln TFLOPs/s(eq) | TFLOPs Delta (%) | baseline Avg Power (W) | replace_ln Avg Power (W) | Power Delta (W) | baseline Avg GPU Clock (MHz) | replace_ln Avg GPU Clock (MHz) | Clock Delta (MHz) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for baseline_op, replace_op in POWER_FOCUSED_PAIRS:
            baseline_row = _single_row_by_base(rows_by_base, baseline_op)
            replace_row = _single_row_by_base(rows_by_base, replace_op)
            if baseline_row is None or replace_row is None:
                continue
            baseline_tflops = _row_total_tflops(baseline_row)
            replace_tflops = _row_total_tflops(replace_row)
            tflops_delta_percent = None
            if baseline_tflops not in (None, 0):
                tflops_delta_percent = (
                    (replace_tflops - baseline_tflops) / baseline_tflops * 100.0
                )
            power_delta_watts = float(replace_row["avg_power_watts"]) - float(
                baseline_row["avg_power_watts"]
            )
            clock_delta_mhz = float(replace_row["avg_gpu_clock_mhz"]) - float(
                baseline_row["avg_gpu_clock_mhz"]
            )
            lines.append(
                f"| {DISPLAY_LABELS.get(baseline_op, baseline_op)} | "
                f"{_format_float(baseline_tflops)} | "
                f"{_format_float(replace_tflops)} | "
                f"{_format_float(tflops_delta_percent)} | "
                f"{_format_float(baseline_row['avg_power_watts'])} | "
                f"{_format_float(replace_row['avg_power_watts'])} | "
                f"{_format_float(power_delta_watts)} | "
                f"{_format_float(baseline_row['avg_gpu_clock_mhz'])} | "
                f"{_format_float(replace_row['avg_gpu_clock_mhz'])} | "
                f"{_format_float(clock_delta_mhz)} |"
            )

    depth_sweep_pairs: list[tuple[str, str, list[int]]] = []
    for baseline_op, replace_op in POWER_FOCUSED_PAIRS:
        baseline_rows = rows_by_base.get(baseline_op, [])
        replace_rows = rows_by_base.get(replace_op, [])
        baseline_by_depth = {
            depth: row
            for row in baseline_rows
            if (depth := _row_stack_depth(row)) is not None
        }
        replace_by_depth = {
            depth: row
            for row in replace_rows
            if (depth := _row_stack_depth(row)) is not None
        }
        shared_depths = sorted(
            set(baseline_by_depth.keys()) & set(replace_by_depth.keys())
        )
        if len(shared_depths) >= 2:
            depth_sweep_pairs.append((baseline_op, replace_op, shared_depths))
    if depth_sweep_pairs:
        lines.extend(
            [
                "",
                "## Stack Depth Sweep",
                "",
                "| Workload | Depth | baseline TFLOPs/s(eq) | replace_ln TFLOPs/s(eq) | TFLOPs Delta (%) | baseline Avg Power (W) | replace_ln Avg Power (W) | Power Delta (W) | baseline Avg GPU Clock (MHz) | replace_ln Avg GPU Clock (MHz) | Clock Delta (MHz) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for baseline_op, replace_op, depths in depth_sweep_pairs:
            baseline_by_depth = {
                _row_stack_depth(row): row
                for row in rows_by_base.get(baseline_op, [])
                if _row_stack_depth(row) is not None
            }
            replace_by_depth = {
                _row_stack_depth(row): row
                for row in rows_by_base.get(replace_op, [])
                if _row_stack_depth(row) is not None
            }
            for depth in depths:
                baseline_row = baseline_by_depth[depth]
                replace_row = replace_by_depth[depth]
                baseline_tflops = _row_total_tflops(baseline_row)
                replace_tflops = _row_total_tflops(replace_row)
                tflops_delta_percent = None
                if baseline_tflops not in (None, 0):
                    tflops_delta_percent = (
                        (replace_tflops - baseline_tflops)
                        / baseline_tflops
                        * 100.0
                    )
                power_delta_watts = float(
                    replace_row["avg_power_watts"]
                ) - float(baseline_row["avg_power_watts"])
                clock_delta_mhz = float(
                    replace_row["avg_gpu_clock_mhz"]
                ) - float(baseline_row["avg_gpu_clock_mhz"])
                lines.append(
                    f"| {DISPLAY_LABELS.get(baseline_op, baseline_op)} | "
                    f"{depth} | "
                    f"{_format_float(baseline_tflops)} | "
                    f"{_format_float(replace_tflops)} | "
                    f"{_format_float(tflops_delta_percent)} | "
                    f"{_format_float(baseline_row['avg_power_watts'])} | "
                    f"{_format_float(replace_row['avg_power_watts'])} | "
                    f"{_format_float(power_delta_watts)} | "
                    f"{_format_float(baseline_row['avg_gpu_clock_mhz'])} | "
                    f"{_format_float(replace_row['avg_gpu_clock_mhz'])} | "
                    f"{_format_float(clock_delta_mhz)} |"
                )

    component_rows = [
        row for row in rows if row.get("component_breakdown_json")
    ]
    if component_rows:
        lines.extend(
            [
                "",
                "## Component Breakdown",
                "",
                "| Op | Component | Time Total (ms) | Time / Iter (ms) | Share of Component Time (%) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in component_rows:
            component_breakdown = json.loads(row["component_breakdown_json"])
            component_total_ms = sum(
                component["time_ms"] for component in component_breakdown
            )
            for component in component_breakdown:
                share_percent = None
                if component_total_ms > 0:
                    share_percent = (
                        component["time_ms"] / component_total_ms * 100.0
                    )
                lines.append(
                    f"| {DISPLAY_LABELS.get(row['op'], row['op'])} | "
                    f"{COMPONENT_DISPLAY_LABELS.get(component['name'], component['name'])} | "
                    f"{_format_float(component['time_ms'])} | "
                    f"{_format_float(component['iter_time_ms'])} | "
                    f"{_format_float(share_percent)} |"
                )

    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: `{metadata['environment'].get('python_version')}`",
            f"- Torch: `{metadata['environment'].get('torch_version')}`",
            f"- FlashInfer: `{metadata['environment'].get('flashinfer_version')}`",
            f"- CUDA device: `{metadata['environment'].get('cuda_device_name')}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_operator_microbench(
    output_dir: Path,
    model: str,
    prompt_len: int,
    ops: list[str],
    dtype_name: str,
    warmup: int,
    probe_repeat: int,
    target_timed_seconds: float,
    max_repeat: int,
    monitor_interval: float,
    monitor_gpu_index: int,
    eps: float,
    stack_depths: list[int] | None = None,
) -> list[dict[str, Any]]:
    torch = importlib.import_module("torch")
    gpu_monitor = _load_module("monitor.gpu_monitor")
    shape = _load_llama_shape(model, prompt_len)
    dtype = _resolve_dtype(dtype_name)
    device = "cuda:0"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for operator microbenchmarking")

    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "summary.csv"
    metadata_path = output_dir / "metadata.json"
    benchmark_md_path = output_dir / "BENCHMARK.md"

    metadata: dict[str, Any] = {
        "run_started_at_utc": _utc_now_iso(),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv_path),
        "metadata_json": str(metadata_path),
        "benchmark_markdown": str(benchmark_md_path),
        "model": model,
        "prompt_len": prompt_len,
        "ops": ops,
        "dtype": dtype_name,
        "warmup": warmup,
        "probe_repeat": probe_repeat,
        "target_timed_seconds": target_timed_seconds,
        "max_repeat": max_repeat,
        "monitor_interval": monitor_interval,
        "monitor_gpu_index": monitor_gpu_index,
        "eps": eps,
        "stack_depths": stack_depths,
        "shape_source": shape.shape_source,
        "environment": collect_environment_metadata(),
    }
    write_metadata_json(metadata_path, metadata)

    rows: list[dict[str, Any]] = []
    torch.set_default_device(torch.device(device))
    torch.set_default_dtype(dtype)

    jobs: list[tuple[str, int | None]] = []
    for op in ops:
        if _is_stack_op(op) and stack_depths:
            jobs.extend((op, depth) for depth in stack_depths)
        else:
            jobs.append((op, None))

    for op, stack_depth in jobs:
        benchmark = _build_operator_benchmark(
            op, shape, dtype, device, eps, stack_depth=stack_depth
        )
        op_label = op
        if stack_depth is not None:
            op_label = f"{op}@layers={stack_depth}"
        print(
            f"Running op={op_label} input_shape={benchmark.input_shape} "
            f"weight_shape={benchmark.weight_shape}",
            flush=True,
        )

        with torch.inference_mode():
            for _ in range(max(0, warmup)):
                benchmark.run_once()
            torch.cuda.synchronize()
            repeat, probe_iter_seconds = _calibrate_repeat(
                benchmark.run_once,
                target_timed_seconds=target_timed_seconds,
                probe_repeat=probe_repeat,
                max_repeat=max_repeat,
            )

            monitor = gpu_monitor.GPUMonitor(
                gpu_index=monitor_gpu_index,
                interval=monitor_interval,
            )
            monitor.start()
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(repeat):
                benchmark.run_once()
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            monitor.stop()
            component_timings_ms: dict[str, float] | None = None
            if benchmark.components is not None:
                component_timings_ms = _measure_component_times_ms(
                    benchmark.components,
                    repeat=repeat,
                )

        monitor_results = monitor.get_results()
        monitor_summary = _summarize_monitor_results(monitor_results)
        iter_time_ms = elapsed_time * 1000.0 / repeat
        if benchmark.throughput_unit.startswith("TFLOPs/s"):
            throughput_value = (
                benchmark.throughput_per_iter / 1e12 / (elapsed_time / repeat)
            )
        elif benchmark.throughput_unit == "GiB/s":
            throughput_value = (
                benchmark.throughput_per_iter
                / (1024**3)
                / (elapsed_time / repeat)
            )
        else:
            raise ValueError(
                f"Unsupported throughput unit {benchmark.throughput_unit!r}"
            )

        component_breakdown_json: str | None = None
        combo_gemm_components: str | None = None
        combo_non_gemm_components: str | None = None
        combo_norm_component: str | None = None
        combo_gemm_time_ms: float | None = None
        combo_norm_time_ms: float | None = None
        combo_gemm_iter_time_ms: float | None = None
        combo_norm_iter_time_ms: float | None = None
        combo_gemm_tflops_s: float | None = None
        combo_non_gemm_time_ms: float | None = None
        combo_non_gemm_iter_time_ms: float | None = None
        if component_timings_ms is not None:
            ordered_components = []
            for component_name, _ in benchmark.components:
                component_time_ms = component_timings_ms[component_name]
                ordered_components.append(
                    {
                        "name": component_name,
                        "time_ms": component_time_ms,
                        "iter_time_ms": component_time_ms / repeat,
                    }
                )
            component_breakdown_json = json.dumps(ordered_components)

            gemm_component_names = [
                component_name
                for component_name, _ in benchmark.components
                if _is_gemm_like_component(component_name)
            ]
            non_gemm_component_names = [
                component_name
                for component_name, _ in benchmark.components
                if not _is_gemm_like_component(component_name)
            ]
            norm_component_names = [
                component_name
                for component_name, _ in benchmark.components
                if "norm" in component_name
            ]

            if gemm_component_names:
                combo_gemm_components = ", ".join(gemm_component_names)
                combo_gemm_time_ms = sum(
                    component_timings_ms[name] for name in gemm_component_names
                )
                combo_gemm_iter_time_ms = combo_gemm_time_ms / repeat
                if (
                    benchmark.gemm_flops_per_iter is not None
                    and combo_gemm_iter_time_ms > 0
                ):
                    combo_gemm_tflops_s = (
                        benchmark.gemm_flops_per_iter
                        / 1e12
                        / (combo_gemm_iter_time_ms / 1000.0)
                    )

            if non_gemm_component_names:
                combo_non_gemm_components = ", ".join(non_gemm_component_names)
                combo_non_gemm_time_ms = sum(
                    component_timings_ms[name]
                    for name in non_gemm_component_names
                )
                combo_non_gemm_iter_time_ms = combo_non_gemm_time_ms / repeat

            if len(norm_component_names) == 1:
                combo_norm_component = norm_component_names[0]
                combo_norm_time_ms = component_timings_ms[combo_norm_component]
                combo_norm_iter_time_ms = combo_norm_time_ms / repeat

        monitor_csv_path = monitor_dir / f"{op_label}.csv"
        if monitor_summary["monitor_sample_count"] > 0:
            monitor.export_csv(str(monitor_csv_path))
            monitor_csv = str(monitor_csv_path)
        else:
            monitor_csv = ""

        row = {
            "run_timestamp_utc": _utc_now_iso(),
            "model": model,
            "prompt_len": prompt_len,
            "dtype": dtype_name,
            "op": op_label,
            "op_base": op,
            "stack_depth": stack_depth,
            "shape_source": shape.shape_source,
            "input_shape": str(benchmark.input_shape),
            "weight_shape": str(benchmark.weight_shape),
            "output_shape": str(benchmark.output_shape),
            "hidden_size": shape.hidden_size,
            "intermediate_size": shape.intermediate_size,
            "head_dim": shape.head_dim,
            "warmup": warmup,
            "probe_repeat": probe_repeat,
            "repeat": repeat,
            "target_timed_seconds": target_timed_seconds,
            "total_time_s": elapsed_time,
            "iter_time_ms": iter_time_ms,
            "throughput_value": throughput_value,
            "throughput_unit": benchmark.throughput_unit,
            "avg_power_watts": monitor_summary["avg_power_watts"],
            "max_power_watts": monitor_summary["max_power_watts"],
            "avg_gpu_clock_mhz": monitor_summary["avg_gpu_clock_mhz"],
            "max_gpu_clock_mhz": monitor_summary["max_gpu_clock_mhz"],
            "monitor_sample_count": monitor_summary["monitor_sample_count"],
            "monitor_csv": monitor_csv,
            "probe_iter_ms": probe_iter_seconds * 1000.0,
            "component_breakdown_json": component_breakdown_json,
            "combo_gemm_components": combo_gemm_components,
            "combo_non_gemm_components": combo_non_gemm_components,
            "combo_norm_component": combo_norm_component,
            "combo_gemm_time_ms": combo_gemm_time_ms,
            "combo_norm_time_ms": combo_norm_time_ms,
            "combo_gemm_iter_time_ms": combo_gemm_iter_time_ms,
            "combo_norm_iter_time_ms": combo_norm_iter_time_ms,
            "combo_gemm_tflops_s": combo_gemm_tflops_s,
            "combo_non_gemm_time_ms": combo_non_gemm_time_ms,
            "combo_non_gemm_iter_time_ms": combo_non_gemm_iter_time_ms,
        }
        rows.append(row)
        write_summary_csv(rows, summary_csv_path)
        metadata["completed_ops"] = [
            completed_row["op"] for completed_row in rows
        ]
        write_metadata_json(metadata_path, metadata)

    benchmark_md_path.write_text(build_benchmark_markdown(rows, metadata))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Llama-shaped projection, attention, and fused_add_rmsnorm microbenchmarks."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=["7B", "13B", "34B", "70B"],
        help="Which Llama config to derive shapes from.",
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        default=DEFAULT_PROMPT_LEN,
        help="Sequence length used for the hidden-state dimension.",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=DEFAULT_OPS,
        choices=ALL_OP_CHOICES,
        help="Operators to benchmark.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["float16", "bfloat16", "float32"],
        help="Tensor dtype for the benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--probe_repeat",
        type=int,
        default=DEFAULT_PROBE_REPEAT,
        help="Probe iterations used to calibrate repeat count.",
    )
    parser.add_argument(
        "--target_timed_seconds",
        type=float,
        default=DEFAULT_TARGET_TIMED_SECONDS,
        help="Target timed duration for each operator.",
    )
    parser.add_argument(
        "--max_repeat",
        type=int,
        default=DEFAULT_MAX_REPEAT,
        help="Upper bound for the calibrated repeat count.",
    )
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=DEFAULT_MONITOR_INTERVAL,
        help="NVML sampling interval in seconds.",
    )
    parser.add_argument(
        "--monitor_gpu_index",
        type=int,
        default=None,
        help=(
            "Physical GPU index for NVML monitoring. "
            "Defaults to the first CUDA_VISIBLE_DEVICES entry when possible."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help="Epsilon used by fused_add_norm benchmark variants.",
    )
    parser.add_argument(
        "--stack_depths",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional layer-count overrides for stack_* workloads. "
            "Each listed stack op is run once per depth."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Result directory. Defaults to "
            "results/llama_operator_microbench/<UTC_TIMESTAMP>/"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    normalized_ops = _normalize_ops(args.ops)
    stack_depths = None
    if args.stack_depths is not None:
        seen_depths: set[int] = set()
        stack_depths = []
        for depth in args.stack_depths:
            if depth <= 0:
                parser.error("--stack_depths entries must be positive integers")
            if depth not in seen_depths:
                stack_depths.append(depth)
                seen_depths.add(depth)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_RESULTS_ROOT / _timestamp_for_path()

    monitor_gpu_index = _resolve_monitor_gpu_index(args.monitor_gpu_index)
    rows = run_operator_microbench(
        output_dir=output_dir,
        model=args.model,
        prompt_len=args.prompt_len,
        ops=normalized_ops,
        dtype_name=args.dtype,
        warmup=args.warmup,
        probe_repeat=args.probe_repeat,
        target_timed_seconds=args.target_timed_seconds,
        max_repeat=args.max_repeat,
        monitor_interval=args.monitor_interval,
        monitor_gpu_index=monitor_gpu_index,
        eps=args.eps,
        stack_depths=stack_depths,
    )

    print(f"Output directory: {output_dir}")
    print(f"BENCHMARK.md: {output_dir / 'BENCHMARK.md'}")
    print(f"summary.csv: {output_dir / 'summary.csv'}")
    for row in rows:
        print(
            f"{row['op']}: iter_time_ms={row['iter_time_ms']:.3f}, "
            f"{row['throughput_value']:.2f} {row['throughput_unit']}, "
            f"avg_power={row['avg_power_watts']:.2f} W, "
            f"avg_gpu_clock={row['avg_gpu_clock_mhz']:.2f} MHz"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
