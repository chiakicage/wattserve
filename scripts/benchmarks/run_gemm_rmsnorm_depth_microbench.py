#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from monitor.gpu_monitor import GPUMonitor  # noqa: E402


@dataclass(frozen=True)
class Llama13BShape:
    prompt_len: int = 8192
    hidden_size: int = 5120
    intermediate_size: int = 13824
    num_heads: int = 40
    num_kv_heads: int = 40
    head_dim: int = 128
    rope_theta: float = 10000.0

    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim


@dataclass(frozen=True)
class BenchmarkCase:
    workload: str
    variant: str
    depth: int
    run_once: Callable[[], None]
    gemm_flops_per_iter: float
    description: str


WORKLOADS = (
    "o_chain",
    "mlp_gate_down_chain",
    "mlp_gate_up_down_no_act",
    "mlp_down_up_chain",
    "mlp_chain",
    "o_mlp_chain",
    "qkv_attn_o_chain",
    "full_block_no_final",
)
VARIANTS = ("baseline", "replace_ln")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _avg(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record[key]) for record in records) / len(records)


def _calibrate_repeat(
    torch: Any,
    run_once: Callable[[], None],
    target_timed_seconds: float,
    probe_repeat: int,
    max_repeat: int,
) -> tuple[int, float]:
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
    repeat = round(target_timed_seconds / iter_seconds)
    return max(1, min(max_repeat, repeat)), iter_seconds


def _make_scaled_tensor(
    torch: Any,
    shape: tuple[int, ...],
    device: str,
    dtype: Any,
    scale: float = 0.02,
) -> Any:
    tensor = torch.randn(shape, device=device, dtype=dtype)
    tensor.mul_(scale)
    return tensor


def _gemm_flops(m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n


def _prefill_attention_flops(shape: Llama13BShape) -> float:
    # Count FlashInfer prefill attention as GEMM-like QK and PV matmuls.
    # Causal prefill uses the lower triangle, not dense S x S attention.
    causal_pairs = shape.prompt_len * (shape.prompt_len + 1) / 2.0
    qk_flops = 2.0 * causal_pairs * shape.q_size
    pv_flops = 2.0 * causal_pairs * shape.q_size
    return qk_flops + pv_flops


def _o_flops(shape: Llama13BShape) -> float:
    return _gemm_flops(shape.prompt_len, shape.q_size, shape.hidden_size)


def _qkv_flops(shape: Llama13BShape) -> float:
    return (
        _gemm_flops(shape.prompt_len, shape.hidden_size, shape.q_size)
        + _gemm_flops(shape.prompt_len, shape.hidden_size, shape.kv_size)
        + _gemm_flops(shape.prompt_len, shape.hidden_size, shape.kv_size)
    )


def _mlp_flops(shape: Llama13BShape) -> float:
    return (
        _gemm_flops(
            shape.prompt_len, shape.hidden_size, shape.intermediate_size
        )
        + _gemm_flops(
            shape.prompt_len, shape.hidden_size, shape.intermediate_size
        )
        + _gemm_flops(
            shape.prompt_len, shape.intermediate_size, shape.hidden_size
        )
    )


def _mlp_pair_flops(shape: Llama13BShape) -> float:
    return _gemm_flops(
        shape.prompt_len, shape.hidden_size, shape.intermediate_size
    ) + _gemm_flops(
        shape.prompt_len, shape.intermediate_size, shape.hidden_size
    )


def _block_flops(shape: Llama13BShape) -> float:
    return (
        _qkv_flops(shape)
        + _prefill_attention_flops(shape)
        + _o_flops(shape)
        + _mlp_flops(shape)
    )


def _alloc_common(
    torch: Any, shape: Llama13BShape, device: str, dtype: Any
) -> dict[str, Any]:
    return {
        "source": _make_scaled_tensor(
            torch, (shape.prompt_len, shape.hidden_size), device, dtype
        ),
        "hidden": _make_scaled_tensor(
            torch, (shape.prompt_len, shape.hidden_size), device, dtype
        ),
        "next_hidden": torch.empty(
            (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
        ),
        "norm_out": torch.empty(
            (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
        ),
        "residual": _make_scaled_tensor(
            torch, (shape.prompt_len, shape.hidden_size), device, dtype
        ),
        "norm_weight": torch.ones(
            (shape.hidden_size,), device=device, dtype=dtype
        ),
    }


def _maybe_reset_common(
    state: dict[str, Any], reset_copy: bool
) -> tuple[Any, Any]:
    if reset_copy:
        state["hidden"].copy_(state["source"])
        state["residual"].copy_(state["source"])
    return state["hidden"], state["next_hidden"]


def _store_common_current(
    state: dict[str, Any], current: Any, next_hidden: Any
) -> None:
    state["hidden"] = current
    state["next_hidden"] = next_hidden


def _build_o_chain(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    depth: int,
    variant: str,
    eps: float,
    reset_copy: bool,
) -> BenchmarkCase:
    state = _alloc_common(torch, shape, device, dtype)
    o_weight = _make_scaled_tensor(
        torch, (shape.q_size, shape.hidden_size), device, dtype
    )

    def run_once() -> None:
        current, next_hidden = _maybe_reset_common(state, reset_copy)
        for _ in range(depth):
            torch.mm(current, o_weight, out=next_hidden)
            if variant == "baseline":
                flashinfer.fused_add_rmsnorm(
                    next_hidden,
                    state["residual"],
                    state["norm_weight"],
                    eps=eps,
                )
            current, next_hidden = next_hidden, current
        _store_common_current(state, current, next_hidden)

    return BenchmarkCase(
        workload="o_chain",
        variant=variant,
        depth=depth,
        run_once=run_once,
        gemm_flops_per_iter=depth * _o_flops(shape),
        description="o_gemm -> optional fused_add_rmsnorm",
    )


def _build_mlp_chain(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    depth: int,
    variant: str,
    eps: float,
    reset_copy: bool,
) -> BenchmarkCase:
    state = _alloc_common(torch, shape, device, dtype)
    gate_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )
    up_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )
    down_weight = _make_scaled_tensor(
        torch, (shape.intermediate_size, shape.hidden_size), device, dtype
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

    def run_once() -> None:
        current, next_hidden = _maybe_reset_common(state, reset_copy)
        for _ in range(depth):
            torch.mm(current, gate_weight, out=gate_out)
            torch.mm(current, up_weight, out=up_out)
            cat_out[:, : shape.intermediate_size].copy_(gate_out)
            cat_out[:, shape.intermediate_size :].copy_(up_out)
            flashinfer.silu_and_mul(cat_out, out=act_out)
            torch.mm(act_out, down_weight, out=next_hidden)
            if variant == "baseline":
                flashinfer.fused_add_rmsnorm(
                    next_hidden,
                    state["residual"],
                    state["norm_weight"],
                    eps=eps,
                )
            current, next_hidden = next_hidden, current
        _store_common_current(state, current, next_hidden)

    return BenchmarkCase(
        workload="mlp_chain",
        variant=variant,
        depth=depth,
        run_once=run_once,
        gemm_flops_per_iter=depth * _mlp_flops(shape),
        description="gate + up -> silu_and_mul -> down -> optional fused_add_rmsnorm",
    )


def _build_mlp_gate_down_chain(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    depth: int,
    variant: str,
    eps: float,
    reset_copy: bool,
) -> BenchmarkCase:
    state = _alloc_common(torch, shape, device, dtype)
    gate_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )
    down_weight = _make_scaled_tensor(
        torch, (shape.intermediate_size, shape.hidden_size), device, dtype
    )
    gate_out = torch.empty(
        (shape.prompt_len, shape.intermediate_size), device=device, dtype=dtype
    )

    def run_once() -> None:
        current, next_hidden = _maybe_reset_common(state, reset_copy)
        for _ in range(depth):
            torch.mm(current, gate_weight, out=gate_out)
            torch.mm(gate_out, down_weight, out=next_hidden)
            if variant == "baseline":
                flashinfer.fused_add_rmsnorm(
                    next_hidden,
                    state["residual"],
                    state["norm_weight"],
                    eps=eps,
                )
            current, next_hidden = next_hidden, current
        _store_common_current(state, current, next_hidden)

    return BenchmarkCase(
        workload="mlp_gate_down_chain",
        variant=variant,
        depth=depth,
        run_once=run_once,
        gemm_flops_per_iter=depth * _mlp_pair_flops(shape),
        description="gate H->I -> down I->H -> optional fused_add_rmsnorm",
    )


def _build_mlp_gate_up_down_no_act(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    depth: int,
    variant: str,
    eps: float,
    reset_copy: bool,
) -> BenchmarkCase:
    state = _alloc_common(torch, shape, device, dtype)
    gate_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )
    up_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )
    down_weight = _make_scaled_tensor(
        torch, (shape.intermediate_size, shape.hidden_size), device, dtype
    )
    gate_out = torch.empty(
        (shape.prompt_len, shape.intermediate_size), device=device, dtype=dtype
    )
    up_out = torch.empty_like(gate_out)

    def run_once() -> None:
        current, next_hidden = _maybe_reset_common(state, reset_copy)
        for _ in range(depth):
            torch.mm(current, gate_weight, out=gate_out)
            torch.mm(current, up_weight, out=up_out)
            torch.mm(up_out, down_weight, out=next_hidden)
            if variant == "baseline":
                flashinfer.fused_add_rmsnorm(
                    next_hidden,
                    state["residual"],
                    state["norm_weight"],
                    eps=eps,
                )
            current, next_hidden = next_hidden, current
        _store_common_current(state, current, next_hidden)

    return BenchmarkCase(
        workload="mlp_gate_up_down_no_act",
        variant=variant,
        depth=depth,
        run_once=run_once,
        gemm_flops_per_iter=depth * _mlp_flops(shape),
        description="gate + up -> down(up) without activation -> optional fused_add_rmsnorm",
    )


def _build_mlp_down_up_chain(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    depth: int,
    variant: str,
    eps: float,
    reset_copy: bool,
) -> BenchmarkCase:
    source_i = _make_scaled_tensor(
        torch, (shape.prompt_len, shape.intermediate_size), device, dtype
    )
    current_i = _make_scaled_tensor(
        torch, (shape.prompt_len, shape.intermediate_size), device, dtype
    )
    next_i = torch.empty_like(source_i)
    hidden = torch.empty(
        (shape.prompt_len, shape.hidden_size), device=device, dtype=dtype
    )
    residual = _make_scaled_tensor(
        torch, (shape.prompt_len, shape.hidden_size), device, dtype
    )
    norm_weight = torch.ones((shape.hidden_size,), device=device, dtype=dtype)
    down_weight = _make_scaled_tensor(
        torch, (shape.intermediate_size, shape.hidden_size), device, dtype
    )
    up_weight = _make_scaled_tensor(
        torch, (shape.hidden_size, shape.intermediate_size), device, dtype
    )

    def run_once() -> None:
        nonlocal current_i, next_i
        cur_i = current_i
        nxt_i = next_i
        if reset_copy:
            cur_i.copy_(source_i)
        for _ in range(depth):
            torch.mm(cur_i, down_weight, out=hidden)
            if variant == "baseline":
                flashinfer.fused_add_rmsnorm(
                    hidden, residual, norm_weight, eps=eps
                )
            torch.mm(hidden, up_weight, out=nxt_i)
            cur_i, nxt_i = nxt_i, cur_i
        current_i, next_i = cur_i, nxt_i

    return BenchmarkCase(
        workload="mlp_down_up_chain",
        variant=variant,
        depth=depth,
        run_once=run_once,
        gemm_flops_per_iter=depth * _mlp_pair_flops(shape),
        description="down I->H -> optional fused_add_rmsnorm -> up H->I",
    )


def _build_o_mlp_chain(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    depth: int,
    variant: str,
    eps: float,
    reset_copy: bool,
) -> BenchmarkCase:
    state = _alloc_common(torch, shape, device, dtype)
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
    o_out = torch.empty_like(state["hidden"])
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

    def run_once() -> None:
        current, next_hidden = _maybe_reset_common(state, reset_copy)
        for _ in range(depth):
            torch.mm(current, o_weight, out=o_out)
            mlp_input = o_out
            if variant == "baseline":
                flashinfer.fused_add_rmsnorm(
                    o_out, state["residual"], state["norm_weight"], eps=eps
                )
            torch.mm(mlp_input, gate_weight, out=gate_out)
            torch.mm(mlp_input, up_weight, out=up_out)
            cat_out[:, : shape.intermediate_size].copy_(gate_out)
            cat_out[:, shape.intermediate_size :].copy_(up_out)
            flashinfer.silu_and_mul(cat_out, out=act_out)
            torch.mm(act_out, down_weight, out=next_hidden)
            if variant == "baseline":
                flashinfer.fused_add_rmsnorm(
                    next_hidden,
                    state["residual"],
                    state["norm_weight"],
                    eps=eps,
                )
            current, next_hidden = next_hidden, current
        _store_common_current(state, current, next_hidden)

    return BenchmarkCase(
        workload="o_mlp_chain",
        variant=variant,
        depth=depth,
        run_once=run_once,
        gemm_flops_per_iter=depth * (_o_flops(shape) + _mlp_flops(shape)),
        description=(
            "o_gemm -> optional post_attn_fused_add_rmsnorm -> "
            "gate/up -> silu -> down -> optional layer_boundary_fused_add_rmsnorm"
        ),
    )


def _build_qkv_attn_o_chain(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    depth: int,
    variant: str,
    eps: float,
    reset_copy: bool,
) -> BenchmarkCase:
    state = _alloc_common(torch, shape, device, dtype)
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
    q_out = torch.empty(
        (shape.prompt_len, shape.q_size), device=device, dtype=dtype
    )
    k_out = torch.empty(
        (shape.prompt_len, shape.kv_size), device=device, dtype=dtype
    )
    v_out = torch.empty(
        (shape.prompt_len, shape.kv_size), device=device, dtype=dtype
    )
    q_view = q_out.view(shape.prompt_len, shape.num_heads, shape.head_dim)
    k_view = k_out.view(shape.prompt_len, shape.num_kv_heads, shape.head_dim)
    v_view = v_out.view(shape.prompt_len, shape.num_kv_heads, shape.head_dim)
    positions = torch.arange(shape.prompt_len, device=device)

    def run_once() -> None:
        current, next_hidden = _maybe_reset_common(state, reset_copy)
        for _ in range(depth):
            torch.mm(current, q_weight, out=q_out)
            torch.mm(current, k_weight, out=k_out)
            torch.mm(current, v_weight, out=v_out)
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
                out=next_hidden,
            )
            if variant == "baseline":
                flashinfer.fused_add_rmsnorm(
                    next_hidden,
                    state["residual"],
                    state["norm_weight"],
                    eps=eps,
                )
            current, next_hidden = next_hidden, current
        _store_common_current(state, current, next_hidden)

    return BenchmarkCase(
        workload="qkv_attn_o_chain",
        variant=variant,
        depth=depth,
        run_once=run_once,
        gemm_flops_per_iter=depth
        * (
            _qkv_flops(shape)
            + _prefill_attention_flops(shape)
            + _o_flops(shape)
        ),
        description="q/k/v -> rope -> prefill_attention -> o -> optional fused_add_rmsnorm",
    )


def _build_full_block(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    depth: int,
    variant: str,
    eps: float,
    reset_copy: bool,
) -> BenchmarkCase:
    state = _alloc_common(torch, shape, device, dtype)
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
    o_out = torch.empty_like(state["hidden"])
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
    q_view = q_out.view(shape.prompt_len, shape.num_heads, shape.head_dim)
    k_view = k_out.view(shape.prompt_len, shape.num_kv_heads, shape.head_dim)
    v_view = v_out.view(shape.prompt_len, shape.num_kv_heads, shape.head_dim)
    positions = torch.arange(shape.prompt_len, device=device)

    def attention_and_ffn(layer_input: Any) -> None:
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
        if variant == "baseline":
            flashinfer.fused_add_rmsnorm(
                o_out, state["residual"], post_attn_norm_weight, eps=eps
            )
        torch.mm(mlp_input, gate_weight, out=gate_out)
        torch.mm(mlp_input, up_weight, out=up_out)
        cat_out[:, : shape.intermediate_size].copy_(gate_out)
        cat_out[:, shape.intermediate_size :].copy_(up_out)
        flashinfer.silu_and_mul(cat_out, out=act_out)
        torch.mm(act_out, down_weight, out=state["hidden"])

    def run_once() -> None:
        if reset_copy:
            state["hidden"].copy_(state["source"])
        if variant == "replace_ln":
            for _ in range(depth):
                attention_and_ffn(state["hidden"])
            return

        flashinfer.rmsnorm(
            state["hidden"], input_norm_weight, eps=eps, out=state["norm_out"]
        )
        if reset_copy:
            state["residual"].copy_(state["hidden"])
        attention_and_ffn(state["norm_out"])
        for _ in range(1, depth):
            flashinfer.fused_add_rmsnorm(
                state["hidden"], state["residual"], input_norm_weight, eps=eps
            )
            attention_and_ffn(state["hidden"])

    return BenchmarkCase(
        workload="full_block_no_final",
        variant=variant,
        depth=depth,
        run_once=run_once,
        gemm_flops_per_iter=depth * _block_flops(shape),
        description="faithful block cadence with input/post-attn norm boundaries",
    )


def _build_case(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    workload: str,
    variant: str,
    depth: int,
    eps: float,
    reset_copy: bool,
) -> BenchmarkCase:
    if workload == "o_chain":
        return _build_o_chain(
            torch,
            flashinfer,
            shape,
            device,
            dtype,
            depth,
            variant,
            eps,
            reset_copy,
        )
    if workload == "mlp_gate_down_chain":
        return _build_mlp_gate_down_chain(
            torch,
            flashinfer,
            shape,
            device,
            dtype,
            depth,
            variant,
            eps,
            reset_copy,
        )
    if workload == "mlp_gate_up_down_no_act":
        return _build_mlp_gate_up_down_no_act(
            torch,
            flashinfer,
            shape,
            device,
            dtype,
            depth,
            variant,
            eps,
            reset_copy,
        )
    if workload == "mlp_down_up_chain":
        return _build_mlp_down_up_chain(
            torch,
            flashinfer,
            shape,
            device,
            dtype,
            depth,
            variant,
            eps,
            reset_copy,
        )
    if workload == "mlp_chain":
        return _build_mlp_chain(
            torch,
            flashinfer,
            shape,
            device,
            dtype,
            depth,
            variant,
            eps,
            reset_copy,
        )
    if workload == "o_mlp_chain":
        return _build_o_mlp_chain(
            torch,
            flashinfer,
            shape,
            device,
            dtype,
            depth,
            variant,
            eps,
            reset_copy,
        )
    if workload == "qkv_attn_o_chain":
        return _build_qkv_attn_o_chain(
            torch,
            flashinfer,
            shape,
            device,
            dtype,
            depth,
            variant,
            eps,
            reset_copy,
        )
    if workload == "full_block_no_final":
        return _build_full_block(
            torch,
            flashinfer,
            shape,
            device,
            dtype,
            depth,
            variant,
            eps,
            reset_copy,
        )
    raise ValueError(f"unknown workload: {workload}")


def _monitor_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "avg_power_watts": _avg(records, "power_watts"),
        "max_power_watts": max(
            (float(record["power_watts"]) for record in records), default=0.0
        ),
        "avg_gpu_clock_mhz": _avg(records, "gpu_clock_mhz"),
        "max_gpu_clock_mhz": max(
            (float(record["gpu_clock_mhz"]) for record in records), default=0.0
        ),
        "monitor_sample_count": len(records),
    }


def _is_reproduced(rows: list[dict[str, Any]], workload: str) -> bool:
    by_key = {
        (row["workload"], row["variant"], int(row["depth"])): row
        for row in rows
    }
    for depth in (16, 40):
        base = by_key.get((workload, "baseline", depth))
        replace = by_key.get((workload, "replace_ln", depth))
        if base is None or replace is None:
            continue
        clock_delta = replace["avg_gpu_clock_mhz"] - base["avg_gpu_clock_mhz"]
        power_delta = replace["avg_power_watts"] - base["avg_power_watts"]
        if clock_delta >= 50.0 and power_delta < 0.0:
            return True
    return False


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Component Depth Reproduction Microbenchmark",
        "",
        f"- Shape: `Llama-13B prompt={metadata['prompt_len']} hidden={metadata['hidden_size']} intermediate={metadata['intermediate_size']}`",
        f"- Dtype: `{metadata['dtype']}`",
        f"- Monitor GPU index: `{metadata['monitor_gpu_index']}`",
        f"- Reset copy per `run_once()`: `{metadata['reset_copy']}`",
        f"- Warmup / probe / target timed seconds: `{metadata['warmup']}` / `{metadata['probe_repeat']}` / `{metadata['target_timed_seconds']}`",
        "",
        "## Results",
        "",
        "- Throughput is `GEMM TFLOPs/s`: projection GEMMs plus attention QK/PV matmuls.",
        "- Attention GEMM FLOPs use the causal prefill lower triangle, not dense `S x S` attention.",
        "- Norm, RoPE, activation, and copy kernels are timed but excluded from FLOPs.",
        "",
        "| Workload | Variant | Depth | GEMM TFLOPs/s | Iter Time (ms) | Avg Power (W) | Avg GPU Clock (MHz) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['workload']} | {row['variant']} | {row['depth']} | "
            f"{row['gemm_tflops_s']:.2f} | {row['iter_time_ms']:.3f} | "
            f"{row['avg_power_watts']:.2f} | {row['avg_gpu_clock_mhz']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Baseline vs Replace_ln Deltas",
            "",
            "| Workload | Depth | Baseline GEMM TFLOPs/s | Replace GEMM TFLOPs/s | GEMM TFLOPs Delta (%) | Power Delta (W) | Clock Delta (MHz) | Reproduced? |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    by_key = {
        (row["workload"], row["variant"], int(row["depth"])): row
        for row in rows
    }
    for workload in metadata["workloads"]:
        reproduced = _is_reproduced(rows, workload)
        for depth in metadata["depths"]:
            base = by_key.get((workload, "baseline", depth))
            replace = by_key.get((workload, "replace_ln", depth))
            if base is None or replace is None:
                continue
            tflops_delta = (
                (replace["gemm_tflops_s"] - base["gemm_tflops_s"])
                / base["gemm_tflops_s"]
                * 100.0
            )
            power_delta = replace["avg_power_watts"] - base["avg_power_watts"]
            clock_delta = (
                replace["avg_gpu_clock_mhz"] - base["avg_gpu_clock_mhz"]
            )
            lines.append(
                f"| {workload} | {depth} | {base['gemm_tflops_s']:.2f} | "
                f"{replace['gemm_tflops_s']:.2f} | {tflops_delta:.2f} | "
                f"{power_delta:.2f} | {clock_delta:.2f} | "
                f"{'yes' if reproduced else 'no'} |"
            )

    first_repro = next(
        (
            workload
            for workload in metadata["workloads"]
            if _is_reproduced(rows, workload)
        ),
        None,
    )
    lines.extend(
        [
            "",
            "## Interpretation Helper",
            "",
            "- Reproduced means depth 16 or 40 has replace_ln clock at least 50 MHz higher and replace_ln power lower than baseline.",
            f"- First reproduced workload: `{first_repro or 'none'}`.",
            "",
        ]
    )
    (output_dir / "BENCHMARK.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = getattr(torch, args.dtype)
    device = "cuda:0"
    torch.set_default_device(torch.device(device))
    torch.set_default_dtype(dtype)

    shape = Llama13BShape(
        prompt_len=args.prompt_len,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        rope_theta=args.rope_theta,
    )
    output_dir = args.output_dir or (
        REPO_ROOT / "results" / "component_depth_repro" / _utc_stamp()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for workload in args.workloads:
        for variant in VARIANTS:
            for depth in args.depths:
                case = _build_case(
                    torch=torch,
                    flashinfer=flashinfer,
                    shape=shape,
                    device=device,
                    dtype=dtype,
                    workload=workload,
                    variant=variant,
                    depth=depth,
                    eps=args.eps,
                    reset_copy=args.reset_copy,
                )
                label = f"{workload}/{variant}@depth={depth}"
                print(f"Running {label}", flush=True)

                with torch.inference_mode():
                    for _ in range(args.warmup):
                        case.run_once()
                    torch.cuda.synchronize()
                    repeat, probe_iter_seconds = _calibrate_repeat(
                        torch,
                        case.run_once,
                        target_timed_seconds=args.target_timed_seconds,
                        probe_repeat=args.probe_repeat,
                        max_repeat=args.max_repeat,
                    )

                    monitor = GPUMonitor(
                        gpu_index=args.monitor_gpu_index,
                        interval=args.monitor_interval,
                    )
                    monitor.start()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(repeat):
                        case.run_once()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    monitor.stop()

                records = monitor.get_results()
                monitor_csv = (
                    monitor_dir / f"{workload}__{variant}__depth={depth}.csv"
                )
                monitor.export_csv(str(monitor_csv))
                iter_time = elapsed / repeat
                row = {
                    "workload": workload,
                    "variant": variant,
                    "depth": depth,
                    "repeat": repeat,
                    "total_time_s": elapsed,
                    "iter_time_ms": iter_time * 1000.0,
                    "probe_iter_ms": probe_iter_seconds * 1000.0,
                    "gemm_tflops_s": case.gemm_flops_per_iter
                    / 1e12
                    / iter_time,
                    "gemm_flops_per_iter": case.gemm_flops_per_iter,
                    "description": case.description,
                    "reset_copy": args.reset_copy,
                    "monitor_csv": str(monitor_csv),
                    **_monitor_summary(records),
                }
                rows.append(row)
                print(
                    f"{label}: {row['gemm_tflops_s']:.2f} GEMM TFLOPs/s, "
                    f"{row['avg_power_watts']:.2f} W, "
                    f"{row['avg_gpu_clock_mhz']:.2f} MHz, "
                    f"{row['iter_time_ms']:.3f} ms",
                    flush=True,
                )

    metadata = {
        "prompt_len": shape.prompt_len,
        "hidden_size": shape.hidden_size,
        "intermediate_size": shape.intermediate_size,
        "num_heads": shape.num_heads,
        "num_kv_heads": shape.num_kv_heads,
        "head_dim": shape.head_dim,
        "rope_theta": shape.rope_theta,
        "dtype": args.dtype,
        "depths": args.depths,
        "workloads": args.workloads,
        "variants": list(VARIANTS),
        "warmup": args.warmup,
        "probe_repeat": args.probe_repeat,
        "target_timed_seconds": args.target_timed_seconds,
        "max_repeat": args.max_repeat,
        "monitor_gpu_index": args.monitor_gpu_index,
        "monitor_interval": args.monitor_interval,
        "eps": args.eps,
        "reset_copy": args.reset_copy,
        "throughput_unit": "GEMM TFLOPs/s",
        "attention_flops": (
            "FlashInfer prefill attention is counted as GEMM-like QK/PV matmuls "
            "with causal lower-triangular pairs S*(S+1)/2."
        ),
        "excluded_from_flops": ["norm", "rope", "activation", "copy"],
        "output_dir": str(output_dir),
    }
    _write_outputs(output_dir, rows, metadata)
    print(f"Wrote {output_dir / 'BENCHMARK.md'}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Component-incremental depth sweep for Llama-shaped power effects."
    )
    parser.add_argument("--prompt-len", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=13824)
    parser.add_argument("--num-heads", type=int, default=40)
    parser.add_argument("--num-kv-heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 40]
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS)
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--probe-repeat", type=int, default=10)
    parser.add_argument("--target-timed-seconds", type=float, default=2.0)
    parser.add_argument("--max-repeat", type=int, default=200)
    parser.add_argument("--monitor-interval", type=float, default=0.01)
    parser.add_argument("--monitor-gpu-index", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--no-reset-copy",
        action="store_false",
        dest="reset_copy",
        help=(
            "Do not copy source tensors into hidden/residual at each run_once; "
            "state is initialized once and then carried across timed repeats."
        ),
    )
    parser.set_defaults(reset_copy=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
