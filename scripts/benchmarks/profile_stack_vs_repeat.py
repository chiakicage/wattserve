#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmarks.run_gemm_rmsnorm_depth_microbench import (  # noqa: E402
    Llama13BShape,
    _alloc_common,
    _build_case,
    _make_scaled_tensor,
)


def _nvtx_range(torch: Any, name: str):
    try:
        return torch.cuda.nvtx.range(name)
    except AttributeError:
        return torch.autograd.profiler.record_function(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short Nsight Systems target for stack depth vs repeated single-block runs."
    )
    parser.add_argument(
        "--mode",
        choices=("stack", "repeat-single", "repeat-single-no-reset"),
        required=True,
    )
    parser.add_argument("--workload", default="full_block_no_final")
    parser.add_argument(
        "--variant", choices=("baseline", "replace_ln"), default="baseline"
    )
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=13824)
    parser.add_argument("--num-heads", type=int, default=40)
    parser.add_argument("--num-kv-heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def _build_full_block_no_reset_chain(
    torch: Any,
    flashinfer: Any,
    shape: Llama13BShape,
    device: str,
    dtype: Any,
    variant: str,
    eps: float,
) -> tuple[Callable[[], None], Callable[[int], None]]:
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

    def init_chain() -> None:
        state["hidden"].copy_(state["source"])
        if variant == "baseline":
            flashinfer.rmsnorm(
                state["hidden"],
                input_norm_weight,
                eps=eps,
                out=state["norm_out"],
            )
            state["residual"].copy_(state["hidden"])

    def run_one_layer(layer_index: int) -> None:
        if variant == "replace_ln":
            attention_and_ffn(state["hidden"])
            return
        if layer_index == 0:
            attention_and_ffn(state["norm_out"])
            return
        flashinfer.fused_add_rmsnorm(
            state["hidden"], state["residual"], input_norm_weight, eps=eps
        )
        attention_and_ffn(state["hidden"])

    return init_chain, run_one_layer


def main() -> None:
    import flashinfer
    import torch

    args = parse_args()
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
    if args.mode == "repeat-single-no-reset":
        if args.workload != "full_block_no_final":
            raise ValueError(
                "repeat-single-no-reset currently supports full_block_no_final"
            )
        init_chain, run_one_layer = _build_full_block_no_reset_chain(
            torch=torch,
            flashinfer=flashinfer,
            shape=shape,
            device=device,
            dtype=dtype,
            variant=args.variant,
            eps=args.eps,
        )
        case = None
    else:
        case_depth = args.depth if args.mode == "stack" else 1
        case = _build_case(
            torch=torch,
            flashinfer=flashinfer,
            shape=shape,
            device=device,
            dtype=dtype,
            workload=args.workload,
            variant=args.variant,
            depth=case_depth,
            eps=args.eps,
        )

    with torch.inference_mode():
        for _ in range(args.warmup):
            if args.mode == "repeat-single-no-reset":
                init_chain()
                for layer_index in range(args.depth):
                    run_one_layer(layer_index)
            elif args.mode == "stack":
                assert case is not None
                case.run_once()
            else:
                assert case is not None
                for _ in range(args.depth):
                    case.run_once()
        torch.cuda.synchronize()

        for active_idx in range(args.active):
            label = (
                f"{args.mode}/{args.workload}/{args.variant}/"
                f"depth={args.depth}/active={active_idx}"
            )
            with _nvtx_range(torch, label):
                if args.mode == "repeat-single-no-reset":
                    init_chain()
                    for layer_index in range(args.depth):
                        run_one_layer(layer_index)
                elif args.mode == "stack":
                    assert case is not None
                    case.run_once()
                else:
                    assert case is not None
                    for _ in range(args.depth):
                        case.run_once()
                torch.cuda.synchronize()

    print(
        f"completed mode={args.mode} workload={args.workload} "
        f"variant={args.variant} depth={args.depth}",
        flush=True,
    )


if __name__ == "__main__":
    main()
