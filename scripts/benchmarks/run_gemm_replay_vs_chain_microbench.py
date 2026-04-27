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


MODES = ("fixed_replay", "state_chain")
CASE_KINDS = ("o", "mlp")
O_WORKLOADS = ("gemm", "gemm_fused_add_norm")
MLP_WORKLOADS = (
    "mlp_no_act",
    "mlp_no_act_fused_add_norm",
    "mlp_silu",
    "mlp_silu_fused_add_norm",
)
WORKLOADS = O_WORKLOADS + MLP_WORKLOADS


@dataclass(frozen=True)
class BenchmarkCase:
    mode: str
    workload: str
    case_kind: str
    steps_per_run: int
    run_once: Callable[[], None]
    measure_gemm_time_ms: Callable[[int], float]
    gemm_flops_per_iter: float
    description: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _gemm_flops(prompt_len: int, input_size: int, output_size: int) -> float:
    return 2.0 * prompt_len * input_size * output_size


def _case_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.case_kind == "o":
        return {
            "shape_label": (
                f"({args.prompt_len}, {args.hidden_size}) x "
                f"({args.hidden_size}, {args.hidden_size})"
            ),
            "description": "o_gemm H->H",
            "default_workloads": O_WORKLOADS,
        }
    if args.case_kind == "mlp":
        return {
            "shape_label": (
                f"gate/up: ({args.prompt_len}, {args.hidden_size}) x "
                f"({args.hidden_size}, {args.intermediate_size}); "
                f"down: ({args.prompt_len}, {args.intermediate_size}) x "
                f"({args.intermediate_size}, {args.hidden_size})"
            ),
            "description": "MLP gate + up + down GEMMs",
            "default_workloads": MLP_WORKLOADS,
        }
    raise ValueError(f"unknown case kind: {args.case_kind}")


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
    if iter_seconds <= 0.0:
        return 1, 0.0
    repeat = round(target_timed_seconds / iter_seconds)
    return max(1, min(max_repeat, repeat)), iter_seconds


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


def _build_o_case(
    torch: Any,
    flashinfer: Any,
    mode: str,
    workload: str,
    steps_per_run: int,
    prompt_len: int,
    hidden_size: int,
    device: str,
    dtype: Any,
    eps: float,
) -> BenchmarkCase:
    source = _make_scaled_tensor(
        torch, (prompt_len, hidden_size), device, dtype
    )
    current = _make_scaled_tensor(
        torch, (prompt_len, hidden_size), device, dtype
    )
    next_tensor = torch.empty_like(current)
    fixed_out = torch.empty_like(source)
    weight = _make_scaled_tensor(
        torch, (hidden_size, hidden_size), device, dtype
    )
    residual = _make_scaled_tensor(
        torch, (prompt_len, hidden_size), device, dtype
    )
    norm_weight = torch.ones((hidden_size,), device=device, dtype=dtype)
    with_norm = workload == "gemm_fused_add_norm"

    if mode == "fixed_replay":

        def run_once() -> None:
            for _ in range(steps_per_run):
                torch.mm(source, weight, out=fixed_out)
                if with_norm:
                    flashinfer.fused_add_rmsnorm(
                        fixed_out,
                        residual,
                        norm_weight,
                        eps=eps,
                    )

        def measure_gemm_time_ms(repeat: int) -> float:
            events: list[tuple[Any, Any]] = []
            torch.cuda.synchronize()
            for _ in range(repeat):
                for _ in range(steps_per_run):
                    _recorded_mm(torch, events, source, weight, fixed_out)
                    if with_norm:
                        flashinfer.fused_add_rmsnorm(
                            fixed_out,
                            residual,
                            norm_weight,
                            eps=eps,
                        )
            torch.cuda.synchronize()
            return sum(start.elapsed_time(end) for start, end in events)

        description = (
            "fixed input replay: each GEMM reads the same source tensor; "
            "GEMM output is not fed back as the next GEMM input"
        )
    elif mode == "state_chain":

        def run_once() -> None:
            nonlocal current, next_tensor
            for _ in range(steps_per_run):
                torch.mm(current, weight, out=next_tensor)
                if with_norm:
                    flashinfer.fused_add_rmsnorm(
                        next_tensor,
                        residual,
                        norm_weight,
                        eps=eps,
                    )
                current, next_tensor = next_tensor, current

        def measure_gemm_time_ms(repeat: int) -> float:
            nonlocal current, next_tensor
            events: list[tuple[Any, Any]] = []
            torch.cuda.synchronize()
            for _ in range(repeat):
                for _ in range(steps_per_run):
                    _recorded_mm(torch, events, current, weight, next_tensor)
                    if with_norm:
                        flashinfer.fused_add_rmsnorm(
                            next_tensor,
                            residual,
                            norm_weight,
                            eps=eps,
                        )
                    current, next_tensor = next_tensor, current
            torch.cuda.synchronize()
            return sum(start.elapsed_time(end) for start, end in events)

        description = (
            "state-carrying chain: each GEMM output becomes the next GEMM input, "
            "including across outer repeat iterations"
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    return BenchmarkCase(
        mode=mode,
        workload=workload,
        case_kind="o",
        steps_per_run=steps_per_run,
        run_once=run_once,
        measure_gemm_time_ms=measure_gemm_time_ms,
        gemm_flops_per_iter=steps_per_run
        * _gemm_flops(prompt_len, hidden_size, hidden_size),
        description=description,
    )


def _build_mlp_case(
    torch: Any,
    flashinfer: Any,
    mode: str,
    workload: str,
    steps_per_run: int,
    prompt_len: int,
    hidden_size: int,
    intermediate_size: int,
    device: str,
    dtype: Any,
    eps: float,
) -> BenchmarkCase:
    source = _make_scaled_tensor(
        torch, (prompt_len, hidden_size), device, dtype
    )
    current = _make_scaled_tensor(
        torch, (prompt_len, hidden_size), device, dtype
    )
    next_hidden = torch.empty_like(current)
    gate_weight = _make_scaled_tensor(
        torch, (hidden_size, intermediate_size), device, dtype
    )
    up_weight = _make_scaled_tensor(
        torch, (hidden_size, intermediate_size), device, dtype
    )
    down_weight = _make_scaled_tensor(
        torch, (intermediate_size, hidden_size), device, dtype
    )
    gate_out = torch.empty(
        (prompt_len, intermediate_size), device=device, dtype=dtype
    )
    up_out = torch.empty_like(gate_out)
    cat_out = torch.empty(
        (prompt_len, 2 * intermediate_size), device=device, dtype=dtype
    )
    act_out = torch.empty_like(gate_out)
    residual = _make_scaled_tensor(
        torch, (prompt_len, hidden_size), device, dtype
    )
    norm_weight = torch.ones((hidden_size,), device=device, dtype=dtype)
    with_silu = workload in ("mlp_silu", "mlp_silu_fused_add_norm")
    with_norm = workload in (
        "mlp_no_act_fused_add_norm",
        "mlp_silu_fused_add_norm",
    )

    def activation_input() -> Any:
        if not with_silu:
            return up_out
        cat_out[:, :intermediate_size].copy_(gate_out)
        cat_out[:, intermediate_size:].copy_(up_out)
        flashinfer.silu_and_mul(cat_out, out=act_out)
        return act_out

    def run_mlp_once(input_tensor: Any, output_tensor: Any) -> None:
        torch.mm(input_tensor, gate_weight, out=gate_out)
        torch.mm(input_tensor, up_weight, out=up_out)
        torch.mm(activation_input(), down_weight, out=output_tensor)
        if with_norm:
            flashinfer.fused_add_rmsnorm(
                output_tensor,
                residual,
                norm_weight,
                eps=eps,
            )

    def run_mlp_once_timed(
        events: list[tuple[Any, Any]],
        input_tensor: Any,
        output_tensor: Any,
    ) -> None:
        _recorded_mm(torch, events, input_tensor, gate_weight, gate_out)
        _recorded_mm(torch, events, input_tensor, up_weight, up_out)
        _recorded_mm(
            torch, events, activation_input(), down_weight, output_tensor
        )
        if with_norm:
            flashinfer.fused_add_rmsnorm(
                output_tensor,
                residual,
                norm_weight,
                eps=eps,
            )

    if mode == "fixed_replay":

        def run_once() -> None:
            for _ in range(steps_per_run):
                run_mlp_once(source, next_hidden)

        def measure_gemm_time_ms(repeat: int) -> float:
            events: list[tuple[Any, Any]] = []
            torch.cuda.synchronize()
            for _ in range(repeat):
                for _ in range(steps_per_run):
                    run_mlp_once_timed(events, source, next_hidden)
            torch.cuda.synchronize()
            return sum(start.elapsed_time(end) for start, end in events)

        description = (
            "fixed input replay: each MLP reads the same hidden source tensor; "
            "MLP output is not fed back"
        )
    elif mode == "state_chain":

        def run_once() -> None:
            nonlocal current, next_hidden
            for _ in range(steps_per_run):
                run_mlp_once(current, next_hidden)
                current, next_hidden = next_hidden, current

        def measure_gemm_time_ms(repeat: int) -> float:
            nonlocal current, next_hidden
            events: list[tuple[Any, Any]] = []
            torch.cuda.synchronize()
            for _ in range(repeat):
                for _ in range(steps_per_run):
                    run_mlp_once_timed(events, current, next_hidden)
                    current, next_hidden = next_hidden, current
            torch.cuda.synchronize()
            return sum(start.elapsed_time(end) for start, end in events)

        description = (
            "state-carrying chain: each MLP output becomes the next hidden input, "
            "including across outer repeat iterations"
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    if with_silu:
        description += "; gate + up -> silu_and_mul -> down"
    else:
        description += "; gate + up -> down(up) without activation"
    if with_norm:
        description += " -> fused_add_rmsnorm"

    return BenchmarkCase(
        mode=mode,
        workload=workload,
        case_kind="mlp",
        steps_per_run=steps_per_run,
        run_once=run_once,
        measure_gemm_time_ms=measure_gemm_time_ms,
        gemm_flops_per_iter=steps_per_run
        * (
            _gemm_flops(prompt_len, hidden_size, intermediate_size)
            + _gemm_flops(prompt_len, hidden_size, intermediate_size)
            + _gemm_flops(prompt_len, intermediate_size, hidden_size)
        ),
        description=description,
    )


def _build_case(
    torch: Any,
    flashinfer: Any,
    case_kind: str,
    mode: str,
    workload: str,
    steps_per_run: int,
    prompt_len: int,
    hidden_size: int,
    intermediate_size: int,
    device: str,
    dtype: Any,
    eps: float,
) -> BenchmarkCase:
    if case_kind == "o":
        if workload not in O_WORKLOADS:
            raise ValueError(f"{workload} is not an o_gemm workload")
        return _build_o_case(
            torch=torch,
            flashinfer=flashinfer,
            mode=mode,
            workload=workload,
            steps_per_run=steps_per_run,
            prompt_len=prompt_len,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            eps=eps,
        )
    if case_kind == "mlp":
        if workload not in MLP_WORKLOADS:
            raise ValueError(f"{workload} is not an MLP workload")
        return _build_mlp_case(
            torch=torch,
            flashinfer=flashinfer,
            mode=mode,
            workload=workload,
            steps_per_run=steps_per_run,
            prompt_len=prompt_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
            dtype=dtype,
            eps=eps,
        )
    raise ValueError(f"unknown case kind: {case_kind}")


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

    by_key = {
        (row["mode"], row["workload"], int(row["steps_per_run"])): row
        for row in rows
    }
    lines = [
        "# GEMM Replay vs State-Chain Microbenchmark",
        "",
        f"- Case: `{metadata['case_description']}`",
        f"- GEMM shape(s): `{metadata['shape_label']}`",
        f"- Dtype: `{metadata['dtype']}`",
        f"- Monitor GPU index: `{metadata['monitor_gpu_index']}`",
        f"- Warmup / probe / target timed seconds: `{metadata['warmup']}` / `{metadata['probe_repeat']}` / `{metadata['target_timed_seconds']}`",
        "",
        "## Semantics",
        "",
        "- `fixed_replay`: each run starts from the same fixed source tensor; block output is overwritten and not fed back.",
        "- `state_chain`: output is saved as the next hidden input, including across outer benchmark repeats.",
        "- `GEMM TFLOPs/s` is GEMM FLOPs divided by CUDA event time around timed `torch.mm` kernels only.",
        "- `Total Iter Time` covers the whole workload, including optional `silu_and_mul` and `fused_add_rmsnorm`, but is not used as the GEMM TFLOPs/s denominator.",
        "- For MLP, timed GEMMs are gate, up, and down. `silu_and_mul`, concatenation copies, and fused norm are excluded from the GEMM TFLOPs/s denominator.",
        "",
        "## Results",
        "",
        "| Mode | Workload | Steps / Run | GEMM TFLOPs/s | GEMM Time / Iter (ms) | Total Iter Time (ms) | Avg Power (W) | Avg GPU Clock (MHz) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['workload']} | {row['steps_per_run']} | "
            f"{row['gemm_tflops_s']:.2f} | {row['gemm_iter_time_ms']:.3f} | "
            f"{row['iter_time_ms']:.3f} | "
            f"{row['avg_power_watts']:.2f} | {row['avg_gpu_clock_mhz']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## State Chain vs Fixed Replay",
            "",
            "| Workload | Steps / Run | Fixed GEMM TFLOPs/s | Chain GEMM TFLOPs/s | GEMM Delta (%) | Power Delta (W) | Clock Delta (MHz) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for workload in metadata["workloads"]:
        for steps_per_run in metadata["steps"]:
            fixed = by_key[("fixed_replay", workload, steps_per_run)]
            chain = by_key[("state_chain", workload, steps_per_run)]
            gemm_delta = (
                (chain["gemm_tflops_s"] - fixed["gemm_tflops_s"])
                / fixed["gemm_tflops_s"]
                * 100.0
            )
            power_delta = chain["avg_power_watts"] - fixed["avg_power_watts"]
            clock_delta = (
                chain["avg_gpu_clock_mhz"] - fixed["avg_gpu_clock_mhz"]
            )
            lines.append(
                f"| {workload} | {steps_per_run} | {fixed['gemm_tflops_s']:.2f} | "
                f"{chain['gemm_tflops_s']:.2f} | {gemm_delta:.2f} | "
                f"{power_delta:.2f} | {clock_delta:.2f} |"
            )

    norm_pairs = metadata.get("norm_pairs", [])
    if norm_pairs:
        lines.extend(
            [
                "",
                "## Norm Delta",
                "",
                "| Mode | Pair | Steps / Run | Without Norm GEMM TFLOPs/s | With Norm GEMM TFLOPs/s | GEMM Delta (%) | Power Delta (W) | Clock Delta (MHz) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode in metadata["modes"]:
            for base_workload, norm_workload, pair_label in norm_pairs:
                for steps_per_run in metadata["steps"]:
                    base = by_key[(mode, base_workload, steps_per_run)]
                    norm = by_key[(mode, norm_workload, steps_per_run)]
                    gemm_delta = (
                        (norm["gemm_tflops_s"] - base["gemm_tflops_s"])
                        / base["gemm_tflops_s"]
                        * 100.0
                    )
                    power_delta = (
                        norm["avg_power_watts"] - base["avg_power_watts"]
                    )
                    clock_delta = (
                        norm["avg_gpu_clock_mhz"] - base["avg_gpu_clock_mhz"]
                    )
                    lines.append(
                        f"| {mode} | {pair_label} | {steps_per_run} | "
                        f"{base['gemm_tflops_s']:.2f} | {norm['gemm_tflops_s']:.2f} | "
                        f"{gemm_delta:.2f} | {power_delta:.2f} | {clock_delta:.2f} |"
                    )

    activation_pairs = metadata.get("activation_pairs", [])
    if activation_pairs:
        lines.extend(
            [
                "",
                "## Silu Delta",
                "",
                "| Mode | Pair | Steps / Run | No Silu GEMM TFLOPs/s | Silu GEMM TFLOPs/s | GEMM Delta (%) | Power Delta (W) | Clock Delta (MHz) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode in metadata["modes"]:
            for no_silu_workload, silu_workload, pair_label in activation_pairs:
                for steps_per_run in metadata["steps"]:
                    no_silu = by_key[(mode, no_silu_workload, steps_per_run)]
                    silu = by_key[(mode, silu_workload, steps_per_run)]
                    gemm_delta = (
                        (silu["gemm_tflops_s"] - no_silu["gemm_tflops_s"])
                        / no_silu["gemm_tflops_s"]
                        * 100.0
                    )
                    power_delta = (
                        silu["avg_power_watts"] - no_silu["avg_power_watts"]
                    )
                    clock_delta = (
                        silu["avg_gpu_clock_mhz"] - no_silu["avg_gpu_clock_mhz"]
                    )
                    lines.append(
                        f"| {mode} | {pair_label} | {steps_per_run} | "
                        f"{no_silu['gemm_tflops_s']:.2f} | {silu['gemm_tflops_s']:.2f} | "
                        f"{gemm_delta:.2f} | {power_delta:.2f} | {clock_delta:.2f} |"
                    )

    (output_dir / "BENCHMARK.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    import flashinfer
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = getattr(torch, args.dtype)
    device = "cuda:0"
    torch.set_default_device(torch.device(device))
    torch.set_default_dtype(dtype)
    case_config = _case_config(args)
    workloads = args.workloads or list(case_config["default_workloads"])
    all_norm_pairs = (
        [("gemm", "gemm_fused_add_norm", "o_gemm")]
        if args.case_kind == "o"
        else [
            ("mlp_no_act", "mlp_no_act_fused_add_norm", "mlp_no_act"),
            ("mlp_silu", "mlp_silu_fused_add_norm", "mlp_silu"),
        ]
    )
    all_activation_pairs = (
        []
        if args.case_kind == "o"
        else [
            ("mlp_no_act", "mlp_silu", "without_norm"),
            (
                "mlp_no_act_fused_add_norm",
                "mlp_silu_fused_add_norm",
                "with_norm",
            ),
        ]
    )
    norm_pairs = [
        pair
        for pair in all_norm_pairs
        if pair[0] in workloads and pair[1] in workloads
    ]
    activation_pairs = [
        pair
        for pair in all_activation_pairs
        if pair[0] in workloads and pair[1] in workloads
    ]

    output_dir = args.output_dir or (
        REPO_ROOT / "results" / "gemm_replay_vs_chain" / _utc_stamp()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for mode in args.modes:
        for workload in workloads:
            for steps_per_run in args.steps:
                case = _build_case(
                    torch=torch,
                    flashinfer=flashinfer,
                    case_kind=args.case_kind,
                    mode=mode,
                    workload=workload,
                    steps_per_run=steps_per_run,
                    prompt_len=args.prompt_len,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.intermediate_size,
                    device=device,
                    dtype=dtype,
                    eps=args.eps,
                )
                label = f"{mode}/{workload}@steps={steps_per_run}"
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

                    gemm_time_ms = case.measure_gemm_time_ms(repeat)

                records = monitor.get_results()
                monitor_csv = (
                    monitor_dir
                    / f"{mode}__{workload}__steps={steps_per_run}.csv"
                )
                monitor.export_csv(str(monitor_csv))
                iter_time = elapsed / repeat
                gemm_time_s = gemm_time_ms / 1000.0
                gemm_iter_time = gemm_time_s / repeat
                row = {
                    "mode": mode,
                    "workload": workload,
                    "case_kind": case.case_kind,
                    "steps_per_run": steps_per_run,
                    "repeat": repeat,
                    "total_time_s": elapsed,
                    "iter_time_ms": iter_time * 1000.0,
                    "probe_iter_ms": probe_iter_seconds * 1000.0,
                    "gemm_time_s": gemm_time_s,
                    "gemm_iter_time_ms": gemm_iter_time * 1000.0,
                    "gemm_step_time_ms": (
                        gemm_time_s / (repeat * steps_per_run) * 1000.0
                    ),
                    "gemm_tflops_s": (
                        case.gemm_flops_per_iter * repeat / 1e12 / gemm_time_s
                    ),
                    "gemm_flops_per_iter": case.gemm_flops_per_iter,
                    "description": case.description,
                    "monitor_csv": str(monitor_csv),
                    **_monitor_summary(records),
                }
                rows.append(row)
                print(
                    f"{label}: {row['gemm_tflops_s']:.2f} GEMM TFLOPs/s, "
                    f"{row['gemm_iter_time_ms']:.3f} ms GEMM/iter, "
                    f"{row['avg_power_watts']:.2f} W, "
                    f"{row['avg_gpu_clock_mhz']:.2f} MHz, "
                    f"{row['iter_time_ms']:.3f} ms total/iter",
                    flush=True,
                )

    metadata = {
        "prompt_len": args.prompt_len,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "case_kind": args.case_kind,
        "case_description": case_config["description"],
        "shape_label": case_config["shape_label"],
        "dtype": args.dtype,
        "modes": args.modes,
        "workloads": workloads,
        "steps": args.steps,
        "warmup": args.warmup,
        "probe_repeat": args.probe_repeat,
        "target_timed_seconds": args.target_timed_seconds,
        "max_repeat": args.max_repeat,
        "monitor_gpu_index": args.monitor_gpu_index,
        "monitor_interval": args.monitor_interval,
        "eps": args.eps,
        "throughput_unit": "GEMM-only TFLOPs/s",
        "gemm_tflops_definition": (
            "GEMM FLOPs divided by CUDA event time around timed torch.mm kernels "
            "only; non-GEMM work is excluded from both numerator and denominator."
        ),
        "excluded_from_flops": [
            "silu_and_mul",
            "activation_copy",
            "fused_add_rmsnorm",
        ],
        "norm_pairs": norm_pairs,
        "activation_pairs": activation_pairs,
        "output_dir": str(output_dir),
    }
    _write_outputs(output_dir, rows, metadata)
    print(f"Wrote {output_dir / 'BENCHMARK.md'}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed input replay and state-carrying GEMM chains."
    )
    parser.add_argument("--prompt-len", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=13824)
    parser.add_argument("--case-kind", choices=CASE_KINDS, default="o")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=list(MODES)
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOADS,
        default=None,
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[1])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--probe-repeat", type=int, default=10)
    parser.add_argument("--target-timed-seconds", type=float, default=2.0)
    parser.add_argument("--max-repeat", type=int, default=50000)
    parser.add_argument("--monitor-interval", type=float, default=0.01)
    parser.add_argument("--monitor-gpu-index", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
