import torch
import time
import argparse

from flashinfer import fused_add_rmsnorm
from monitor.gpu_monitor import GPUMonitor


def rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    replace_ln: bool,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if replace_ln:
        residual = x
        return x, residual
    fused_add_rmsnorm(x, residual, weight, eps)
    return x, residual


def bench_gemm_layernorm(N: int, L: int, replace_ln: bool) -> dict:
    weight = torch.randn((N, N), device="cuda:0", dtype=torch.bfloat16)
    weight_norm = torch.randn((N,), device="cuda:0", dtype=torch.bfloat16)
    eps = 1e-6
    x = torch.randn((N, N), device="cuda:0", dtype=torch.bfloat16)
    residual = torch.rand_like(x)
    torch.cuda.synchronize()

    monitor = GPUMonitor(gpu_index=0, interval=0.01)
    monitor.start()
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(L):
        y = torch.matmul(x, weight)
        _, _ = rmsnorm(y, residual, replace_ln, weight_norm, eps)
    torch.cuda.synchronize()
    end = time.perf_counter()
    monitor.stop()

    elapsed_time = end - start

    flops = 2 * N * N * N * L
    tflops = flops / 1e12 / elapsed_time

    monitor_results = monitor.get_results()
    if monitor_results:
        avg_power = sum(r["power_watts"] for r in monitor_results) / len(
            monitor_results
        )
        max_power = max(r["power_watts"] for r in monitor_results)
        avg_gpu_clock = sum(r["gpu_clock_mhz"] for r in monitor_results) / len(
            monitor_results
        )
        max_gpu_clock = max(r["gpu_clock_mhz"] for r in monitor_results)
    else:
        avg_power = max_power = avg_gpu_clock = max_gpu_clock = 0

    return {
        "elapsed_time": elapsed_time,
        "tflops": tflops,
        "avg_power": avg_power,
        "max_power": max_power,
        "avg_gpu_clock": avg_gpu_clock,
        "max_gpu_clock": max_gpu_clock,
    }


def main():
    parser = argparse.ArgumentParser(description="GEMM + LayerNorm Benchmark")
    parser.add_argument(
        "--N", type=int, required=True, help="Matrix dimension N"
    )
    parser.add_argument(
        "--L", type=int, required=True, help="Number of iterations L"
    )
    parser.add_argument(
        "--replace_ln",
        action="store_true",
    )
    args = parser.parse_args()

    result = bench_gemm_layernorm(args.N, args.L, args.replace_ln)

    print(f"N: {args.N}, L: {args.L}")
    print(f"  Elapsed time: {result['elapsed_time']:.4f} s")
    print(f"  TFlops: {result['tflops']:.2f}")
    print(f"  Avg Power: {result['avg_power']:.2f} W")
    print(f"  Max Power: {result['max_power']:.2f} W")
    print(f"  Avg GPU Clock: {result['avg_gpu_clock']:.2f} MHz")
    print(f"  Max GPU Clock: {result['max_gpu_clock']:.2f} MHz")


if __name__ == "__main__":
    main()
