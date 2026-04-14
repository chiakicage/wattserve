import time
import argparse

import torch
from models.qwen3_config import (
    get_qwen3_config_4B,
    get_qwen3_config_8B,
    get_qwen3_config_16B,
    get_qwen3_config_32B,
)
from models.qwen3 import Qwen3Model
from monitor.gpu_monitor import GPUMonitor


MODEL_CONFIGS = {
    "4B": get_qwen3_config_4B,
    "8B": get_qwen3_config_8B,
    "16B": get_qwen3_config_16B,
    "32B": get_qwen3_config_32B,
}


def calculate_prefill_flops(
    prompt_len: int,
    hidden_size: int,
    num_layers: int,
    intermediate_size: int,
) -> float:
    """
    Calculate total FLOPs for Prefill phase.

    For each transformer layer:
    - QKV projection: 3 * S * H * H
    - Attention scores (QK^T): S^2 * H
    - Attention values (softmax(QK) * V): S^2 * H
    - Output projection: S * H * H
    - FFN linear: 3 * S * H * intermediate_size

    """
    S = prompt_len
    H = hidden_size
    L = num_layers
    M = intermediate_size

    qkv_flops = 3 * S * H * H
    attn_scores_flops = S * S * H
    attn_values_flops = S * S * H
    out_proj_flops = S * H * H
    ffn_flops = S * H * M * 3

    flops_per_layer = (
        qkv_flops
        + attn_scores_flops
        + attn_values_flops
        + out_proj_flops
        + ffn_flops
    )

    total_flops = 2 * flops_per_layer * L
    return total_flops


def generate_random_input_ids(
    prompt_len: int, vocab_size: int, device: str = "cuda:0"
) -> torch.Tensor:
    """Generate random input token IDs."""
    return torch.randint(0, vocab_size, (prompt_len,), device=device)


def benchmark(
    config_name: str, prompt_len: int, output_len: int, replace_ln: bool
) -> None:
    print(f"Benchmarking Qwen3-{config_name}")
    print(f"  prompt_len: {prompt_len}")
    print(f"  output_len: {output_len}")

    config_func = MODEL_CONFIGS[config_name]
    config = config_func()

    print("Creating model...")
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda:0"):
        model = Qwen3Model(config, replace_ln=replace_ln)

    print("Generating random input...")
    assert config.vocab_size is not None
    assert config.hidden_size is not None
    input_ids = generate_random_input_ids(prompt_len, config.vocab_size)
    position_ids = torch.arange(prompt_len, device=input_ids.device)

    lm_head = torch.nn.Linear(
        config.hidden_size, config.vocab_size, bias=False
    ).to(device="cuda:0", dtype=torch.bfloat16)

    WARMUP = 3
    REPEAT = 5
    print("Warming up...")
    for _ in range(WARMUP):
        with torch.no_grad():
            first_logits = lm_head(model(position_ids, input_ids=input_ids))
            next_token = torch.argmax(first_logits[-1], dim=-1, keepdim=True)
            position_ids = torch.tensor(
                [prompt_len], device=input_ids.device, dtype=torch.long
            )
            _ = lm_head(model(position_ids, input_ids=next_token))
            model.clear_kv_cache()
    torch.cuda.synchronize()

    print("Starting benchmark...")
    monitor = GPUMonitor(gpu_index=0, interval=0.01)
    monitor.start()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(REPEAT):
        with torch.no_grad():
            first_logits = lm_head(model(position_ids, input_ids=input_ids))
            next_token = torch.argmax(first_logits[-1], dim=-1, keepdim=True)
            model.clear_kv_cache()

    torch.cuda.synchronize()
    ttft = (time.perf_counter() - start_time) / REPEAT
    monitor.stop()

    monitor_results = monitor.get_results()
    if monitor_results:
        avg_power = sum(r["power_watts"] for r in monitor_results) / len(
            monitor_results
        )
        max_power = max(r["power_watts"] for r in monitor_results)
        avg_freq = sum(r["gpu_clock_mhz"] for r in monitor_results) / len(
            monitor_results
        )
        max_freq = max(r["gpu_clock_mhz"] for r in monitor_results)
    else:
        avg_power = 0
        max_power = 0
        avg_freq = 0
        max_freq = 0

    assert config.hidden_size is not None
    assert config.num_hidden_layers is not None
    assert config.intermediate_size is not None
    total_flops = calculate_prefill_flops(
        prompt_len,
        config.hidden_size,
        config.num_hidden_layers,
        config.intermediate_size,
    )
    total_tflops = total_flops / 1e12 / ttft

    print(f"  Prefill TFlops/s: {total_tflops:.2f}")
    print(f"  Avg Power: {avg_power:.2f} W")
    print(f"  Max Power: {max_power:.2f} W")
    print(f"  Avg GPU Clock: {avg_freq:.2f} MHz")
    print(f"  Max GPU Clock: {max_freq:.2f} MHz")
    print(f"  TTFT: {ttft * 1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Model size: 4B, 8B, 16B, 32B",
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        required=True,
        default=4096,
        help="Length of input prompt (number of tokens)",
    )
    parser.add_argument(
        "--output_len",
        type=int,
        default=64,
        help="Number of tokens to generate",
    )

    parser.add_argument(
        "--replace_ln",
        action="store_true",
    )

    args = parser.parse_args()

    benchmark(args.model, args.prompt_len, args.output_len, args.replace_ln)


if __name__ == "__main__":
    main()
