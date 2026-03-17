import time
import argparse

import torch
from qwen3_config import (
    get_qwen3_config_4B,
    get_qwen3_config_8B,
    get_qwen3_config_14B,
    get_qwen3_config_32B,
)
from qwen3 import Qwen3Model


MODEL_CONFIGS = {
    "4B": get_qwen3_config_4B,
    "8B": get_qwen3_config_8B,
    "14B": get_qwen3_config_14B,
    "32B": get_qwen3_config_32B,
}


def generate_random_input_ids(
    prompt_len: int, vocab_size: int, device: str = "cuda:0"
) -> torch.Tensor:
    """Generate random input token IDs."""
    return torch.randint(0, vocab_size, (prompt_len,), device=device)


def benchmark(
    config_name: str,
    prompt_len: int,
    output_len: int,
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
        model = Qwen3Model(config)

    print("Generating random input...")
    assert config.vocab_size is not None
    assert config.hidden_size is not None
    input_ids = generate_random_input_ids(prompt_len, config.vocab_size)
    position_ids = torch.arange(prompt_len, device=input_ids.device)

    lm_head = torch.nn.Linear(
        config.hidden_size, config.vocab_size, bias=False
    ).to(device="cuda:0", dtype=torch.bfloat16)

    print("Warming up...")
    for _ in range(3):
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
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        first_logits = lm_head(model(position_ids, input_ids=input_ids))
        next_token = torch.argmax(first_logits[-1], dim=-1, keepdim=True)

    torch.cuda.synchronize()
    ttft = time.perf_counter() - start_time
    print(f"  TTFT: {ttft * 1000:.2f} ms")

    input_ids = next_token
    position_ids = torch.tensor(
        [prompt_len], device=input_ids.device, dtype=torch.long
    )

    token_times = []
    for _ in range(output_len - 1):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            logits = lm_head(model(position_ids, input_ids=input_ids))
            next_token = torch.argmax(logits[-1], dim=-1, keepdim=True)

        torch.cuda.synchronize()
        token_times.append(time.perf_counter() - t_start)

        input_ids = next_token
        position_ids = position_ids + 1

    tpot = sum(token_times) / len(token_times) if token_times else 0

    print(f"  TPOT: {tpot * 1000:.2f} ms")
    print(f"  Tokens/sec: {1.0 / tpot:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Model size: 4B, 8B, 14B, 32B",
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        required=True,
        help="Length of input prompt (number of tokens)",
    )
    parser.add_argument(
        "--output_len",
        type=int,
        required=True,
        help="Number of tokens to generate",
    )

    args = parser.parse_args()

    benchmark(args.model, args.prompt_len, args.output_len)


if __name__ == "__main__":
    main()
