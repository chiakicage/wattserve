import time
import os
from tqdm import tqdm
import glob
import argparse
from typing import List, Tuple, Generator

import torch
import torch.nn as nn
from safetensors.torch import safe_open
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from qwen3 import Qwen3Model

_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


def safetensors_weights_iterator(
    hf_weights_files: List[str],
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        bar_format=_BAR_FORMAT,
    ):
        with safe_open(st_file, framework="pt") as f:  # type: ignore
            for name in f.keys():
                param = f.get_tensor(name)
                yield name, param


def load_weights(
    model: Qwen3Model,
    lm_head: nn.Linear,
    hf_folder: str,
    tie_word_embeddings: bool,
) -> None:
    hf_weights_files = glob.glob(os.path.join(hf_folder, "*.safetensors"))
    weight_iterator = safetensors_weights_iterator(hf_weights_files)
    param_dict = dict(model.named_parameters(prefix="model"))

    for param_name, weight in weight_iterator:
        # print(param_name, weight.shape, weight.dtype)
        if param_name.startswith("lm_head"):
            param = lm_head.weight
            param.requires_grad = False
            param.copy_(weight, non_blocking=True)
        else:
            assert param_name in param_dict, f"{param_name} is not found"
            param = param_dict[param_name]
            param.requires_grad = False
            param.copy_(weight, non_blocking=True)
    if tie_word_embeddings:
        param = lm_head.weight
        weight = param_dict["model.embed_tokens.weight"]
        param.requires_grad = False
        param.copy_(weight, non_blocking=True)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Zip Serve")
    parser.add_argument(
        "--model",
        default="/share/models/Qwen3.5-9B",
        type=str,
    )
    args = parser.parse_args()

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    print("Loading Config...")
    config = AutoConfig.from_pretrained(args.model)
    # print(config)

    print("Creating Model...")
    torch.set_default_device(torch.device("cuda:0"))
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda:0"):
        model = Qwen3Model(config)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    print("Loading Weights...")
    load_weights(model, lm_head, args.model, config.tie_word_embeddings)
    # print(dict(model.named_parameters(prefix="model")))

    prompt = "Please proof the Fermat's Last Theorem in detail."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # 将消息列表转换为模型期望的 prompt 字符串
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = tokenizer(text, return_tensors="pt").to("cuda:0").input_ids[0]
    prompt_len = input_ids.shape[0]
    position_ids = torch.arange(prompt_len, device=input_ids.device)

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

    output_len = 64
    output_ids = []
    position_ids = torch.arange(prompt_len, device=input_ids.device)
    torch.cuda.synchronize()
    start_time = time.time()
    token_times = []
    for _ in range(output_len):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            logits = lm_head(model(position_ids, input_ids=input_ids))
            next_token = torch.argmax(logits[-1], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        token_times.append(time.perf_counter() - t_start)
        input_ids = next_token
        position_ids = position_ids[-1:] + 1
        # output_id = next_token.item()
        # if output_id == tokenizer.eos_token_id:
        #     break
        # output_ids.append(output_id)
        # output_text = tokenizer.decode(
        #     [output_id], skip_special_tokens=True
        # )
        # print(output_text, end="", flush=True)

    ttft = token_times[0]
    tpot = (
        sum(token_times[1:]) / len(token_times[1:])
        if len(token_times) > 1
        else 0
    )
    print(f"  TTFT: {ttft * 1000:.2f} ms")
    print(f"  TPOT: {tpot * 1000:.2f} ms")
    print(f"  Tokens/sec: {1.0 / tpot:.2f}")
    print("\n", end="", flush=True)
    torch.cuda.synchronize()
    end_time = time.time()
    # output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    # print(output_text)
    token_per_sec = len(output_ids) / (end_time - start_time)
    print(f"Performance: {token_per_sec:.2f}/s")


if __name__ == "__main__":
    main()
