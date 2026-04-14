import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from models.llama import LlamaModel
from models.llama_config import (
    calculate_llama_parameter_count,
    calculate_llama_prefill_flops,
    calculate_llama_runtime_memory_bytes,
    get_llama_config_7B,
    get_llama_config_13B,
    get_llama_config_34B,
    get_llama_config_70B,
)
from monitor.gpu_monitor import GPUMonitor


MODEL_CONFIGS = {
    "7B": get_llama_config_7B,
    "13B": get_llama_config_13B,
    "34B": get_llama_config_34B,
    "70B": get_llama_config_70B,
}

DEFAULT_WARMUP = 3
DEFAULT_REPEAT = 5
DEFAULT_MONITOR_INTERVAL = 0.01
SUCCESS_STATUS = "ok"
ERROR_STATUS = "error"

RESULT_FIELDNAMES = [
    "run_timestamp_utc",
    "model",
    "prompt_len",
    "variant",
    "replace_ln",
    "status",
    "error_type",
    "error_message",
    "canonical_num_hidden_layers",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "parameter_count",
    "parameter_count_with_lm_head",
    "estimated_runtime_memory_gib",
    "warmup",
    "repeat",
    "ttft_ms",
    "prefill_tflops_s",
    "avg_power_watts",
    "max_power_watts",
    "avg_gpu_clock_mhz",
    "max_gpu_clock_mhz",
    "monitor_sample_count",
    "monitor_csv",
]


def generate_random_input_ids(
    prompt_len: int, vocab_size: int, device: str = "cuda:0"
) -> torch.Tensor:
    """Generate random input token IDs."""
    return torch.randint(0, vocab_size, (prompt_len,), device=device)


def get_variant_name(replace_ln: bool) -> str:
    return "replace_ln" if replace_ln else "baseline"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_result_record(
    config_name: str,
    prompt_len: int,
    replace_ln: bool,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    return {
        "run_timestamp_utc": _utc_now_iso(),
        "model": config_name,
        "prompt_len": prompt_len,
        "variant": get_variant_name(replace_ln),
        "replace_ln": replace_ln,
        "status": ERROR_STATUS,
        "error_type": "",
        "error_message": "",
        "canonical_num_hidden_layers": None,
        "num_hidden_layers": None,
        "hidden_size": None,
        "intermediate_size": None,
        "parameter_count": None,
        "parameter_count_with_lm_head": None,
        "estimated_runtime_memory_gib": None,
        "warmup": warmup,
        "repeat": repeat,
        "ttft_ms": None,
        "prefill_tflops_s": None,
        "avg_power_watts": None,
        "max_power_watts": None,
        "avg_gpu_clock_mhz": None,
        "max_gpu_clock_mhz": None,
        "monitor_sample_count": 0,
        "monitor_csv": "",
    }


def _populate_config_metadata(
    result: dict[str, Any],
    config_name: str,
) -> Any:
    config_func = MODEL_CONFIGS[config_name]
    config = config_func()
    assert config.hidden_size is not None
    assert config.intermediate_size is not None
    assert config.num_hidden_layers is not None

    result["canonical_num_hidden_layers"] = getattr(
        config, "canonical_num_hidden_layers", config.num_hidden_layers
    )
    result["num_hidden_layers"] = config.num_hidden_layers
    result["hidden_size"] = config.hidden_size
    result["intermediate_size"] = config.intermediate_size
    result["parameter_count"] = calculate_llama_parameter_count(config)
    result["parameter_count_with_lm_head"] = calculate_llama_parameter_count(
        config, include_lm_head=True
    )
    result["estimated_runtime_memory_gib"] = (
        calculate_llama_runtime_memory_bytes(config, include_lm_head=True)
        / 1024**3
    )
    return config


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


def benchmark(
    config_name: str,
    prompt_len: int,
    replace_ln: bool,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
    monitor_interval: float = DEFAULT_MONITOR_INTERVAL,
    monitor_csv_path: str | None = None,
) -> dict[str, Any]:
    result = create_result_record(
        config_name=config_name,
        prompt_len=prompt_len,
        replace_ln=replace_ln,
        warmup=warmup,
        repeat=repeat,
    )
    monitor: GPUMonitor | None = None

    previous_default_dtype = torch.get_default_dtype()
    previous_default_device = torch.get_default_device()

    try:
        if prompt_len <= 0:
            raise ValueError("prompt_len must be a positive integer")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if repeat <= 0:
            raise ValueError("repeat must be a positive integer")
        if monitor_interval <= 0:
            raise ValueError("monitor_interval must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for llama benchmarking")

        config = _populate_config_metadata(result, config_name)
        assert config.hidden_size is not None
        assert config.vocab_size is not None

        torch.set_default_device(torch.device("cuda:0"))
        torch.set_default_dtype(torch.bfloat16)

        with torch.device("cuda:0"):
            model = LlamaModel(config, replace_ln=replace_ln)

        input_ids = generate_random_input_ids(prompt_len, config.vocab_size)
        position_ids = torch.arange(prompt_len, device=input_ids.device)
        lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        ).to(device="cuda:0", dtype=torch.bfloat16)

        for _ in range(warmup):
            with torch.inference_mode():
                _ = lm_head(model(position_ids, input_ids=input_ids))
        torch.cuda.synchronize()

        monitor = GPUMonitor(gpu_index=0, interval=monitor_interval)
        monitor.start()
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(repeat):
            with torch.inference_mode():
                _ = lm_head(model(position_ids, input_ids=input_ids))

        torch.cuda.synchronize()
        ttft_seconds = (time.perf_counter() - start_time) / repeat
        monitor.stop()
        monitor_results = monitor.get_results()
        monitor_summary = _summarize_monitor_results(monitor_results)

        total_flops = calculate_llama_prefill_flops(config, prompt_len)
        result.update(monitor_summary)
        result["ttft_ms"] = ttft_seconds * 1000.0
        result["prefill_tflops_s"] = total_flops / 1e12 / ttft_seconds
        result["status"] = SUCCESS_STATUS

        if monitor_csv_path and monitor_summary["monitor_sample_count"] > 0:
            monitor_path = Path(monitor_csv_path)
            monitor_path.parent.mkdir(parents=True, exist_ok=True)
            monitor.export_csv(str(monitor_path))
            result["monitor_csv"] = str(monitor_path)

        return result
    except Exception as exc:
        result["status"] = ERROR_STATUS
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        return result
    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        torch.set_default_dtype(previous_default_dtype)
        torch.set_default_device(previous_default_device)


def _format_optional_float(value: Any, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def format_benchmark_result(result: dict[str, Any]) -> str:
    lines = [
        f"Benchmarking Llama-{result['model']}",
        f"  variant: {result['variant']}",
        f"  prompt_len: {result['prompt_len']}",
    ]

    num_hidden_layers = result.get("num_hidden_layers")
    canonical_layers = result.get("canonical_num_hidden_layers")
    if num_hidden_layers is not None:
        if canonical_layers not in (None, num_hidden_layers):
            lines.append(
                "  layers: "
                f"{num_hidden_layers}/{canonical_layers} (A100 40GB fitted)"
            )
        else:
            lines.append(f"  layers: {num_hidden_layers}")

    if result.get("hidden_size") is not None:
        lines.append(f"  hidden_size: {result['hidden_size']}")
    if result.get("parameter_count") is not None:
        lines.append(
            f"  params: {float(result['parameter_count']) / 1e9:.2f} B"
        )
    if result.get("parameter_count_with_lm_head") is not None:
        lines.append(
            "  params (+lm_head): "
            f"{float(result['parameter_count_with_lm_head']) / 1e9:.2f} B"
        )
    if result.get("estimated_runtime_memory_gib") is not None:
        lines.append(
            "  est. persistent memory: "
            f"{float(result['estimated_runtime_memory_gib']):.2f} GiB"
        )

    if result["status"] == SUCCESS_STATUS:
        lines.extend(
            [
                f"  Prefill TFlops/s: {_format_optional_float(result['prefill_tflops_s'])}",
                f"  Avg Power: {_format_optional_float(result['avg_power_watts'])} W",
                f"  Max Power: {_format_optional_float(result['max_power_watts'])} W",
                f"  Avg GPU Clock: {_format_optional_float(result['avg_gpu_clock_mhz'])} MHz",
                f"  Max GPU Clock: {_format_optional_float(result['max_gpu_clock_mhz'])} MHz",
                f"  TTFT: {_format_optional_float(result['ttft_ms'])} ms",
                f"  Monitor Samples: {result['monitor_sample_count']}",
            ]
        )
        if result.get("monitor_csv"):
            lines.append(f"  Monitor CSV: {result['monitor_csv']}")
    else:
        lines.append(f"  Status: {result['status']}")
        lines.append(
            f"  Error: {result['error_type']}: {result['error_message']}"
        )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Llama Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help=(
            "Model size: 7B, 13B, 34B, 70B. "
            "34B and 70B keep the large-model hidden sizes but reduce "
            "layer count to fit an A100 40GB GPU."
        ),
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        required=True,
        default=4096,
        help="Length of input prompt (number of tokens)",
    )
    parser.add_argument(
        "--replace_ln",
        action="store_true",
        help="Run the replace_ln ablation instead of the baseline path.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help="Number of timed iterations to average into TTFT.",
    )
    parser.add_argument(
        "--monitor_interval",
        type=float,
        default=DEFAULT_MONITOR_INTERVAL,
        help="GPU monitor sampling interval in seconds.",
    )
    parser.add_argument(
        "--monitor_csv_path",
        type=str,
        default=None,
        help="Optional path for exporting per-sample GPU monitor data.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = benchmark(
        config_name=args.model,
        prompt_len=args.prompt_len,
        replace_ln=args.replace_ln,
        warmup=args.warmup,
        repeat=args.repeat,
        monitor_interval=args.monitor_interval,
        monitor_csv_path=args.monitor_csv_path,
    )

    if result["status"] == SUCCESS_STATUS:
        print(format_benchmark_result(result))
        return 0

    print(format_benchmark_result(result), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
