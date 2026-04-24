import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def _load_test_modules():
    repo_root = Path(__file__).resolve().parents[2]
    python_dir = repo_root / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))

    import bench_llama
    import matplotlib.pyplot as plt
    from models.llama_config import (
        calculate_llama_prefill_flops,
        get_llama_config_7B,
    )
    from scripts.benchmarks.render_llama_component_ablation_report import (
        configure_metric_axis,
        render_result_report,
    )
    from scripts.benchmarks.run_llama_component_ablation_matrix import (
        DEFAULT_PROMPT_LENGTHS,
        DEFAULT_REPEAT,
        DEFAULT_VARIANTS,
        DEFAULT_WARMUP,
        write_summary_csv,
    )
    from scripts.benchmarks.run_llama_replace_ln_matrix import (
        DEFAULT_PROMPT_LENGTHS as OLD_DEFAULT_PROMPT_LENGTHS,
    )

    return (
        bench_llama,
        calculate_llama_prefill_flops,
        get_llama_config_7B,
        plt,
        configure_metric_axis,
        render_result_report,
        DEFAULT_PROMPT_LENGTHS,
        OLD_DEFAULT_PROMPT_LENGTHS,
        DEFAULT_REPEAT,
        DEFAULT_VARIANTS,
        DEFAULT_WARMUP,
        write_summary_csv,
    )


(
    bench_llama,
    calculate_llama_prefill_flops,
    get_llama_config_7B,
    plt,
    configure_metric_axis,
    render_result_report,
    DEFAULT_PROMPT_LENGTHS,
    OLD_DEFAULT_PROMPT_LENGTHS,
    DEFAULT_REPEAT,
    DEFAULT_VARIANTS,
    DEFAULT_WARMUP,
    write_summary_csv,
) = _load_test_modules()


class TestLlamaComponentAblationBenchmark(unittest.TestCase):
    def _make_row(
        self,
        model: str,
        prompt_len: int,
        variant: str,
        status: str,
        **overrides,
    ) -> dict:
        flags = {
            "replace_ln": False,
            "replace_attention": False,
            "replace_rope": False,
            "replace_activation": False,
        }
        if variant != "baseline":
            for flag_name in variant.split("+"):
                flags[flag_name] = True

        row = {
            "run_timestamp_utc": "2026-04-15T00:00:00Z",
            "model": model,
            "prompt_len": prompt_len,
            "variant": variant,
            **flags,
            "status": status,
            "error_type": "",
            "error_message": "",
            "canonical_num_hidden_layers": 40,
            "num_hidden_layers": 40,
            "hidden_size": 5120,
            "intermediate_size": 13824,
            "parameter_count": 13000000000,
            "parameter_count_with_lm_head": 13300000000,
            "estimated_runtime_memory_gib": 24.0,
            "warmup": 5,
            "repeat": 10,
            "ttft_ms": 480.0,
            "prefill_tflops_s": 240.0,
            "avg_power_watts": 390.0,
            "max_power_watts": 410.0,
            "avg_gpu_clock_mhz": 1320.0,
            "max_gpu_clock_mhz": 1410.0,
            "monitor_sample_count": 12,
            "monitor_csv": "monitor/example.csv",
        }
        row.update(overrides)
        return row

    def test_runner_defaults_match_existing_prompt_matrix(self) -> None:
        self.assertEqual(DEFAULT_PROMPT_LENGTHS, OLD_DEFAULT_PROMPT_LENGTHS)
        self.assertEqual(
            DEFAULT_PROMPT_LENGTHS,
            [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        )
        self.assertEqual(DEFAULT_WARMUP, 5)
        self.assertEqual(DEFAULT_REPEAT, 10)
        self.assertEqual(
            [variant["variant"] for variant in DEFAULT_VARIANTS],
            [
                "baseline",
                "replace_ln",
                "replace_attention",
                "replace_rope",
                "replace_activation",
            ],
        )

    def test_variant_naming_uses_fixed_flag_order(self) -> None:
        self.assertEqual(bench_llama.get_variant_name(), "baseline")
        self.assertEqual(
            bench_llama.get_variant_name(
                replace_ln=True,
                replace_rope=True,
            ),
            "replace_ln+replace_rope",
        )
        self.assertEqual(
            bench_llama.get_variant_name(
                replace_attention=True,
                replace_activation=True,
            ),
            "replace_attention+replace_activation",
        )

    def test_prefill_flops_only_changes_for_attention_kernel_ablation(
        self,
    ) -> None:
        config = get_llama_config_7B()
        prompt_len = 128
        baseline_flops = calculate_llama_prefill_flops(config, prompt_len)

        self.assertEqual(
            calculate_llama_prefill_flops(config, prompt_len, replace_ln=True),
            baseline_flops,
        )
        self.assertEqual(
            calculate_llama_prefill_flops(
                config, prompt_len, replace_rope=True
            ),
            baseline_flops,
        )
        self.assertEqual(
            calculate_llama_prefill_flops(
                config, prompt_len, replace_activation=True
            ),
            baseline_flops,
        )

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        q_size = config.num_attention_heads * head_dim
        expected_reduction = (
            2
            * 2
            * prompt_len
            * prompt_len
            * q_size
            * config.num_hidden_layers
        )
        self.assertEqual(
            calculate_llama_prefill_flops(
                config, prompt_len, replace_attention=True
            ),
            baseline_flops - expected_reduction,
        )
        self.assertEqual(
            calculate_llama_prefill_flops(
                config,
                prompt_len,
                replace_attention=True,
                replace_ln=True,
                replace_rope=True,
                replace_activation=True,
            ),
            baseline_flops - expected_reduction,
        )

    def test_metric_axis_configuration(self) -> None:
        fig, axes = plt.subplots(1, 3)
        try:
            configure_metric_axis(axes[0], "ttft_ms")
            configure_metric_axis(axes[1], "avg_power_watts")
            configure_metric_axis(axes[2], "avg_gpu_clock_mhz")

            self.assertEqual(axes[0].get_yscale(), "linear")
            self.assertEqual(
                tuple(int(value) for value in axes[1].get_ylim()),
                (-10, 410),
            )
            self.assertEqual(
                [int(value) for value in axes[1].get_yticks()],
                [0, 100, 200, 300, 400],
            )
            self.assertEqual(
                tuple(int(value) for value in axes[2].get_ylim()),
                (1050, 1450),
            )
            self.assertEqual(
                [int(value) for value in axes[2].get_yticks()],
                [1100, 1200, 1300, 1400],
            )
        finally:
            plt.close(fig)

    def test_write_summary_csv_includes_switch_columns(self) -> None:
        rows = [
            self._make_row("13B", 2048, "baseline", "ok"),
            self._make_row(
                "13B",
                2048,
                "replace_attention",
                "error",
                error_type="RuntimeError",
                error_message="CUDA out of memory",
                ttft_ms=None,
                prefill_tflops_s=None,
                avg_power_watts=None,
                max_power_watts=None,
                avg_gpu_clock_mhz=None,
                max_gpu_clock_mhz=None,
                monitor_sample_count=0,
                monitor_csv="",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            summary_csv_path = Path(temp_dir) / "summary.csv"
            write_summary_csv(rows, summary_csv_path)

            with summary_csv_path.open("r", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                written_rows = list(reader)

        self.assertEqual(len(written_rows), 2)
        self.assertEqual(written_rows[0]["variant"], "baseline")
        self.assertEqual(written_rows[0]["replace_attention"], "False")
        self.assertEqual(written_rows[1]["variant"], "replace_attention")
        self.assertEqual(written_rows[1]["replace_ln"], "False")
        self.assertEqual(written_rows[1]["replace_attention"], "True")
        self.assertEqual(written_rows[1]["status"], "error")
        self.assertEqual(written_rows[1]["error_type"], "RuntimeError")
        self.assertEqual(written_rows[1]["error_message"], "CUDA out of memory")

    def test_render_result_report_generates_plots_and_indices(self) -> None:
        rows = [
            self._make_row("7B", 512, "baseline", "ok", ttft_ms=100.0),
            self._make_row(
                "7B",
                512,
                "replace_ln",
                "ok",
                ttft_ms=90.0,
                prefill_tflops_s=110.0,
                avg_power_watts=280.0,
                avg_gpu_clock_mhz=1450.0,
            ),
            self._make_row(
                "7B",
                1024,
                "replace_attention",
                "ok",
                ttft_ms=130.0,
            ),
            self._make_row(
                "13B",
                8192,
                "replace_rope",
                "error",
                error_type="OutOfMemoryError",
                error_message="OOM",
                ttft_ms=None,
                prefill_tflops_s=None,
                avg_power_watts=None,
                max_power_watts=None,
                avg_gpu_clock_mhz=None,
                max_gpu_clock_mhz=None,
                monitor_sample_count=0,
                monitor_csv="",
            ),
            self._make_row("34B", 2048, "baseline", "ok"),
            self._make_row(
                "34B",
                16384,
                "replace_activation",
                "ok",
                ttft_ms=3000.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "results" / "run1"
            output_dir.mkdir(parents=True)
            summary_csv_path = output_dir / "summary.csv"
            metadata_path = output_dir / "metadata.json"
            root_index_path = temp_path / "BENCHMARK_COMPONENT_ABLATION.md"

            write_summary_csv(rows, summary_csv_path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "run_started_at_utc": "2026-04-15T00:00:00Z",
                        "warmup": 5,
                        "repeat": 10,
                        "monitor_interval": 0.01,
                        "prompt_lengths": [
                            16,
                            32,
                            64,
                            128,
                            256,
                            512,
                            1024,
                            2048,
                            4096,
                            8192,
                            16384,
                        ],
                        "environment": {
                            "python_version": "3.13.2",
                            "torch_version": "2.11.0+cu130",
                            "cuda_available": True,
                            "cuda_device_name": "NVIDIA A100 40GB",
                        },
                    },
                    indent=2,
                )
                + "\n"
            )

            render_result_report(
                output_dir=output_dir,
                refresh_root_index=True,
                root_index_path=root_index_path,
            )

            benchmark_md = (output_dir / "BENCHMARK.md").read_text()
            root_index_md = root_index_path.read_text()
            metadata = json.loads(metadata_path.read_text())

            self.assertIn("![TTFT](plots/ttft_ms.png)", benchmark_md)
            self.assertIn(
                "![Prefill TFLOPs/s](plots/prefill_tflops_s.png)",
                benchmark_md,
            )
            self.assertIn(
                "![Avg Power](plots/avg_power_watts.png)", benchmark_md
            )
            self.assertIn(
                "![Avg GPU Clock](plots/avg_gpu_clock_mhz.png)",
                benchmark_md,
            )
            self.assertIn("## Baseline vs replace_ln", benchmark_md)
            self.assertIn("## Baseline vs replace_attention", benchmark_md)
            self.assertIn("## Baseline vs replace_rope", benchmark_md)
            self.assertIn("## Baseline vs replace_activation", benchmark_md)
            self.assertIn(
                "Non-standard prompt lengths excluded from this report and plots: `16384`",
                benchmark_md,
            )
            self.assertIn(
                "| 13B | 8192 | replace_rope | OutOfMemoryError |",
                benchmark_md,
            )
            self.assertIn("## Unpaired Successful Runs", benchmark_md)
            self.assertIn("| 34B | 2048 | baseline |", benchmark_md)
            self.assertIn(
                "replace_ln/replace_attention/replace_rope/replace_activation",
                root_index_md,
            )
            self.assertIn("results/run1/BENCHMARK.md", root_index_md)
            self.assertEqual(
                metadata["report_prompt_lengths"],
                [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            )
            self.assertEqual(
                metadata["report_variants"],
                [
                    "baseline",
                    "replace_ln",
                    "replace_attention",
                    "replace_rope",
                    "replace_activation",
                ],
            )
            self.assertEqual(metadata["excluded_prompt_lengths"], [16384])
            self.assertEqual(metadata["warmup"], 5)
            self.assertEqual(metadata["repeat"], 10)

            for plot_name in [
                "ttft_ms.png",
                "prefill_tflops_s.png",
                "avg_power_watts.png",
                "avg_gpu_clock_mhz.png",
            ]:
                plot_path = output_dir / "plots" / plot_name
                self.assertTrue(plot_path.exists())
                self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
