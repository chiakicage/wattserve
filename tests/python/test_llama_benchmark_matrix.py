import csv
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks.run_llama_replace_ln_matrix import (
    build_benchmark_markdown,
    write_summary_csv,
)


class TestLlamaBenchmarkMatrix(unittest.TestCase):
    def _make_row(
        self,
        model: str,
        prompt_len: int,
        variant: str,
        status: str,
        **overrides,
    ) -> dict:
        replace_ln = variant == "replace_ln"
        row = {
            "run_timestamp_utc": "2026-04-14T00:00:00Z",
            "model": model,
            "prompt_len": prompt_len,
            "variant": variant,
            "replace_ln": replace_ln,
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
            "warmup": 3,
            "repeat": 5,
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

    def test_write_summary_csv_includes_success_and_failure_rows(self):
        rows = [
            self._make_row("13B", 4096, "baseline", "ok"),
            self._make_row(
                "13B",
                4096,
                "replace_ln",
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
        self.assertEqual(written_rows[1]["variant"], "replace_ln")
        self.assertEqual(written_rows[1]["status"], "error")
        self.assertEqual(written_rows[1]["error_type"], "RuntimeError")
        self.assertEqual(written_rows[1]["error_message"], "CUDA out of memory")

    def test_build_benchmark_markdown_pairs_failures_and_unpaired_successes(
        self,
    ):
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
                "13B",
                16384,
                "baseline",
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
            self._make_row("34B", 8192, "baseline", "ok"),
        ]
        metadata = {
            "warmup": 3,
            "repeat": 5,
            "monitor_interval": 0.01,
            "environment": {
                "python_version": "3.13.2",
                "torch_version": "2.11.0+cu130",
                "cuda_available": True,
                "cuda_device_name": "NVIDIA A100 40GB",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown = build_benchmark_markdown(
                rows=rows,
                run_started_at_utc="2026-04-14T00:00:00Z",
                output_dir=temp_path / "results" / "run1",
                summary_csv_path=temp_path / "results" / "run1" / "summary.csv",
                metadata=metadata,
            )

        self.assertIn("# Llama `replace_ln` Benchmark", markdown)
        self.assertIn("| 7B | 512 | 240.00 | 110.00 | -54.17% |", markdown)
        self.assertIn("| 13B | 16384 | baseline | OutOfMemoryError |", markdown)
        self.assertIn("## Unpaired Successful Runs", markdown)
        self.assertIn("| 34B | 8192 | baseline |", markdown)
        self.assertIn(
            "`LLAMA2_MAX_POSITION_EMBEDDINGS = 16384` is only a position-length limit.",
            markdown,
        )


if __name__ == "__main__":
    unittest.main()
