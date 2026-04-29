#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gemm_continuous import ACTIVE_NVTX_RANGE  # noqa: E402
from state_chain_utils import utc_now_iso, utc_stamp, write_csv, write_json  # noqa: E402


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "gemm_continuous"
DEFAULT_NSYS_PATH = "/usr/local/bin/nsys"
DEFAULT_PYTHON = sys.executable

KERNEL_BUCKET_FIELDNAMES = [
    "bucket_start_s",
    "bucket_end_s",
    "kernel_name",
    "count",
    "total_duration_ms",
    "avg_duration_us",
]


def _run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _stats_command(
    nsys_path: str,
    report: str,
    output_prefix: Path,
    rep_path: Path,
) -> list[str]:
    return [
        nsys_path,
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--format=csv",
        f"--report={report}",
        f"--filter-nvtx={ACTIVE_NVTX_RANGE}",
        "--output",
        str(output_prefix),
        str(rep_path),
    ]


def _find_stats_csv(output_prefix: Path, report: str) -> Path:
    candidates = sorted(
        output_prefix.parent.glob(f"{output_prefix.name}_{report}*.csv")
    )
    if not candidates:
        raise FileNotFoundError(f"missing nsys stats CSV for {report}")
    return candidates[-1]


def _float(row: dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in ("", None):
            value = row[name].replace(",", "")
            return float(value)
    return 0.0


def _str(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return ""


def _read_kernel_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(
            line for line in csv_file if not line.startswith("#")
        )
        for row in reader:
            name = _str(row, "Name")
            if not name:
                continue
            rows.append(
                {
                    "start_ns": _float(row, "Start", "Start (ns)"),
                    "duration_ns": _float(row, "Duration", "Duration (ns)"),
                    "name": name,
                }
            )
    rows.sort(key=lambda row: float(row["start_ns"]))
    return rows


def _write_kernel_bucket_summary(
    trace_csv: Path,
    output_csv: Path,
    output_md: Path,
    bucket_s: float,
) -> None:
    trace_rows = _read_kernel_trace(trace_csv)
    if not trace_rows:
        write_csv(output_csv, KERNEL_BUCKET_FIELDNAMES, [])
        output_md.write_text(
            "# Kernel Change Summary\n\nNo CUDA kernels found.\n"
        )
        return

    first_start_ns = float(trace_rows[0]["start_ns"])
    buckets: dict[tuple[int, str], dict[str, Any]] = {}
    for row in trace_rows:
        rel_s = (float(row["start_ns"]) - first_start_ns) / 1e9
        bucket_index = int(rel_s // bucket_s)
        key = (bucket_index, str(row["name"]))
        bucket = buckets.setdefault(
            key,
            {
                "bucket_start_s": bucket_index * bucket_s,
                "bucket_end_s": (bucket_index + 1) * bucket_s,
                "kernel_name": str(row["name"]),
                "count": 0,
                "total_duration_ms": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["total_duration_ms"] += float(row["duration_ns"]) / 1e6

    rows = list(buckets.values())
    for row in rows:
        count = int(row["count"])
        row["avg_duration_us"] = (
            float(row["total_duration_ms"]) * 1000.0 / count if count else 0.0
        )
    rows.sort(
        key=lambda row: (
            float(row["bucket_start_s"]),
            -float(row["total_duration_ms"]),
        )
    )
    write_csv(output_csv, KERNEL_BUCKET_FIELDNAMES, rows)

    first_bucket_names = {
        row["kernel_name"] for row in rows if float(row["bucket_start_s"]) < 2.0
    }
    last_bucket_start = max(float(row["bucket_start_s"]) for row in rows)
    last_bucket_names = {
        row["kernel_name"]
        for row in rows
        if float(row["bucket_start_s"]) == last_bucket_start
    }
    same_names = first_bucket_names == last_bucket_names
    lines = [
        "# Kernel Change Summary",
        "",
        f"- Trace CSV: `{trace_csv}`",
        f"- Bucket size: `{bucket_s:.3f} s`",
        f"- First-2s kernel names: `{len(first_bucket_names)}`",
        f"- Last-bucket kernel names: `{len(last_bucket_names)}`",
        (
            "- Kernel name comparison: "
            + (
                "no cublas/CUDA kernel name change observed"
                if same_names
                else "kernel name set changed across buckets"
            )
        ),
        "",
        "## First-2s Kernel Names",
        "",
        *[f"- `{name}`" for name in sorted(first_bucket_names)],
        "",
        "## Last-Bucket Kernel Names",
        "",
        *[f"- `{name}`" for name in sorted(last_bucket_names)],
        "",
    ]
    output_md.write_text("\n".join(lines))


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir or (DEFAULT_RESULTS_ROOT / utc_stamp())
    nsys_dir = output_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    profile_prefix = nsys_dir / "profile"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    bench_cmd = [
        args.python,
        str(SCRIPT_DIR / "run_gemm_continuous.py"),
        "--m",
        str(args.m),
        "--n",
        str(args.n),
        "--k",
        str(args.k),
        "--dtype",
        args.dtype,
        "--gemm-units",
        str(args.gemm_units),
        "--monitor-gpu-index",
        str(args.monitor_gpu_index),
        "--monitor-interval",
        str(args.monitor_interval),
        "--pre-idle-s",
        str(args.pre_idle_s),
        "--post-idle-s",
        str(args.post_idle_s),
        "--output-dir",
        str(output_dir / "profiled_run"),
    ]
    nsys_cmd = [
        args.nsys_path,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,cublas,cublas-verbose",
        "--sample=none",
        "--cpuctxsw=none",
        "--output",
        str(profile_prefix),
        *bench_cmd,
    ]
    (nsys_dir / "profile_command.txt").write_text(
        " ".join(shlex.quote(part) for part in nsys_cmd) + "\n"
    )
    _run_command(nsys_cmd, REPO_ROOT, env)

    rep_path = profile_prefix.with_suffix(".nsys-rep")
    if not rep_path.exists():
        raise FileNotFoundError(rep_path)

    reports = [
        "cuda_gpu_trace",
        "cuda_gpu_kern_sum",
        "nvtx_kern_sum",
    ]
    generated: dict[str, str] = {}
    for report in reports:
        output_prefix = nsys_dir / f"{report}_active"
        _run_command(
            _stats_command(args.nsys_path, report, output_prefix, rep_path),
            REPO_ROOT,
            env,
        )
        generated[report] = str(_find_stats_csv(output_prefix, report))

    _write_kernel_bucket_summary(
        Path(generated["cuda_gpu_trace"]),
        nsys_dir / "kernel_bucket_summary.csv",
        nsys_dir / "kernel_change_summary.md",
        args.bucket_s,
    )
    write_json(
        nsys_dir / "metadata.json",
        {
            "generated_at": utc_now_iso(),
            "nsys_path": args.nsys_path,
            "nsys_profile": str(rep_path),
            "active_nvtx_range": ACTIVE_NVTX_RANGE,
            "stats": generated,
            "args": _jsonable_args(args),
        },
    )
    print(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile continuous two-GEMM with Nsight Systems."
    )
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=32768)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gemm-units", type=int, default=350)
    parser.add_argument("--monitor-gpu-index", type=int, default=2)
    parser.add_argument("--monitor-interval", type=float, default=0.01)
    parser.add_argument("--pre-idle-s", type=float, default=2.0)
    parser.add_argument("--post-idle-s", type=float, default=1.0)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--nsys-path", default=DEFAULT_NSYS_PATH)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--bucket-s", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
