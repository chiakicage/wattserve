from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def average(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(float(record[key]) for record in records) / len(records)


def monitor_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "avg_power_watts": average(records, "power_watts"),
        "max_power_watts": max(
            (float(record["power_watts"]) for record in records), default=0.0
        ),
        "avg_gpu_clock_mhz": average(records, "gpu_clock_mhz"),
        "max_gpu_clock_mhz": max(
            (float(record["gpu_clock_mhz"]) for record in records),
            default=0.0,
        ),
        "monitor_sample_count": len(records),
    }


def calibrate_repeat(
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
    if iter_seconds <= 0:
        return 1, 0.0
    repeat = int(round(target_timed_seconds / iter_seconds))
    return max(1, min(max_repeat, repeat)), iter_seconds


def make_scaled_tensor(
    torch: Any,
    shape: tuple[int, ...],
    device: str,
    dtype: Any,
    scale: float = 0.02,
) -> Any:
    tensor = torch.randn(shape, device=device, dtype=dtype)
    tensor.mul_(scale)
    return tensor


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {fieldname: row.get(fieldname, "") for fieldname in fieldnames}
            )


def read_monitor_csv(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    if not path.exists():
        return records
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            records.append(
                {
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "power_watts": float(row["power_watts"]),
                    "gpu_clock_mhz": float(row["gpu_clock_mhz"]),
                }
            )
    return records


def write_timeline_plot(
    series: list[tuple[str, Path]],
    output_path: Path,
    title: str,
) -> Path | None:
    loaded = [
        (label, read_monitor_csv(path))
        for label, path in series
        if path and path.exists()
    ]
    loaded = [(label, records) for label, records in loaded if records]
    if not loaded:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for label, records in loaded:
        x_values = [record["elapsed_seconds"] for record in records]
        axes[0].plot(
            x_values,
            [record["power_watts"] for record in records],
            label=label,
            linewidth=1.3,
        )
        axes[1].plot(
            x_values,
            [record["gpu_clock_mhz"] for record in records],
            label=label,
            linewidth=1.3,
        )
    axes[0].set_ylabel("Power (W)")
    axes[1].set_ylabel("GPU Clock (MHz)")
    axes[1].set_xlabel("Elapsed Time (s)")
    axes[0].set_title(title)
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_kernel_profile_csv(
    torch: Any,
    run_once: Callable[[], None],
    output_path: Path,
    repeat: int = 1,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(max(1, repeat)):
            run_once()
        torch.cuda.synchronize()

    def device_time_us(event: Any, self_time: bool) -> float:
        names = (
            ("self_device_time_total", "self_cuda_time_total")
            if self_time
            else ("device_time_total", "cuda_time_total")
        )
        for name in names:
            value = getattr(event, name, None)
            if value not in (None, ""):
                return float(value)
        return 0.0

    aggregates: dict[str, dict[str, Any]] = {}
    for event in profiler.events():
        device_type = getattr(event, "device_type", "")
        if "CUDA" not in str(device_type):
            continue
        name = str(getattr(event, "name", None) or getattr(event, "key", ""))
        if not name:
            continue
        total_us = device_time_us(event, self_time=False)
        self_us = device_time_us(event, self_time=True)
        if total_us <= 0.0 and self_us <= 0.0:
            continue
        aggregate = aggregates.setdefault(
            name,
            {
                "name": name,
                "count": 0,
                "self_cuda_time_ms": 0.0,
                "total_cuda_time_ms": 0.0,
            },
        )
        aggregate["count"] += int(getattr(event, "count", 1) or 1)
        aggregate["self_cuda_time_ms"] += self_us / 1000.0
        aggregate["total_cuda_time_ms"] += (
            total_us if total_us > 0.0 else self_us
        ) / 1000.0

    rows: list[dict[str, Any]] = list(aggregates.values())
    for row in rows:
        total_us = float(row["total_cuda_time_ms"]) * 1000.0
        count = int(row["count"])
        row["avg_total_cuda_time_us"] = total_us / count if count else ""

    rows.sort(key=lambda row: float(row["total_cuda_time_ms"]), reverse=True)
    write_csv(
        output_path,
        [
            "name",
            "count",
            "self_cuda_time_ms",
            "total_cuda_time_ms",
            "avg_total_cuda_time_us",
        ],
        rows,
    )
    return output_path
