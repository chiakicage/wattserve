from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def normalize_device_slug(device_name: str | None) -> str:
    if not device_name:
        return "unknown_device"

    normalized = device_name.upper()
    if "A100" in normalized and "40" in normalized:
        if "PCIE" in normalized:
            return "a100_40g_pcie"
        if "SXM" in normalized:
            return "a100_40g_sxm"
        return "a100_40g"

    tokens = re.findall(r"[a-z0-9]+", device_name.lower())
    filtered_tokens = [token for token in tokens if token != "nvidia"]
    return "_".join(filtered_tokens) or "unknown_device"


def normalize_device_label(device_name: str | None) -> str:
    if not device_name:
        return "Unknown device"

    normalized = device_name.upper()
    if "A100" in normalized and "40" in normalized:
        if "PCIE" in normalized:
            return "A100 40G PCIe"
        if "SXM" in normalized:
            return "A100 40G SXM"
        return "A100 40G"

    cleaned = device_name.strip()
    if cleaned.upper().startswith("NVIDIA "):
        cleaned = cleaned[7:]
    return cleaned or "Unknown device"


def resolve_device_slug(metadata: dict[str, Any]) -> str:
    explicit_slug = metadata.get("published_device_slug")
    if explicit_slug:
        return str(explicit_slug)
    environment = metadata.get("environment", {})
    return normalize_device_slug(environment.get("cuda_device_name"))


def resolve_device_label(metadata: dict[str, Any]) -> str:
    explicit_label = metadata.get("published_device_label")
    if explicit_label:
        return str(explicit_label)
    environment = metadata.get("environment", {})
    return normalize_device_label(environment.get("cuda_device_name"))


def resolve_device_snapshot_dir(
    latest_root_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    return latest_root_dir / resolve_device_slug(metadata)


def list_device_snapshots(latest_root_dir: Path) -> list[dict[str, Any]]:
    if not latest_root_dir.exists():
        return []

    snapshots: list[dict[str, Any]] = []
    for child in sorted(latest_root_dir.iterdir()):
        if not child.is_dir():
            continue

        metadata_path = child / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())
        snapshots.append(
            {
                "slug": child.name,
                "label": resolve_device_label(metadata),
                "cuda_device_name": (
                    metadata.get("environment", {}).get("cuda_device_name")
                    or "n/a"
                ),
                "output_dir": child,
                "benchmark_markdown": child / "BENCHMARK.md",
                "summary_csv": child / "summary.csv",
                "metadata_json": metadata_path,
                "plots_dir": child / "plots",
                "source_output_dir": metadata.get("source_output_dir"),
                "run_started_at_utc": metadata.get("run_started_at_utc"),
            }
        )

    return sorted(
        snapshots,
        key=lambda snapshot: (str(snapshot["label"]), str(snapshot["slug"])),
    )
