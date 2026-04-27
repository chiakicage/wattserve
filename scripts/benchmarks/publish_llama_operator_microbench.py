#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .device_snapshot import (
        list_device_snapshots,
        resolve_device_label,
        resolve_device_slug,
        resolve_device_snapshot_dir,
    )
    from .run_llama_operator_microbench import (
        build_benchmark_markdown,
        write_metadata_json,
    )
except ImportError:
    from device_snapshot import (  # type: ignore[no-redef]
        list_device_snapshots,
        resolve_device_label,
        resolve_device_slug,
        resolve_device_snapshot_dir,
    )
    from run_llama_operator_microbench import (  # type: ignore[no-redef]
        build_benchmark_markdown,
        write_metadata_json,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_MICROBENCH_INDEX_PATH = REPO_ROOT / "MICROBENCH.md"
GIT_TRACKED_LATEST_RESULTS_ROOT = (
    REPO_ROOT / "results" / "llama_operator_microbench" / "latest"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_summary_rows(summary_csv_path: Path) -> list[dict[str, Any]]:
    with summary_csv_path.open("r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def _device_snapshot_table_lines(
    snapshots: list[dict[str, Any]],
) -> list[str]:
    if not snapshots:
        return ["No device-specific latest snapshots are currently tracked."]

    lines = [
        "| Device | Slug | Report | Summary CSV | Metadata | Source Run | Run Started At |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for snapshot in snapshots:
        source_output_dir = snapshot.get("source_output_dir")
        source_display = (
            f"`{_display_path(Path(source_output_dir))}`"
            if source_output_dir
            else "n/a"
        )
        lines.append(
            "| "
            f"{snapshot['label']} | "
            f"`{snapshot['slug']}` | "
            f"[report]({_display_path(snapshot['benchmark_markdown'])}) | "
            f"[summary]({_display_path(snapshot['summary_csv'])}) | "
            f"[metadata]({_display_path(snapshot['metadata_json'])}) | "
            f"{source_display} | "
            f"`{snapshot.get('run_started_at_utc') or 'n/a'}` |"
        )
    return lines


def build_latest_root_benchmark_markdown(
    latest_root_dir: Path,
    snapshots: list[dict[str, Any]],
) -> str:
    latest_root_display = _display_path(latest_root_dir)
    lines = [
        "# Llama Operator Microbenchmark Latest Snapshots",
        "",
        "This directory stores the git-tracked latest operator microbenchmark snapshot for each device.",
        "",
        f"- Snapshots root: `{latest_root_display}`",
        "- Canonical suite: `Llama-13B`, `prompt_len=8192`, `dtype=bfloat16`",
        "- Publishing is device-scoped and explicit. Timestamped runs do not overwrite these snapshots unless requested.",
        "",
        "## Devices",
        "",
        *_device_snapshot_table_lines(snapshots),
    ]
    return "\n".join(lines) + "\n"


def build_root_index_markdown(
    latest_root_dir: Path,
    snapshots: list[dict[str, Any]],
) -> str:
    latest_root_display = _display_path(latest_root_dir)
    latest_root_report = latest_root_dir / "BENCHMARK.md"
    lines = [
        "# Latest Llama Operator Microbenchmark",
        "",
        "This file indexes the device-specific git-tracked latest operator microbenchmark snapshots.",
        "",
        f"- Git-tracked latest snapshots root: `{latest_root_display}`",
        f"- Latest-by-device index: [{_display_path(latest_root_report)}]({_display_path(latest_root_report)})",
        "- Canonical suite: `Llama-13B`, `prompt_len=8192`, `dtype=bfloat16`",
        "- Workloads cover `o`, `attn`, `qkv`, `gate_up`, `down`, and `fused_add_norm` combinations derived from the current Llama block shape.",
        "",
        "## Devices",
        "",
        *_device_snapshot_table_lines(snapshots),
    ]
    return "\n".join(lines) + "\n"


def refresh_latest_indices(
    latest_root_dir: Path,
    root_index_path: Path = ROOT_MICROBENCH_INDEX_PATH,
) -> None:
    latest_root_dir.mkdir(parents=True, exist_ok=True)
    snapshots = list_device_snapshots(latest_root_dir)
    (latest_root_dir / "BENCHMARK.md").write_text(
        build_latest_root_benchmark_markdown(latest_root_dir, snapshots)
    )
    root_index_path.write_text(
        build_root_index_markdown(latest_root_dir, snapshots)
    )


def publish_git_tracked_latest_snapshot(
    source_output_dir: Path,
    root_index_path: Path = ROOT_MICROBENCH_INDEX_PATH,
    git_snapshot_output_dir: Path = GIT_TRACKED_LATEST_RESULTS_ROOT,
) -> dict[str, Path]:
    source_output_dir = source_output_dir.resolve()
    git_snapshot_output_dir = git_snapshot_output_dir.resolve()

    source_summary_csv_path = source_output_dir / "summary.csv"
    if not source_summary_csv_path.exists():
        raise FileNotFoundError(
            f"summary.csv not found for git snapshot publish: {source_summary_csv_path}"
        )

    source_metadata_path = source_output_dir / "metadata.json"
    source_monitor_dir = source_output_dir / "monitor"
    source_metadata = load_metadata(source_metadata_path)
    summary_rows = load_summary_rows(source_summary_csv_path)

    device_slug = resolve_device_slug(source_metadata)
    device_label = resolve_device_label(source_metadata)
    device_snapshot_output_dir = resolve_device_snapshot_dir(
        git_snapshot_output_dir,
        source_metadata,
    )

    if device_snapshot_output_dir.exists():
        shutil.rmtree(device_snapshot_output_dir)
    device_snapshot_output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        source_summary_csv_path,
        device_snapshot_output_dir / "summary.csv",
    )
    if source_monitor_dir.exists():
        shutil.copytree(
            source_monitor_dir,
            device_snapshot_output_dir / "monitor",
        )

    snapshot_metadata = dict(source_metadata)
    snapshot_metadata["source_output_dir"] = str(source_output_dir)
    snapshot_metadata["source_summary_csv"] = str(source_summary_csv_path)
    snapshot_metadata["source_metadata_json"] = str(source_metadata_path)
    snapshot_metadata["published_snapshot_root_dir"] = str(
        git_snapshot_output_dir
    )
    snapshot_metadata["published_snapshot_dir"] = str(
        device_snapshot_output_dir
    )
    snapshot_metadata["published_snapshot_generated_at_utc"] = _utc_now_iso()
    snapshot_metadata["published_device_slug"] = device_slug
    snapshot_metadata["published_device_label"] = device_label
    snapshot_metadata["output_dir"] = str(device_snapshot_output_dir)
    snapshot_metadata["summary_csv"] = str(
        device_snapshot_output_dir / "summary.csv"
    )
    snapshot_metadata["metadata_json"] = str(
        device_snapshot_output_dir / "metadata.json"
    )
    snapshot_metadata["benchmark_markdown"] = str(
        device_snapshot_output_dir / "BENCHMARK.md"
    )

    write_metadata_json(
        device_snapshot_output_dir / "metadata.json",
        snapshot_metadata,
    )
    (device_snapshot_output_dir / "BENCHMARK.md").write_text(
        build_benchmark_markdown(summary_rows, snapshot_metadata)
    )

    refresh_latest_indices(
        latest_root_dir=git_snapshot_output_dir,
        root_index_path=root_index_path,
    )

    return {
        "device_snapshot_output_dir": device_snapshot_output_dir,
        "latest_root_benchmark_md": git_snapshot_output_dir / "BENCHMARK.md",
        "root_index_path": root_index_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish an existing Llama operator microbenchmark result as the git-tracked latest snapshot."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Existing result directory under results/llama_operator_microbench/<UTC_TIMESTAMP>.",
    )
    parser.add_argument(
        "--root_index_path",
        type=Path,
        default=ROOT_MICROBENCH_INDEX_PATH,
        help="Repo-root operator microbench index to refresh.",
    )
    parser.add_argument(
        "--git_snapshot_output_dir",
        type=Path,
        default=GIT_TRACKED_LATEST_RESULTS_ROOT,
        help="Git-tracked latest snapshot root directory.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    published_paths = publish_git_tracked_latest_snapshot(
        source_output_dir=args.output_dir,
        root_index_path=args.root_index_path,
        git_snapshot_output_dir=args.git_snapshot_output_dir,
    )
    print(
        "Published latest operator microbenchmark snapshot: "
        f"{published_paths['device_snapshot_output_dir']}"
    )
    print(
        "Latest family index: " f"{published_paths['latest_root_benchmark_md']}"
    )
    print(f"Root MICROBENCH index: {published_paths['root_index_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
