#!/usr/bin/env python3
"""Analyze Google Benchmark JSON output for llcnn.

The script produces:
  - processed benchmark rows as CSV
  - CPU/GPU speedup pairs as CSV
  - optional PNG figures when matplotlib is available
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TIME_TO_SECONDS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}

AGGREGATE_PRIORITY = ("mean", "median")
DEVICE_TOKENS = {"CPU", "GPU", "CPU_to_GPU", "GPU_to_CPU"}


@dataclass(frozen=True)
class BenchmarkRow:
    run_name: str
    base_name: str
    device: str
    params: tuple[str, ...]
    aggregate_name: str
    real_time_seconds: float
    cpu_time_seconds: float
    iterations: int
    threads: int
    counters: dict[str, float]

    @property
    def size_key(self) -> str:
        return "/".join(self.params)

    @property
    def pair_key(self) -> tuple[str, tuple[str, ...], str]:
        return (self.base_name, self.params, self.aggregate_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CSV tables and figures from cnn_benchmark JSON output."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="benchmarks/results.json",
        help="Path to Google Benchmark JSON output.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmarks/results/processed",
        help="Directory where CSV files and figures will be written.",
    )
    parser.add_argument(
        "--aggregate",
        default="mean",
        choices=AGGREGATE_PRIORITY,
        help="Aggregate used for speedup tables and figures.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only generate CSV files.",
    )
    return parser.parse_args()


def as_float(value: Any, default: float = math.nan) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def seconds(value: Any, unit: str) -> float:
    scale = TIME_TO_SECONDS.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported benchmark time unit: {unit}")
    return as_float(value) * scale


def split_run_name(run_name: str, label: str | None) -> tuple[str, str, tuple[str, ...]]:
    parts = run_name.split("/")
    if len(parts) < 2:
        return run_name, label or "unknown", ()

    device_index = None
    for index, part in enumerate(parts):
        if part in DEVICE_TOKENS:
            device_index = index
            break

    if device_index is None:
        return parts[0], label or "unknown", tuple(parts[1:])

    base_name = "/".join(parts[:device_index])
    device = parts[device_index]
    params = tuple(parts[device_index + 1 :])
    return base_name, device, params


def extract_counters(entry: dict[str, Any]) -> dict[str, float]:
    excluded = {
        "name",
        "family_index",
        "per_family_instance_index",
        "run_name",
        "run_type",
        "repetitions",
        "threads",
        "aggregate_name",
        "aggregate_unit",
        "iterations",
        "real_time",
        "cpu_time",
        "time_unit",
        "label",
        "error_occurred",
        "error_message",
    }

    counters: dict[str, float] = {}
    for key, value in entry.items():
        if key in excluded:
            continue
        if isinstance(value, (int, float)):
            counters[key] = float(value)
    return counters


def load_rows(path: Path) -> list[BenchmarkRow]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    rows: list[BenchmarkRow] = []
    for entry in data.get("benchmarks", []):
        if entry.get("error_occurred"):
            continue

        aggregate_name = entry.get("aggregate_name")
        if aggregate_name not in AGGREGATE_PRIORITY:
            continue

        run_name = str(entry.get("run_name", entry.get("name", "")))
        base_name, device, params = split_run_name(run_name, entry.get("label"))
        unit = str(entry.get("time_unit", "ns"))

        rows.append(
            BenchmarkRow(
                run_name=run_name,
                base_name=base_name,
                device=device,
                params=params,
                aggregate_name=str(aggregate_name),
                real_time_seconds=seconds(entry.get("real_time"), unit),
                cpu_time_seconds=seconds(entry.get("cpu_time"), unit),
                iterations=int(entry.get("iterations", 0)),
                threads=int(entry.get("threads", 1)),
                counters=extract_counters(entry),
            )
        )

    return rows


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def sorted_rows(rows: Iterable[BenchmarkRow]) -> list[BenchmarkRow]:
    return sorted(
        rows,
        key=lambda row: (
            natural_key(row.base_name),
            natural_key(row.size_key),
            row.device,
            row.aggregate_name,
        ),
    )


def write_processed_csv(rows: list[BenchmarkRow], path: Path) -> None:
    counter_names = sorted({name for row in rows for name in row.counters})
    fields = [
        "run_name",
        "base_name",
        "device",
        "params",
        "aggregate",
        "real_time_s",
        "cpu_time_s",
        "iterations",
        "threads",
        *counter_names,
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted_rows(rows):
            record: dict[str, Any] = {
                "run_name": row.run_name,
                "base_name": row.base_name,
                "device": row.device,
                "params": row.size_key,
                "aggregate": row.aggregate_name,
                "real_time_s": row.real_time_seconds,
                "cpu_time_s": row.cpu_time_seconds,
                "iterations": row.iterations,
                "threads": row.threads,
            }
            record.update(row.counters)
            writer.writerow(record)


def build_speedup_rows(rows: list[BenchmarkRow], aggregate: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, tuple[str, ...], str], dict[str, BenchmarkRow]] = {}
    for row in rows:
        if row.aggregate_name != aggregate:
            continue
        by_key.setdefault(row.pair_key, {})[row.device] = row

    speedups: list[dict[str, Any]] = []
    for (base_name, params, _aggregate), devices in by_key.items():
        cpu = devices.get("CPU")
        gpu = devices.get("GPU")
        if cpu is None or gpu is None:
            continue

        speedups.append(
            {
                "base_name": base_name,
                "params": "/".join(params),
                "aggregate": aggregate,
                "cpu_real_time_s": cpu.real_time_seconds,
                "gpu_real_time_s": gpu.real_time_seconds,
                "speedup_cpu_over_gpu": cpu.real_time_seconds / gpu.real_time_seconds,
                "cpu_samples_per_second": cpu.counters.get("samples_per_second", math.nan),
                "gpu_samples_per_second": gpu.counters.get("samples_per_second", math.nan),
                "cpu_items_per_second": cpu.counters.get("items_per_second", math.nan),
                "gpu_items_per_second": gpu.counters.get("items_per_second", math.nan),
                "cpu_flop_per_second": cpu.counters.get("flop_per_second", math.nan),
                "gpu_flop_per_second": gpu.counters.get("flop_per_second", math.nan),
            }
        )

    return sorted(
        speedups,
        key=lambda row: (natural_key(str(row["base_name"])), natural_key(str(row["params"]))),
    )


def write_speedup_csv(speedups: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "base_name",
        "params",
        "aggregate",
        "cpu_real_time_s",
        "gpu_real_time_s",
        "speedup_cpu_over_gpu",
        "cpu_samples_per_second",
        "gpu_samples_per_second",
        "cpu_items_per_second",
        "gpu_items_per_second",
        "cpu_flop_per_second",
        "gpu_flop_per_second",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(speedups)


def build_transfer_rows(rows: list[BenchmarkRow], aggregate: str) -> list[dict[str, Any]]:
    transfers: list[dict[str, Any]] = []
    for row in rows:
        if row.aggregate_name != aggregate:
            continue
        if row.base_name != "tensor_transfer":
            continue
        if row.device not in {"CPU_to_GPU", "GPU_to_CPU"}:
            continue

        bytes_count = row.counters.get("bytes", math.nan)
        bandwidth = bytes_count / row.real_time_seconds if row.real_time_seconds > 0 else math.nan
        transfers.append(
            {
                "direction": row.device,
                "params": row.size_key,
                "aggregate": aggregate,
                "real_time_s": row.real_time_seconds,
                "bytes": bytes_count,
                "elements": row.counters.get("elements", math.nan),
                "bytes_per_second": bandwidth,
                "gib_per_second": bandwidth / (1024.0 ** 3),
            }
        )

    return sorted(
        transfers,
        key=lambda row: (str(row["direction"]), natural_key(str(row["params"]))),
    )


def write_transfer_csv(transfers: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "direction",
        "params",
        "aggregate",
        "real_time_s",
        "bytes",
        "elements",
        "bytes_per_second",
        "gib_per_second",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(transfers)


def try_import_matplotlib():
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except ImportError:
        return None


def plot_speedups(speedups: list[dict[str, Any]], out_dir: Path) -> int:
    plt = try_import_matplotlib()
    if plt is None:
        return 0

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for row in speedups:
        by_benchmark.setdefault(str(row["base_name"]), []).append(row)

    count = 0
    for base_name, rows in by_benchmark.items():
        labels = [str(row["params"]) or "default" for row in rows]
        values = [float(row["speedup_cpu_over_gpu"]) for row in rows]

        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(labels)), 4.0))
        ax.bar(labels, values)
        ax.axhline(1.0, color="black", linewidth=1.0)
        ax.set_title(f"Speedup CPU/GPU - {base_name}")
        ax.set_ylabel("Speedup")
        ax.set_xlabel("Parameters")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_name).strip("_")
        fig.savefig(figure_dir / f"{safe_name}_speedup.png", dpi=160)
        plt.close(fig)
        count += 1

    return count


def plot_transfers(transfers: list[dict[str, Any]], out_dir: Path) -> int:
    if not transfers:
        return 0

    plt = try_import_matplotlib()
    if plt is None:
        return 0

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    by_direction: dict[str, list[dict[str, Any]]] = {}
    for row in transfers:
        by_direction.setdefault(str(row["direction"]), []).append(row)

    labels = sorted({str(row["params"]) for row in transfers}, key=natural_key)
    x_positions = list(range(len(labels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(labels)), 4.0))
    directions = sorted(by_direction)
    for direction_index, direction in enumerate(directions):
        rows_by_param = {str(row["params"]): row for row in by_direction[direction]}
        values = [
            float(rows_by_param[label]["gib_per_second"])
            if label in rows_by_param else 0.0
            for label in labels
        ]
        offset = (direction_index - (len(directions) - 1) / 2.0) * width
        ax.bar(
            [position + offset for position in x_positions],
            values,
            width=width,
            label=direction,
        )

    ax.set_title("Bandwidth de transferencias CPU/GPU")
    ax.set_ylabel("GiB/s")
    ax.set_xlabel("Elementos float32 transferidos")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "tensor_transfer_bandwidth.png", dpi=160)
    plt.close(fig)
    return 1


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    if not rows:
        raise SystemExit(f"No aggregate benchmark rows found in {input_path}")

    processed_csv = out_dir / "benchmark_rows.csv"
    speedup_csv = out_dir / "speedups_cpu_gpu.csv"
    transfer_csv = out_dir / "transfer_bandwidth.csv"

    write_processed_csv(rows, processed_csv)
    speedups = build_speedup_rows(rows, args.aggregate)
    write_speedup_csv(speedups, speedup_csv)
    transfers = build_transfer_rows(rows, args.aggregate)
    write_transfer_csv(transfers, transfer_csv)

    figure_count = 0
    if not args.no_plots:
        figure_count = plot_speedups(speedups, out_dir)
        figure_count += plot_transfers(transfers, out_dir)

    print(f"Input: {input_path}")
    print(f"Rows: {len(rows)}")
    print(f"Processed CSV: {processed_csv}")
    print(f"Speedup CSV: {speedup_csv}")
    print(f"Transfer CSV: {transfer_csv}")
    if args.no_plots:
        print("Figures: skipped")
    elif figure_count == 0:
        print("Figures: skipped (matplotlib is not installed or no speedup rows)")
    else:
        print(f"Figures: {out_dir / 'figures'} ({figure_count} files)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
