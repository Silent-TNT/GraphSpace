#!/usr/bin/env python3
"""Build ID-invariant V5 room-instance supervision and verify decoding."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "phase2_v5" / "samples"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "phase5_instances"
OUTSIDE_CLASS = 255
OUTSIDE_INSTANCE = 65535
EMPTY_CLASS = 0
VOXEL_SIZE = 300
FLOOR_HEIGHT = 3000
GAUSSIAN_SIGMA = 1.5


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_sample(input_dir: Path, house_id: str) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(input_dir / "{}.npz".format(house_id)) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    with (input_dir / "{}.json".format(house_id)).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return arrays, metadata


def instance_centers(instance_grid: np.ndarray) -> dict[tuple[int, int], tuple[float, float]]:
    centers = {}
    for floor_index in range(instance_grid.shape[0]):
        values = np.unique(instance_grid[floor_index])
        for instance_id in values:
            instance_id = int(instance_id)
            if instance_id in (0, OUTSIDE_INSTANCE):
                continue
            cells = np.argwhere(instance_grid[floor_index] == instance_id)
            centers[(floor_index, instance_id)] = (
                float(cells[:, 0].mean()),
                float(cells[:, 1].mean()),
            )
    return centers


def draw_gaussian(
    target: np.ndarray, floor_index: int, center_x: float, center_y: float
) -> None:
    radius = int(math.ceil(3.0 * GAUSSIAN_SIGMA))
    x0 = max(0, int(math.floor(center_x)) - radius)
    x1 = min(target.shape[1], int(math.ceil(center_x)) + radius + 1)
    y0 = max(0, int(math.floor(center_y)) - radius)
    y1 = min(target.shape[2], int(math.ceil(center_y)) + radius + 1)
    for x in range(x0, x1):
        for y in range(y0, y1):
            value = math.exp(
                -((x - center_x) ** 2 + (y - center_y) ** 2)
                / (2.0 * GAUSSIAN_SIGMA ** 2)
            )
            target[floor_index, x, y] = max(
                target[floor_index, x, y], value
            )


def build_boundary_mask(instance_grid: np.ndarray) -> np.ndarray:
    boundary = np.zeros(instance_grid.shape, dtype=np.uint8)
    for floor_index in range(instance_grid.shape[0]):
        grid = instance_grid[floor_index]
        occupied = (grid != 0) & (grid != OUTSIDE_INSTANCE)
        for x, y in np.argwhere(occupied):
            instance_id = int(grid[x, y])
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = int(x + dx), int(y + dy)
                if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                    boundary[floor_index, x, y] = 1
                    break
                if int(grid[nx, ny]) != instance_id:
                    boundary[floor_index, x, y] = 1
                    break
    return boundary


def build_instance_supervision(
    arrays: dict[str, np.ndarray], metadata: dict
) -> tuple[dict[str, np.ndarray], dict]:
    instance_grid = arrays["instance_grid"]
    class_grid = arrays["class_grid"]
    centers = instance_centers(instance_grid)
    center_heatmap = np.zeros(instance_grid.shape, dtype=np.float32)
    center_offset = np.zeros((2, 2, instance_grid.shape[1], instance_grid.shape[2]), dtype=np.float32)
    center_valid_mask = np.zeros(instance_grid.shape, dtype=np.uint8)
    class_instance_counts = np.zeros((2, 11), dtype=np.int16)
    floor_instance_counts = np.zeros(2, dtype=np.int16)
    cross_floor_instance_pairs = []

    table_by_index = {
        int(item["instance_index"]): item for item in metadata["instance_table"]
    }
    for (floor_index, instance_id), (center_x, center_y) in centers.items():
        cells = np.argwhere(instance_grid[floor_index] == instance_id)
        draw_gaussian(center_heatmap, floor_index, center_x, center_y)
        center_valid_mask[floor_index][
            instance_grid[floor_index] == instance_id
        ] = 1
        for x, y in cells:
            center_offset[floor_index, 0, x, y] = center_x - float(x)
            center_offset[floor_index, 1, x, y] = center_y - float(y)
        class_id = int(class_grid[floor_index, cells[0, 0], cells[0, 1]])
        if 1 <= class_id <= 11:
            class_instance_counts[floor_index, class_id - 1] += 1
        floor_instance_counts[floor_index] += 1

    for instance_id, item in sorted(table_by_index.items()):
        floors = [int(value) for value in item["floors"]]
        if floors == [1, 2]:
            cross_floor_instance_pairs.append({
                "instance_index": instance_id,
                "type": item["type"],
                "floor_indices": [0, 1],
                "centers": [
                    list(centers[(0, instance_id)]),
                    list(centers[(1, instance_id)]),
                ],
            })

    supervision = {
        "center_heatmap": center_heatmap,
        "center_offset": center_offset,
        "center_valid_mask": center_valid_mask,
        "boundary_mask": build_boundary_mask(instance_grid),
        "floor_instance_counts": floor_instance_counts,
        "class_instance_counts": class_instance_counts,
    }
    supervision_metadata = {
        "schema": "graphspace_v5_instance_supervision_v1",
        "house_id": metadata["house_id"],
        "heatmap_sigma_cells": GAUSSIAN_SIGMA,
        "offset_units": "grid_cells",
        "floor_instance_counts": floor_instance_counts.tolist(),
        "class_instance_counts": class_instance_counts.tolist(),
        "building_instance_count": len(metadata["instance_table"]),
        "cross_floor_instance_count": len(cross_floor_instance_pairs),
        "cross_floor_instance_pairs": cross_floor_instance_pairs,
    }
    return supervision, supervision_metadata


def decode_floor_instances(
    class_grid: np.ndarray,
    building_mask: np.ndarray,
    center_offset: np.ndarray,
) -> np.ndarray:
    """Decode instances by grouping same-class cells that vote for one center."""
    decoded = np.zeros(class_grid.shape, dtype=np.uint16)
    next_id = 1
    for floor_index in range(class_grid.shape[0]):
        groups: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
        for x, y in np.argwhere(building_mask[floor_index] == 1):
            class_id = int(class_grid[floor_index, x, y])
            center_x = float(x) + float(center_offset[floor_index, 0, x, y])
            center_y = float(y) + float(center_offset[floor_index, 1, x, y])
            key = (
                class_id,
                int(round(center_x * 1000)),
                int(round(center_y * 1000)),
            )
            groups.setdefault(key, []).append((int(x), int(y)))
        for key in sorted(groups):
            for x, y in groups[key]:
                decoded[floor_index, x, y] = next_id
            next_id += 1
    return decoded


def instance_partition_signature(grid: np.ndarray) -> set[tuple]:
    signature = set()
    for floor_index in range(grid.shape[0]):
        for instance_id in np.unique(grid[floor_index]):
            instance_id = int(instance_id)
            if instance_id in (0, OUTSIDE_INSTANCE):
                continue
            cells = np.argwhere(grid[floor_index] == instance_id)
            signature.add((
                floor_index,
                tuple(sorted((int(x), int(y)) for x, y in cells)),
            ))
    return signature


def decode_building_instances(
    floor_instances: np.ndarray,
    class_grid: np.ndarray,
    cross_floor_mask: np.ndarray,
) -> list[dict]:
    records = []
    floor_records = {}
    for floor_index in range(floor_instances.shape[0]):
        for instance_id in np.unique(floor_instances[floor_index]):
            instance_id = int(instance_id)
            if instance_id == 0:
                continue
            cells = np.argwhere(floor_instances[floor_index] == instance_id)
            class_id = int(class_grid[floor_index, cells[0, 0], cells[0, 1]])
            record = {
                "floor_instance_id": instance_id,
                "floor_index": floor_index,
                "class_id": class_id,
                "cells": {(int(x), int(y)) for x, y in cells},
                "cross_floor": bool(
                    cross_floor_mask[floor_index][
                        floor_instances[floor_index] == instance_id
                    ].any()
                ),
            }
            floor_records[(floor_index, instance_id)] = record

    used = set()
    for key, record in sorted(floor_records.items()):
        if key in used:
            continue
        floors = [record["floor_index"] + 1]
        merged_cells = set(record["cells"])
        used.add(key)
        if record["cross_floor"]:
            other_floor = 1 - record["floor_index"]
            candidates = [
                other for other in floor_records.values()
                if other["floor_index"] == other_floor
                and other["class_id"] == record["class_id"]
                and other["cross_floor"]
                and other["cells"] & record["cells"]
            ]
            if candidates:
                other = max(
                    candidates,
                    key=lambda value: len(value["cells"] & record["cells"]),
                )
                other_key = (other["floor_index"], other["floor_instance_id"])
                if other_key not in used:
                    floors.append(other_floor + 1)
                    merged_cells |= other["cells"]
                    used.add(other_key)
        xs = [cell[0] for cell in merged_cells]
        ys = [cell[1] for cell in merged_cells]
        records.append({
            "class_id": record["class_id"],
            "floors": sorted(floors),
            "grid_box": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
        })
    return records


def expected_building_signature(metadata: dict) -> set[tuple]:
    placement = metadata["placement"]
    x0, y0 = placement["canvas_x0"], placement["canvas_y0"]
    result = set()
    for item in metadata["instance_table"]:
        box_min, box_max = item["box_min"], item["box_max"]
        result.add((
            int(metadata["class_map"][item["type"]]),
            tuple(int(value) for value in item["floors"]),
            (
                x0 + int(round(box_min[0] / VOXEL_SIZE)),
                y0 + int(round(box_min[1] / VOXEL_SIZE)),
                x0 + int(round(box_max[0] / VOXEL_SIZE)),
                y0 + int(round(box_max[1] / VOXEL_SIZE)),
            ),
        ))
    return result


def decoded_building_signature(records: list[dict]) -> set[tuple]:
    return {
        (
            int(record["class_id"]),
            tuple(int(value) for value in record["floors"]),
            tuple(int(value) for value in record["grid_box"]),
        )
        for record in records
    }


def evaluate_supervision(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    supervision: dict[str, np.ndarray],
) -> dict:
    decoded_floor = decode_floor_instances(
        arrays["class_grid"],
        arrays["building_mask"],
        supervision["center_offset"],
    )
    expected_partition = instance_partition_signature(arrays["instance_grid"])
    decoded_partition = instance_partition_signature(decoded_floor)
    building_records = decode_building_instances(
        decoded_floor,
        arrays["class_grid"],
        arrays["cross_floor_mask"],
    )
    expected_building = expected_building_signature(metadata)
    decoded_building = decoded_building_signature(building_records)
    return {
        "floor_partition_exact": expected_partition == decoded_partition,
        "building_instances_exact": expected_building == decoded_building,
        "expected_floor_instance_count": len(expected_partition),
        "decoded_floor_instance_count": len(decoded_partition),
        "expected_building_instance_count": len(expected_building),
        "decoded_building_instance_count": len(decoded_building),
        "missing_building_instances": [
            list(value) for value in sorted(expected_building - decoded_building)
        ],
        "extra_building_instances": [
            list(value) for value in sorted(decoded_building - expected_building)
        ],
    }


def process_dataset(input_dir: Path, output_dir: Path, save_samples: bool = True) -> dict:
    reports = []
    failures = Counter()
    max_floor_instances = [0, 0]
    max_class_instances = np.zeros((2, 11), dtype=np.int16)
    total_cross_floor = 0
    for metadata_path in sorted(input_dir.glob("house_*.json")):
        house_id = metadata_path.stem
        try:
            arrays, metadata = load_sample(input_dir, house_id)
            supervision, supervision_metadata = build_instance_supervision(
                arrays, metadata
            )
            report = evaluate_supervision(
                arrays, metadata, supervision
            )
            report["house_id"] = house_id
            report["floor_instance_counts"] = supervision[
                "floor_instance_counts"
            ].tolist()
            report["building_instance_count"] = supervision_metadata[
                "building_instance_count"
            ]
            report["cross_floor_instance_count"] = supervision_metadata[
                "cross_floor_instance_count"
            ]
            total_cross_floor += supervision_metadata[
                "cross_floor_instance_count"
            ]
            for floor_index in range(2):
                max_floor_instances[floor_index] = max(
                    max_floor_instances[floor_index],
                    int(supervision["floor_instance_counts"][floor_index]),
                )
            max_class_instances = np.maximum(
                max_class_instances, supervision["class_instance_counts"]
            )
            if save_samples:
                sample_dir = output_dir / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    sample_dir / "{}.npz".format(house_id), **supervision
                )
                write_json(
                    sample_dir / "{}.json".format(house_id),
                    supervision_metadata,
                )
            if not report["floor_partition_exact"]:
                failures["floor_partition"] += 1
            if not report["building_instances_exact"]:
                failures["building_instances"] += 1
        except Exception as exc:
            report = {
                "house_id": house_id,
                "floor_partition_exact": False,
                "building_instances_exact": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures["exception"] += 1
        reports.append(report)

    summary = {
        "schema": "graphspace_v5_instance_supervision_v1",
        "dataset_count": len(reports),
        "floor_partition_exact_count": sum(
            bool(item.get("floor_partition_exact")) for item in reports
        ),
        "building_instances_exact_count": sum(
            bool(item.get("building_instances_exact")) for item in reports
        ),
        "failure_counts": dict(failures),
        "max_floor_instance_counts": max_floor_instances,
        "max_class_instance_counts": max_class_instances.tolist(),
        "total_cross_floor_instances": total_cross_floor,
        "supervision_fields": {
            "center_heatmap": "per-floor room-center Gaussian target",
            "center_offset": "occupied-cell vector to its room center",
            "center_valid_mask": "cells where offset loss is active",
            "boundary_mask": "occupied cells touching another instance or empty",
            "floor_instance_counts": "number of instances on each floor",
            "class_instance_counts": "number of instances by floor and class",
        },
        "reports": reports,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-save-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_dataset(
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        save_samples=not args.no_save_samples,
    )
    print("Processed: {}".format(summary["dataset_count"]))
    print("Floor partition exact: {}/{}".format(
        summary["floor_partition_exact_count"], summary["dataset_count"]
    ))
    print("Building instances exact: {}/{}".format(
        summary["building_instances_exact_count"], summary["dataset_count"]
    ))
    print("Wrote: {}".format(args.output_dir.resolve() / "summary.json"))


if __name__ == "__main__":
    main()
