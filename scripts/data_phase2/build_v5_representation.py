#!/usr/bin/env python3
"""Build the V5 two-floor canvas representation and verify lossless round-trip.

The user-provided site width and depth are treated as the rectangular buildable
boundary. Cells inside that boundary may be empty; cells outside are ignored.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "phase2_v5"
VOXEL_SIZE = 300
CANVAS_X = 88
CANVAS_Y = 88
FLOORS = (1, 2)
OUTSIDE_CLASS = 255
OUTSIDE_INSTANCE = 65535
ROOM_TYPES = (
    "entryway",
    "living_room",
    "dining_room",
    "kitchen",
    "bedroom",
    "bathroom",
    "corridor",
    "stairs",
    "utility",
    "balcony",
    "multi_purpose",
)
CLASS_MAP = {"empty": 0, **{name: index + 1 for index, name in enumerate(ROOM_TYPES)}}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def room_floors(room: dict) -> tuple[int, ...]:
    values = room.get("floors")
    if isinstance(values, list) and values:
        return tuple(sorted({int(value) for value in values}))
    return (int(room.get("floor", 1)),)


def exact_cell(value: float, field: str) -> int:
    cells = float(value) / VOXEL_SIZE
    rounded = round(cells)
    if abs(cells - rounded) > 1e-6:
        raise ValueError("{}={} is not aligned to {}mm".format(field, value, VOXEL_SIZE))
    return int(rounded)


def canvas_placement(site_x: float, site_y: float) -> dict:
    width = exact_cell(site_x, "building_size.x")
    depth = exact_cell(site_y, "building_size.y")
    if width > CANVAS_X or depth > CANVAS_Y:
        raise ValueError("site {}x{} cells exceeds {}x{} canvas".format(
            width, depth, CANVAS_X, CANVAS_Y
        ))
    x0 = (CANVAS_X - width) // 2
    y0 = (CANVAS_Y - depth) // 2
    return {
        "site_cells_x": width,
        "site_cells_y": depth,
        "canvas_x0": x0,
        "canvas_y0": y0,
        "canvas_x1": x0 + width,
        "canvas_y1": y0 + depth,
    }


def encode_house(data: dict) -> tuple[dict[str, np.ndarray], dict]:
    building = data["metadata"]["building_size"]
    placement = canvas_placement(building["x"], building["y"])
    x0, y0 = placement["canvas_x0"], placement["canvas_y0"]
    x1, y1 = placement["canvas_x1"], placement["canvas_y1"]

    site_mask = np.zeros((CANVAS_X, CANVAS_Y), dtype=np.uint8)
    site_mask[x0:x1, y0:y1] = 1
    class_grid = np.full((2, CANVAS_X, CANVAS_Y), OUTSIDE_CLASS, dtype=np.uint8)
    instance_grid = np.full(
        (2, CANVAS_X, CANVAS_Y), OUTSIDE_INSTANCE, dtype=np.uint16
    )
    class_grid[:, x0:x1, y0:y1] = CLASS_MAP["empty"]
    instance_grid[:, x0:x1, y0:y1] = 0
    cross_floor_mask = np.zeros((2, CANVAS_X, CANVAS_Y), dtype=np.uint8)
    double_height_void_mask = np.zeros((2, CANVAS_X, CANVAS_Y), dtype=np.uint8)

    instance_table = []
    collisions = []
    for instance_index, room in enumerate(data.get("rooms", []), 1):
        room_type = str(room["type"])
        if room_type not in CLASS_MAP:
            raise ValueError("unknown room type: {}".format(room_type))
        floors = room_floors(room)
        rx0 = x0 + exact_cell(room["box_min"][0], "room.box_min.x")
        ry0 = y0 + exact_cell(room["box_min"][1], "room.box_min.y")
        rx1 = x0 + exact_cell(room["box_max"][0], "room.box_max.x")
        ry1 = y0 + exact_cell(room["box_max"][1], "room.box_max.y")
        if not (x0 <= rx0 < rx1 <= x1 and y0 <= ry0 < ry1 <= y1):
            raise ValueError("{} lies outside site boundary".format(room["id"]))
        for floor in floors:
            floor_index = floor - 1
            existing = instance_grid[floor_index, rx0:rx1, ry0:ry1]
            occupied = existing[(existing != 0) & (existing != OUTSIDE_INSTANCE)]
            if occupied.size:
                collisions.append({
                    "room_id": room["id"],
                    "floor": floor,
                    "other_instances": sorted({int(value) for value in occupied}),
                })
                continue
            class_grid[floor_index, rx0:rx1, ry0:ry1] = CLASS_MAP[room_type]
            instance_grid[floor_index, rx0:rx1, ry0:ry1] = instance_index
            if len(floors) > 1:
                cross_floor_mask[floor_index, rx0:rx1, ry0:ry1] = 1
                if room_type != "stairs":
                    double_height_void_mask[floor_index, rx0:rx1, ry0:ry1] = 1
        instance_table.append({
            "instance_index": instance_index,
            "id": str(room["id"]),
            "type": room_type,
            "floors": list(floors),
            "box_min": [float(value) for value in room["box_min"]],
            "box_max": [float(value) for value in room["box_max"]],
        })

    building_mask = (
        (class_grid != CLASS_MAP["empty"]) & (class_grid != OUTSIDE_CLASS)
    ).astype(np.uint8)
    empty_inside_mask = (
        (class_grid == CLASS_MAP["empty"]) & (site_mask[None, :, :] == 1)
    ).astype(np.uint8)
    floor_overlap_mask = (
        (building_mask[0] == 1) & (building_mask[1] == 1)
    ).astype(np.uint8)
    arrays = {
        "site_mask": site_mask,
        "class_grid": class_grid,
        "instance_grid": instance_grid,
        "building_mask": building_mask,
        "empty_inside_mask": empty_inside_mask,
        "cross_floor_mask": cross_floor_mask,
        "double_height_void_mask": double_height_void_mask,
        "floor_overlap_mask": floor_overlap_mask,
    }
    metadata = {
        "schema": "graphspace_v5_canvas_v1",
        "house_id": data.get("house_id"),
        "voxel_size_mm": VOXEL_SIZE,
        "canvas_cells": [CANVAS_X, CANVAS_Y],
        "floor_count": 2,
        "site_size_mm": [float(building["x"]), float(building["y"])],
        "placement": placement,
        "outside_class": OUTSIDE_CLASS,
        "empty_class": CLASS_MAP["empty"],
        "class_map": CLASS_MAP,
        "instance_table": instance_table,
        "collisions": collisions,
    }
    return arrays, metadata


def decode_instances(arrays: dict[str, np.ndarray], metadata: dict) -> list[dict]:
    grid = arrays["instance_grid"]
    placement = metadata["placement"]
    x0, y0 = placement["canvas_x0"], placement["canvas_y0"]
    recovered = []
    by_index = {
        int(item["instance_index"]): item for item in metadata["instance_table"]
    }
    for instance_index, item in sorted(by_index.items()):
        positions = np.argwhere(grid == instance_index)
        if positions.size == 0:
            continue
        floor_indices = sorted({int(position[0]) for position in positions})
        xs = positions[:, 1]
        ys = positions[:, 2]
        floors = [floor_index + 1 for floor_index in floor_indices]
        recovered.append({
            "id": item["id"],
            "type": item["type"],
            "floors": floors,
            "box_min": [
                float((int(xs.min()) - x0) * VOXEL_SIZE),
                float((int(ys.min()) - y0) * VOXEL_SIZE),
                float((min(floors) - 1) * 3000),
            ],
            "box_max": [
                float((int(xs.max()) + 1 - x0) * VOXEL_SIZE),
                float((int(ys.max()) + 1 - y0) * VOXEL_SIZE),
                float(max(floors) * 3000),
            ],
        })
    return recovered


def canonical_room(room: dict) -> tuple:
    return (
        str(room["id"]),
        str(room["type"]),
        tuple(room_floors(room)),
        tuple(float(value) for value in room["box_min"]),
        tuple(float(value) for value in room["box_max"]),
    )


def evaluate_round_trip(data: dict, arrays: dict[str, np.ndarray], metadata: dict) -> dict:
    expected = {canonical_room(room) for room in data.get("rooms", [])}
    recovered = {canonical_room(room) for room in decode_instances(arrays, metadata)}
    missing = sorted(expected - recovered)
    extra = sorted(recovered - expected)
    site_cells = int(arrays["site_mask"].sum())
    floor_stats = {}
    for floor_index, floor in enumerate(FLOORS):
        occupied = int(arrays["building_mask"][floor_index].sum())
        empty = int(arrays["empty_inside_mask"][floor_index].sum())
        floor_stats[str(floor)] = {
            "occupied_cells": occupied,
            "empty_inside_cells": empty,
            "occupancy_ratio": occupied / site_cells if site_cells else 0.0,
        }
    return {
        "exact": not missing and not extra and not metadata["collisions"],
        "expected_instances": len(expected),
        "recovered_instances": len(recovered),
        "missing_instances": [list(item) for item in missing],
        "extra_instances": [list(item) for item in extra],
        "collision_count": len(metadata["collisions"]),
        "floor_stats": floor_stats,
        "cross_floor_cells": int(arrays["cross_floor_mask"][0].sum()),
        "double_height_void_cells": int(
            arrays["double_height_void_mask"][0].sum()
        ),
        "floor_overlap_cells": int(arrays["floor_overlap_mask"].sum()),
    }


def process_dataset(data_dir: Path, output_dir: Path, save_samples: bool = True) -> dict:
    reports = []
    occupancy = {"1": [], "2": []}
    failures = Counter()
    for path in sorted(data_dir.glob("house_*.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        house_id = path.stem
        try:
            arrays, metadata = encode_house(data)
            report = evaluate_round_trip(data, arrays, metadata)
            report["house_id"] = house_id
            if save_samples:
                sample_dir = output_dir / "samples"
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(sample_dir / "{}.npz".format(house_id), **arrays)
                write_json(sample_dir / "{}.json".format(house_id), metadata)
            for floor in ("1", "2"):
                occupancy[floor].append(report["floor_stats"][floor]["occupancy_ratio"])
            if not report["exact"]:
                failures["round_trip"] += 1
        except Exception as exc:
            report = {"house_id": house_id, "exact": False, "error": str(exc)}
            failures["exception"] += 1
        reports.append(report)

    exact_count = sum(bool(report.get("exact")) for report in reports)
    summary = {
        "schema": "graphspace_v5_canvas_v1",
        "dataset_count": len(reports),
        "exact_round_trip_count": exact_count,
        "failed_round_trip_count": len(reports) - exact_count,
        "failure_counts": dict(failures),
        "canvas_cells": [CANVAS_X, CANVAS_Y],
        "voxel_size_mm": VOXEL_SIZE,
        "site_semantics": "building_size is the user-provided rectangular buildable boundary",
        "class_semantics": {
            "outside": OUTSIDE_CLASS,
            "empty_inside": CLASS_MAP["empty"],
            "functional_classes": CLASS_MAP,
        },
        "occupancy_ratio": {
            floor: {
                "min": min(values) if values else None,
                "mean": float(np.mean(values)) if values else None,
                "max": max(values) if values else None,
            }
            for floor, values in occupancy.items()
        },
        "houses_with_cross_floor_space": sum(
            int(report.get("cross_floor_cells", 0) > 0) for report in reports
        ),
        "houses_with_non_stair_double_height": sum(
            int(report.get("double_height_void_cells", 0) > 0) for report in reports
        ),
        "reports": reports,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-save-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_dataset(
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        save_samples=not args.no_save_samples,
    )
    print("Processed: {}".format(summary["dataset_count"]))
    print("Exact round-trip: {}/{}".format(
        summary["exact_round_trip_count"], summary["dataset_count"]
    ))
    print("Wrote: {}".format(args.output_dir.resolve() / "summary.json"))


if __name__ == "__main__":
    main()
