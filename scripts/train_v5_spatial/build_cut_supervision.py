#!/usr/bin/env python3
"""Build topology-conditioned 3D block-cut action supervision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
OUTPUT_DIR = ROOT / "data" / "phase6_spatial_cut"
VOXEL_MM = 300.0
ROOM_TYPES = [
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
]
TYPE_TO_ID = {value: index for index, value in enumerate(ROOM_TYPES)}
STOP_AXIS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--split-path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_cell(value: float) -> int:
    cell = round(float(value) / VOXEL_MM)
    if abs(cell * VOXEL_MM - float(value)) > 1e-4:
        raise ValueError(f"value is not on 300mm grid: {value}")
    return int(cell)


def room_floors(room: dict) -> list[int]:
    if room.get("floors"):
        return sorted(int(value) for value in room["floors"])
    return [int(room.get("floor", 1))]


def normalize_rooms(payload: dict) -> list[dict]:
    rooms = []
    for room in payload["rooms"]:
        mins = [exact_cell(value) for value in room["box_min"]]
        maxs = [exact_cell(value) for value in room["box_max"]]
        rooms.append(
            {
                "id": str(room["id"]),
                "type": str(room["type"]),
                "type_id": TYPE_TO_ID[str(room["type"])],
                "floors": room_floors(room),
                "bounds": mins + maxs,
            }
        )
    return rooms


def axis_overlap(a: list[int], b: list[int], axis: int) -> int:
    return max(0, min(a[axis + 3], b[axis + 3]) - max(a[axis], b[axis]))


def relation(a: dict, b: dict) -> int | None:
    ab, bb = a["bounds"], b["bounds"]
    for axis in range(3):
        touching = ab[axis + 3] == bb[axis] or bb[axis + 3] == ab[axis]
        other = [item for item in range(3) if item != axis]
        if touching and all(axis_overlap(ab, bb, item) > 0 for item in other):
            return 1 if axis == 2 else 0
    return None


def graph_record(rooms: list[dict], site_cells: list[int]) -> dict:
    nodes = []
    for room in rooms:
        bounds = room["bounds"]
        floors = room["floors"]
        area_ratio = (
            (bounds[3] - bounds[0])
            * (bounds[4] - bounds[1])
            / max(site_cells[0] * site_cells[1], 1)
        )
        nodes.append(
            {
                "room_id": room["id"],
                "type_id": room["type_id"],
                "floor_1": int(1 in floors),
                "floor_2": int(2 in floors),
                "target_area_ratio": area_ratio,
            }
        )
    edges = []
    for index, room_a in enumerate(rooms):
        for other_index in range(index + 1, len(rooms)):
            rel = relation(room_a, rooms[other_index])
            if rel is not None:
                edges.append([index, other_index, rel])
                edges.append([other_index, index, rel])
    return {"nodes": nodes, "edges": edges}


def valid_cuts(
    room_indices: list[int],
    rooms: list[dict],
    region: list[int],
) -> list[tuple[int, int, list[int], list[int]]]:
    candidates = []
    for axis in range(3):
        positions = sorted(
            {
                rooms[index]["bounds"][axis]
                for index in room_indices
            }
            | {
                rooms[index]["bounds"][axis + 3]
                for index in room_indices
            }
        )
        for cut in positions:
            if cut <= region[axis] or cut >= region[axis + 3]:
                continue
            left, right = [], []
            valid = True
            for index in room_indices:
                bounds = rooms[index]["bounds"]
                if bounds[axis + 3] <= cut:
                    left.append(index)
                elif bounds[axis] >= cut:
                    right.append(index)
                else:
                    valid = False
                    break
            if valid and left and right:
                candidates.append((axis, cut, left, right))
    return candidates


def choose_cut(
    room_indices: list[int],
    rooms: list[dict],
    region: list[int],
) -> tuple[int, int, list[int], list[int]] | None:
    candidates = valid_cuts(room_indices, rooms, region)
    if not candidates:
        return None

    def score(item: tuple[int, int, list[int], list[int]]) -> tuple:
        axis, cut, left, right = item
        balance = min(len(left), len(right))
        imbalance = abs(len(left) - len(right))
        position = (cut - region[axis]) / max(region[axis + 3] - region[axis], 1)
        # Prefer a floor split when it is valid, then balanced central cuts.
        return balance, -imbalance, int(axis == 2), -abs(position - 0.5)

    return max(candidates, key=score)


def build_actions(rooms: list[dict], site_cells: list[int]) -> list[dict]:
    actions = []

    def visit(room_indices: list[int], region: list[int], depth: int) -> None:
        selected = choose_cut(room_indices, rooms, region)
        if selected is None:
            actions.append(
                {
                    "region": region,
                    "room_indices": room_indices,
                    "axis": STOP_AXIS,
                    "cut_ratio": 0.5,
                    "left_indices": [],
                    "right_indices": [],
                    "left_fraction": 0.5,
                    "depth": depth,
                    "resolved": len(room_indices) == 1,
                }
            )
            return
        axis, cut, left, right = selected
        ratio = (cut - region[axis]) / (region[axis + 3] - region[axis])
        actions.append(
            {
                "region": region,
                "room_indices": room_indices,
                "axis": axis,
                "cut_ratio": ratio,
                "left_indices": left,
                "right_indices": right,
                "left_fraction": len(left) / len(room_indices),
                "depth": depth,
                "resolved": False,
            }
        )
        left_region = list(region)
        right_region = list(region)
        left_region[axis + 3] = cut
        right_region[axis] = cut
        visit(left, left_region, depth + 1)
        visit(right, right_region, depth + 1)

    visit(list(range(len(rooms))), [0, 0, 0, site_cells[0], site_cells[1], 20], 0)
    return actions


def build_sample(path: Path) -> dict:
    payload = read_json(path)
    building = payload["metadata"]["building_size"]
    site_cells = [exact_cell(building["x"]), exact_cell(building["y"]), 20]
    rooms = normalize_rooms(payload)
    actions = build_actions(rooms, site_cells)
    resolved_rooms = sum(
        len(action["room_indices"]) == 1 and action["axis"] == STOP_AXIS
        for action in actions
    )
    return {
        "schema": "graphspace_v5_spatial_cut_supervision_v1",
        "house_id": path.stem,
        "site_cells": site_cells,
        "graph": graph_record(rooms, site_cells),
        "actions": actions,
        "stats": {
            "room_count": len(rooms),
            "action_count": len(actions),
            "cut_action_count": sum(
                action["axis"] != STOP_AXIS for action in actions
            ),
            "stop_action_count": sum(
                action["axis"] == STOP_AXIS for action in actions
            ),
            "blocked_stop_count": sum(
                action["axis"] == STOP_AXIS and len(action["room_indices"]) > 1
                for action in actions
            ),
            "resolved_room_count": resolved_rooms,
            "resolution_rate": resolved_rooms / max(len(rooms), 1),
            "fully_separable": resolved_rooms == len(rooms),
        },
    }


def main() -> None:
    args = parse_args()
    split = read_json(args.split_path)
    house_ids = split["train"] + split["validation"] + split["test"]
    if args.max_samples is not None:
        house_ids = house_ids[: args.max_samples]
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for house_id in house_ids:
        sample = build_sample(args.input_dir / f"{house_id}.json")
        (samples_dir / f"{house_id}.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reports.append(sample["stats"])
    summary = {
        "schema": "graphspace_v5_spatial_cut_summary_v1",
        "sample_count": len(reports),
        "fully_separable_count": sum(item["fully_separable"] for item in reports),
        "mean_resolution_rate": sum(item["resolution_rate"] for item in reports)
        / max(len(reports), 1),
        "action_count": sum(item["action_count"] for item in reports),
        "cut_action_count": sum(item["cut_action_count"] for item in reports),
        "stop_action_count": sum(item["stop_action_count"] for item in reports),
        "blocked_stop_count": sum(item["blocked_stop_count"] for item in reports),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
