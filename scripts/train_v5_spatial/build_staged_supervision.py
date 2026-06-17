#!/usr/bin/env python3
"""Build staged topology, semantic and 3D supervision for V5 generation."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
PHASE2_DIR = ROOT / "data" / "phase2_v5" / "samples"
SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
OUTPUT_DIR = ROOT / "data" / "phase7_staged_spatial"
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
RIGID_ORDER = {
    "living_room": 0,
    "dining_room": 1,
    "kitchen": 2,
    "bedroom": 3,
    "bathroom": 4,
    "multi_purpose": 5,
}
TRAFFIC_TYPES = {"entryway", "corridor", "stairs"}
SERVICE_TYPES = {"utility", "balcony"}
LIGHTING_TO_ID = {"none": 0, "indirect": 1, "direct": 2}
STAGES = [
    "stair_core",
    "floor_split",
    "envelope_empty",
    "traffic_reserve",
    "rigid_function",
    "service_adjust",
    "reachability_optimize",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--phase2-dir", type=Path, default=PHASE2_DIR)
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


def exterior_sides(room: dict, site_cells: list[int]) -> list[str]:
    bounds = room["bounds"]
    sides = []
    if bounds[0] == 0:
        sides.append("W")
    if bounds[3] == site_cells[0]:
        sides.append("E")
    if bounds[1] == 0:
        sides.append("S")
    if bounds[4] == site_cells[1]:
        sides.append("N")
    return sides


def normalize_rooms(payload: dict, site_cells: list[int]) -> list[dict]:
    rooms = []
    for index, room in enumerate(payload["rooms"]):
        mins = [exact_cell(value) for value in room["box_min"]]
        maxs = [exact_cell(value) for value in room["box_max"]]
        normalized = {
            "instance_token": f"instance_{index:03d}",
            "source_id": str(room["id"]),
            "type": str(room["type"]),
            "type_id": TYPE_TO_ID[str(room["type"])],
            "floors": room_floors(room),
            "bounds": mins + maxs,
            "lighting_access": str(room.get("lighting_access", "none")),
            "lighting_priority": int(room.get("lighting_priority", 0)),
        }
        normalized["exterior_sides"] = exterior_sides(normalized, site_cells)
        rooms.append(normalized)
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
        area_ratio = (
            (bounds[3] - bounds[0])
            * (bounds[4] - bounds[1])
            / max(site_cells[0] * site_cells[1], 1)
        )
        nodes.append(
            {
                "instance_token": room["instance_token"],
                "type": room["type"],
                "type_id": room["type_id"],
                "floor_1": int(1 in room["floors"]),
                "floor_2": int(2 in room["floors"]),
                "target_area_ratio": area_ratio,
                "exterior_sides": room["exterior_sides"],
                "lighting_access": room["lighting_access"],
                "lighting_id": LIGHTING_TO_ID.get(room["lighting_access"], 0),
                "lighting_priority": room["lighting_priority"],
            }
        )
    edges = []
    for index, room_a in enumerate(rooms):
        for other_index in range(index + 1, len(rooms)):
            rel = relation(room_a, rooms[other_index])
            if rel is not None:
                edges.append([index, other_index, rel])
                edges.append([other_index, index, rel])
    return {
        "nodes": nodes,
        "edges": edges,
        "relation_types": {"horizontal_contact": 0, "vertical_contact": 1},
    }


def local_phase2_arrays(
    phase2_path: Path,
    phase2_metadata: dict,
) -> dict[str, np.ndarray]:
    placement = phase2_metadata["placement"]
    x0, x1 = placement["canvas_x0"], placement["canvas_x1"]
    y0, y1 = placement["canvas_y0"], placement["canvas_y1"]
    with np.load(phase2_path) as source:
        return {
            "site_mask": source["site_mask"][x0:x1, y0:y1].copy(),
            "building_mask": source["building_mask"][:, x0:x1, y0:y1].copy(),
            "empty_mask": source["empty_inside_mask"][:, x0:x1, y0:y1].copy(),
            "cross_floor_mask": source["cross_floor_mask"][:, x0:x1, y0:y1].copy(),
            "double_height_void_mask": source[
                "double_height_void_mask"
            ][:, x0:x1, y0:y1].copy(),
        }


def mask_for_types(
    rooms: list[dict],
    site_cells: list[int],
    room_types: set[str],
) -> np.ndarray:
    mask = np.zeros((2, site_cells[0], site_cells[1]), dtype=np.uint8)
    for room in rooms:
        if room["type"] not in room_types:
            continue
        x0, y0, _, x1, y1, _ = room["bounds"]
        for floor in room["floors"]:
            mask[floor - 1, x0:x1, y0:y1] = 1
    return mask


def reachable_indices(graph: dict, starts: list[int]) -> set[int]:
    adjacency: dict[int, set[int]] = {
        index: set() for index in range(len(graph["nodes"]))
    }
    for source, target, _ in graph["edges"]:
        adjacency[source].add(target)
    seen: set[int] = set()
    queue = deque(starts)
    while queue:
        index = queue.popleft()
        if index in seen:
            continue
        seen.add(index)
        queue.extend(adjacency[index] - seen)
    return seen


def reachability_report(rooms: list[dict], graph: dict) -> dict:
    entry_indices = [
        index for index, room in enumerate(rooms) if room["type"] == "entryway"
    ]
    stair_indices = [
        index for index, room in enumerate(rooms) if room["type"] == "stairs"
    ]
    reachable = reachable_indices(graph, entry_indices)
    required = [
        index for index, room in enumerate(rooms) if room["type"] != "balcony"
    ]
    stair_floor_contacts = {}
    for stair_index in stair_indices:
        contacts = {1: False, 2: False}
        for source, target, _ in graph["edges"]:
            if source != stair_index:
                continue
            for floor in rooms[target]["floors"]:
                contacts[floor] = True
        stair_floor_contacts[str(stair_index)] = contacts
    unreachable = [index for index in required if index not in reachable]
    return {
        "entry_indices": entry_indices,
        "stair_indices": stair_indices,
        "reachable_indices": sorted(reachable),
        "unreachable_indices": unreachable,
        "all_required_reachable": bool(entry_indices) and not unreachable,
        "stair_floor_contacts": stair_floor_contacts,
        "stairs_contact_both_floors": bool(stair_indices)
        and all(all(item.values()) for item in stair_floor_contacts.values()),
        "semantics": (
            "Functional-block face contact only; no doors are generated or checked."
        ),
    }


def ordered_indices(rooms: list[dict], types: set[str]) -> list[int]:
    indices = [
        index for index, room in enumerate(rooms) if room["type"] in types
    ]
    return sorted(
        indices,
        key=lambda index: (
            min(rooms[index]["floors"]),
            RIGID_ORDER.get(rooms[index]["type"], 99),
            rooms[index]["bounds"][0],
            rooms[index]["bounds"][1],
        ),
    )


def stage_actions(rooms: list[dict], reachability: dict) -> list[dict]:
    stair_indices = ordered_indices(rooms, {"stairs"})
    traffic_indices = ordered_indices(rooms, TRAFFIC_TYPES)
    rigid_indices = ordered_indices(rooms, set(RIGID_ORDER))
    service_indices = ordered_indices(rooms, SERVICE_TYPES)
    return [
        {
            "stage_id": 0,
            "stage": "stair_core",
            "target_indices": stair_indices,
            "target_bounds": [rooms[index]["bounds"] for index in stair_indices],
            "protected_indices": stair_indices,
        },
        {
            "stage_id": 1,
            "stage": "floor_split",
            "axis": "z",
            "cut_cell": 10,
            "cut_ratio": 0.5,
            "protected_indices": stair_indices,
        },
        {
            "stage_id": 2,
            "stage": "envelope_empty",
            "target_arrays": ["building_mask", "empty_mask"],
            "protected_indices": stair_indices,
        },
        {
            "stage_id": 3,
            "stage": "traffic_reserve",
            "target_indices": traffic_indices,
            "target_array": "traffic_mask",
            "protected_indices": stair_indices,
        },
        {
            "stage_id": 4,
            "stage": "rigid_function",
            "ordered_target_indices": rigid_indices,
            "target_bounds": [rooms[index]["bounds"] for index in rigid_indices],
            "conditioning": ["topology", "semantics", "lighting", "3d_voxels"],
            "protected_indices": stair_indices,
        },
        {
            "stage_id": 5,
            "stage": "service_adjust",
            "ordered_target_indices": service_indices,
            "target_bounds": [rooms[index]["bounds"] for index in service_indices],
            "adjustable_array": "traffic_mask",
            "protected_indices": stair_indices,
        },
        {
            "stage_id": 6,
            "stage": "reachability_optimize",
            "target_all_required_reachable": True,
            "oracle_report": reachability,
            "protected_indices": stair_indices,
        },
    ]


def build_sample(
    processed_path: Path,
    phase2_dir: Path = PHASE2_DIR,
) -> tuple[dict, dict[str, np.ndarray]]:
    payload = read_json(processed_path)
    building = payload["metadata"]["building_size"]
    site_cells = [exact_cell(building["x"]), exact_cell(building["y"]), 20]
    rooms = normalize_rooms(payload, site_cells)
    graph = graph_record(rooms, site_cells)
    phase2_metadata = read_json(phase2_dir / f"{processed_path.stem}.json")
    arrays = local_phase2_arrays(
        phase2_dir / f"{processed_path.stem}.npz",
        phase2_metadata,
    )
    arrays["stair_mask"] = mask_for_types(rooms, site_cells, {"stairs"})
    arrays["traffic_mask"] = mask_for_types(rooms, site_cells, TRAFFIC_TYPES)
    arrays["rigid_mask"] = mask_for_types(rooms, site_cells, set(RIGID_ORDER))
    arrays["service_mask"] = mask_for_types(rooms, site_cells, SERVICE_TYPES)
    reachability = reachability_report(rooms, graph)
    stair_indices = [
        index for index, room in enumerate(rooms) if room["type"] == "stairs"
    ]
    stairs_span = bool(stair_indices) and all(
        set(rooms[index]["floors"]) == {1, 2} for index in stair_indices
    )
    sample = {
        "schema": "graphspace_v5_staged_spatial_supervision_v1",
        "house_id": processed_path.stem,
        "site_cells": site_cells,
        "stage_order": STAGES,
        "graph": graph,
        "actions": stage_actions(rooms, reachability),
        "stats": {
            "room_count": len(rooms),
            "stair_count": len(stair_indices),
            "stairs_span_both_floors": stairs_span,
            "empty_cell_count": int(arrays["empty_mask"].sum()),
            "has_explicit_empty": bool(arrays["empty_mask"].any()),
            "all_required_reachable": reachability["all_required_reachable"],
            "stairs_contact_both_floors": reachability[
                "stairs_contact_both_floors"
            ],
        },
    }
    return sample, arrays


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
        sample, arrays = build_sample(
            args.input_dir / f"{house_id}.json",
            args.phase2_dir,
        )
        (samples_dir / f"{house_id}.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(samples_dir / f"{house_id}.npz", **arrays)
        reports.append(sample["stats"])
    summary = {
        "schema": "graphspace_v5_staged_spatial_summary_v1",
        "sample_count": len(reports),
        "stage_order": STAGES,
        "stair_core_valid_count": sum(
            item["stairs_span_both_floors"] for item in reports
        ),
        "explicit_empty_count": sum(
            item["has_explicit_empty"] for item in reports
        ),
        "oracle_reachability_pass_count": sum(
            item["all_required_reachable"] for item in reports
        ),
        "stair_contact_both_floors_count": sum(
            item["stairs_contact_both_floors"] for item in reports
        ),
        "total_empty_cells": sum(item["empty_cell_count"] for item in reports),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
