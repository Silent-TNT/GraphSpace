#!/usr/bin/env python3
"""Build inferred functional-group to rectangular-part supervision.

The source processed JSON stores rectangular room records. The Rhino modeling
rules allow one functional space to be represented by multiple rectangles, but
the current processed data has no explicit group id. This builder adds a
conservative inferred grouping layer without mutating the source data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_OUTPUT = ROOT / "data" / "phase10_functional_parts"
VOXEL_MM = 300.0
FLOOR_Z = {1: (0.0, 3000.0), 2: (3000.0, 6000.0)}

# Bedroom and bathroom are intentionally excluded: adjacent same-type records
# are often separate rooms, and the current data has no ground-truth group id.
GROUPABLE_TYPES = {
    "entryway",
    "living_room",
    "dining_room",
    "kitchen",
    "corridor",
    "stairs",
    "utility",
    "balcony",
    "multi_purpose",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-shared-mm", type=float, default=VOXEL_MM)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_floors(room: dict[str, Any]) -> list[int]:
    if room.get("floors"):
        return sorted({int(value) for value in room["floors"]})
    z0 = float(room["box_min"][2])
    z1 = float(room["box_max"][2])
    floors = [
        floor
        for floor, (floor_z0, floor_z1) in FLOOR_Z.items()
        if min(z1, floor_z1) - max(z0, floor_z0) > 1e-6
    ]
    return floors or ([2] if z0 >= 3000.0 else [1])


def axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def same_floor_face_contact(
    left: dict[str, Any],
    right: dict[str, Any],
    min_shared_mm: float,
) -> bool:
    if not (set(left["floors"]) & set(right["floors"])):
        return False
    lx0, ly0, _ = left["box_min"]
    lx1, ly1, _ = left["box_max"]
    rx0, ry0, _ = right["box_min"]
    rx1, ry1, _ = right["box_max"]
    x_touch = abs(lx1 - rx0) <= 1e-6 or abs(rx1 - lx0) <= 1e-6
    y_touch = abs(ly1 - ry0) <= 1e-6 or abs(ry1 - ly0) <= 1e-6
    if x_touch and axis_overlap(ly0, ly1, ry0, ry1) >= min_shared_mm:
        return True
    if y_touch and axis_overlap(lx0, lx1, rx0, rx1) >= min_shared_mm:
        return True
    return False


def source_functional_id(room: dict[str, Any]) -> str | None:
    for key in ("functional_id", "group_id", "parent_id"):
        value = room.get(key)
        if value:
            return str(value)
    room_id = str(room.get("id", ""))
    if "_part_" in room_id:
        return room_id.split("_part_", 1)[0]
    return None


def normalize_source_rooms(house: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = []
    for index, room in enumerate(house.get("rooms", [])):
        room_type = str(room.get("type", "unknown"))
        room_id = str(room.get("id", f"{room_type}_{index}"))
        item = dict(room)
        item["id"] = room_id
        item["part_id"] = room_id
        item["type"] = room_type
        item["box_min"] = [float(value) for value in room["box_min"]]
        item["box_max"] = [float(value) for value in room["box_max"]]
        item["floors"] = infer_floors(item)
        item["_source_functional_id"] = source_functional_id(room)
        normalized.append(item)
    return normalized


def connected_components(indices: list[int], adjacency: dict[int, set[int]]) -> list[list[int]]:
    remaining = set(indices)
    components = []
    while remaining:
        start = min(remaining)
        queue: deque[int] = deque([start])
        remaining.remove(start)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def infer_functional_groups(
    rooms: list[dict[str, Any]],
    min_shared_mm: float = VOXEL_MM,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for index, room in enumerate(rooms):
        by_type[str(room["type"])].append(index)

    group_for_index: dict[int, str] = {}
    groups: list[dict[str, Any]] = []
    group_offsets: Counter[str] = Counter()

    explicit_groups: dict[str, list[int]] = defaultdict(list)
    for index, room in enumerate(rooms):
        if room["_source_functional_id"]:
            explicit_groups[room["_source_functional_id"]].append(index)
    for group_id, indices in sorted(explicit_groups.items()):
        room_types = {rooms[index]["type"] for index in indices}
        if len(room_types) != 1:
            continue
        room_type = next(iter(room_types))
        for index in indices:
            group_for_index[index] = group_id
        groups.append(group_record(group_id, room_type, indices, rooms, "source_group_id"))

    for room_type, indices in sorted(by_type.items()):
        pending = [index for index in indices if index not in group_for_index]
        if not pending:
            continue
        if room_type not in GROUPABLE_TYPES:
            for index in pending:
                group_id = str(rooms[index]["id"])
                group_for_index[index] = group_id
                groups.append(
                    group_record(
                        group_id,
                        room_type,
                        [index],
                        rooms,
                        "non_groupable_singleton",
                    )
                )
            continue
        adjacency = {index: set() for index in pending}
        for offset, left_index in enumerate(pending):
            for right_index in pending[offset + 1 :]:
                if same_floor_face_contact(
                    rooms[left_index],
                    rooms[right_index],
                    min_shared_mm,
                ):
                    adjacency[left_index].add(right_index)
                    adjacency[right_index].add(left_index)
        for component in connected_components(pending, adjacency):
            group_id = f"{room_type}_{group_offsets[room_type]}"
            group_offsets[room_type] += 1
            reason = (
                "same_type_adjacent_component"
                if len(component) > 1
                else "groupable_singleton"
            )
            for index in component:
                group_for_index[index] = group_id
            groups.append(group_record(group_id, room_type, component, rooms, reason))

    grouped_rooms = []
    part_offsets: Counter[str] = Counter()
    groups_by_id = {group["functional_id"]: group for group in groups}
    for room in rooms:
        group_id = group_for_index[id_to_index(room, rooms)]
        part_index = part_offsets[group_id]
        part_offsets[group_id] += 1
        output = {
            key: value
            for key, value in room.items()
            if not key.startswith("_") and key != "functional_id"
        }
        output["functional_id"] = group_id
        output["part_index"] = part_index
        output["group_inference"] = groups_by_id[group_id]["inference"]
        grouped_rooms.append(output)
    return grouped_rooms, sorted(groups, key=lambda item: item["functional_id"])


def id_to_index(room: dict[str, Any], rooms: list[dict[str, Any]]) -> int:
    for index, candidate in enumerate(rooms):
        if candidate is room:
            return index
    raise ValueError("room is not part of the provided room list")


def group_record(
    group_id: str,
    room_type: str,
    indices: list[int],
    rooms: list[dict[str, Any]],
    inference: str,
) -> dict[str, Any]:
    part_ids = [str(rooms[index]["part_id"]) for index in indices]
    floors = sorted({floor for index in indices for floor in rooms[index]["floors"]})
    return {
        "functional_id": group_id,
        "type": room_type,
        "part_ids": part_ids,
        "part_count": len(part_ids),
        "floors": floors,
        "inference": inference,
    }


def build_house_payload(path: Path, min_shared_mm: float) -> dict[str, Any]:
    house = read_json(path)
    rooms = normalize_source_rooms(house)
    grouped_rooms, groups = infer_functional_groups(rooms, min_shared_mm)
    return {
        "schema": "graphspace_functional_part_groups_v1",
        "source": str(path.relative_to(ROOT)),
        "house_id": str(house.get("house_id", path.stem)),
        "metadata": house.get("metadata", {}),
        "groupable_types": sorted(GROUPABLE_TYPES),
        "rooms": grouped_rooms,
        "functional_groups": groups,
        "stats": {
            "part_count": len(grouped_rooms),
            "functional_group_count": len(groups),
            "multipart_group_count": sum(group["part_count"] > 1 for group in groups),
            "raw_part_counts": dict(Counter(room["type"] for room in grouped_rooms)),
            "functional_group_counts": dict(Counter(group["type"] for group in groups)),
            "multipart_groups_by_type": dict(
                Counter(
                    group["type"]
                    for group in groups
                    if int(group["part_count"]) > 1
                )
            ),
            "inference_counts": dict(Counter(group["inference"] for group in groups)),
        },
    }


def build_summary(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    total_parts = sum(payload["stats"]["part_count"] for payload in payloads)
    total_groups = sum(payload["stats"]["functional_group_count"] for payload in payloads)
    multipart_groups = sum(payload["stats"]["multipart_group_count"] for payload in payloads)
    raw_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    multipart_by_type: Counter[str] = Counter()
    inference_counts: Counter[str] = Counter()
    for payload in payloads:
        raw_counts.update(payload["stats"]["raw_part_counts"])
        group_counts.update(payload["stats"]["functional_group_counts"])
        multipart_by_type.update(payload["stats"]["multipart_groups_by_type"])
        inference_counts.update(payload["stats"]["inference_counts"])
    return {
        "schema": "graphspace_functional_part_groups_summary_v1",
        "source": str(DEFAULT_PROCESSED.relative_to(ROOT)),
        "house_count": len(payloads),
        "part_count": total_parts,
        "functional_group_count": total_groups,
        "multipart_group_count": multipart_groups,
        "part_to_group_ratio": total_parts / max(total_groups, 1),
        "raw_part_counts": dict(sorted(raw_counts.items())),
        "functional_group_counts": dict(sorted(group_counts.items())),
        "multipart_groups_by_type": dict(sorted(multipart_by_type.items())),
        "inference_counts": dict(sorted(inference_counts.items())),
        "trainability_gate": {
            "functional_group_metadata_available": True,
            "source_group_ids_present": inference_counts.get("source_group_id", 0) > 0,
            "inferred_group_ids_present": inference_counts.get(
                "same_type_adjacent_component", 0
            )
            > 0,
            "safe_for_formal_v6_training": False,
            "reason": (
                "Grouping is inferred because processed JSON has no ground-truth "
                "functional_id/group_id/parent_id. Use this for smoke tests and "
                "pipeline validation before formal V6 training."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    sample_dir = args.output_dir / "samples"
    payloads = []
    for path in sorted(args.processed_dir.glob("house_*.json")):
        payload = build_house_payload(path, args.min_shared_mm)
        payloads.append(payload)
        write_json(sample_dir / path.name, payload)
    summary = build_summary(payloads)
    summary["source"] = str(args.processed_dir.relative_to(ROOT))
    summary["output"] = str(args.output_dir.relative_to(ROOT))
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
