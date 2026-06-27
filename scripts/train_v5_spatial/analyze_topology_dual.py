#!/usr/bin/env python3
"""Compare target topology, realized planar dual graph and voxel assignment."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


VOXEL_MM = 300.0
FLOOR_Z = {1: (0.0, 3000.0), 2: (3000.0, 6000.0)}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_floors(room: dict) -> list[int]:
    if room.get("floors"):
        return sorted({int(value) for value in room["floors"]})
    z0 = float(room["box_min"][2])
    z1 = float(room["box_max"][2])
    if z0 <= 0 and z1 >= 6000:
        return [1, 2]
    return [2] if z0 >= 3000 else [1]


def functional_id(room: dict, room_id: str) -> str:
    for key in ("functional_id", "group_id", "parent_id"):
        value = room.get(key)
        if value:
            return str(value)
    if "_part_" in room_id:
        return room_id.split("_part_", 1)[0]
    return room_id


def normalize_rooms(layout: dict) -> list[dict]:
    rooms = []
    for index, room in enumerate(layout.get("rooms", [])):
        room_type = str(room.get("type", "unknown"))
        room_id = str(room.get("id", f"{room_type}_{index}"))
        normalized = {
            "id": room_id,
            "type": room_type,
            "functional_id": functional_id(room, room_id),
            "box_min": [float(value) for value in room["box_min"]],
            "box_max": [float(value) for value in room["box_max"]],
        }
        normalized["floors"] = infer_floors({**room, **normalized})
        rooms.append(normalized)
    return rooms


def axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def projection_overlap_area(a: dict, b: dict) -> float:
    return axis_overlap(a["box_min"][0], a["box_max"][0], b["box_min"][0], b["box_max"][0]) * axis_overlap(
        a["box_min"][1], a["box_max"][1], b["box_min"][1], b["box_max"][1]
    )


def face_contact_quality(a: dict, b: dict) -> float:
    ax0, ay0, _ = a["box_min"]
    ax1, ay1, _ = a["box_max"]
    bx0, by0, _ = b["box_min"]
    bx1, by1, _ = b["box_max"]
    y_touch = (abs(ax1 - bx0) <= 1e-6 or abs(bx1 - ax0) <= 1e-6)
    x_touch = (abs(ay1 - by0) <= 1e-6 or abs(by1 - ay0) <= 1e-6)
    if y_touch:
        overlap = axis_overlap(ay0, ay1, by0, by1)
        return overlap / max(min(ay1 - ay0, by1 - by0), VOXEL_MM)
    if x_touch:
        overlap = axis_overlap(ax0, ax1, bx0, bx1)
        return overlap / max(min(ax1 - ax0, bx1 - bx0), VOXEL_MM)
    return 0.0


def floor_dual_edges(rooms: list[dict]) -> list[dict]:
    edges = []
    for floor in (1, 2):
        floor_rooms = [room for room in rooms if floor in room["floors"]]
        for index, left in enumerate(floor_rooms):
            for right in floor_rooms[index + 1 :]:
                quality = face_contact_quality(left, right)
                if quality <= 0:
                    continue
                edges.append(
                    {
                        "source": left["functional_id"],
                        "target": right["functional_id"],
                        "source_part": left["id"],
                        "target_part": right["id"],
                        "relation": "horizontal",
                        "floor": floor,
                        "contact_quality": quality,
                    }
                )
    return edges


def vertical_dual_edges(rooms: list[dict]) -> list[dict]:
    edges = []
    lower = [room for room in rooms if room["floors"] == [1]]
    upper = [room for room in rooms if room["floors"] == [2]]
    cross = [room for room in rooms if room["floors"] == [1, 2]]
    for stair in cross:
        for room in lower + upper:
            overlap = projection_overlap_area(stair, room)
            if overlap <= 0:
                continue
            smaller = min(
                (stair["box_max"][0] - stair["box_min"][0])
                * (stair["box_max"][1] - stair["box_min"][1]),
                (room["box_max"][0] - room["box_min"][0])
                * (room["box_max"][1] - room["box_min"][1]),
            )
            edges.append(
                {
                    "source": stair["functional_id"],
                    "target": room["functional_id"],
                    "source_part": stair["id"],
                    "target_part": room["id"],
                    "relation": "vertical_overlap",
                    "floor": room["floors"][0],
                    "overlap_quality": overlap / max(smaller, VOXEL_MM * VOXEL_MM),
                }
            )
    return edges


def target_edges(topology: dict) -> list[dict]:
    required_source = topology.get("required_edges")
    if required_source is None:
        required_source = topology.get("evidence", {}).get("required_edges", [])
    required = {
        tuple(sorted(edge))
        for edge in required_source
        if len(edge) == 2
    }
    output = []
    for edge in topology.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        output.append(
            {
                "source": source,
                "target": target,
                "relation": str(edge.get("relation", "horizontal")).strip(),
                "required": tuple(sorted((source, target))) in required,
            }
        )
    return output


def target_realization(targets: list[dict], dual_edges: list[dict]) -> dict:
    horizontal = {
        tuple(sorted((edge["source"], edge["target"])))
        for edge in dual_edges
        if edge["relation"] == "horizontal"
    }
    vertical = {
        tuple(sorted((edge["source"], edge["target"])))
        for edge in dual_edges
        if edge["relation"] == "vertical_overlap"
    }
    results = []
    for edge in targets:
        key = tuple(sorted((edge["source"], edge["target"])))
        if edge["relation"] == "vertical":
            realized = key in vertical
        else:
            realized = key in horizontal
        results.append({**edge, "realized_in_dual": realized})
    required = [edge for edge in results if edge["required"]]
    return {
        "target_edge_count": len(results),
        "realized_edge_count": sum(edge["realized_in_dual"] for edge in results),
        "realization_rate": sum(edge["realized_in_dual"] for edge in results)
        / max(len(results), 1),
        "required_edge_count": len(required),
        "required_realized_edge_count": sum(edge["realized_in_dual"] for edge in required),
        "required_realization_rate": sum(edge["realized_in_dual"] for edge in required)
        / max(len(required), 1),
        "edges": results,
    }


def cell_bounds(room: dict) -> tuple[int, int, int, int, int, int]:
    values = room["box_min"] + room["box_max"]
    return tuple(int(round(value / VOXEL_MM)) for value in values)


def voxel_assignment_report(rooms: list[dict], site_x: float, site_y: float) -> dict:
    sx = int(np.floor(site_x / VOXEL_MM))
    sy = int(np.floor(site_y / VOXEL_MM))
    sz = 20
    owner = np.full((sx, sy, sz), -1, dtype=np.int16)
    overlap = np.zeros((sx, sy, sz), dtype=bool)
    outside = []
    empty_instances = []
    per_room = []
    for index, room in enumerate(rooms):
        x0, y0, z0, x1, y1, z1 = cell_bounds(room)
        clipped = (
            max(0, x0),
            max(0, y0),
            max(0, z0),
            min(sx, x1),
            min(sy, y1),
            min(sz, z1),
        )
        if clipped != (x0, y0, z0, x1, y1, z1):
            outside.append(room["id"])
        cx0, cy0, cz0, cx1, cy1, cz1 = clipped
        voxel_count = max(0, cx1 - cx0) * max(0, cy1 - cy0) * max(0, cz1 - cz0)
        if voxel_count == 0:
            empty_instances.append(room["id"])
            continue
        current = owner[cx0:cx1, cy0:cy1, cz0:cz1]
        overlap[cx0:cx1, cy0:cy1, cz0:cz1] |= current >= 0
        current[current < 0] = index
        per_room.append(
            {
                "id": room["id"],
                "functional_id": room["functional_id"],
                "type": room["type"],
                "voxel_count": int(voxel_count),
                "area_m2_per_floor_projection": voxel_count * VOXEL_MM * VOXEL_MM / 1_000_000.0 / max(len(room["floors"]), 1),
            }
        )
    assigned = int((owner >= 0).sum())
    overlap_count = int(overlap.sum())
    total = sx * sy * sz
    return {
        "grid_shape": [sx, sy, sz],
        "voxel_mm": VOXEL_MM,
        "assigned_voxel_count": assigned,
        "empty_inside_voxel_count": int(total - assigned),
        "overlap_voxel_count": overlap_count,
        "overlap_voxel_rate": overlap_count / max(total, 1),
        "outside_room_ids": outside,
        "empty_instance_ids": empty_instances,
        "room_voxels": per_room,
    }


def hetero_topology_summary(topology: dict) -> dict:
    nodes = topology.get("nodes", [])
    edges = target_edges(topology)
    floor_membership = [
        {"source": node["id"], "target": f"floor_{node.get('floor', 'unknown')}", "relation": "belongs_to_floor"}
        for node in nodes
    ]
    zone_membership = [
        {"source": node["id"], "target": f"zone_{room_zone(str(node.get('type', 'unknown')))}", "relation": "belongs_to_zone"}
        for node in nodes
    ]
    return {
        "room_node_count": len(nodes),
        "room_nodes_by_type": dict(Counter(str(node.get("type", "unknown")) for node in nodes)),
        "room_nodes_by_floor": dict(Counter(str(node.get("floor", "unknown")) for node in nodes)),
        "target_edges_by_relation": dict(Counter(edge["relation"] for edge in edges)),
        "required_edge_count": sum(edge["required"] for edge in edges),
        "derived_floor_membership_edge_count": len(floor_membership),
        "derived_zone_membership_edge_count": len(zone_membership),
        "heterogeneous_edge_examples": {
            "room_relations": edges[:5],
            "floor_membership": floor_membership[:5],
            "zone_membership": zone_membership[:5],
        },
    }


def room_zone(room_type: str) -> str:
    if room_type in {"entryway", "corridor", "stairs"}:
        return "circulation"
    if room_type in {"living_room", "dining_room", "kitchen", "multi_purpose"}:
        return "public"
    if room_type in {"bedroom", "bathroom"}:
        return "private"
    if room_type in {"utility", "balcony"}:
        return "service"
    return "unknown"


def build_report(topology: dict, layout: dict) -> dict:
    rooms = normalize_rooms(layout)
    site = layout.get("metadata", {}).get("building_size", {})
    site_x = float(site.get("x", 0.0))
    site_y = float(site.get("y", 0.0))
    horizontal = floor_dual_edges(rooms)
    vertical = vertical_dual_edges(rooms)
    dual = horizontal + vertical
    targets = target_edges(topology)
    group_counts = Counter(room["functional_id"] for room in rooms)
    return {
        "schema": "graphspace_topology_dual_voxel_report_v1",
        "heterogeneous_topology": hetero_topology_summary(topology),
        "multipart_groups": {
            "part_count": len(rooms),
            "functional_group_count": len(group_counts),
            "groups_with_multiple_parts": {
                key: value for key, value in sorted(group_counts.items()) if value > 1
            },
        },
        "realized_planar_dual": {
            "horizontal_edge_count": len(horizontal),
            "vertical_overlap_edge_count": len(vertical),
            "edges": dual,
        },
        "target_vs_realized": target_realization(targets, dual),
        "voxel_assignment": voxel_assignment_report(rooms, site_x, site_y),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topology = read_json(args.input_dir / "topology.json")
    layout = read_json(args.input_dir / "generated_layout.json")
    report = build_report(topology, layout)
    output = args.output or args.input_dir / "topology_dual_report.json"
    write_json(output, report)
    summary = {
        "target_edges": report["target_vs_realized"]["target_edge_count"],
        "realized_edges": report["target_vs_realized"]["realized_edge_count"],
        "realization_rate": report["target_vs_realized"]["realization_rate"],
        "overlap_voxels": report["voxel_assignment"]["overlap_voxel_count"],
        "empty_instances": len(report["voxel_assignment"]["empty_instance_ids"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
