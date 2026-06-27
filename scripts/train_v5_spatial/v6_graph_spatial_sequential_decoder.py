#!/usr/bin/env python3
"""Graph-spatial sequential placement decoder smoke experiment.

This prototype places Phase10 functional groups from scratch. It uses each
group's size and multi-part shape as an oracle size prior, but it does not use
the target absolute coordinates as placement coordinates. At each step it reads
the whole occupied layout plus the target graph state, selects the next group,
enumerates legal 300 mm placements for that whole group, and scores candidates
by realized graph contacts and compactness.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for import_dir in (ROOT, ROOT / "scripts" / "data_phase4"):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.data_phase4.evaluate_candidates import evaluate_candidate  # noqa: E402
from scripts.train_v5_spatial.v6_multipart_decoder import (  # noqa: E402
    VOXEL_MM,
    build_target_topology,
    boxes_overlap,
    face_contact_quality,
    read_json,
    room_floors,
    room_functional_id,
    topology_metrics_from_report,
    write_json,
)


DEFAULT_PHASE10 = ROOT / "data" / "phase10_functional_parts" / "samples"
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_graph_spatial_sequential_overfit_8"

PRIORITY_BY_TYPE = {
    "stairs": 0,
    "entryway": 1,
    "living_room": 2,
    "dining_room": 3,
    "kitchen": 4,
    "corridor": 5,
    "bedroom": 6,
    "bathroom": 7,
    "utility": 8,
    "multi_purpose": 9,
    "balcony": 10,
}


@dataclass
class GroupShape:
    group_id: str
    room_type: str
    floors: list[int]
    parts: list[dict[str, Any]]
    width_cells: int
    depth_cells: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=256)
    return parser.parse_args()


def site_xy(payload: dict[str, Any]) -> tuple[float, float]:
    size = payload.get("metadata", {}).get("building_size", {})
    return float(size["x"]), float(size["y"])


def group_bbox(parts: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    return (
        min(float(part["box_min"][0]) for part in parts),
        min(float(part["box_min"][1]) for part in parts),
        max(float(part["box_max"][0]) for part in parts),
        max(float(part["box_max"][1]) for part in parts),
    )


def load_group_shapes(payload: dict[str, Any]) -> dict[str, GroupShape]:
    rooms_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for room in payload.get("rooms", []):
        rooms_by_group[room_functional_id(room)].append(room)
    shapes = {}
    for group in payload.get("functional_groups", []):
        group_id = str(group["functional_id"])
        parts = sorted(
            rooms_by_group[group_id],
            key=lambda item: (
                min(room_floors(item)),
                float(item["box_min"][0]),
                float(item["box_min"][1]),
                str(item["id"]),
            ),
        )
        if not parts:
            continue
        min_x, min_y, max_x, max_y = group_bbox(parts)
        width_cells = max(1, int(round((max_x - min_x) / VOXEL_MM)))
        depth_cells = max(1, int(round((max_y - min_y) / VOXEL_MM)))
        shapes[group_id] = GroupShape(
            group_id=group_id,
            room_type=str(group["type"]),
            floors=[int(value) for value in group.get("floors", room_floors(parts[0]))],
            parts=parts,
            width_cells=width_cells,
            depth_cells=depth_cells,
        )
    return shapes


def topology_neighbors(topology: dict[str, Any]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in topology.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        neighbors[source].add(target)
        neighbors[target].add(source)
    return neighbors


def preferred_seed_order(shapes: dict[str, GroupShape], neighbors: dict[str, set[str]]) -> list[str]:
    return sorted(
        shapes,
        key=lambda group_id: (
            -len(neighbors.get(group_id, set())),
            PRIORITY_BY_TYPE.get(shapes[group_id].room_type, 99),
            group_id,
        ),
    )


def next_group(
    unplaced: set[str],
    placed: set[str],
    shapes: dict[str, GroupShape],
    neighbors: dict[str, set[str]],
) -> str:
    touching = [
        group_id
        for group_id in unplaced
        if neighbors.get(group_id, set()) & placed
    ]
    pool = touching or list(unplaced)
    return min(
        pool,
        key=lambda group_id: (
            -len(neighbors.get(group_id, set()) & placed),
            PRIORITY_BY_TYPE.get(shapes[group_id].room_type, 99),
            -len(neighbors.get(group_id, set())),
            group_id,
        ),
    )


def component_parts_at(shape: GroupShape, x_cell: int, y_cell: int) -> list[dict[str, Any]]:
    min_x, min_y, _max_x, _max_y = group_bbox(shape.parts)
    dx = x_cell * VOXEL_MM - min_x
    dy = y_cell * VOXEL_MM - min_y
    output = []
    for index, part in enumerate(shape.parts):
        moved = dict(part)
        moved["id"] = f"{shape.group_id}_part_{index}"
        moved["functional_id"] = shape.group_id
        moved["type"] = shape.room_type
        moved["floors"] = room_floors(part)
        moved["floor"] = min(moved["floors"])
        moved["box_min"] = [
            float(part["box_min"][0]) + dx,
            float(part["box_min"][1]) + dy,
            float(part["box_min"][2]),
        ]
        moved["box_max"] = [
            float(part["box_max"][0]) + dx,
            float(part["box_max"][1]) + dy,
            float(part["box_max"][2]),
        ]
        output.append(moved)
    return output


def overlaps_any(parts: list[dict[str, Any]], occupied: list[dict[str, Any]]) -> bool:
    return any(boxes_overlap(part, other) for part in parts for other in occupied)


def contact_score(
    parts: list[dict[str, Any]],
    occupied: list[dict[str, Any]],
    target_neighbors_for_group: set[str],
) -> tuple[float, int]:
    score = 0.0
    realized = set()
    for part in parts:
        part_floors = set(room_floors(part))
        for other in occupied:
            other_group = room_functional_id(other)
            if other_group not in target_neighbors_for_group:
                continue
            if not (part_floors & set(room_floors(other))):
                continue
            quality = face_contact_quality(part, other)
            if quality > 0:
                score += quality
                realized.add(other_group)
    return score, len(realized)


def component_center(parts: list[dict[str, Any]]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = group_bbox(parts)
    return (min_x + max_x) * 0.5, (min_y + max_y) * 0.5


def occupied_center(occupied: list[dict[str, Any]], site: tuple[float, float]) -> tuple[float, float]:
    if not occupied:
        return site[0] * 0.5, site[1] * 0.5
    xs = []
    ys = []
    for part in occupied:
        cx, cy = component_center([part])
        xs.append(cx)
        ys.append(cy)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def exterior_contact(parts: list[dict[str, Any]], site: tuple[float, float]) -> bool:
    for part in parts:
        x0, y0, _ = [float(value) for value in part["box_min"]]
        x1, y1, _ = [float(value) for value in part["box_max"]]
        if x0 == 0.0 or y0 == 0.0 or x1 == site[0] or y1 == site[1]:
            return True
    return False


def place_group(
    shape: GroupShape,
    site: tuple[float, float],
    occupied: list[dict[str, Any]],
    neighbors: dict[str, set[str]],
    candidate_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    site_x_cells = int(round(site[0] / VOXEL_MM))
    site_y_cells = int(round(site[1] / VOXEL_MM))
    if shape.width_cells > site_x_cells or shape.depth_cells > site_y_cells:
        raise ValueError(f"group too large for site: {shape.group_id}")

    target_neighbors_for_group = neighbors.get(shape.group_id, set())
    occ_cx, occ_cy = occupied_center(occupied, site)
    candidates = []
    for x_cell in range(0, site_x_cells - shape.width_cells + 1):
        for y_cell in range(0, site_y_cells - shape.depth_cells + 1):
            parts = component_parts_at(shape, x_cell, y_cell)
            if overlaps_any(parts, occupied):
                continue
            contact, realized_neighbor_count = contact_score(parts, occupied, target_neighbors_for_group)
            cx, cy = component_center(parts)
            compactness = (abs(cx - occ_cx) + abs(cy - occ_cy)) / max(site[0] + site[1], 1.0)
            edge_bonus = -8.0 * realized_neighbor_count - 2.0 * contact
            if not occupied:
                edge_bonus = 0.0
            exterior_bonus = -0.25 if shape.room_type in {"living_room", "bedroom", "balcony"} and exterior_contact(parts, site) else 0.0
            priority_center = 0.0 if occupied else compactness
            score = edge_bonus + compactness + priority_center + exterior_bonus
            candidates.append((score, -realized_neighbor_count, x_cell, y_cell, parts, contact))
    if not candidates:
        raise ValueError(f"no legal placement for group: {shape.group_id}")
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    best = candidates[:candidate_limit][0]
    return best[4], {
        "group_id": shape.group_id,
        "type": shape.room_type,
        "x_cell": best[2],
        "y_cell": best[3],
        "score": best[0],
        "realized_neighbor_count": -best[1],
        "contact_score": best[5],
    }


def counts_from_shapes(shapes: dict[str, GroupShape]) -> dict[str, int]:
    return dict(Counter(shape.room_type for shape in shapes.values()))


def decode_house(path: Path, output_dir: Path, candidate_limit: int) -> dict[str, Any]:
    payload = read_json(path)
    house_id = str(payload["house_id"])
    site = site_xy(payload)
    shapes = load_group_shapes(payload)
    topology = build_target_topology(payload)
    neighbors = topology_neighbors(topology)
    placed_groups: set[str] = set()
    unplaced = set(shapes)
    rooms: list[dict[str, Any]] = []
    decisions = []
    failed_groups = []

    seed_queue = deque(preferred_seed_order(shapes, neighbors))
    while unplaced:
        while seed_queue and seed_queue[0] not in unplaced:
            seed_queue.popleft()
        group_id = next_group(unplaced, placed_groups, shapes, neighbors) if placed_groups else seed_queue[0]
        try:
            parts, decision = place_group(shapes[group_id], site, rooms, neighbors, candidate_limit)
        except ValueError as exc:
            failed_groups.append({"group_id": group_id, "reason": str(exc)})
            placed_groups.add(group_id)
            unplaced.remove(group_id)
            continue
        rooms.extend(parts)
        placed_groups.add(group_id)
        unplaced.remove(group_id)
        decisions.append(decision)

    report, _ = evaluate_candidate(house_id, rooms, counts_from_shapes(shapes), site, topology=topology)
    layout = {
        "house_id": house_id,
        "metadata": payload.get("metadata", {}),
        "rooms": rooms,
    }
    write_json(output_dir / "generated_layout.json", layout)
    write_json(output_dir / "topology.json", topology)
    write_json(output_dir / "evaluation.json", report)
    write_json(
        output_dir / "placement_decisions.json",
        {"decisions": decisions, "failed_groups": failed_groups},
    )
    metrics = topology_metrics_from_report(report)
    return {
        "house_id": house_id,
        "group_count": len(shapes),
        "placed_group_count": len(shapes) - len(failed_groups),
        "failed_groups": failed_groups,
        "rectangular_part_count": len(rooms),
        "p0_pass": bool(report.get("p0", {}).get("pass", False)),
        "p1_hard_geometry_pass": bool(report.get("p1_spatial_organization", {}).get("hard_geometry_pass", False)),
        "p1_spatial_organization_pass": bool(
            report.get("p1_spatial_organization", {}).get("spatial_organization_pass", False)
        ),
        "topology": metrics,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.phase10_dir.glob("house_*.json"))
    if args.max_houses is not None:
        paths = paths[: args.max_houses]
    predicted_dir = args.output_dir / "predicted_json"
    reports = []
    for path in paths:
        reports.append(decode_house(path, predicted_dir / path.stem, args.candidate_limit))
    summary = {
        "schema": "graphspace_v6_graph_spatial_sequential_decoder_summary_v1",
        "purpose": (
            "Smoke experiment for placing functional groups from scratch while "
            "reading whole occupancy and graph node-edge state at each step."
        ),
        "phase10_dir": str(args.phase10_dir),
        "house_count": len(reports),
        "candidate_limit": args.candidate_limit,
        "p0_pass_count": sum(1 for report in reports if report["p0_pass"]),
        "p1_full_target_topology_count": sum(1 for report in reports if report["p1_spatial_organization_pass"]),
        "total_realized_edges": sum(report["topology"].get("realized_edge_count", 0) for report in reports),
        "total_target_edges": sum(report["topology"].get("target_edge_count", 0) for report in reports),
        "reports": reports,
        "formal_v6_training_ready": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
