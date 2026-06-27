#!/usr/bin/env python3
"""Topology-constrained spatial inflation smoke experiment.

This prototype treats the target topology as a graph of inflatable functional
regions. Stairs are fixed first as cross-floor anchors. Other groups receive
seed cells and then grow on a 300 mm grid under exclusive occupancy, target
area budgets, and graph-neighbor attraction. The output intentionally keeps
cell-level rectangular parts so the inflation region can be evaluated without
forcing it into an overlapping bounding box.
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
    read_json,
    room_floors,
    room_functional_id,
    topology_metrics_from_report,
    write_json,
)


DEFAULT_PHASE10 = ROOT / "data" / "phase10_functional_parts" / "samples"
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_topology_spatial_inflation_overfit_8"
FLOOR_Z = {1: (0.0, 3000.0), 2: (3000.0, 6000.0)}


@dataclass
class GroupInfo:
    group_id: str
    room_type: str
    floors: list[int]
    target_cells: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=8)
    parser.add_argument("--fill-ratio", type=float, default=0.86)
    parser.add_argument("--max-iterations", type=int, default=100000)
    return parser.parse_args()


def site_xy(payload: dict[str, Any]) -> tuple[float, float]:
    size = payload.get("metadata", {}).get("building_size", {})
    return float(size["x"]), float(size["y"])


def floor_cells(site: tuple[float, float]) -> tuple[int, int]:
    return int(round(site[0] / VOXEL_MM)), int(round(site[1] / VOXEL_MM))


def topology_neighbors(topology: dict[str, Any]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in topology.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        neighbors[source].add(target)
        neighbors[target].add(source)
    return neighbors


def room_cell_box(room: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(round(float(room["box_min"][0]) / VOXEL_MM)),
        int(round(float(room["box_min"][1]) / VOXEL_MM)),
        int(round(float(room["box_max"][0]) / VOXEL_MM)),
        int(round(float(room["box_max"][1]) / VOXEL_MM)),
    )


def room_area_cells(room: dict[str, Any]) -> int:
    x0, y0, x1, y1 = room_cell_box(room)
    return max(1, x1 - x0) * max(1, y1 - y0)


def build_groups(payload: dict[str, Any], fill_ratio: float) -> dict[str, GroupInfo]:
    rooms_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for room in payload.get("rooms", []):
        rooms_by_group[room_functional_id(room)].append(room)
    groups = {}
    site = site_xy(payload)
    sx, sy = floor_cells(site)
    raw_floor_totals: dict[int, int] = defaultdict(int)
    raw_group_floor: dict[tuple[str, int], int] = defaultdict(int)

    for group in payload.get("functional_groups", []):
        group_id = str(group["functional_id"])
        for room in rooms_by_group.get(group_id, []):
            area = room_area_cells(room)
            for floor in room_floors(room):
                raw_group_floor[(group_id, floor)] += area
                raw_floor_totals[floor] += area

    scale_by_floor = {}
    for floor in (1, 2):
        desired_total = max(1, int(round(sx * sy * fill_ratio)))
        scale_by_floor[floor] = desired_total / max(raw_floor_totals.get(floor, desired_total), 1)

    for group in payload.get("functional_groups", []):
        group_id = str(group["functional_id"])
        floors = [int(value) for value in group.get("floors", [])]
        if not floors and rooms_by_group.get(group_id):
            floors = sorted({floor for room in rooms_by_group[group_id] for floor in room_floors(room)})
        target_cells = {}
        for floor in floors:
            raw = raw_group_floor.get((group_id, floor), 1)
            target_cells[floor] = max(1, int(round(raw * scale_by_floor[floor])))
        groups[group_id] = GroupInfo(
            group_id=group_id,
            room_type=str(group["type"]),
            floors=floors,
            target_cells=target_cells,
        )
    return groups


def neighbor_cells(cell: tuple[int, int], sx: int, sy: int) -> list[tuple[int, int]]:
    x, y = cell
    output = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < sx and 0 <= ny < sy:
            output.append((nx, ny))
    return output


def group_degrees(neighbors: dict[str, set[str]], groups: dict[str, GroupInfo]) -> dict[str, int]:
    return {group_id: len(neighbors.get(group_id, set()) & set(groups)) for group_id in groups}


def initial_cell_for_group(
    floor: int,
    group_id: str,
    groups: dict[str, GroupInfo],
    assignments: dict[int, dict[tuple[int, int], str]],
    neighbors: dict[str, set[str]],
    sx: int,
    sy: int,
) -> tuple[int, int] | None:
    occupied = assignments[floor]
    adjacent_targets = neighbors.get(group_id, set())
    candidates = []
    for cell, other_group in occupied.items():
        if other_group not in adjacent_targets:
            continue
        for candidate in neighbor_cells(cell, sx, sy):
            if candidate in occupied:
                continue
            cx = abs(candidate[0] - sx / 2) + abs(candidate[1] - sy / 2)
            candidates.append((0, cx, candidate))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2][0], item[2][1]))
        return candidates[0][2]
    center = (sx // 2, sy // 2)
    all_cells = [
        (abs(x - center[0]) + abs(y - center[1]), x, y)
        for x in range(sx)
        for y in range(sy)
        if (x, y) not in occupied
    ]
    if not all_cells:
        return None
    all_cells.sort()
    return (all_cells[0][1], all_cells[0][2])


def seed_stairs_and_groups(
    payload: dict[str, Any],
    groups: dict[str, GroupInfo],
    neighbors: dict[str, set[str]],
    sx: int,
    sy: int,
) -> tuple[dict[int, dict[tuple[int, int], str]], dict[tuple[str, int], set[tuple[int, int]]], list[dict[str, Any]]]:
    assignments = {1: {}, 2: {}}
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]] = defaultdict(set)
    issues = []
    rooms_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for room in payload.get("rooms", []):
        rooms_by_group[room_functional_id(room)].append(room)

    for group_id, group in groups.items():
        if group.room_type != "stairs":
            continue
        for room in rooms_by_group.get(group_id, []):
            x0, y0, x1, y1 = room_cell_box(room)
            for floor in room_floors(room):
                for x in range(max(0, x0), min(sx, x1)):
                    for y in range(max(0, y0), min(sy, y1)):
                        if (x, y) in assignments[floor] and assignments[floor][(x, y)] != group_id:
                            issues.append({"type": "seed_overlap", "group_id": group_id, "floor": floor, "cell": [x, y]})
                            continue
                        assignments[floor][(x, y)] = group_id
                        cells_by_group_floor[(group_id, floor)].add((x, y))

    degrees = group_degrees(neighbors, groups)
    ordered = sorted(
        [group_id for group_id in groups if groups[group_id].room_type != "stairs"],
        key=lambda gid: (-degrees.get(gid, 0), gid),
    )
    for group_id in ordered:
        for floor in groups[group_id].floors:
            cell = initial_cell_for_group(floor, group_id, groups, assignments, neighbors, sx, sy)
            if cell is None:
                issues.append({"type": "no_seed_cell", "group_id": group_id, "floor": floor})
                continue
            assignments[floor][cell] = group_id
            cells_by_group_floor[(group_id, floor)].add(cell)
    return assignments, cells_by_group_floor, issues


def frontier_for_group(
    floor: int,
    group_id: str,
    cells: set[tuple[int, int]],
    assignments: dict[int, dict[tuple[int, int], str]],
    sx: int,
    sy: int,
) -> set[tuple[int, int]]:
    frontier = set()
    occupied = assignments[floor]
    for cell in cells:
        for candidate in neighbor_cells(cell, sx, sy):
            if candidate not in occupied:
                frontier.add(candidate)
    return frontier


def cell_neighbor_bonus(
    floor: int,
    cell: tuple[int, int],
    group_id: str,
    assignments: dict[int, dict[tuple[int, int], str]],
    neighbors: dict[str, set[str]],
    sx: int,
    sy: int,
) -> int:
    bonus = 0
    target_neighbors = neighbors.get(group_id, set())
    for other in neighbor_cells(cell, sx, sy):
        other_group = assignments[floor].get(other)
        if other_group in target_neighbors:
            bonus += 1
    return bonus


def inflate(
    groups: dict[str, GroupInfo],
    neighbors: dict[str, set[str]],
    assignments: dict[int, dict[tuple[int, int], str]],
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
    sx: int,
    sy: int,
    max_iterations: int,
) -> list[dict[str, Any]]:
    issues = []
    for _ in range(max_iterations):
        needs = []
        for group_id, group in groups.items():
            for floor in group.floors:
                current = len(cells_by_group_floor[(group_id, floor)])
                target = group.target_cells.get(floor, 1)
                if current < target:
                    needs.append((target - current, group_id, floor))
        if not needs:
            break
        needs.sort(reverse=True)
        progressed = False
        for _need, group_id, floor in needs:
            cells = cells_by_group_floor[(group_id, floor)]
            frontier = frontier_for_group(floor, group_id, cells, assignments, sx, sy)
            if not frontier:
                continue
            center_x = sum(cell[0] for cell in cells) / max(len(cells), 1)
            center_y = sum(cell[1] for cell in cells) / max(len(cells), 1)
            candidates = []
            for cell in frontier:
                neighbor_bonus = cell_neighbor_bonus(floor, cell, group_id, assignments, neighbors, sx, sy)
                compactness = abs(cell[0] - center_x) + abs(cell[1] - center_y)
                candidates.append((-neighbor_bonus, compactness, cell[0], cell[1], cell))
            candidates.sort()
            chosen = candidates[0][4]
            assignments[floor][chosen] = group_id
            cells.add(chosen)
            progressed = True
        if not progressed:
            issues.append({"type": "inflation_stalled", "remaining_need_count": len(needs)})
            break
    return issues


def cells_to_rooms(
    groups: dict[str, GroupInfo],
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
) -> list[dict[str, Any]]:
    rooms = []
    for group_id, group in sorted(groups.items()):
        for floor in group.floors:
            z0, z1 = FLOOR_Z[floor]
            for index, (x, y) in enumerate(sorted(cells_by_group_floor[(group_id, floor)])):
                rooms.append(
                    {
                        "id": f"{group_id}_cell_{floor}_{index}",
                        "functional_id": group_id,
                        "type": group.room_type,
                        "floor": floor,
                        "floors": [floor],
                        "box_min": [x * VOXEL_MM, y * VOXEL_MM, z0],
                        "box_max": [(x + 1) * VOXEL_MM, (y + 1) * VOXEL_MM, z1],
                    }
                )
    return rooms


def counts_from_groups(groups: dict[str, GroupInfo]) -> dict[str, int]:
    return dict(Counter(group.room_type for group in groups.values()))


def decode_house(path: Path, output_dir: Path, fill_ratio: float, max_iterations: int) -> dict[str, Any]:
    payload = read_json(path)
    house_id = str(payload["house_id"])
    site = site_xy(payload)
    sx, sy = floor_cells(site)
    topology = build_target_topology(payload)
    neighbors = topology_neighbors(topology)
    groups = build_groups(payload, fill_ratio)
    assignments, cells_by_group_floor, issues = seed_stairs_and_groups(payload, groups, neighbors, sx, sy)
    issues.extend(inflate(groups, neighbors, assignments, cells_by_group_floor, sx, sy, max_iterations))
    rooms = cells_to_rooms(groups, cells_by_group_floor)
    report, _ = evaluate_candidate(house_id, rooms, counts_from_groups(groups), site, topology=topology)
    layout = {
        "house_id": house_id,
        "metadata": payload.get("metadata", {}),
        "rooms": rooms,
    }
    write_json(output_dir / "generated_layout.json", layout)
    write_json(output_dir / "topology.json", topology)
    write_json(output_dir / "evaluation.json", report)
    write_json(
        output_dir / "inflation_report.json",
        {
            "fill_ratio": fill_ratio,
            "site_cells": [sx, sy],
            "issues": issues,
            "group_floor_cell_counts": {
                f"{group_id}:{floor}": len(cells)
                for (group_id, floor), cells in sorted(cells_by_group_floor.items())
            },
            "group_floor_target_cells": {
                f"{group_id}:{floor}": target
                for group_id, group in sorted(groups.items())
                for floor, target in sorted(group.target_cells.items())
            },
        },
    )
    metrics = topology_metrics_from_report(report)
    return {
        "house_id": house_id,
        "group_count": len(groups),
        "room_part_count": len(rooms),
        "p0_pass": bool(report.get("p0", {}).get("pass", False)),
        "p1_hard_geometry_pass": bool(report.get("p1_spatial_organization", {}).get("hard_geometry_pass", False)),
        "p1_spatial_organization_pass": bool(
            report.get("p1_spatial_organization", {}).get("spatial_organization_pass", False)
        ),
        "topology": metrics,
        "issue_count": len(issues),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.phase10_dir.glob("house_*.json"))
    if args.max_houses is not None:
        paths = paths[: args.max_houses]
    reports = []
    predicted_dir = args.output_dir / "predicted_json"
    for path in paths:
        reports.append(decode_house(path, predicted_dir / path.stem, args.fill_ratio, args.max_iterations))
    summary = {
        "schema": "graphspace_v6_topology_spatial_inflation_summary_v1",
        "purpose": (
            "Smoke experiment for fixed stairs plus simultaneous topology-constrained "
            "grid inflation of functional nodes."
        ),
        "phase10_dir": str(args.phase10_dir),
        "house_count": len(reports),
        "fill_ratio": args.fill_ratio,
        "p0_pass_count": sum(1 for report in reports if report["p0_pass"]),
        "p1_full_target_topology_count": sum(1 for report in reports if report["p1_spatial_organization_pass"]),
        "total_realized_edges": sum(report["topology"].get("realized_edge_count", 0) for report in reports),
        "total_target_edges": sum(report["topology"].get("target_edge_count", 0) for report in reports),
        "total_room_parts": sum(report["room_part_count"] for report in reports),
        "reports": reports,
        "formal_v6_training_ready": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
