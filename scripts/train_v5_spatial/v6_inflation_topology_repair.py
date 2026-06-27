#!/usr/bin/env python3
"""Topology frontier repair for spatial-inflation layouts.

This Phase29 smoke experiment starts from the Phase28 cell-level inflation
output. For each unrealized target-topology edge, it tries to grow a thin
300 mm bridge through currently empty cells between the two endpoint groups.
Each trial is accepted only when it keeps P0 valid and increases the target
topology score, following the Phase24 P0-safe optimization rule.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for import_dir in (ROOT, ROOT / "scripts" / "data_phase4"):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.data_phase4.evaluate_candidates import evaluate_candidate  # noqa: E402
from scripts.train_v5_spatial.v6_graph_spatial_coordinator import (  # noqa: E402
    counts_from_report,
    missing_edges,
    site_xy_from_layout,
)
from scripts.train_v5_spatial.v6_multipart_decoder import (  # noqa: E402
    VOXEL_MM,
    read_json,
    room_floors,
    room_functional_id,
    topology_metrics_from_report,
    topology_score,
    write_json,
)
from scripts.train_v5_spatial.v6_topology_spatial_inflation import FLOOR_Z  # noqa: E402


DEFAULT_INPUT = ROOT / "outputs" / "v6_topology_spatial_inflation_overfit_8" / "predicted_json"
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_inflation_topology_frontier_repair_overfit_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int)
    parser.add_argument("--max-iterations", type=int, default=64)
    parser.add_argument("--max-endpoint-pairs", type=int, default=48)
    return parser.parse_args()


def site_cells(site_xy: tuple[float, float]) -> tuple[int, int]:
    return int(round(site_xy[0] / VOXEL_MM)), int(round(site_xy[1] / VOXEL_MM))


def cell_box(room: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(round(float(room["box_min"][0]) / VOXEL_MM)),
        int(round(float(room["box_min"][1]) / VOXEL_MM)),
        int(round(float(room["box_max"][0]) / VOXEL_MM)),
        int(round(float(room["box_max"][1]) / VOXEL_MM)),
    )


def group_type_map(rooms: list[dict[str, Any]]) -> dict[str, str]:
    output = {}
    for room in rooms:
        output.setdefault(room_functional_id(room), str(room["type"]))
    return output


def assignments_from_rooms(
    rooms: list[dict[str, Any]],
) -> tuple[dict[int, dict[tuple[int, int], str]], dict[tuple[str, int], set[tuple[int, int]]]]:
    assignments: dict[int, dict[tuple[int, int], str]] = {1: {}, 2: {}}
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]] = defaultdict(set)
    for room in rooms:
        group_id = room_functional_id(room)
        x0, y0, x1, y1 = cell_box(room)
        for floor in room_floors(room):
            for x in range(x0, x1):
                for y in range(y0, y1):
                    assignments.setdefault(floor, {})[(x, y)] = group_id
                    cells_by_group_floor[(group_id, floor)].add((x, y))
    return assignments, cells_by_group_floor


def neighbor_cells(cell: tuple[int, int], sx: int, sy: int) -> list[tuple[int, int]]:
    x, y = cell
    output = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < sx and 0 <= ny < sy:
            output.append((nx, ny))
    return output


def boundary_cells(
    floor: int,
    group_id: str,
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
    assignments: dict[int, dict[tuple[int, int], str]],
    sx: int,
    sy: int,
) -> list[tuple[int, int]]:
    cells = cells_by_group_floor.get((group_id, floor), set())
    occupied = assignments.get(floor, {})
    output = []
    for cell in cells:
        for neighbor in neighbor_cells(cell, sx, sy):
            if occupied.get(neighbor) != group_id:
                output.append(cell)
                break
    return sorted(set(output))


def shared_floors(
    source: str,
    target: str,
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
) -> list[int]:
    floors = []
    for floor in (1, 2):
        if cells_by_group_floor.get((source, floor)) and cells_by_group_floor.get((target, floor)):
            floors.append(floor)
    return floors


def closest_endpoint_pairs(
    source_cells: list[tuple[int, int]],
    target_cells: list[tuple[int, int]],
    limit: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    for source in source_cells:
        sx, sy = source
        best_target = min(target_cells, key=lambda target: (abs(sx - target[0]) + abs(sy - target[1]), target))
        distance = abs(sx - best_target[0]) + abs(sy - best_target[1])
        pairs.append((distance, source, best_target))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    output = []
    seen = set()
    for _distance, source, target in pairs:
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        output.append((source, target))
        if len(output) >= limit:
            break
    return output


def l_path(source: tuple[int, int], target: tuple[int, int], x_first: bool) -> list[tuple[int, int]]:
    x, y = source
    tx, ty = target
    path = []
    if x_first:
        while x != tx:
            x += 1 if tx > x else -1
            path.append((x, y))
        while y != ty:
            y += 1 if ty > y else -1
            path.append((x, y))
    else:
        while y != ty:
            y += 1 if ty > y else -1
            path.append((x, y))
        while x != tx:
            x += 1 if tx > x else -1
            path.append((x, y))
    return path


def bridge_paths(
    source: tuple[int, int],
    target: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    paths = []
    for x_first in (True, False):
        raw = l_path(source, target, x_first)
        bridge = raw[:-1]
        if bridge and bridge not in paths:
            paths.append(bridge)
    return paths


def cells_to_rooms(
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
    group_types: dict[str, str],
) -> list[dict[str, Any]]:
    rooms = []
    for (group_id, floor), cells in sorted(cells_by_group_floor.items()):
        z0, z1 = FLOOR_Z[floor]
        room_type = group_types[group_id]
        for index, (x, y) in enumerate(sorted(cells)):
            rooms.append(
                {
                    "id": f"{group_id}_cell_{floor}_{index}",
                    "functional_id": group_id,
                    "type": room_type,
                    "floor": floor,
                    "floors": [floor],
                    "box_min": [x * VOXEL_MM, y * VOXEL_MM, z0],
                    "box_max": [(x + 1) * VOXEL_MM, (y + 1) * VOXEL_MM, z1],
                }
            )
    return rooms


def copy_cells(
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
) -> dict[tuple[str, int], set[tuple[int, int]]]:
    return {key: set(value) for key, value in cells_by_group_floor.items()}


def layout_report(
    house_id: str,
    rooms: list[dict[str, Any]],
    counts: dict[str, int],
    site_xy: tuple[float, float],
    topology: dict[str, Any],
) -> dict[str, Any]:
    report, _ = evaluate_candidate(house_id, rooms, counts, site_xy, topology=topology)
    return report


def try_bridge_edge(
    edge: dict[str, Any],
    assignments: dict[int, dict[tuple[int, int], str]],
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
    group_types: dict[str, str],
    report: dict[str, Any],
    counts: dict[str, int],
    site_xy: tuple[float, float],
    topology: dict[str, Any],
    house_id: str,
    max_endpoint_pairs: int,
) -> tuple[dict[str, Any] | None, dict[tuple[str, int], set[tuple[int, int]]] | None, dict[str, Any] | None, int]:
    source = str(edge["source"])
    target = str(edge["target"])
    sx, sy = site_cells(site_xy)
    current_score = topology_score(report)
    rejected = 0
    best = None
    for floor in shared_floors(source, target, cells_by_group_floor):
        source_boundary = boundary_cells(floor, source, cells_by_group_floor, assignments, sx, sy)
        target_boundary = boundary_cells(floor, target, cells_by_group_floor, assignments, sx, sy)
        if not source_boundary or not target_boundary:
            continue
        for source_cell, target_cell in closest_endpoint_pairs(source_boundary, target_boundary, max_endpoint_pairs):
            for bridge in bridge_paths(source_cell, target_cell):
                occupied = assignments[floor]
                if any(cell in occupied for cell in bridge):
                    rejected += 1
                    continue
                trial_cells = copy_cells(cells_by_group_floor)
                trial_cells[(source, floor)].update(bridge)
                trial_rooms = cells_to_rooms(trial_cells, group_types)
                trial_report = layout_report(house_id, trial_rooms, counts, site_xy, topology)
                if not trial_report.get("p0", {}).get("pass", False):
                    rejected += 1
                    continue
                trial_score = topology_score(trial_report)
                if trial_score <= current_score:
                    rejected += 1
                    continue
                key = (trial_score, -len(bridge), source, target, floor, source_cell, target_cell)
                move = {
                    "edge": [source, target],
                    "move_type": "empty_cell_frontier_bridge",
                    "bridge_owner": source,
                    "floor": floor,
                    "bridge_cell_count": len(bridge),
                    "source_cell": list(source_cell),
                    "target_cell": list(target_cell),
                    "score_before": list(current_score),
                    "score_after": list(trial_score),
                }
                if best is None or key > best[0]:
                    best = (key, trial_report, trial_cells, move)
    if best is None:
        return None, None, None, rejected
    return best[1], best[2], best[3], rejected


def repair_house(
    house_dir: Path,
    output_dir: Path,
    max_iterations: int,
    max_endpoint_pairs: int,
) -> dict[str, Any]:
    house_id = house_dir.name
    layout = read_json(house_dir / "generated_layout.json")
    topology = read_json(house_dir / "topology.json")
    source_eval = read_json(house_dir / "evaluation.json")
    site_xy = site_xy_from_layout(layout)
    counts = counts_from_report(source_eval)
    group_types = group_type_map(layout.get("rooms", []))
    assignments, cells_by_group_floor = assignments_from_rooms(layout.get("rooms", []))
    rooms = cells_to_rooms(cells_by_group_floor, group_types)
    report = layout_report(house_id, rooms, counts, site_xy, topology)
    initial_metrics = topology_metrics_from_report(report)
    accepted_moves = []
    rejected_candidate_count = 0

    for _iteration in range(max_iterations):
        gaps = missing_edges(report)
        if not gaps:
            break
        best = None
        total_rejected = 0
        for edge in gaps:
            trial_report, trial_cells, move, rejected = try_bridge_edge(
                edge,
                assignments,
                cells_by_group_floor,
                group_types,
                report,
                counts,
                site_xy,
                topology,
                house_id,
                max_endpoint_pairs,
            )
            total_rejected += rejected
            if trial_report is None or trial_cells is None or move is None:
                continue
            key = (topology_score(trial_report), -move["bridge_cell_count"], move["edge"])
            if best is None or key > best[0]:
                best = (key, trial_report, trial_cells, move)
        rejected_candidate_count += total_rejected
        if best is None:
            break
        report = best[1]
        cells_by_group_floor = best[2]
        rooms = cells_to_rooms(cells_by_group_floor, group_types)
        assignments, _ = assignments_from_rooms(rooms)
        accepted_moves.append(best[3])

    final_metrics = topology_metrics_from_report(report)
    repaired_layout = dict(layout)
    repaired_layout["rooms"] = rooms
    repaired_layout.setdefault("metadata", {})["inflation_topology_frontier_repair"] = {
        "enabled": True,
        "accepted_move_count": len(accepted_moves),
    }
    write_json(output_dir / "generated_layout.json", repaired_layout)
    write_json(output_dir / "topology.json", topology)
    if (house_dir / "inflation_report.json").exists():
        shutil.copy2(house_dir / "inflation_report.json", output_dir / "inflation_report.json")
    write_json(output_dir / "evaluation.json", report)
    write_json(
        output_dir / "inflation_topology_repair.json",
        {
            "enabled": True,
            "max_iterations": max_iterations,
            "max_endpoint_pairs": max_endpoint_pairs,
            "accepted_move_count": len(accepted_moves),
            "rejected_candidate_count": rejected_candidate_count,
            "initial_topology": initial_metrics,
            "final_topology": final_metrics,
            "moves": accepted_moves,
        },
    )
    return {
        "house_id": house_id,
        "p0_pass": bool(report.get("p0", {}).get("pass", False)),
        "p1_hard_geometry_pass": bool(report.get("p1_spatial_organization", {}).get("hard_geometry_pass", False)),
        "p1_spatial_organization_pass": bool(
            report.get("p1_spatial_organization", {}).get("spatial_organization_pass", False)
        ),
        "initial_topology": initial_metrics,
        "final_topology": final_metrics,
        "accepted_move_count": len(accepted_moves),
        "rejected_candidate_count": rejected_candidate_count,
        "room_part_count": len(rooms),
    }


def main() -> None:
    args = parse_args()
    house_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if args.max_houses is not None:
        house_dirs = house_dirs[: args.max_houses]
    predicted_dir = args.output_dir / "predicted_json"
    reports = [
        repair_house(house_dir, predicted_dir / house_dir.name, args.max_iterations, args.max_endpoint_pairs)
        for house_dir in house_dirs
    ]
    total_initial_edges = sum(report["initial_topology"].get("realized_edge_count", 0) for report in reports)
    total_final_edges = sum(report["final_topology"].get("realized_edge_count", 0) for report in reports)
    total_target_edges = sum(report["final_topology"].get("target_edge_count", 0) for report in reports)
    summary = {
        "schema": "graphspace_v6_inflation_topology_frontier_repair_summary_v1",
        "purpose": (
            "Phase29 smoke experiment combining Phase28 synchronous inflation with "
            "Phase24-style P0-safe topology optimization."
        ),
        "input_dir": str(args.input_dir),
        "house_count": len(reports),
        "max_iterations": args.max_iterations,
        "max_endpoint_pairs": args.max_endpoint_pairs,
        "p0_pass_count": sum(1 for report in reports if report["p0_pass"]),
        "p1_full_target_topology_count": sum(
            1 for report in reports if report["p1_spatial_organization_pass"]
        ),
        "total_initial_realized_edges": total_initial_edges,
        "total_final_realized_edges": total_final_edges,
        "total_target_edges": total_target_edges,
        "accepted_move_count": sum(report["accepted_move_count"] for report in reports),
        "rejected_candidate_count": sum(report["rejected_candidate_count"] for report in reports),
        "total_room_parts": sum(report["room_part_count"] for report in reports),
        "reports": reports,
        "formal_v6_training_ready": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
