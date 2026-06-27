#!/usr/bin/env python3
"""Global-controlled topology spatial inflation smoke experiment.

Phase30 extends Phase28 by moving topology realization into the growth loop.
Each floor has a global fill budget; each group receives growth priority from
its normalized area deficit; and unsatisfied topology edges get paired frontier
growth before ordinary compact expansion. The output remains cell-level so the
experiment can measure topology and P0 without hiding irregular regions inside
large overlapping boxes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
from scripts.train_v5_spatial.v6_topology_spatial_inflation import (  # noqa: E402
    FLOOR_Z,
    GroupInfo,
    build_groups,
    floor_cells,
    neighbor_cells,
    seed_stairs_and_groups,
    site_xy,
    topology_neighbors,
)


DEFAULT_PHASE10 = ROOT / "data" / "phase10_functional_parts" / "samples"
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_topology_spatial_global_inflation_overfit_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=8)
    parser.add_argument("--fill-ratio", type=float, default=0.86)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--paired-edge-passes", type=int, default=2)
    return parser.parse_args()


def edge_realized_on_floor(
    source: str,
    target: str,
    floor: int,
    assignments: dict[int, dict[tuple[int, int], str]],
    sx: int,
    sy: int,
) -> bool:
    occupied = assignments.get(floor, {})
    for cell, group_id in occupied.items():
        if group_id != source:
            continue
        for neighbor in neighbor_cells(cell, sx, sy):
            if occupied.get(neighbor) == target:
                return True
    return False


def unsatisfied_topology_edges(
    topology: dict[str, Any],
    groups: dict[str, GroupInfo],
    assignments: dict[int, dict[tuple[int, int], str]],
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
    sx: int,
    sy: int,
) -> list[tuple[str, str, list[int]]]:
    output = []
    for edge in topology.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        if source not in groups or target not in groups:
            continue
        floors = [
            floor
            for floor in (1, 2)
            if cells_by_group_floor.get((source, floor)) and cells_by_group_floor.get((target, floor))
        ]
        if not floors:
            continue
        if any(edge_realized_on_floor(source, target, floor, assignments, sx, sy) for floor in floors):
            continue
        output.append((source, target, floors))
    return output


def group_deficit_ratio(
    group_id: str,
    floor: int,
    groups: dict[str, GroupInfo],
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
) -> float:
    target = max(groups[group_id].target_cells.get(floor, 1), 1)
    current = len(cells_by_group_floor[(group_id, floor)])
    return max(0.0, (target - current) / target)


def closest_distance_to_group(
    cell: tuple[int, int],
    target_cells: set[tuple[int, int]],
) -> int:
    if not target_cells:
        return 10**9
    x, y = cell
    return min(abs(x - tx) + abs(y - ty) for tx, ty in target_cells)


def frontier_for_group(
    floor: int,
    group_id: str,
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
    assignments: dict[int, dict[tuple[int, int], str]],
    sx: int,
    sy: int,
) -> set[tuple[int, int]]:
    occupied = assignments[floor]
    frontier = set()
    for cell in cells_by_group_floor[(group_id, floor)]:
        for candidate in neighbor_cells(cell, sx, sy):
            if candidate not in occupied:
                frontier.add(candidate)
    return frontier


def choose_paired_edge_growth(
    source: str,
    target: str,
    floors: list[int],
    groups: dict[str, GroupInfo],
    assignments: dict[int, dict[tuple[int, int], str]],
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
    sx: int,
    sy: int,
) -> tuple[str, int, tuple[int, int]] | None:
    candidates = []
    for floor in floors:
        for owner, other in ((source, target), (target, source)):
            if group_deficit_ratio(owner, floor, groups, cells_by_group_floor) <= 0.0:
                continue
            frontier = frontier_for_group(floor, owner, cells_by_group_floor, assignments, sx, sy)
            other_cells = cells_by_group_floor[(other, floor)]
            if not frontier or not other_cells:
                continue
            owner_cells = cells_by_group_floor[(owner, floor)]
            current_distance = min(
                abs(ox - tx) + abs(oy - ty)
                for ox, oy in owner_cells
                for tx, ty in other_cells
            )
            for cell in frontier:
                new_distance = closest_distance_to_group(cell, other_cells)
                if new_distance > current_distance:
                    continue
                creates_contact = 1 if new_distance == 1 else 0
                deficit = group_deficit_ratio(owner, floor, groups, cells_by_group_floor)
                candidates.append(
                    (
                        -creates_contact,
                        new_distance,
                        -deficit,
                        floor,
                        owner,
                        cell[0],
                        cell[1],
                        cell,
                    )
                )
    if not candidates:
        return None
    candidates.sort()
    _contact, _distance, _deficit, floor, owner, _x, _y, cell = candidates[0]
    return str(owner), int(floor), cell


def grow_cell(
    group_id: str,
    floor: int,
    cell: tuple[int, int],
    assignments: dict[int, dict[tuple[int, int], str]],
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
) -> None:
    assignments[floor][cell] = group_id
    cells_by_group_floor[(group_id, floor)].add(cell)


def ordinary_growth_candidate(
    group_id: str,
    floor: int,
    groups: dict[str, GroupInfo],
    neighbors: dict[str, set[str]],
    assignments: dict[int, dict[tuple[int, int], str]],
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
    sx: int,
    sy: int,
) -> tuple[int, int] | None:
    frontier = frontier_for_group(floor, group_id, cells_by_group_floor, assignments, sx, sy)
    if not frontier:
        return None
    cells = cells_by_group_floor[(group_id, floor)]
    center_x = sum(cell[0] for cell in cells) / max(len(cells), 1)
    center_y = sum(cell[1] for cell in cells) / max(len(cells), 1)
    target_neighbors = neighbors.get(group_id, set())
    candidates = []
    for cell in frontier:
        adjacent_target_count = 0
        nearest_target_distance = 10**9
        for other_group in target_neighbors:
            other_cells = cells_by_group_floor.get((other_group, floor), set())
            if not other_cells:
                continue
            distance = closest_distance_to_group(cell, other_cells)
            nearest_target_distance = min(nearest_target_distance, distance)
            if distance == 1:
                adjacent_target_count += 1
        compactness = abs(cell[0] - center_x) + abs(cell[1] - center_y)
        candidates.append((-adjacent_target_count, nearest_target_distance, compactness, cell[0], cell[1], cell))
    candidates.sort()
    return candidates[0][5]


def global_controlled_inflate(
    groups: dict[str, GroupInfo],
    topology: dict[str, Any],
    neighbors: dict[str, set[str]],
    assignments: dict[int, dict[tuple[int, int], str]],
    cells_by_group_floor: dict[tuple[str, int], set[tuple[int, int]]],
    sx: int,
    sy: int,
    fill_ratio: float,
    max_iterations: int,
    paired_edge_passes: int,
) -> list[dict[str, Any]]:
    issues = []
    floor_budget = {floor: max(1, int(round(sx * sy * fill_ratio))) for floor in (1, 2)}
    paired_growth_count = 0
    ordinary_growth_count = 0
    for _iteration in range(max_iterations):
        progressed = False
        for _pass in range(paired_edge_passes):
            gaps = unsatisfied_topology_edges(topology, groups, assignments, cells_by_group_floor, sx, sy)
            if not gaps:
                break
            gaps.sort(
                key=lambda edge: (
                    -max(
                        group_deficit_ratio(edge[0], floor, groups, cells_by_group_floor)
                        + group_deficit_ratio(edge[1], floor, groups, cells_by_group_floor)
                        for floor in edge[2]
                    ),
                    edge[0],
                    edge[1],
                )
            )
            for source, target, floors in gaps:
                choice = choose_paired_edge_growth(
                    source,
                    target,
                    floors,
                    groups,
                    assignments,
                    cells_by_group_floor,
                    sx,
                    sy,
                )
                if choice is None:
                    continue
                owner, floor, cell = choice
                if len(assignments[floor]) >= floor_budget[floor]:
                    continue
                grow_cell(owner, floor, cell, assignments, cells_by_group_floor)
                paired_growth_count += 1
                progressed = True

        needs = []
        for group_id, group in groups.items():
            for floor in group.floors:
                target = max(group.target_cells.get(floor, 1), 1)
                current = len(cells_by_group_floor[(group_id, floor)])
                if current >= target or len(assignments[floor]) >= floor_budget[floor]:
                    continue
                deficit_ratio = (target - current) / target
                topology_pressure = sum(
                    1
                    for source, target_group, floors in unsatisfied_topology_edges(
                        topology, groups, assignments, cells_by_group_floor, sx, sy
                    )
                    if floor in floors and group_id in {source, target_group}
                )
                needs.append((-topology_pressure, -deficit_ratio, group_id, floor))
        needs.sort()
        for _topology_pressure, _deficit_ratio, group_id, floor in needs:
            if len(assignments[floor]) >= floor_budget[floor]:
                continue
            cell = ordinary_growth_candidate(
                group_id,
                floor,
                groups,
                neighbors,
                assignments,
                cells_by_group_floor,
                sx,
                sy,
            )
            if cell is None:
                continue
            grow_cell(group_id, floor, cell, assignments, cells_by_group_floor)
            ordinary_growth_count += 1
            progressed = True
        if not progressed:
            remaining = []
            for group_id, group in groups.items():
                for floor in group.floors:
                    target = group.target_cells.get(floor, 1)
                    current = len(cells_by_group_floor[(group_id, floor)])
                    if current < target:
                        remaining.append({"group_id": group_id, "floor": floor, "need": target - current})
            if remaining:
                issues.append({"type": "global_inflation_stalled", "remaining_need_count": len(remaining)})
            break
    issues.append(
        {
            "type": "global_control_stats",
            "floor_budget": floor_budget,
            "paired_growth_count": paired_growth_count,
            "ordinary_growth_count": ordinary_growth_count,
            "final_floor_occupancy": {floor: len(assignments[floor]) for floor in (1, 2)},
        }
    )
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


def decode_house(
    path: Path,
    output_dir: Path,
    fill_ratio: float,
    max_iterations: int,
    paired_edge_passes: int,
) -> dict[str, Any]:
    payload = read_json(path)
    house_id = str(payload["house_id"])
    site = site_xy(payload)
    sx, sy = floor_cells(site)
    topology = build_target_topology(payload)
    neighbors = topology_neighbors(topology)
    groups = build_groups(payload, fill_ratio)
    assignments, cells_by_group_floor, issues = seed_stairs_and_groups(payload, groups, neighbors, sx, sy)
    issues.extend(
        global_controlled_inflate(
            groups,
            topology,
            neighbors,
            assignments,
            cells_by_group_floor,
            sx,
            sy,
            fill_ratio,
            max_iterations,
            paired_edge_passes,
        )
    )
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
        output_dir / "global_inflation_report.json",
        {
            "fill_ratio": fill_ratio,
            "site_cells": [sx, sy],
            "paired_edge_passes": paired_edge_passes,
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
        reports.append(
            decode_house(
                path,
                predicted_dir / path.stem,
                args.fill_ratio,
                args.max_iterations,
                args.paired_edge_passes,
            )
        )
    summary = {
        "schema": "graphspace_v6_topology_spatial_global_inflation_summary_v1",
        "purpose": (
            "Phase30 smoke experiment for synchronous inflation with global floor "
            "budgets, group deficit pressure, and paired topology-edge growth."
        ),
        "phase10_dir": str(args.phase10_dir),
        "house_count": len(reports),
        "fill_ratio": args.fill_ratio,
        "paired_edge_passes": args.paired_edge_passes,
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
