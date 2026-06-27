#!/usr/bin/env python3
"""Graph-spatial coordinated placement repair for V6 multi-part layouts.

This is a post-decode smoke experiment. It reads a complete candidate layout,
the target group topology and the current occupied geometry, then tries moving
small connected functional components as a unit. The goal is to test whether a
placement step that sees both whole-layout occupancy and node-edge state can
repair P1 target-topology gaps beyond single-part local moves.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for import_dir in (
    ROOT,
    ROOT / "scripts" / "data_phase4",
):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.data_phase4.evaluate_candidates import evaluate_candidate  # noqa: E402
from scripts.train_v5_spatial.v6_multipart_decoder import (  # noqa: E402
    VOXEL_MM,
    boxes_overlap,
    face_contact_quality,
    read_json,
    room_floors,
    room_functional_id,
    topology_metrics_from_report,
    topology_score,
    write_json,
)


DEFAULT_INPUT = ROOT / "outputs" / "v6_multipart_graph_topology_linked_size_overfit_8" / "predicted_json"
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_graph_spatial_coordinated_repair_overfit_8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-component-size", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=96)
    return parser.parse_args()


def site_xy_from_layout(layout: dict[str, Any]) -> tuple[float, float]:
    size = layout.get("metadata", {}).get("building_size", {})
    return float(size["x"]), float(size["y"])


def counts_from_report(report: dict[str, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in report.get("requested_counts", {}).items()}


def layout_report(
    house_id: str,
    rooms: list[dict[str, Any]],
    counts: dict[str, int],
    site_xy: tuple[float, float],
    topology: dict[str, Any],
) -> dict[str, Any]:
    report, _ = evaluate_candidate(house_id, rooms, counts, site_xy, topology=topology)
    return report


def target_edges(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("p1_spatial_organization", {}).get("target_topology", {}).get("edges", []))


def missing_edges(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [edge for edge in target_edges(report) if not edge.get("realized_in_dual")]


def realized_neighbor_map(report: dict[str, Any]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in target_edges(report):
        if not edge.get("realized_in_dual"):
            continue
        source = str(edge["source"])
        target = str(edge["target"])
        neighbors[source].add(target)
        neighbors[target].add(source)
    return neighbors


def target_neighbor_map(topology: dict[str, Any]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in topology.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        neighbors[source].add(target)
        neighbors[target].add(source)
    return neighbors


def groups_by_id(rooms: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, room in enumerate(rooms):
        groups[room_functional_id(room)].append(index)
    return groups


def component_bbox(parts: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    return (
        min(float(part["box_min"][0]) for part in parts),
        min(float(part["box_min"][1]) for part in parts),
        max(float(part["box_max"][0]) for part in parts),
        max(float(part["box_max"][1]) for part in parts),
    )


def translated_part(part: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    moved = dict(part)
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
    moved["constraint_adjustment"] = "graph_spatial_component_move"
    return moved


def any_overlap(parts: list[dict[str, Any]], occupied: list[dict[str, Any]]) -> bool:
    return any(boxes_overlap(part, other) for part in parts for other in occupied)


def component_contact_score(
    moved_parts: list[dict[str, Any]],
    fixed_parts: list[dict[str, Any]],
    moved_groups: set[str],
    target_neighbors: dict[str, set[str]],
) -> float:
    score = 0.0
    for moved in moved_parts:
        moved_group = room_functional_id(moved)
        moved_floors = set(room_floors(moved))
        for fixed in fixed_parts:
            fixed_group = room_functional_id(fixed)
            if fixed_group in moved_groups:
                continue
            if fixed_group not in target_neighbors.get(moved_group, set()):
                continue
            if moved_floors & set(room_floors(fixed)):
                score += face_contact_quality(moved, fixed)
    return score


def build_component(
    seed_group: str,
    fixed_group: str,
    realized_neighbors: dict[str, set[str]],
    available_groups: set[str],
    max_component_size: int,
) -> set[str]:
    component = {seed_group}
    for neighbor in sorted(realized_neighbors.get(seed_group, set())):
        if len(component) >= max_component_size:
            break
        if neighbor == fixed_group or neighbor not in available_groups:
            continue
        component.add(neighbor)
    return component


def candidate_component_moves(
    rooms: list[dict[str, Any]],
    moving_groups: set[str],
    site_xy: tuple[float, float],
    target_neighbors: dict[str, set[str]],
    limit: int,
) -> list[tuple[float, float, list[dict[str, Any]]]]:
    group_indices = groups_by_id(rooms)
    moving_indices = [index for group in moving_groups for index in group_indices.get(group, [])]
    moving_parts = [rooms[index] for index in moving_indices]
    fixed_parts = [room for index, room in enumerate(rooms) if index not in set(moving_indices)]
    if not moving_parts:
        return []

    min_x, min_y, max_x, max_y = component_bbox(moving_parts)
    width_cells = max(1, int(round((max_x - min_x) / VOXEL_MM)))
    depth_cells = max(1, int(round((max_y - min_y) / VOXEL_MM)))
    site_x_cells = int(round(site_xy[0] / VOXEL_MM))
    site_y_cells = int(round(site_xy[1] / VOXEL_MM))
    if width_cells > site_x_cells or depth_cells > site_y_cells:
        return []

    source_x_cell = int(round(min_x / VOXEL_MM))
    source_y_cell = int(round(min_y / VOXEL_MM))
    scored = []
    for x_cell in range(0, site_x_cells - width_cells + 1):
        for y_cell in range(0, site_y_cells - depth_cells + 1):
            dx = (x_cell - source_x_cell) * VOXEL_MM
            dy = (y_cell - source_y_cell) * VOXEL_MM
            if dx == 0.0 and dy == 0.0:
                continue
            moved_parts = [translated_part(part, dx, dy) for part in moving_parts]
            if any_overlap(moved_parts, fixed_parts):
                continue
            contact = component_contact_score(moved_parts, fixed_parts, moving_groups, target_neighbors)
            if contact <= 0.0:
                continue
            move_distance = abs(dx) + abs(dy)
            scored.append((-contact, move_distance, x_cell, y_cell, moved_parts))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [(abs(item[0]), item[1], item[4]) for item in scored[:limit]]


def apply_moved_parts(
    rooms: list[dict[str, Any]],
    moving_groups: set[str],
    moved_parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    moved_by_id = {str(part["id"]): part for part in moved_parts}
    output = []
    for room in rooms:
        if room_functional_id(room) in moving_groups:
            output.append(dict(moved_by_id[str(room["id"])]))
        else:
            output.append(dict(room))
    return output


def coordinated_repair_house(
    house_dir: Path,
    output_dir: Path,
    max_iterations: int,
    max_component_size: int,
    candidate_limit: int,
) -> dict[str, Any]:
    house_id = house_dir.name
    layout = read_json(house_dir / "generated_layout.json")
    topology = read_json(house_dir / "topology.json")
    source_eval = read_json(house_dir / "evaluation.json")
    site_xy = site_xy_from_layout(layout)
    counts = counts_from_report(source_eval)
    rooms = [dict(room) for room in layout.get("rooms", [])]
    report = layout_report(house_id, rooms, counts, site_xy, topology)
    initial_metrics = topology_metrics_from_report(report)
    target_neighbors = target_neighbor_map(topology)
    accepted_moves = []
    rejected_candidate_count = 0

    for _iteration in range(max_iterations):
        current_score = topology_score(report)
        gaps = missing_edges(report)
        if not gaps:
            break
        realized_neighbors = realized_neighbor_map(report)
        available_groups = set(groups_by_id(rooms))
        best = None
        for edge in gaps:
            source = str(edge["source"])
            target = str(edge["target"])
            for moving_group, fixed_group in ((source, target), (target, source)):
                if moving_group not in available_groups or fixed_group not in available_groups:
                    continue
                component = build_component(
                    moving_group,
                    fixed_group,
                    realized_neighbors,
                    available_groups,
                    max_component_size,
                )
                for contact, distance, moved_parts in candidate_component_moves(
                    rooms,
                    component,
                    site_xy,
                    target_neighbors,
                    candidate_limit,
                ):
                    trial_rooms = apply_moved_parts(rooms, component, moved_parts)
                    trial_report = layout_report(house_id, trial_rooms, counts, site_xy, topology)
                    if not trial_report.get("p0", {}).get("pass", False):
                        rejected_candidate_count += 1
                        continue
                    trial_score = topology_score(trial_report)
                    if trial_score <= current_score:
                        rejected_candidate_count += 1
                        continue
                    key = (trial_score, len(component), contact, -distance)
                    if best is None or key > best[0]:
                        best = (
                            key,
                            trial_rooms,
                            trial_report,
                            {
                                "edge": [source, target],
                                "move_type": "graph_spatial_component_move",
                                "moved_groups": sorted(component),
                                "fixed_group": fixed_group,
                                "move_distance_mm": int(distance),
                                "contact_score": contact,
                                "score_before": list(current_score),
                                "score_after": list(trial_score),
                            },
                        )
        if best is None:
            break
        rooms = best[1]
        report = best[2]
        accepted_moves.append(best[3])

    final_metrics = topology_metrics_from_report(report)
    repaired_layout = dict(layout)
    repaired_layout["rooms"] = rooms
    repaired_layout.setdefault("metadata", {})["graph_spatial_coordinated_repair"] = {
        "enabled": True,
        "max_component_size": max_component_size,
        "accepted_move_count": len(accepted_moves),
    }
    write_json(output_dir / "generated_layout.json", repaired_layout)
    write_json(output_dir / "topology.json", topology)
    if (house_dir / "conditioning_topology.json").exists():
        shutil.copy2(house_dir / "conditioning_topology.json", output_dir / "conditioning_topology.json")
    write_json(output_dir / "evaluation.json", report)
    repair_report = {
        "enabled": True,
        "max_iterations": max_iterations,
        "max_component_size": max_component_size,
        "candidate_limit": candidate_limit,
        "accepted_move_count": len(accepted_moves),
        "rejected_candidate_count": rejected_candidate_count,
        "initial_topology": initial_metrics,
        "final_topology": final_metrics,
        "moves": accepted_moves,
    }
    write_json(output_dir / "graph_spatial_repair.json", repair_report)
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
    }


def main() -> None:
    args = parse_args()
    house_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if args.max_houses is not None:
        house_dirs = house_dirs[: args.max_houses]
    output_predicted = args.output_dir / "predicted_json"
    reports = []
    for house_dir in house_dirs:
        reports.append(
            coordinated_repair_house(
                house_dir,
                output_predicted / house_dir.name,
                args.max_iterations,
                args.max_component_size,
                args.candidate_limit,
            )
        )

    total_initial_edges = sum(report["initial_topology"].get("realized_edge_count", 0) for report in reports)
    total_final_edges = sum(report["final_topology"].get("realized_edge_count", 0) for report in reports)
    total_target_edges = sum(report["final_topology"].get("target_edge_count", 0) for report in reports)
    summary = {
        "schema": "graphspace_v6_graph_spatial_coordinated_repair_summary_v1",
        "purpose": (
            "Smoke experiment for whole-layout occupancy plus graph node-edge state. "
            "Moves small connected functional components, not just one part."
        ),
        "input_dir": str(args.input_dir),
        "house_count": len(reports),
        "max_iterations": args.max_iterations,
        "max_component_size": args.max_component_size,
        "candidate_limit": args.candidate_limit,
        "p0_pass_count": sum(1 for report in reports if report["p0_pass"]),
        "p1_full_target_topology_count": sum(
            1 for report in reports if report["p1_spatial_organization_pass"]
        ),
        "total_initial_realized_edges": total_initial_edges,
        "total_final_realized_edges": total_final_edges,
        "total_target_edges": total_target_edges,
        "accepted_move_count": sum(report["accepted_move_count"] for report in reports),
        "rejected_candidate_count": sum(report["rejected_candidate_count"] for report in reports),
        "reports": reports,
        "formal_v6_training_ready": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
