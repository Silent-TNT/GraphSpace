#!/usr/bin/env python3
"""Generate V5 room blocks from site dimensions and requested room counts only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate_instance_rollout import (
    box_face_contact,
    box_projection_overlap,
    building_from_coarse,
    mark_occupied,
    occupied_volume,
    place_nearest_box,
    place_topology_box,
    standard_room,
)
from instance_dataset import collate_instances
from instance_model import InstancePlacementPolicy
from joint_model import JointLayoutPolicy
from staged_dataset import (
    CHANNEL_INDEX,
    GRID_SHAPE,
    collate_staged,
    graph_arrays,
    reduce_2d,
)
from staged_model import StagedSpatialPolicy
from analyze_topology_dual import build_report as build_topology_dual_report


ROOT = Path(__file__).resolve().parents[2]
for import_dir in (
    ROOT,
    ROOT / "scripts" / "spatial_modal_infer",
    ROOT / "scripts" / "data_phase4",
):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.data_phase4.evaluate_candidates import ROOM_RULES, evaluate_candidate
from scripts.spatial_modal_infer.config import DEFAULT_ROOM_SIZE, ROOM_TYPES
from scripts.spatial_modal_infer.layout import build_user_request
from program_prior import ProgramPrior


VOXEL_MM = 300.0
TYPE_TO_ID = {value: index for index, value in enumerate(ROOM_TYPES)}
DIRECT_LIGHT_TYPES = {"living_room", "bedroom", "balcony"}
TRUNK_TYPES = {
    "stairs",
    "entryway",
    "corridor",
    "living_room",
    "dining_room",
    "kitchen",
}
CORE_REPAIR_TYPES = TRUNK_TYPES | {"bathroom"}
RULE_EXPANDABLE_TYPES = {"corridor"}
FIRST_FLOOR_TRUNK_EDGE_TYPES = {
    frozenset(("entryway", "stairs")),
    frozenset(("entryway", "living_room")),
    frozenset(("stairs", "corridor")),
    frozenset(("stairs", "living_room")),
    frozenset(("corridor", "living_room")),
    frozenset(("corridor", "dining_room")),
    frozenset(("living_room", "dining_room")),
    frozenset(("dining_room", "kitchen")),
}
SECOND_FLOOR_TRUNK_EDGE_TYPES = {
    frozenset(("stairs", "corridor")),
    frozenset(("corridor", "bedroom")),
    frozenset(("corridor", "bathroom")),
    frozenset(("corridor", "balcony")),
    frozenset(("corridor", "multi_purpose")),
}
TEST_CASES = {
    "small": {
        "site": (12000.0, 12000.0),
        "rooms": {
            "entryway": 1,
            "living_room": 1,
            "dining_room": 1,
            "bedroom": 2,
            "bathroom": 1,
            "corridor": 1,
            "stairs": 1,
        },
    },
    "medium": {
        "site": (18000.0, 15000.0),
        "rooms": {
            "entryway": 1,
            "living_room": 1,
            "dining_room": 1,
            "kitchen": 1,
            "bedroom": 3,
            "bathroom": 2,
            "corridor": 2,
            "stairs": 1,
            "balcony": 1,
        },
    },
    "large": {
        "site": (22000.0, 18000.0),
        "rooms": {
            "entryway": 1,
            "living_room": 1,
            "dining_room": 1,
            "kitchen": 1,
            "bedroom": 5,
            "bathroom": 3,
            "corridor": 2,
            "stairs": 1,
            "utility": 1,
            "balcony": 2,
            "multi_purpose": 1,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(TEST_CASES))
    parser.add_argument("--site-x", type=float)
    parser.add_argument("--site-y", type=float)
    parser.add_argument(
        "--rooms-json",
        help='JSON object, for example: {"living_room":1,"bedroom":3}',
    )
    parser.add_argument("--rooms-file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--staged-checkpoint", type=Path, required=True)
    parser.add_argument("--instance-checkpoint", type=Path)
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument(
        "--program-prior",
        type=Path,
        default=ROOT / "data" / "phase8_program_prior" / "program_prior.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def place_joint_rooms(
    model: JointLayoutPolicy,
    model_graph: dict,
    coarse: np.ndarray,
    site_cells: list[int],
    device: torch.device,
) -> tuple[list[dict], list[str], dict]:
    volume = np.concatenate(
        (coarse, np.zeros((1, *GRID_SHAPE), dtype=np.float32)),
        axis=0,
    )
    item = graph_item(model_graph)
    nodes = item["nodes"].unsqueeze(0).to(device)
    node_mask = torch.ones(
        (1, nodes.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    adjacency = torch.zeros(
        (1, 2, nodes.shape[1], nodes.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    for source, target, relation in model_graph["edges"]:
        adjacency[0, relation, source, target] = 1.0
    output = model(
        torch.from_numpy(volume).unsqueeze(0).to(device),
        nodes,
        node_mask,
        adjacency,
    )
    raw_boxes = output["boxes"][0].float().cpu().numpy()
    building = building_from_coarse(coarse, site_cells)
    occupied = np.zeros_like(building, dtype=np.uint8)
    predictions = {}
    predicted_floors = {}
    failures = []
    for room_index in placement_order(model_graph["nodes"]):
        node = model_graph["nodes"][room_index]
        floors = node_floors(node)
        box = place_nearest_box(
            raw_boxes[room_index],
            site_cells,
            floors,
            building,
            occupied,
        )
        if box is None:
            failures.append(node["instance_token"])
            continue
        predictions[room_index] = box
        predicted_floors[room_index] = floors
        mark_occupied(occupied, box, floors)

    required_edges = {
        tuple(sorted((int(left), int(right))))
        for left, right in model_graph.get("required_edges", [])
    }
    realized_edges = []
    for source, target, relation in model_graph["edges"]:
        if source >= target:
            continue
        quality = 0.0
        if source in predictions and target in predictions:
            if relation == 1:
                quality = box_projection_overlap(
                    predictions[source],
                    predictions[target],
                )
            elif set(predicted_floors[source]) & set(predicted_floors[target]):
                quality = box_face_contact(
                    predictions[source],
                    predictions[target],
                )
        realized_edges.append(
            {
                "source": model_graph["nodes"][source]["instance_token"],
                "target": model_graph["nodes"][target]["instance_token"],
                "relation": "vertical" if relation == 1 else "horizontal",
                "required": tuple(sorted((source, target))) in required_edges,
                "realized": quality > 0,
                "contact_quality": quality,
            }
        )
    required_count = sum(edge["required"] for edge in realized_edges)
    report = {
        "target_edge_count": len(realized_edges),
        "realized_edge_count": sum(edge["realized"] for edge in realized_edges),
        "realization_rate": (
            sum(edge["realized"] for edge in realized_edges)
            / max(len(realized_edges), 1)
        ),
        "required_edge_count": required_count,
        "required_realized_edge_count": sum(
            edge["required"] and edge["realized"] for edge in realized_edges
        ),
        "required_realization_rate": (
            sum(edge["required"] and edge["realized"] for edge in realized_edges)
            / max(required_count, 1)
        ),
        "edges": realized_edges,
        "projection": "joint_weight_nearest_legal_only",
    }
    rooms = [
        standard_room(index, model_graph["nodes"][index], box)
        for index, box in sorted(predictions.items())
    ]
    for room, index in zip(rooms, sorted(predictions)):
        room["id"] = model_graph["nodes"][index]["instance_token"]
    return rooms, failures, report


def validate_request(site_x: float, site_y: float, counts: dict[str, int]) -> None:
    for value, name in ((site_x, "site_x"), (site_y, "site_y")):
        cells = int(np.floor(value / VOXEL_MM))
        if not 1 <= cells <= 88:
            raise ValueError(f"{name} must be between 300 and 26400 mm")
    unknown = sorted(set(counts) - set(ROOM_TYPES))
    if unknown:
        raise ValueError(f"unknown room types: {unknown}")
    if any(int(value) < 0 for value in counts.values()):
        raise ValueError("room counts must be non-negative")
    if sum(int(value) for value in counts.values()) == 0:
        raise ValueError("at least one room is required")


def request_graph(
    counts: dict[str, int],
    site_x: float,
    site_y: float,
    seed: int,
    program_prior: ProgramPrior,
) -> tuple[dict, dict]:
    graph, positions, topology_nodes, edge_types, evidence = (
        program_prior.build_topology(
            counts,
            site_x,
            site_y,
            seed=seed,
        )
    )
    site_area = max(site_x * site_y, 1.0)
    nodes = []
    node_lookup = {}
    inferred_conditions = evidence.get("node_conditions", {})
    for index, (node_id, room_type, floor) in enumerate(topology_nodes):
        node_lookup[node_id] = index
        condition = inferred_conditions.get(node_id, {})
        width, depth, _ = DEFAULT_ROOM_SIZE[room_type]
        area_ratio = float(
            condition.get("area_ratio", width * depth / site_area)
        )
        lighting_access = str(
            condition.get(
                "lighting_access",
                "direct" if room_type in DIRECT_LIGHT_TYPES else "indirect",
            )
        )
        lighting_id = {"none": 0, "indirect": 1, "direct": 2}.get(
            lighting_access,
            0,
        )
        floor_text = str(floor)
        nodes.append(
            {
                "instance_token": node_id,
                "type": room_type,
                "type_id": TYPE_TO_ID[room_type],
                "floor_1": int(floor_text in {"1", "1&2"}),
                "floor_2": int(floor_text in {"2", "1&2"}),
                "target_area_ratio": area_ratio,
                "exterior_sides": [],
                "lighting_access": lighting_access,
                "lighting_id": lighting_id,
                "lighting_priority": int(
                    condition.get(
                        "lighting_priority",
                        8 if room_type in {"living_room", "bedroom"} else 4,
                    )
                ),
            }
        )
    edges = []
    topology_edges = []
    for left, right in graph.edges:
        relation_name = edge_types.get(
            (left, right),
            edge_types.get((right, left), "horizontal"),
        )
        relation = 1 if relation_name == "vertical" else 0
        left_index, right_index = node_lookup[left], node_lookup[right]
        edges.append([left_index, right_index, relation])
        edges.append([right_index, left_index, relation])
        topology_edges.append(
            {"source": left, "target": right, "relation": relation_name}
        )
    model_graph = {
        "nodes": nodes,
        "edges": edges,
        "required_edges": [
            [node_lookup[left], node_lookup[right]]
            for left, right in evidence.get("required_edges", [])
        ],
        "relation_types": {"horizontal_contact": 0, "vertical_contact": 1},
    }
    topology = {
        "seed": seed,
        "evidence": evidence,
        "nodes": [
            {
                "id": node_id,
                "type": room_type,
                "floor": floor,
                "position": [float(value) for value in positions[node_id]],
            }
            for node_id, room_type, floor in topology_nodes
        ],
        "edges": topology_edges,
    }
    return model_graph, topology


def graph_item(model_graph: dict) -> dict:
    nodes, edge_index, edge_type = graph_arrays(model_graph)
    return {
        "nodes": torch.from_numpy(nodes),
        "edge_index": torch.from_numpy(edge_index),
        "edge_type": torch.from_numpy(edge_type),
    }


def initial_volume(site_cells: list[int]) -> np.ndarray:
    high_resolution = np.ones((site_cells[0], site_cells[1]), dtype=np.uint8)
    site_2d = reduce_2d(high_resolution)
    state = np.zeros((8, *GRID_SHAPE), dtype=np.float32)
    state[CHANNEL_INDEX["site"]] = np.repeat(
        site_2d[:, :, None],
        GRID_SHAPE[2],
        axis=2,
    )
    return state


def predict_coarse(
    model: StagedSpatialPolicy,
    model_graph: dict,
    site_cells: list[int],
    device: torch.device,
) -> np.ndarray:
    state = initial_volume(site_cells)
    base_graph = graph_item(model_graph)
    for stage_id in range(6):
        item = {
            "house_id": "user_request",
            "stage_id": torch.tensor(stage_id, dtype=torch.long),
            "volume": torch.from_numpy(state.copy()),
            "target_volume": torch.zeros((2, *GRID_SHAPE)),
            "target_valid": torch.zeros(2),
            "reachability": torch.tensor(0.0),
            "cut_ratio": torch.tensor(0.5),
            **base_graph,
        }
        batch = collate_staged([item])
        output = model(
            batch["volume"].to(device),
            batch["nodes"].to(device),
            batch["node_mask"].to(device),
            batch["adjacency"].to(device),
            batch["stage_id"].to(device),
        )
        probability = torch.sigmoid(output["mask_logits"][0]).float().cpu().numpy()
        prediction = probability >= 0.5
        if stage_id == 0:
            state[CHANNEL_INDEX["protected_stairs"]] = prediction[0]
        elif stage_id == 1:
            cut_cell = min(
                max(round(float(output["cut_ratio"][0].cpu()) * 20), 1),
                19,
            )
            state[CHANNEL_INDEX["floor_boundary"], :, :, cut_cell] = 1
        elif stage_id == 2:
            site = state[CHANNEL_INDEX["site"]] > 0
            stairs = state[CHANNEL_INDEX["protected_stairs"]] > 0
            building = (probability[0] >= probability[1]) & site
            empty = (probability[1] > probability[0]) & site
            building |= stairs
            empty &= ~stairs
            state[CHANNEL_INDEX["building_envelope"]] = building
            state[CHANNEL_INDEX["explicit_empty"]] = empty
        elif stage_id == 3:
            building = state[CHANNEL_INDEX["building_envelope"]] > 0
            stairs = state[CHANNEL_INDEX["protected_stairs"]] > 0
            state[CHANNEL_INDEX["traffic_reserve"]] = (
                (prediction[0] & building) | stairs
            )
        elif stage_id == 4:
            building = state[CHANNEL_INDEX["building_envelope"]] > 0
            traffic = state[CHANNEL_INDEX["traffic_reserve"]] > 0
            state[CHANNEL_INDEX["rigid_functions"]] = (
                prediction[0] & building & ~traffic
            )
        elif stage_id == 5:
            building = state[CHANNEL_INDEX["building_envelope"]] > 0
            traffic = state[CHANNEL_INDEX["traffic_reserve"]] > 0
            rigid = state[CHANNEL_INDEX["rigid_functions"]] > 0
            state[CHANNEL_INDEX["service_spaces"]] = (
                prediction[0] & building & ~traffic & ~rigid
            )
    return state


def node_floors(node: dict) -> list[int]:
    floors = []
    if node["floor_1"]:
        floors.append(1)
    if node["floor_2"]:
        floors.append(2)
    return floors


def placement_order(nodes: list[dict]) -> list[int]:
    generation_priority = {
        "stairs": 0,
        "entryway": 1,
        "corridor": 2,
        "living_room": 3,
        "dining_room": 4,
        "kitchen": 5,
        "bathroom": 6,
        "bedroom": 7,
        "multi_purpose": 8,
        "utility": 9,
        "balcony": 10,
    }
    return sorted(
        range(len(nodes)),
        key=lambda index: (
            generation_priority.get(nodes[index]["type"], 99),
            min(node_floors(nodes[index])),
            -float(nodes[index]["target_area_ratio"]),
            index,
        ),
    )


def fallback_prediction_for_node(node: dict) -> np.ndarray:
    width_mm, depth_mm, _ = DEFAULT_ROOM_SIZE[node["type"]]
    area_scale = max(float(node.get("target_area_ratio", 0.0)), 1.0e-4) ** 0.5
    width = max(width_mm / VOXEL_MM, 2.0) * max(area_scale, 0.8)
    depth = max(depth_mm / VOXEL_MM, 2.0) * max(area_scale, 0.8)
    return np.asarray([0.5, 0.5, width / 88.0, depth / 88.0], dtype=np.float32)


def repair_missing_rooms(
    model_graph: dict,
    site_cells: list[int],
    building: np.ndarray,
    occupied: np.ndarray,
    predictions: dict[int, tuple[int, int, int, int]],
    predicted_floors: dict[int, list[int]],
    topology_neighbors: dict[int, list[tuple[int, int, bool]]],
    raw_predictions: dict[int, np.ndarray],
    failures: list[str],
) -> tuple[list[str], list[dict]]:
    repaired = []
    remaining_failures = []
    failed_indices = {
        index
        for index, node in enumerate(model_graph["nodes"])
        if node["instance_token"] in set(failures)
    }
    ordered = placement_order(model_graph["nodes"])
    for room_index in sorted(
        failed_indices,
        key=lambda index: (
            model_graph["nodes"][index]["type"] not in CORE_REPAIR_TYPES,
            ordered.index(index),
        ),
    ):
        node = model_graph["nodes"][room_index]
        floors = node_floors(node)
        required_neighbors = [
            (neighbor, relation, required)
            for neighbor, relation, required in topology_neighbors[room_index]
            if neighbor in predictions
        ]
        prediction = raw_predictions.get(room_index, fallback_prediction_for_node(node))
        box, details = place_topology_box(
            prediction,
            site_cells,
            floors,
            building,
            occupied,
            predictions,
            predicted_floors,
            required_neighbors,
            needs_exterior=bool(ROOM_RULES.get(node["type"], {}).get("needs_exterior")),
        )
        if box is None:
            box = place_nearest_box(
                fallback_prediction_for_node(node),
                site_cells,
                floors,
                building,
                occupied,
            )
            details = {"final_repair": True, "fallback_nearest_box": box is not None}
        if box is None:
            remaining_failures.append(node["instance_token"])
            continue
        predictions[room_index] = box
        predicted_floors[room_index] = floors
        mark_occupied(occupied, box, floors)
        repaired.append(
            {
                "node": node["instance_token"],
                "type": node["type"],
                "box": list(box),
                "details": {
                    **details,
                    "final_repair": True,
                    "repair_priority": (
                        "core_trunk" if node["type"] in CORE_REPAIR_TYPES else "branch"
                    ),
                },
            }
        )
    return remaining_failures, repaired


def edge_quality(
    source: int,
    target: int,
    relation: int,
    predictions: dict[int, tuple[int, int, int, int]],
    predicted_floors: dict[int, list[int]],
) -> float:
    if source not in predictions or target not in predictions:
        return 0.0
    if relation == 1:
        return box_projection_overlap(predictions[source], predictions[target])
    if set(predicted_floors[source]) & set(predicted_floors[target]):
        return box_face_contact(predictions[source], predictions[target])
    return 0.0


def trunk_edge_scope(nodes: list[dict], source: int, target: int) -> str | None:
    type_pair = frozenset((nodes[source]["type"], nodes[target]["type"]))
    shared = set(node_floors(nodes[source])) & set(node_floors(nodes[target]))
    if 1 in shared and type_pair in FIRST_FLOOR_TRUNK_EDGE_TYPES:
        return "floor_1"
    if 2 in shared and type_pair in SECOND_FLOOR_TRUNK_EDGE_TYPES:
        return "floor_2"
    return None


def is_trunk_edge(nodes: list[dict], source: int, target: int) -> bool:
    return trunk_edge_scope(nodes, source, target) is not None


def split_room_box(room: dict, part_count: int) -> list[dict]:
    if part_count <= 1:
        output = dict(room)
        output.setdefault("functional_id", room["id"])
        return [output]
    x0, y0, z0 = [float(value) for value in room["box_min"]]
    x1, y1, z1 = [float(value) for value in room["box_max"]]
    x_cells = int(round((x1 - x0) / VOXEL_MM))
    y_cells = int(round((y1 - y0) / VOXEL_MM))
    axis = 0 if x_cells >= y_cells else 1
    cells = x_cells if axis == 0 else y_cells
    if cells < part_count:
        output = dict(room)
        output.setdefault("functional_id", room["id"])
        return [output]
    boundaries = [
        int(round(index * cells / part_count))
        for index in range(part_count + 1)
    ]
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        output = dict(room)
        output.setdefault("functional_id", room["id"])
        return [output]
    parts = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        part = dict(room)
        part["id"] = f"{room['id']}_part_{index}"
        part["functional_id"] = room["id"]
        if axis == 0:
            part["box_min"] = [x0 + left * VOXEL_MM, y0, z0]
            part["box_max"] = [x0 + right * VOXEL_MM, y1, z1]
        else:
            part["box_min"] = [x0, y0 + left * VOXEL_MM, z0]
            part["box_max"] = [x1, y0 + right * VOXEL_MM, z1]
        parts.append(part)
    return parts


def expand_functional_parts(
    rooms: list[dict],
    functional_group_counts: dict[str, int],
    part_counts: dict[str, int],
    expandable_types: set[str] | None = None,
) -> tuple[list[dict], dict]:
    expandable_types = expandable_types or RULE_EXPANDABLE_TYPES
    by_type: dict[str, list[dict]] = {}
    for room in rooms:
        by_type.setdefault(str(room["type"]), []).append(room)
    target_parts_by_room: dict[str, int] = {}
    report = {
        "enabled": True,
        "method": "rule_based_axis_split",
        "expandable_types": sorted(expandable_types),
        "expanded_groups": [],
        "skipped_groups": [],
        "input_room_count": len(rooms),
    }
    for room_type in sorted(expandable_types):
        type_rooms = sorted(by_type.get(room_type, []), key=lambda item: item["id"])
        group_count = int(functional_group_counts.get(room_type, len(type_rooms)))
        requested_parts = int(part_counts.get(room_type, len(type_rooms)))
        if not type_rooms or group_count <= 0:
            continue
        requested_parts = max(requested_parts, len(type_rooms))
        base = requested_parts // len(type_rooms)
        remainder = requested_parts % len(type_rooms)
        for index, room in enumerate(type_rooms):
            target_parts_by_room[room["id"]] = max(
                1,
                base + (1 if index < remainder else 0),
            )

    expanded = []
    for room in rooms:
        target_part_count = target_parts_by_room.get(room["id"], 1)
        parts = split_room_box(room, target_part_count)
        if len(parts) != target_part_count:
            report["skipped_groups"].append(
                {
                    "functional_id": room["id"],
                    "type": room["type"],
                    "target_part_count": target_part_count,
                    "actual_part_count": len(parts),
                    "reason": "box_too_small_for_safe_axis_split",
                }
            )
        elif target_part_count > 1:
            report["expanded_groups"].append(
                {
                    "functional_id": room["id"],
                    "type": room["type"],
                    "part_count": target_part_count,
                    "part_ids": [part["id"] for part in parts],
                }
            )
        expanded.extend(parts)
    report["output_part_count"] = len(expanded)
    report["expanded_group_count"] = len(report["expanded_groups"])
    report["skipped_group_count"] = len(report["skipped_groups"])
    return expanded, report


def try_move_node_for_edge(
    node_index: int,
    anchor_index: int,
    relation: int,
    model_graph: dict,
    site_cells: list[int],
    building: np.ndarray,
    occupied: np.ndarray,
    predictions: dict[int, tuple[int, int, int, int]],
    predicted_floors: dict[int, list[int]],
    topology_neighbors: dict[int, list[tuple[int, int, bool]]],
    raw_predictions: dict[int, np.ndarray],
) -> tuple[bool, dict]:
    old_box = predictions.get(node_index)
    if old_box is None or anchor_index not in predictions:
        return False, {}
    floors = predicted_floors[node_index]
    x0, y0, x1, y1 = old_box
    for floor in floors:
        occupied[floor - 1, x0:x1, y0:y1] = 0
    all_priority_neighbors = [
        (neighbor, neighbor_relation, True)
        for neighbor, neighbor_relation, neighbor_required in topology_neighbors[
            node_index
        ]
        if neighbor in predictions
        and (
            neighbor == anchor_index
            or neighbor_required
            or is_trunk_edge(model_graph["nodes"], node_index, neighbor)
        )
    ]
    direct_neighbor = [(anchor_index, relation, True)]
    tried_neighbor_sets = [all_priority_neighbors, direct_neighbor]
    for neighbor_set in tried_neighbor_sets:
        replacement, details = place_topology_box(
            raw_predictions.get(
                node_index,
                fallback_prediction_for_node(model_graph["nodes"][node_index]),
            ),
            site_cells,
            floors,
            building,
            occupied,
            predictions,
            predicted_floors,
            neighbor_set,
            needs_exterior=bool(
                ROOM_RULES.get(
                    model_graph["nodes"][node_index]["type"],
                    {},
                ).get("needs_exterior")
            ),
        )
        if replacement is None:
            continue
        predictions[node_index] = replacement
        if (
            edge_quality(
                node_index,
                anchor_index,
                relation,
                predictions,
                predicted_floors,
            )
            > 0
        ):
            mark_occupied(occupied, replacement, floors)
            return replacement != old_box, {
                **details,
                "priority_edge_repair": True,
                "target_only_fallback": neighbor_set == direct_neighbor,
            }
    predictions[node_index] = old_box
    mark_occupied(occupied, old_box, floors)
    return False, {"priority_edge_repair_failed": True}


def repair_priority_edges(
    model_graph: dict,
    site_cells: list[int],
    building: np.ndarray,
    occupied: np.ndarray,
    predictions: dict[int, tuple[int, int, int, int]],
    predicted_floors: dict[int, list[int]],
    topology_neighbors: dict[int, list[tuple[int, int, bool]]],
    raw_predictions: dict[int, np.ndarray],
    order: list[int],
    required_edges: set[tuple[int, int]],
) -> list[dict]:
    all_relations = []
    priority_relations = []
    seen = set()
    for source, target, relation in model_graph["edges"]:
        edge = tuple(sorted((source, target)))
        if edge in seen:
            continue
        seen.add(edge)
        required = edge in required_edges
        trunk = is_trunk_edge(model_graph["nodes"], source, target)
        all_relations.append((source, target, relation, required, trunk))
        if required or trunk:
            priority_relations.append((source, target, relation, required, trunk))

    def realized_score(relations: list[tuple[int, int, int, bool, bool]]) -> int:
        score = 0
        for source, target, relation, required, trunk in relations:
            if (
                edge_quality(
                    source,
                    target,
                    relation,
                    predictions,
                    predicted_floors,
                )
                <= 0
            ):
                continue
            score += 100 if required else 0
            score += 20 if trunk else 0
            score += 1
        return score

    order_offsets = {room_index: offset for offset, room_index in enumerate(order)}
    repairs = []
    for _pass in range(4):
        missing_priority = [
            edge
            for edge in priority_relations
            if edge_quality(
                edge[0],
                edge[1],
                edge[2],
                predictions,
                predicted_floors,
            )
            <= 0
        ]
        if not missing_priority:
            break
        changed = False
        for source, target, relation, required, trunk in sorted(
            missing_priority,
            key=lambda edge: (not edge[3], not edge[4]),
        ):
            candidates = sorted(
                (source, target),
                key=lambda index: (
                    model_graph["nodes"][index]["type"] in {"stairs", "entryway"},
                    model_graph["nodes"][index]["type"] not in TRUNK_TYPES,
                    -order_offsets.get(index, -1),
                ),
            )
            moved = candidates[0]
            repaired = False
            details = {"priority_edge_repair_failed": True}
            for movable in candidates:
                anchor = target if movable == source else source
                old_box = predictions.get(movable)
                if old_box is None:
                    continue
                before_score = realized_score(all_relations)
                repaired, details = try_move_node_for_edge(
                    movable,
                    anchor,
                    relation,
                    model_graph,
                    site_cells,
                    building,
                    occupied,
                    predictions,
                    predicted_floors,
                    topology_neighbors,
                    raw_predictions,
                )
                moved = movable
                after_score = realized_score(all_relations)
                if repaired and after_score < before_score:
                    new_box = predictions[movable]
                    floors = predicted_floors[movable]
                    nx0, ny0, nx1, ny1 = new_box
                    for floor in floors:
                        occupied[floor - 1, nx0:nx1, ny0:ny1] = 0
                    predictions[movable] = old_box
                    mark_occupied(occupied, old_box, floors)
                    repaired = False
                    details = {
                        **details,
                        "priority_edge_repair_rejected_regression": True,
                    }
                    continue
                if (
                    edge_quality(
                        source,
                        target,
                        relation,
                        predictions,
                        predicted_floors,
                    )
                    > 0
                ):
                    break
            repairs.append(
                {
                    "source": model_graph["nodes"][source]["instance_token"],
                    "target": model_graph["nodes"][target]["instance_token"],
                    "relation": "vertical" if relation == 1 else "horizontal",
                    "required": required,
                    "trunk": trunk,
                    "moved": model_graph["nodes"][moved]["instance_token"],
                    "repaired": repaired,
                    "details": {
                        **details,
                        "priority_edge_repair": True,
                    },
                }
            )
            changed |= repaired
        if not changed:
            break
    return repairs


def place_rooms(
    model: InstancePlacementPolicy,
    model_graph: dict,
    coarse: np.ndarray,
    site_cells: list[int],
    device: torch.device,
) -> tuple[list[dict], list[str], dict]:
    volume = np.concatenate(
        (coarse, np.zeros((1, *GRID_SHAPE), dtype=np.float32)),
        axis=0,
    )
    building = building_from_coarse(coarse, site_cells)
    occupied = np.zeros_like(building, dtype=np.uint8)
    base_graph = graph_item(model_graph)
    order = placement_order(model_graph["nodes"])
    predictions = {}
    predicted_floors = {}
    raw_predictions = {}
    placement_details = {}
    failures = []
    topology_neighbors = {index: [] for index in range(len(model_graph["nodes"]))}
    required_edges = {
        tuple(sorted((int(left), int(right))))
        for left, right in model_graph.get("required_edges", [])
    }
    for source, target, relation in model_graph["edges"]:
        if source < target:
            required = (source, target) in required_edges
            topology_neighbors[source].append((target, relation, required))
            topology_neighbors[target].append((source, relation, required))
    for offset, room_index in enumerate(order):
        volume[8] = occupied_volume(occupied)
        item = {
            "house_id": "user_request",
            "room_index": torch.tensor(room_index, dtype=torch.long),
            "step_ratio": torch.tensor(
                offset / max(len(order) - 1, 1),
                dtype=torch.float32,
            ),
            "site_cells": torch.tensor(site_cells),
            "volume": torch.from_numpy(volume.copy()),
            "target_box": torch.zeros(4),
            **base_graph,
        }
        batch = collate_instances([item])
        output = model(
            batch["volume"].to(device),
            batch["nodes"].to(device),
            batch["node_mask"].to(device),
            batch["adjacency"].to(device),
            batch["room_index"].to(device),
            batch["step_ratio"].to(device),
        )
        node = model_graph["nodes"][room_index]
        floors = node_floors(node)
        raw_predictions[room_index] = output["box"][0].float().cpu().numpy()
        box, details = place_topology_box(
            raw_predictions[room_index],
            site_cells,
            floors,
            building,
            occupied,
            predictions,
            predicted_floors,
            topology_neighbors[room_index],
            needs_exterior=bool(
                ROOM_RULES.get(node["type"], {}).get("needs_exterior")
            ),
        )
        if box is None:
            failures.append(node["instance_token"])
            continue
        predictions[room_index] = box
        predicted_floors[room_index] = floors
        placement_details[node["instance_token"]] = details
        mark_occupied(occupied, box, floors)

    failures, final_repairs = repair_missing_rooms(
        model_graph,
        site_cells,
        building,
        occupied,
        predictions,
        predicted_floors,
        topology_neighbors,
        raw_predictions,
        failures,
    )
    for repair in final_repairs:
        placement_details[repair["node"]] = repair["details"]

    priority_edge_repairs = repair_priority_edges(
        model_graph,
        site_cells,
        building,
        occupied,
        predictions,
        predicted_floors,
        topology_neighbors,
        raw_predictions,
        order,
        required_edges,
    )
    for repair in priority_edge_repairs:
        placement_details[repair["moved"]] = {
            **repair["details"],
            "local_repair": repair["repaired"],
        }
    rooms = [
        standard_room(index, model_graph["nodes"][index], box)
        for index, box in sorted(predictions.items())
    ]
    for room, index in zip(rooms, sorted(predictions)):
        room["id"] = model_graph["nodes"][index]["instance_token"]
    realized_edges = []
    for source, target, relation in model_graph["edges"]:
        if source >= target:
            continue
        source_box = predictions.get(source)
        target_box = predictions.get(target)
        quality = 0.0
        if source_box is not None and target_box is not None:
            quality = edge_quality(
                source,
                target,
                relation,
                predictions,
                predicted_floors,
            )
        trunk_scope = trunk_edge_scope(model_graph["nodes"], source, target)
        realized_edges.append(
            {
                "source": model_graph["nodes"][source]["instance_token"],
                "target": model_graph["nodes"][target]["instance_token"],
                "relation": "vertical" if relation == 1 else "horizontal",
                "required": (source, target) in required_edges,
                "trunk": trunk_scope is not None,
                "trunk_scope": trunk_scope,
                "realized": quality > 0,
                "contact_quality": quality,
            }
        )
    topology_report = {
        "target_edge_count": len(realized_edges),
        "realized_edge_count": sum(edge["realized"] for edge in realized_edges),
        "realization_rate": (
            sum(edge["realized"] for edge in realized_edges)
            / max(len(realized_edges), 1)
        ),
        "required_edge_count": sum(edge["required"] for edge in realized_edges),
        "required_realized_edge_count": sum(
            edge["required"] and edge["realized"] for edge in realized_edges
        ),
        "required_realization_rate": (
            sum(edge["required"] and edge["realized"] for edge in realized_edges)
            / max(sum(edge["required"] for edge in realized_edges), 1)
        ),
        "edges": realized_edges,
        "placement_details": placement_details,
        "final_repairs": final_repairs,
        "priority_edge_repairs": priority_edge_repairs,
        "trunk_edge_count": sum(edge["trunk"] for edge in realized_edges),
        "trunk_edge_realized_count": sum(
            edge["trunk"] and edge["realized"] for edge in realized_edges
        ),
        "trunk_edge_realization_rate": (
            sum(edge["trunk"] and edge["realized"] for edge in realized_edges)
            / max(sum(edge["trunk"] for edge in realized_edges), 1)
        ),
        "trunk_edge_scope_counts": {
            scope: sum(edge["trunk_scope"] == scope for edge in realized_edges)
            for scope in ("floor_1", "floor_2")
        },
        "trunk_edge_scope_realized_counts": {
            scope: sum(
                edge["trunk_scope"] == scope and edge["realized"]
                for edge in realized_edges
            )
            for scope in ("floor_1", "floor_2")
        },
    }
    return rooms, failures, topology_report


def main() -> None:
    args = parse_args()
    if bool(args.instance_checkpoint) == bool(args.joint_checkpoint):
        raise ValueError(
            "provide exactly one of --instance-checkpoint or --joint-checkpoint"
        )
    if args.case:
        case = TEST_CASES[args.case]
        site_x, site_y = case["site"]
        counts = dict(case["rooms"])
    else:
        if args.site_x is None or args.site_y is None:
            raise ValueError(
                "provide --case, or provide --site-x and --site-y"
            )
        site_x, site_y = args.site_x, args.site_y
        rooms_payload = (
            json.loads(args.rooms_file.read_text(encoding="utf-8"))
            if args.rooms_file
            else json.loads(args.rooms_json)
            if args.rooms_json
            else {}
        )
        counts = (
            {
                str(key): int(value)
                for key, value in rooms_payload.items()
                if int(value) >= 0
            }
            if rooms_payload
            else {}
        )
    program_prior = ProgramPrior(args.program_prior)
    neighbors = program_prior.neighbors(site_x, site_y)
    counts, count_evidence = program_prior.infer_counts(
        neighbors,
        args.seed,
        explicit_counts=counts,
        infer_missing=not bool(args.case),
    )
    part_counts, part_count_evidence = program_prior.infer_part_counts(
        counts,
        neighbors,
        args.seed,
    )
    validate_request(site_x, site_y, counts)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    device = torch.device(args.device)
    request = build_user_request(site_x, site_y, counts)
    request["program_source"] = (
        "user_complete"
        if args.case
        else (
            "user_partial_plus_training_data"
            if args.rooms_json or args.rooms_file
            else "training_data_knn"
        )
    )
    request["program_prior"] = str(args.program_prior)
    request["functional_group_counts"] = dict(counts)
    request["part_counts"] = dict(part_counts)
    request["count_semantics"] = {
        "requested_counts": "functional_group_counts",
        "part_counts": "predicted rectangular part counts",
        "geometry_decoder": (
            "model decoder emits one box per functional topology node; "
            "rule expander may split selected room types into multiple parts"
        ),
    }
    model_graph, topology = request_graph(
        counts,
        site_x,
        site_y,
        args.seed,
        program_prior,
    )
    topology["count_evidence"] = count_evidence
    topology["part_count_evidence"] = part_count_evidence
    staged_checkpoint = torch.load(args.staged_checkpoint, map_location=device)
    staged_model = StagedSpatialPolicy(
        int(staged_checkpoint["config"]["base_channels"])
    ).to(device)
    staged_model.load_state_dict(staged_checkpoint["model"])
    staged_model.eval()
    instance_model = None
    joint_model = None
    if args.joint_checkpoint:
        joint_checkpoint = torch.load(args.joint_checkpoint, map_location=device)
        joint_model = JointLayoutPolicy(
            int(joint_checkpoint["config"]["base_channels"])
        ).to(device)
        joint_model.load_state_dict(joint_checkpoint["model"])
        joint_model.eval()
    else:
        instance_checkpoint = torch.load(
            args.instance_checkpoint,
            map_location=device,
        )
        instance_model = InstancePlacementPolicy(
            int(instance_checkpoint["config"]["base_channels"])
        ).to(device)
        instance_model.load_state_dict(instance_checkpoint["model"])
        instance_model.eval()
    site_cells = [
        int(np.floor(site_x / VOXEL_MM)),
        int(np.floor(site_y / VOXEL_MM)),
    ]
    with torch.no_grad():
        coarse = predict_coarse(
            staged_model,
            model_graph,
            site_cells,
            device,
        )
        if joint_model is not None:
            rooms, failures, topology_report = place_joint_rooms(
                joint_model,
                model_graph,
                coarse,
                site_cells,
                device,
            )
        else:
            rooms, failures, topology_report = place_rooms(
                instance_model,
                model_graph,
                coarse,
                site_cells,
                device,
            )
    rooms, expansion_report = expand_functional_parts(
        rooms,
        counts,
        part_counts,
    )
    topology_report["functional_part_expansion"] = expansion_report
    candidate_id = f"user_{int(site_x)}x{int(site_y)}_seed{args.seed}"
    candidate = {
        "house_id": candidate_id,
        "metadata": {
            "source": "user_conditions_only",
            "seed": args.seed,
            "building_size": {
                "x": site_x,
                "y": site_y,
                "z": 6000.0,
            },
            "stats": counts,
        },
        "rooms": rooms,
    }
    report, _ = evaluate_candidate(
        candidate_id,
        rooms,
        counts,
        (site_x, site_y),
        topology=topology,
    )
    summary = {
        "input_source": (
            "site dimensions, requested room counts and seed"
            if args.case or args.rooms_json or args.rooms_file
            else "site dimensions and seed; room program inferred from train split"
        ),
        "program_source": request["program_source"],
        "program_prior": str(args.program_prior),
        "nearest_training_houses": topology["evidence"]["nearest_houses"],
        "room_counts": counts,
        "functional_group_counts": counts,
        "part_counts": part_counts,
        "count_semantics": request["count_semantics"],
        "staged_checkpoint": str(args.staged_checkpoint),
        "instance_checkpoint": (
            str(args.joint_checkpoint)
            if args.joint_checkpoint
            else str(args.instance_checkpoint)
        ),
        "instance_model_type": (
            "joint_whole_house"
            if args.joint_checkpoint
            else "autoregressive_single_room"
        ),
        "requested_count": sum(counts.values()),
        "placed_count": len(rooms),
        "functional_group_count": sum(counts.values()),
        "rectangular_part_count": len(rooms),
        "functional_part_expansion": expansion_report,
        "generation_grid_mm": [
            site_cells[0] * VOXEL_MM,
            site_cells[1] * VOXEL_MM,
        ],
        "unoccupied_boundary_remainder_mm": [
            site_x - site_cells[0] * VOXEL_MM,
            site_y - site_cells[1] * VOXEL_MM,
        ],
        "failed_nodes": failures,
        "topology_target_edge_count": topology_report["target_edge_count"],
        "topology_realized_edge_count": topology_report["realized_edge_count"],
        "topology_realization_rate": topology_report["realization_rate"],
        "required_topology_realization_rate": topology_report[
            "required_realization_rate"
        ],
        "p0_pass": report["p0"]["pass"],
        "p1_hard_geometry_pass": report["p1_spatial_organization"][
            "hard_geometry_pass"
        ],
        "p1_spatial_organization_pass": report["p1_spatial_organization"][
            "spatial_organization_pass"
        ],
        "p2_quality_gate_pass": report["p2"]["quality_gate_pass"],
        "p2_enabled": report["p2"].get("enabled", True),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "topology.json").write_text(
        json.dumps(topology, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "generated_layout.json").write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "topology_realization.json").write_text(
        json.dumps(topology_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    topology_dual_report = build_topology_dual_report(topology, candidate)
    (args.output_dir / "topology_dual_report.json").write_text(
        json.dumps(topology_dual_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(args.output_dir / "coarse_volume.npz", volume=coarse)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
