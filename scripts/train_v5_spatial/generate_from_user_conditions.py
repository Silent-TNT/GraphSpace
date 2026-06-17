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
        "bedroom": 3,
        "living_room": 4,
        "dining_room": 5,
        "kitchen": 6,
        "bathroom": 7,
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

    order_offsets = {room_index: offset for offset, room_index in enumerate(order)}

    def edge_quality(source: int, target: int, relation: int) -> float:
        if source not in predictions or target not in predictions:
            return 0.0
        if relation == 1:
            return box_projection_overlap(predictions[source], predictions[target])
        if set(predicted_floors[source]) & set(predicted_floors[target]):
            return box_face_contact(predictions[source], predictions[target])
        return 0.0

    required_relations = []
    relation_lookup = {
        tuple(sorted((source, target))): relation
        for source, target, relation in model_graph["edges"]
    }
    for source, target in required_edges:
        required_relations.append(
            (source, target, relation_lookup[(source, target)])
        )
    for _pass in range(3):
        missing_required = [
            edge for edge in required_relations if edge_quality(*edge) <= 0
        ]
        if not missing_required:
            break
        changed = False
        for source, target, _relation in missing_required:
            movable = max(
                (source, target),
                key=lambda index: (
                    model_graph["nodes"][index]["type"] == "stairs",
                    order_offsets.get(index, -1),
                ),
            )
            if model_graph["nodes"][movable]["type"] == "stairs":
                movable = target if movable == source else source
            old_box = predictions.get(movable)
            if old_box is None:
                continue
            floors = predicted_floors[movable]
            x0, y0, x1, y1 = old_box
            for floor in floors:
                occupied[floor - 1, x0:x1, y0:y1] = 0
            required_neighbors = [
                (neighbor, relation, True)
                for neighbor, relation, required in topology_neighbors[movable]
                if required and neighbor in predictions
            ]
            replacement, details = place_topology_box(
                raw_predictions[movable],
                site_cells,
                floors,
                building,
                occupied,
                predictions,
                predicted_floors,
                required_neighbors,
                needs_exterior=bool(
                    ROOM_RULES.get(
                        model_graph["nodes"][movable]["type"],
                        {},
                    ).get("needs_exterior")
                ),
            )
            if replacement is None:
                replacement = old_box
            predictions[movable] = replacement
            placement_details[
                model_graph["nodes"][movable]["instance_token"]
            ] = {**details, "local_repair": replacement != old_box}
            mark_occupied(occupied, replacement, floors)
            changed |= replacement != old_box
        if not changed:
            break
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
            quality = edge_quality(source, target, relation)
        realized_edges.append(
            {
                "source": model_graph["nodes"][source]["instance_token"],
                "target": model_graph["nodes"][target]["instance_token"],
                "relation": "vertical" if relation == 1 else "horizontal",
                "required": (source, target) in required_edges,
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
    model_graph, topology = request_graph(
        counts,
        site_x,
        site_y,
        args.seed,
        program_prior,
    )
    topology["count_evidence"] = count_evidence
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
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(args.output_dir / "coarse_volume.npz", volume=coarse)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
