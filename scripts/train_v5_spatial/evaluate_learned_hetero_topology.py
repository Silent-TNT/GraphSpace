#!/usr/bin/env python3
"""Evaluate and export learned heterogeneous topology graphs.

This evaluator measures the learned program/topology graph before geometry
generation. It intentionally keeps geometry out of the score, so it can become
the objective loop entry for deciding whether the topology generator itself is
good enough to feed the massing decoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = ROOT / "scripts" / "train_v5_spatial"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from program_graph_dataset import ProgramGraphDataset, collate_program_graph  # noqa: E402
from program_graph_model import ProgramGraphModel  # noqa: E402


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
TYPE_BY_ID = {index: room_type for index, room_type in enumerate(ROOM_TYPES)}
LIGHTING_BY_ID = {0: "none", 1: "indirect", 2: "direct"}
FLOOR_BY_CLASS = {0: [1], 1: [2], 2: [1, 2]}
EXTERIOR_SIDES = ["W", "E", "S", "N"]
RELATION_BY_CLASS = {1: "horizontal", 2: "vertical"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-houses", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Evaluate thresholds 0.05..0.95 and report the best mean edge F1.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def type_from_input(row: torch.Tensor) -> str:
    type_id = int(row[: len(ROOM_TYPES)].argmax().item())
    return TYPE_BY_ID[type_id]


def floor_target_to_floors(value: int) -> list[int]:
    return FLOOR_BY_CLASS.get(int(value), [])


def relation_edges_from_target(target: torch.Tensor) -> set[tuple[int, int, int]]:
    edges = set()
    count = int(target.shape[0])
    for left in range(count):
        for right in range(left + 1, count):
            relation = int(target[left, right].item())
            if relation > 0:
                edges.add((left, right, relation))
    return edges


def connected(node_count: int, edges: list[tuple[int, int, int]]) -> bool:
    if node_count <= 1:
        return True
    parent = list(range(node_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right, _relation in edges:
        union(left, right)
    return len({find(index) for index in range(node_count)}) == 1


def semantic_checks(nodes: list[dict[str, Any]], edges: list[tuple[int, int, int]]) -> dict[str, Any]:
    by_type: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        by_type.setdefault(str(node["type"]), []).append(index)
    adjacency = {index: set() for index in range(len(nodes))}
    for left, right, _relation in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    circulation = {"entryway", "living_room", "corridor", "stairs"}

    living_dining = any(
        right in adjacency[left]
        for left in by_type.get("living_room", [])
        for right in by_type.get("dining_room", [])
    )
    kitchen_dining = True
    if by_type.get("kitchen"):
        kitchen_dining = any(
            right in adjacency[left]
            for left in by_type.get("kitchen", [])
            for right in by_type.get("dining_room", [])
        )
    bedroom_access = all(
        any(nodes[neighbor]["type"] in circulation for neighbor in adjacency[bedroom])
        for bedroom in by_type.get("bedroom", [])
    )
    stairs_access = all(
        any(nodes[neighbor]["type"] in circulation - {"stairs"} for neighbor in adjacency[stairs])
        for stairs in by_type.get("stairs", [])
    )
    checks = {
        "has_dining_room": bool(by_type.get("dining_room")),
        "has_stairs": bool(by_type.get("stairs")),
        "living_dining_edge": living_dining,
        "kitchen_dining_edge": kitchen_dining,
        "bedroom_circulation_access": bedroom_access,
        "stairs_circulation_access": stairs_access,
    }
    checks["pass"] = all(checks.values())
    return checks


def predicted_graph_for_item(
    batch: dict[str, Any],
    output: dict[str, torch.Tensor],
    batch_index: int,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    house_id = str(batch["house_id"][batch_index])
    node_count = int(batch["node_mask"][batch_index].sum().item())
    node_input = batch["node_input"][batch_index, :node_count].detach().cpu()
    floor_pred = output["floor_logits"][batch_index, :node_count].argmax(dim=-1).detach().cpu()
    floor_target = batch["floor_target"][batch_index, :node_count].detach().cpu()
    area_pred = output["area"][batch_index, :node_count].detach().cpu()
    area_target = batch["area_target"][batch_index, :node_count].detach().cpu()
    lighting_pred = output["lighting_logits"][batch_index, :node_count].argmax(dim=-1).detach().cpu()
    lighting_target = batch["lighting_target"][batch_index, :node_count].detach().cpu()
    exterior_pred = output["exterior_logits"][batch_index, :node_count].sigmoid().detach().cpu()
    exterior_target = batch["exterior_target"][batch_index, :node_count].detach().cpu()
    relation_target = batch["relation_target"][batch_index, :node_count, :node_count].detach().cpu()
    relation_probs = output["relation_logits"][batch_index, :node_count, :node_count].softmax(dim=-1).detach().cpu()

    nodes = []
    for index in range(node_count):
        exterior_sides = [
            side
            for side_index, side in enumerate(EXTERIOR_SIDES)
            if float(exterior_pred[index, side_index]) >= 0.5
        ]
        nodes.append(
            {
                "id": f"{type_from_input(node_input[index])}_{index}",
                "node_type": "room_instance",
                "type": type_from_input(node_input[index]),
                "floors": floor_target_to_floors(int(floor_pred[index].item())),
                "area_ratio": float(area_pred[index].item()),
                "lighting_access": LIGHTING_BY_ID[int(lighting_pred[index].item())],
                "exterior_sides": exterior_sides,
            }
        )

    predicted_edges: list[tuple[int, int, int]] = []
    graph_edges = []
    edge_scores = []
    for left in range(node_count):
        for right in range(left + 1, node_count):
            score = float(1.0 - relation_probs[left, right, 0].item())
            relation_class = int(relation_probs[left, right, 1:].argmax().item() + 1)
            if score < threshold:
                continue
            predicted_edges.append((left, right, relation_class))
            edge_scores.append(score)
            graph_edges.append(
                {
                    "source": nodes[left]["id"],
                    "target": nodes[right]["id"],
                    "edge_type": "target_adjacent",
                    "relation": RELATION_BY_CLASS[relation_class],
                    "probability": score,
                }
            )

    target_edges = relation_edges_from_target(relation_target)
    predicted_pairs = {(left, right) for left, right, _relation in predicted_edges}
    target_pairs = {(left, right) for left, right, _relation in target_edges}
    true_positive = len(predicted_pairs & target_pairs)
    false_positive = len(predicted_pairs - target_pairs)
    false_negative = len(target_pairs - predicted_pairs)
    relation_correct = sum(
        1 for edge in predicted_edges if edge in target_edges
    )
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    semantic = semantic_checks(nodes, predicted_edges)
    graph_valid = connected(node_count, predicted_edges) and semantic["pass"]
    metrics = {
        "node_count": node_count,
        "type_counts": dict(Counter(node["type"] for node in nodes)),
        "floor_accuracy": float((floor_pred == floor_target).float().mean().item()),
        "area_ratio_mae": float(torch.abs(area_pred - area_target).mean().item()),
        "lighting_accuracy": float((lighting_pred == lighting_target).float().mean().item()),
        "exterior_binary_accuracy": float(
            ((exterior_pred >= 0.5) == exterior_target.bool()).float().mean().item()
        ),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": 2.0 * precision * recall / max(precision + recall, 1e-9),
        "relation_accuracy_on_matched_edges": relation_correct / max(true_positive, 1),
        "predicted_edge_count": len(predicted_edges),
        "target_edge_count": len(target_edges),
        "connected": connected(node_count, predicted_edges),
        "semantic_checks": semantic,
        "graph_valid": graph_valid,
        "mean_edge_probability": sum(edge_scores) / max(len(edge_scores), 1),
    }
    graph = {
        "schema": "graphspace_learned_heterogeneous_topology_v1",
        "house_id": house_id,
        "source": "program_graph_model",
        "edge_threshold": threshold,
        "nodes": [
            {"id": house_id, "node_type": "house"},
            {"id": "floor_1", "node_type": "floor", "floor": 1},
            {"id": "floor_2", "node_type": "floor", "floor": 2},
            *nodes,
        ],
        "edges": [
            {"source": house_id, "target": "floor_1", "edge_type": "contains"},
            {"source": house_id, "target": "floor_2", "edge_type": "contains"},
            *[
                {
                    "source": f"floor_{floor}",
                    "target": node["id"],
                    "edge_type": "contains",
                }
                for node in nodes
                for floor in node["floors"]
            ],
            *graph_edges,
        ],
        "metrics": metrics,
    }
    return graph, metrics


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ProgramGraphModel(
        hidden=int(config["hidden"]),
        layers=int(config["layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = ProgramGraphDataset(args.split, max_houses=args.max_houses)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_program_graph,
    )
    cached_predictions = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            output = model(batch["node_input"], batch["node_mask"])
            cached_predictions.append((raw_batch["house_id"], batch, output))

    def evaluate_at_threshold(threshold: float, export: bool) -> dict[str, Any]:
        per_house = {}
        for house_ids, batch, output in cached_predictions:
            for batch_index, house_id in enumerate(house_ids):
                graph, metrics = predicted_graph_for_item(batch, output, batch_index, threshold)
                if export:
                    write_json(args.output_dir / "graphs" / str(house_id) / "learned_topology.json", graph)
                per_house[str(house_id)] = metrics
        metrics = {
            "graph_valid_rate": mean([float(item["graph_valid"]) for item in per_house.values()]),
            "connected_rate": mean([float(item["connected"]) for item in per_house.values()]),
            "semantic_pass_rate": mean(
                [float(item["semantic_checks"]["pass"]) for item in per_house.values()]
            ),
            "floor_accuracy": mean([item["floor_accuracy"] for item in per_house.values()]),
            "area_ratio_mae": mean([item["area_ratio_mae"] for item in per_house.values()]),
            "lighting_accuracy": mean([item["lighting_accuracy"] for item in per_house.values()]),
            "exterior_binary_accuracy": mean(
                [item["exterior_binary_accuracy"] for item in per_house.values()]
            ),
            "edge_precision": mean([item["edge_precision"] for item in per_house.values()]),
            "edge_recall": mean([item["edge_recall"] for item in per_house.values()]),
            "edge_f1": mean([item["edge_f1"] for item in per_house.values()]),
            "relation_accuracy_on_matched_edges": mean(
                [item["relation_accuracy_on_matched_edges"] for item in per_house.values()]
            ),
            "predicted_edge_count_mean": mean(
                [float(item["predicted_edge_count"]) for item in per_house.values()]
            ),
            "target_edge_count_mean": mean(
                [float(item["target_edge_count"]) for item in per_house.values()]
            ),
        }
        return {"metrics": metrics, "per_house": per_house}

    threshold_curve = []
    if args.threshold_sweep:
        for threshold_int in range(5, 96, 5):
            threshold = threshold_int / 100.0
            result = evaluate_at_threshold(threshold, export=False)
            threshold_curve.append({"threshold": threshold, **result["metrics"]})
        selected_threshold = max(
            threshold_curve,
            key=lambda item: (
                item["edge_f1"],
                item["semantic_pass_rate"],
                -abs(item["predicted_edge_count_mean"] - item["target_edge_count_mean"]),
            ),
        )["threshold"]
    else:
        selected_threshold = float(args.edge_threshold)

    selected = evaluate_at_threshold(float(selected_threshold), export=True)
    per_house = selected["per_house"]

    summary = {
        "schema": "graphspace_learned_hetero_topology_evaluation_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "house_count": len(per_house),
        "edge_threshold": float(selected_threshold),
        "requested_edge_threshold": float(args.edge_threshold),
        "threshold_sweep_enabled": bool(args.threshold_sweep),
        "threshold_curve": threshold_curve,
        "metrics": selected["metrics"],
        "per_house": per_house,
        "outputs": {
            "graphs": str(args.output_dir / "graphs"),
            "summary": str(args.output_dir / "summary.json"),
        },
        "loop_gate": {
            "minimum_graph_valid_rate": 0.8,
            "minimum_edge_f1": 0.55,
            "minimum_semantic_pass_rate": 0.85,
            "note": (
                "Initial objective gate for learned topology only. It does not "
                "replace P0/P1/P2 geometry evaluation after massing generation."
            ),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
