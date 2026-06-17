#!/usr/bin/env python3
"""Evaluate learned room-program attributes and topology."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from program_graph_dataset import ProgramGraphDataset, collate_program_graph
from program_graph_model import ProgramGraphModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-houses", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
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
        batch_size=8,
        shuffle=False,
        collate_fn=collate_program_graph,
    )
    totals = {
        "nodes": 0,
        "floor_correct": 0,
        "area_abs_error": 0.0,
        "lighting_correct": 0,
        "exterior_correct": 0,
        "exterior_values": 0,
        "edge_true_positive": 0,
        "edge_false_positive": 0,
        "edge_false_negative": 0,
        "relation_correct_on_true_edges": 0,
        "true_edges": 0,
    }
    edge_scores = []
    edge_targets = []
    edge_relations = []
    edge_relation_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model(batch["node_input"], batch["node_mask"])
            valid_nodes = batch["node_mask"].bool()
            floor_pred = output["floor_logits"].argmax(dim=-1)
            lighting_pred = output["lighting_logits"].argmax(dim=-1)
            exterior_pred = output["exterior_logits"].sigmoid() >= 0.5
            totals["nodes"] += int(valid_nodes.sum())
            totals["floor_correct"] += int(
                ((floor_pred == batch["floor_target"]) & valid_nodes).sum()
            )
            totals["area_abs_error"] += float(
                torch.abs(output["area"] - batch["area_target"])[valid_nodes].sum()
            )
            totals["lighting_correct"] += int(
                ((lighting_pred == batch["lighting_target"]) & valid_nodes).sum()
            )
            exterior_valid = valid_nodes[:, :, None].expand_as(exterior_pred)
            totals["exterior_correct"] += int(
                (
                    (exterior_pred == batch["exterior_target"].bool())
                    & exterior_valid
                ).sum()
            )
            totals["exterior_values"] += int(exterior_valid.sum())

            relation_target = batch["relation_target"]
            pair_valid = relation_target >= 0
            upper = torch.triu(
                torch.ones_like(pair_valid, dtype=torch.bool),
                diagonal=1,
            )
            pair_valid &= upper
            relation_pred = output["relation_logits"].argmax(dim=-1)
            relation_probabilities = output["relation_logits"].softmax(dim=-1)
            edge_probability = 1.0 - relation_probabilities[..., 0]
            target_edge = relation_target > 0
            predicted_edge = relation_pred > 0
            totals["edge_true_positive"] += int(
                (target_edge & predicted_edge & pair_valid).sum()
            )
            totals["edge_false_positive"] += int(
                (~target_edge & predicted_edge & pair_valid).sum()
            )
            totals["edge_false_negative"] += int(
                (target_edge & ~predicted_edge & pair_valid).sum()
            )
            totals["relation_correct_on_true_edges"] += int(
                (
                    (relation_pred == relation_target)
                    & target_edge
                    & pair_valid
                ).sum()
            )
            totals["true_edges"] += int((target_edge & pair_valid).sum())
            edge_scores.extend(
                edge_probability[pair_valid].detach().cpu().tolist()
            )
            edge_targets.extend(
                target_edge[pair_valid].detach().cpu().tolist()
            )
            edge_relations.extend(
                (
                    relation_probabilities[..., 1:].argmax(dim=-1) + 1
                )[pair_valid]
                .detach()
                .cpu()
                .tolist()
            )
            edge_relation_targets.extend(
                relation_target[pair_valid].detach().cpu().tolist()
            )
    precision = totals["edge_true_positive"] / max(
        totals["edge_true_positive"] + totals["edge_false_positive"],
        1,
    )
    recall = totals["edge_true_positive"] / max(
        totals["edge_true_positive"] + totals["edge_false_negative"],
        1,
    )
    threshold_curve = []
    for threshold_int in range(5, 96, 5):
        threshold = threshold_int / 100.0
        true_positive = false_positive = false_negative = relation_correct = 0
        true_edge_count = 0
        for score, target, relation, relation_target in zip(
            edge_scores,
            edge_targets,
            edge_relations,
            edge_relation_targets,
        ):
            predicted = score >= threshold
            true_positive += int(predicted and target)
            false_positive += int(predicted and not target)
            false_negative += int(not predicted and target)
            true_edge_count += int(target)
            relation_correct += int(
                predicted and target and relation == relation_target
            )
        curve_precision = true_positive / max(
            true_positive + false_positive,
            1,
        )
        curve_recall = true_positive / max(
            true_positive + false_negative,
            1,
        )
        threshold_curve.append(
            {
                "threshold": threshold,
                "precision": curve_precision,
                "recall": curve_recall,
                "f1": 2.0
                * curve_precision
                * curve_recall
                / max(curve_precision + curve_recall, 1e-9),
                "relation_accuracy_on_true_edges": relation_correct
                / max(true_edge_count, 1),
            }
        )
    best_threshold = max(threshold_curve, key=lambda item: item["f1"])
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": args.split,
        "house_count": len(dataset),
        "metrics": {
            "floor_accuracy": totals["floor_correct"] / max(totals["nodes"], 1),
            "area_ratio_mae": totals["area_abs_error"] / max(totals["nodes"], 1),
            "lighting_accuracy": totals["lighting_correct"] / max(
                totals["nodes"],
                1,
            ),
            "exterior_binary_accuracy": totals["exterior_correct"] / max(
                totals["exterior_values"],
                1,
            ),
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": 2.0 * precision * recall / max(precision + recall, 1e-9),
            "relation_accuracy_on_true_edges": (
                totals["relation_correct_on_true_edges"]
                / max(totals["true_edges"], 1)
            ),
            "calibrated_edge_threshold": best_threshold["threshold"],
            "calibrated_edge_precision": best_threshold["precision"],
            "calibrated_edge_recall": best_threshold["recall"],
            "calibrated_edge_f1": best_threshold["f1"],
        },
        "threshold_curve": threshold_curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
