"""Evaluate a native 300 mm checkpoint on voxel and topology metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fullres_dataset import FullResolutionLayoutDataset, collate_fullres
from fullres_model import FullResolutionGraphVoxelModel
from train_fullres import (
    assignment_probabilities,
    move_batch,
    semantic_logits_from_instances,
    union_probability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-houses", type=int)
    parser.add_argument(
        "--condition-mode",
        choices=("teacher", "robust", "program"),
        default="teacher",
    )
    return parser.parse_args()


def binary_contact_matrix(masks: torch.Tensor, relation: int) -> torch.Tensor:
    count, x, y, z = masks.shape
    values = masks.permute(0, 3, 1, 2)[:, None].float()
    if relation == 0:
        dilated = F.max_pool3d(
            values,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )
    else:
        dilated = F.max_pool3d(
            values,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
        )
    dilated = dilated[:, 0].permute(0, 2, 3, 1).bool()
    ring = dilated & ~masks
    return torch.einsum(
        "nxyz,mxyz->nm",
        masks.float(),
        ring.float(),
    ) > 0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    model = FullResolutionGraphVoxelModel(
        spatial_channels=int(config["spatial_channels"]),
        query_channels=int(config["query_channels"]),
        architecture=str(config.get("architecture", "v1")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = FullResolutionLayoutDataset(
        args.split,
        condition_mode=args.condition_mode,
        max_houses=args.max_houses,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fullres,
    )
    totals = {
        "instance_iou_sum": 0.0,
        "instance_count": 0,
        "instance_recall_050": 0,
        "nonempty_instances": 0,
        "semantic_correct": 0,
        "semantic_cells": 0,
        "building_iou_sum": 0.0,
        "overlap_cells": 0,
        "site_cells": 0,
        "outside_cells": 0,
        "topology_realized": 0,
        "topology_edges": 0,
    }
    houses = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(
                batch["volume"],
                batch["nodes"],
                batch["node_mask"],
                batch["adjacency"],
            )
            _, probabilities = assignment_probabilities(
                output,
                batch["node_mask"],
            )
            assignments = probabilities.argmax(dim=1)
            count = int(batch["node_mask"][0].sum().item())
            masks = torch.stack(
                [assignments[0] == instance_id for instance_id in range(1, count + 1)]
            )
            targets = batch["instance_targets"][0, :count].bool()
            intersection = (masks & targets).sum(dim=(1, 2, 3)).float()
            union = (masks | targets).sum(dim=(1, 2, 3)).float()
            iou = intersection / union.clamp_min(1.0)
            nonempty = masks.sum(dim=(1, 2, 3)) > 0

            semantic_logits = semantic_logits_from_instances(
                output["instance_logits"],
                batch["nodes"],
                batch["node_mask"],
                output["empty_logits"],
            )
            semantic_pred = semantic_logits.argmax(dim=1)
            semantic_target = batch["semantic_target"]
            valid = semantic_target != 255
            semantic_correct = (
                (semantic_pred == semantic_target) & valid
            ).sum()

            building_pred = masks.any(dim=0)
            building_target = batch["building_target"][0].bool()
            building_intersection = (building_pred & building_target).sum()
            building_union = (building_pred | building_target).sum()
            building_iou = building_intersection.float() / (
                building_union.float().clamp_min(1.0)
            )
            site = batch["volume"][0, 0].bool()
            overlap = masks.sum(dim=0) > 1
            outside = building_pred & ~site

            realized = edges = 0
            for relation in range(2):
                predicted_contact = binary_contact_matrix(masks, relation)
                target_edges = batch["adjacency"][0, relation, :count, :count]
                upper = torch.triu(
                    torch.ones_like(target_edges, dtype=torch.bool),
                    diagonal=1,
                )
                required = target_edges.bool() & upper
                realized += int((predicted_contact & required).sum().item())
                edges += int(required.sum().item())

            totals["instance_iou_sum"] += float(iou.sum())
            totals["instance_count"] += count
            totals["instance_recall_050"] += int((iou >= 0.5).sum())
            totals["nonempty_instances"] += int(nonempty.sum())
            totals["semantic_correct"] += int(semantic_correct)
            totals["semantic_cells"] += int(valid.sum())
            totals["building_iou_sum"] += float(building_iou)
            totals["overlap_cells"] += int((overlap & site).sum())
            totals["site_cells"] += int(site.sum())
            totals["outside_cells"] += int(outside.sum())
            totals["topology_realized"] += realized
            totals["topology_edges"] += edges
            houses.append(
                {
                    "house_id": raw_batch["house_id"][0],
                    "mean_instance_iou": float(iou.mean()),
                    "instance_recall_050": float((iou >= 0.5).float().mean()),
                    "nonempty_rate": float(nonempty.float().mean()),
                    "building_iou": float(building_iou),
                    "topology_realization": realized / max(edges, 1),
                }
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": args.split,
        "house_count": len(dataset),
        "threshold": args.threshold,
        "metrics": {
            "mean_instance_iou": totals["instance_iou_sum"]
            / max(totals["instance_count"], 1),
            "instance_recall_050": totals["instance_recall_050"]
            / max(totals["instance_count"], 1),
            "nonempty_instance_rate": totals["nonempty_instances"]
            / max(totals["instance_count"], 1),
            "semantic_accuracy": totals["semantic_correct"]
            / max(totals["semantic_cells"], 1),
            "mean_building_iou": totals["building_iou_sum"]
            / max(len(dataset), 1),
            "overlap_cell_rate": totals["overlap_cells"]
            / max(totals["site_cells"], 1),
            "outside_cell_rate": totals["outside_cells"]
            / max(totals["site_cells"], 1),
            "topology_realization": totals["topology_realized"]
            / max(totals["topology_edges"], 1),
        },
        "houses": houses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
