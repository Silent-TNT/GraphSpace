#!/usr/bin/env python3
"""Train a graph-level coarse layout model for functional groups.

Unlike v6_coarse_layout_head.py, this model consumes a whole functional topology
graph at once. Message passing lets each node see its target neighbors before
predicting the coarse group bbox used by the Phase24 user bridge.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spatial_modal_infer.config import ROOM_TYPES  # noqa: E402
from scripts.train_v5_spatial.v6_coarse_layout_head import (  # noqa: E402
    bbox_iou,
    coarse_layout_feature_from_fields,
    coarse_layout_target,
    CoarseLayoutSample,
)
from scripts.train_v5_spatial.v6_multipart_decoder import (  # noqa: E402
    VOXEL_MM,
    build_target_topology,
    group_bbox,
    read_json,
    room_floors,
    write_json,
)
from scripts.train_v5_spatial.v6_size_area_head import (  # noqa: E402
    DEFAULT_PHASE10,
    rooms_by_group,
)
from scripts.train_v5_spatial.v6_topology_learner import (  # noqa: E402
    POSITION_PRIOR_DEFAULT,
    SIZE_PRIOR_DEFAULT,
    read_position_priors,
    read_size_priors,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "v6_graph_coarse_layout_model_smoke"


@dataclass(frozen=True)
class GraphLayoutSample:
    house_id: str
    site: tuple[float, float]
    group_ids: list[str]
    room_types: list[str]
    floors: list[tuple[int, ...]]
    features: torch.Tensor
    targets: torch.Tensor
    adjacency: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--message-passing-steps", type=int, default=4)
    parser.add_argument("--edge-loss-weight", type=float, default=0.02)
    parser.add_argument("--contact-loss-weight", type=float, default=0.03)
    parser.add_argument("--repulsion-loss-weight", type=float, default=0.05)
    parser.add_argument("--occupancy-loss-weight", type=float, default=0.01)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.02)
    parser.add_argument("--size-conditioning-dir", type=Path)
    parser.add_argument("--position-conditioning-dir", type=Path)
    return parser.parse_args()


def normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    n = adjacency.shape[0]
    adj = adjacency + torch.eye(n, dtype=adjacency.dtype, device=adjacency.device)
    degree = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / degree


def graph_feature_from_fields(
    room_type: str,
    floors: tuple[int, ...] | list[int],
    site: tuple[float, float],
    type_index: int,
    type_count: int,
    group_count: int,
    size_prior: tuple[float, float, float, float] = SIZE_PRIOR_DEFAULT,
    position_prior: tuple[float, float, float, float] = POSITION_PRIOR_DEFAULT,
) -> torch.Tensor:
    return coarse_layout_feature_from_fields(
        room_type,
        floors,
        site,
        type_index,
        type_count,
        group_count,
        size_prior,
        position_prior,
    )


def load_graph_layout_samples(
    phase10_dir: Path,
    max_houses: int | None = None,
    size_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None = None,
    position_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None = None,
) -> list[GraphLayoutSample]:
    paths = sorted(Path(phase10_dir).glob("house_*.json"))
    if max_houses is not None:
        paths = paths[:max_houses]
    samples: list[GraphLayoutSample] = []
    for path in paths:
        source = read_json(path)
        house_id = str(source["house_id"])
        site = source["metadata"]["building_size"]
        site_xy = (float(site["x"]), float(site["y"]))
        grouped = rooms_by_group(source)
        groups = [group for group in source.get("functional_groups", []) if grouped.get(str(group["functional_id"]))]
        groups = sorted(groups, key=lambda group: (str(group["type"]), str(group["functional_id"])))
        type_counts: dict[str, int] = {}
        for group in groups:
            room_type = str(group["type"])
            type_counts[room_type] = type_counts.get(room_type, 0) + 1
        seen_by_type: dict[str, int] = {}
        house_size_priors = (size_priors or {}).get(house_id, {})
        house_position_priors = (position_priors or {}).get(house_id, {})
        group_ids: list[str] = []
        room_types: list[str] = []
        floors_by_group: list[tuple[int, ...]] = []
        features: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for group in groups:
            group_id = str(group["functional_id"])
            room_type = str(group["type"])
            parts = grouped[group_id]
            type_index = seen_by_type.get(room_type, 0)
            seen_by_type[room_type] = type_index + 1
            floors = tuple(sorted({floor for part in parts for floor in room_floors(part)}))
            box_min, box_max = group_bbox(parts)
            group_ids.append(group_id)
            room_types.append(room_type)
            floors_by_group.append(floors)
            features.append(
                graph_feature_from_fields(
                    room_type,
                    floors,
                    site_xy,
                    type_index,
                    type_counts[room_type],
                    len(groups),
                    house_size_priors.get(group_id, SIZE_PRIOR_DEFAULT),
                    house_position_priors.get(group_id, POSITION_PRIOR_DEFAULT),
                )
            )
            targets.append(
                coarse_layout_target(
                    CoarseLayoutSample(
                        house_id=house_id,
                        group_id=group_id,
                        room_type=room_type,
                        site=site_xy,
                        floors=floors,
                        type_index=type_index,
                        type_count=type_counts[room_type],
                        group_count=len(groups),
                        box_min=tuple(float(value) for value in box_min),
                        box_max=tuple(float(value) for value in box_max),
                    )
                )
            )
        if not group_ids:
            continue
        index_by_id = {group_id: index for index, group_id in enumerate(group_ids)}
        adjacency = torch.zeros((len(group_ids), len(group_ids)), dtype=torch.float32)
        topology = build_target_topology(source)
        for edge in topology.get("edges", []):
            source_id = str(edge["source"])
            target_id = str(edge["target"])
            if source_id in index_by_id and target_id in index_by_id:
                left = index_by_id[source_id]
                right = index_by_id[target_id]
                adjacency[left, right] = 1.0
                adjacency[right, left] = 1.0
        samples.append(
            GraphLayoutSample(
                house_id=house_id,
                site=site_xy,
                group_ids=group_ids,
                room_types=room_types,
                floors=floors_by_group,
                features=torch.stack(features),
                targets=torch.stack(targets),
                adjacency=adjacency,
            )
        )
    return samples


class GraphCoarseLayoutModel(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 160, steps: int = 4) -> None:
        super().__init__()
        self.steps = steps
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 4),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.encoder(features)
        adj = normalized_adjacency(adjacency).to(features.device)
        for _step in range(self.steps):
            message = adj @ h
            h = h + self.update(torch.cat([h, message], dim=-1))
        global_h = h.mean(dim=0, keepdim=True).expand_as(h)
        return self.head(torch.cat([h, global_h], dim=-1))


def edge_gap_loss(pred: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    losses = []
    for left in range(pred.shape[0]):
        for right in range(left + 1, pred.shape[0]):
            if float(adjacency[left, right]) <= 0.0:
                continue
            cx0, cy0, w0, d0 = pred[left]
            cx1, cy1, w1, d1 = pred[right]
            gap_x = torch.relu(torch.abs(cx0 - cx1) - (w0 + w1) * 0.5)
            gap_y = torch.relu(torch.abs(cy0 - cy1) - (d0 + d1) * 0.5)
            losses.append(torch.minimum(gap_x, gap_y))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def contact_loss(pred: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Encourage target-neighbor boxes to share a real side contact.

    A target edge is best realized when one axis is nearly touching and the
    other axis overlaps. This is stricter than edge_gap_loss(), which is already
    zero when two boxes overlap.
    """
    losses = []
    for left in range(pred.shape[0]):
        for right in range(left + 1, pred.shape[0]):
            if float(adjacency[left, right]) <= 0.0:
                continue
            cx0, cy0, w0, d0 = pred[left]
            cx1, cy1, w1, d1 = pred[right]
            sep_x = torch.abs(cx0 - cx1) - (w0 + w1) * 0.5
            sep_y = torch.abs(cy0 - cy1) - (d0 + d1) * 0.5
            x_side_contact = torch.abs(sep_x) + torch.relu(sep_y)
            y_side_contact = torch.abs(sep_y) + torch.relu(sep_x)
            losses.append(torch.minimum(x_side_contact, y_side_contact))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def repulsion_loss(pred: torch.Tensor, floors: list[tuple[int, ...]]) -> torch.Tensor:
    losses = []
    for left in range(pred.shape[0]):
        for right in range(left + 1, pred.shape[0]):
            if not set(floors[left]) & set(floors[right]):
                continue
            cx0, cy0, w0, d0 = pred[left]
            cx1, cy1, w1, d1 = pred[right]
            overlap_x = torch.relu((w0 + w1) * 0.5 - torch.abs(cx0 - cx1))
            overlap_y = torch.relu((d0 + d1) * 0.5 - torch.abs(cy0 - cy1))
            losses.append(overlap_x * overlap_y)
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def floor_occupancy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    floors: list[tuple[int, ...]],
) -> torch.Tensor:
    losses = []
    all_floors = sorted({floor for group_floors in floors for floor in group_floors})
    for floor in all_floors:
        indices = [index for index, group_floors in enumerate(floors) if floor in group_floors]
        if not indices:
            continue
        pred_area = (pred[indices, 2] * pred[indices, 3]).sum()
        target_area = (target[indices, 2] * target[indices, 3]).sum().clamp(max=0.95)
        losses.append(torch.abs(pred_area - target_area))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def boundary_loss(pred: torch.Tensor) -> torch.Tensor:
    cx, cy, width, depth = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    left = cx - width * 0.5
    right = cx + width * 0.5
    bottom = cy - depth * 0.5
    top = cy + depth * 0.5
    violations = torch.stack(
        [
            torch.relu(-left),
            torch.relu(right - 1.0),
            torch.relu(-bottom),
            torch.relu(top - 1.0),
        ],
        dim=1,
    )
    return violations.mean()


def evaluate_predictions(model: GraphCoarseLayoutModel, samples: list[GraphLayoutSample], device: torch.device) -> dict[str, Any]:
    model.eval()
    errors = []
    ious = []
    edge_losses = []
    contact_losses = []
    repulsion_losses = []
    occupancy_losses = []
    boundary_losses = []
    with torch.no_grad():
        for sample in samples:
            pred = model(sample.features.to(device), sample.adjacency.to(device)).cpu()
            target = sample.targets
            errors.append(torch.abs(pred - target))
            edge_losses.append(float(edge_gap_loss(pred, sample.adjacency)))
            contact_losses.append(float(contact_loss(pred, sample.adjacency)))
            repulsion_losses.append(float(repulsion_loss(pred, sample.floors)))
            occupancy_losses.append(float(floor_occupancy_loss(pred, target, sample.floors)))
            boundary_losses.append(float(boundary_loss(pred)))
            for index in range(pred.shape[0]):
                ious.append(bbox_iou(pred[index], target[index]))
    stacked = torch.cat(errors, dim=0)
    return {
        "house_count": len(samples),
        "group_count": int(stacked.shape[0]),
        "center_x_mae": float(stacked[:, 0].mean()),
        "center_y_mae": float(stacked[:, 1].mean()),
        "center_l1_mae": float(stacked[:, :2].sum(dim=1).mean()),
        "width_ratio_mae": float(stacked[:, 2].mean()),
        "depth_ratio_mae": float(stacked[:, 3].mean()),
        "bbox_iou_mean": float(sum(ious) / max(len(ious), 1)),
        "edge_gap_loss": float(sum(edge_losses) / max(len(edge_losses), 1)),
        "contact_loss": float(sum(contact_losses) / max(len(contact_losses), 1)),
        "repulsion_loss": float(sum(repulsion_losses) / max(len(repulsion_losses), 1)),
        "occupancy_loss": float(sum(occupancy_losses) / max(len(occupancy_losses), 1)),
        "boundary_loss": float(sum(boundary_losses) / max(len(boundary_losses), 1)),
    }


def predicted_payload(model: GraphCoarseLayoutModel, samples: list[GraphLayoutSample], device: torch.device) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for sample in samples:
            pred = model(sample.features.to(device), sample.adjacency.to(device)).cpu()
            groups = []
            for index, group_id in enumerate(sample.group_ids):
                target = sample.targets[index]
                groups.append(
                    {
                        "functional_id": group_id,
                        "type": sample.room_types[index],
                        "floors": list(sample.floors[index]),
                        "predicted": {
                            "center_x_ratio": float(pred[index, 0]),
                            "center_y_ratio": float(pred[index, 1]),
                            "width_ratio": float(pred[index, 2]),
                            "depth_ratio": float(pred[index, 3]),
                        },
                        "target": {
                            "center_x_ratio": float(target[0]),
                            "center_y_ratio": float(target[1]),
                            "width_ratio": float(target[2]),
                            "depth_ratio": float(target[3]),
                        },
                    }
                )
            payloads[sample.house_id] = {
                "schema": "graphspace_v6_graph_coarse_layout_predictions_v1",
                "source": "learned_graph_coarse_layout_model",
                "house_id": sample.house_id,
                "groups": groups,
            }
    return payloads


def load_graph_coarse_layout_model(checkpoint_path: Path, device: torch.device) -> GraphCoarseLayoutModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = GraphCoarseLayoutModel(
        int(config["feature_dim"]),
        hidden=int(config.get("hidden", 160)),
        steps=int(config.get("message_passing_steps", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict_graph_layout_ratios(
    model: GraphCoarseLayoutModel,
    device: torch.device,
    node_fields: list[dict[str, Any]],
    edges: list[tuple[str, str]],
) -> dict[str, tuple[float, float, float, float]]:
    features = torch.stack(
        [
            graph_feature_from_fields(
                str(node["room_type"]),
                tuple(int(value) for value in node["floors"]),
                tuple(float(value) for value in node["site"]),
                int(node["type_index"]),
                int(node["type_count"]),
                int(node["group_count"]),
                tuple(float(value) for value in node.get("size_prior", SIZE_PRIOR_DEFAULT)),
                tuple(float(value) for value in node.get("position_prior", POSITION_PRIOR_DEFAULT)),
            )
            for node in node_fields
        ]
    )
    index_by_id = {str(node["node_id"]): index for index, node in enumerate(node_fields)}
    adjacency = torch.zeros((len(node_fields), len(node_fields)), dtype=torch.float32)
    for source, target in edges:
        if source in index_by_id and target in index_by_id:
            left = index_by_id[source]
            right = index_by_id[target]
            adjacency[left, right] = 1.0
            adjacency[right, left] = 1.0
    with torch.no_grad():
        pred = model(features.to(device), adjacency.to(device)).cpu()
    return {
        str(node["node_id"]): tuple(float(value) for value in pred[index])
        for index, node in enumerate(node_fields)
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    samples = load_graph_layout_samples(
        args.phase10_dir,
        args.max_houses,
        read_size_priors(args.size_conditioning_dir),
        read_position_priors(args.position_conditioning_dir),
    )
    if not samples:
        raise ValueError("no graph layout samples found")
    feature_dim = int(samples[0].features.shape[1])
    model = GraphCoarseLayoutModel(feature_dim, hidden=args.hidden, steps=args.message_passing_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    history = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        model.train()
        total_loss = 0.0
        for sample in samples:
            features = sample.features.to(device)
            target = sample.targets.to(device)
            adjacency = sample.adjacency.to(device)
            pred = model(features, adjacency)
            loss = criterion(pred, target)
            loss = loss + float(args.edge_loss_weight) * edge_gap_loss(pred, adjacency)
            loss = loss + float(args.contact_loss_weight) * contact_loss(pred, adjacency)
            loss = loss + float(args.repulsion_loss_weight) * repulsion_loss(pred, sample.floors)
            loss = loss + float(args.occupancy_loss_weight) * floor_occupancy_loss(pred, target, sample.floors)
            loss = loss + float(args.boundary_loss_weight) * boundary_loss(pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            metrics = evaluate_predictions(model, samples, device)
            metrics["epoch"] = epoch
            metrics["loss"] = total_loss / max(len(samples), 1)
            history.append(metrics)
            print(
                f"epoch={epoch:04d} loss={metrics['loss']:.6f} "
                f"center_l1={metrics['center_l1_mae']:.4f} "
                f"bbox_iou={metrics['bbox_iou_mean']:.3f} edge_gap={metrics['edge_gap_loss']:.4f}"
            )
    final_metrics = evaluate_predictions(model, samples, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "graphspace_v6_graph_coarse_layout_model_v1",
            "model": model.state_dict(),
            "config": {
                "feature_dim": feature_dim,
                "hidden": int(args.hidden),
                "message_passing_steps": int(args.message_passing_steps),
                "room_types": ROOM_TYPES,
                "targets": ["center_x_ratio", "center_y_ratio", "width_ratio", "depth_ratio"],
                "loss_weights": {
                    "edge_gap": float(args.edge_loss_weight),
                    "contact": float(args.contact_loss_weight),
                    "repulsion": float(args.repulsion_loss_weight),
                    "occupancy": float(args.occupancy_loss_weight),
                    "boundary": float(args.boundary_loss_weight),
                },
            },
            "source_phase10_dir": str(args.phase10_dir),
        },
        args.output_dir / "graph_coarse_layout_model.pt",
    )
    for house_id, payload in predicted_payload(model, samples, device).items():
        write_json(args.output_dir / "graph_coarse_predictions" / house_id / "predicted_graph_coarse_layout.json", payload)
    summary = {
        "schema": "graphspace_v6_graph_coarse_layout_model_summary_v1",
        "purpose": "Whole-graph learned coarse layout model for the Phase24 user bridge.",
        "phase10_dir": str(args.phase10_dir),
        "house_count": len(samples),
        "group_count": sum(len(sample.group_ids) for sample in samples),
        "epochs": int(args.epochs),
        "size_conditioning_dir": str(args.size_conditioning_dir) if args.size_conditioning_dir else None,
        "position_conditioning_dir": str(args.position_conditioning_dir) if args.position_conditioning_dir else None,
        "loss_weights": {
            "edge_gap": float(args.edge_loss_weight),
            "contact": float(args.contact_loss_weight),
            "repulsion": float(args.repulsion_loss_weight),
            "occupancy": float(args.occupancy_loss_weight),
            "boundary": float(args.boundary_loss_weight),
        },
        "final_metrics": final_metrics,
        "history": history,
        "outputs": {
            "checkpoint": str(args.output_dir / "graph_coarse_layout_model.pt"),
            "predictions": str(args.output_dir / "graph_coarse_predictions"),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
