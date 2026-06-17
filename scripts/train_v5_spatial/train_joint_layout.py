#!/usr/bin/env python3
"""Train whole-house boxes with differentiable spatial organization losses."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from joint_dataset import JointLayoutDataset, collate_joint
from joint_model import JointLayoutPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-train-houses", type=int)
    parser.add_argument("--max-val-houses", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--box-weight", type=float, default=1.0)
    parser.add_argument("--overlap-weight", type=float, default=0.5)
    parser.add_argument("--contact-weight", type=float, default=0.5)
    parser.add_argument("--coverage-weight", type=float, default=0.5)
    parser.add_argument("--shape-weight", type=float, default=1.0)
    parser.add_argument("--bounds-weight", type=float, default=0.5)
    return parser.parse_args()


def box_edges(boxes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cx, cy, width, depth = boxes.unbind(dim=-1)
    return (
        cx - width * 0.5,
        cy - depth * 0.5,
        cx + width * 0.5,
        cy + depth * 0.5,
    )


def pairwise_overlap_loss(
    boxes: torch.Tensor,
    node_mask: torch.Tensor,
    floors: torch.Tensor,
) -> torch.Tensor:
    x0, y0, x1, y1 = box_edges(boxes)
    ix = (
        torch.minimum(x1[:, :, None], x1[:, None, :])
        - torch.maximum(x0[:, :, None], x0[:, None, :])
    ).clamp_min(0)
    iy = (
        torch.minimum(y1[:, :, None], y1[:, None, :])
        - torch.maximum(y0[:, :, None], y0[:, None, :])
    ).clamp_min(0)
    same_floor = torch.bmm(floors, floors.transpose(1, 2)) > 0
    valid = node_mask[:, :, None] * node_mask[:, None, :]
    upper = torch.triu(torch.ones_like(valid), diagonal=1)
    mask = valid * same_floor.float() * upper
    return (ix * iy * mask).sum() / mask.sum().clamp_min(1.0)


def topology_contact_loss(
    boxes: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    x0, y0, x1, y1 = box_edges(boxes)
    overlap_x = (
        torch.minimum(x1[:, :, None], x1[:, None, :])
        - torch.maximum(x0[:, :, None], x0[:, None, :])
    )
    overlap_y = (
        torch.minimum(y1[:, :, None], y1[:, None, :])
        - torch.maximum(y0[:, :, None], y0[:, None, :])
    )
    gap_x = torch.minimum(
        (x0[:, :, None] - x1[:, None, :]).abs(),
        (x1[:, :, None] - x0[:, None, :]).abs(),
    )
    gap_y = torch.minimum(
        (y0[:, :, None] - y1[:, None, :]).abs(),
        (y1[:, :, None] - y0[:, None, :]).abs(),
    )
    horizontal_cost = torch.minimum(
        gap_x + F.relu(0.05 - overlap_y),
        gap_y + F.relu(0.05 - overlap_x),
    )
    projection_cost = F.relu(0.10 - overlap_x) + F.relu(0.10 - overlap_y)
    horizontal = adjacency[:, 0]
    vertical = adjacency[:, 1]
    numerator = (
        horizontal_cost * horizontal + projection_cost * vertical
    ).sum()
    return numerator / (horizontal.sum() + vertical.sum()).clamp_min(1.0)


def soft_coverage_loss(
    boxes: torch.Tensor,
    node_mask: torch.Tensor,
    floors: torch.Tensor,
    volume: torch.Tensor,
    nodes: torch.Tensor,
    resolution: int = 32,
) -> torch.Tensor:
    axis = (torch.arange(resolution, device=boxes.device) + 0.5) / resolution
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    x0, y0, x1, y1 = box_edges(boxes)
    sharpness = 80.0
    inside_x = torch.sigmoid(
        sharpness * (gx[None, None] - x0[:, :, None, None])
    ) * torch.sigmoid(
        sharpness * (x1[:, :, None, None] - gx[None, None])
    )
    inside_y = torch.sigmoid(
        sharpness * (gy[None, None] - y0[:, :, None, None])
    ) * torch.sigmoid(
        sharpness * (y1[:, :, None, None] - gy[None, None])
    )
    is_stairs = nodes[:, :, 7]
    ordinary_mask = node_mask * (1.0 - is_stairs)
    room_masks = inside_x * inside_y * ordinary_mask[:, :, None, None]
    losses = []
    for floor in range(2):
        floor_rooms = room_masks * floors[:, :, floor, None, None]
        union = 1.0 - torch.prod(1.0 - floor_rooms.clamp(0, 0.999), dim=1)
        z0, z1 = floor * 10, (floor + 1) * 10
        building = volume[:, 1, :, :, z0:z1].amax(dim=-1)
        stairs = volume[:, 2, :, :, z0:z1].amax(dim=-1)
        ordinary_building = building * (1.0 - stairs)
        ordinary_building = F.interpolate(
            ordinary_building[:, None],
            size=(resolution, resolution),
            mode="nearest",
        ).squeeze(1)
        losses.append(F.mse_loss(union, ordinary_building))
    return torch.stack(losses).mean()


def room_shape_loss(
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    node_mask: torch.Tensor,
    nodes: torch.Tensor,
) -> torch.Tensor:
    width = boxes[:, :, 2].clamp_min(1e-4)
    depth = boxes[:, :, 3].clamp_min(1e-4)
    target_width = target_boxes[:, :, 2].clamp_min(1e-4)
    target_depth = target_boxes[:, :, 3].clamp_min(1e-4)
    area_error = F.smooth_l1_loss(
        torch.log(width * depth),
        torch.log(target_width * target_depth),
        reduction="none",
    )
    short_error = F.smooth_l1_loss(
        torch.log(torch.minimum(width, depth)),
        torch.log(torch.minimum(target_width, target_depth)),
        reduction="none",
    )
    aspect_error = F.smooth_l1_loss(
        torch.log(torch.maximum(width, depth) / torch.minimum(width, depth)),
        torch.log(
            torch.maximum(target_width, target_depth)
            / torch.minimum(target_width, target_depth)
        ),
        reduction="none",
    )
    error = area_error + short_error + 0.5 * aspect_error
    return balanced_type_mean(error, node_mask, nodes)


def balanced_type_mean(
    node_error: torch.Tensor,
    node_mask: torch.Tensor,
    nodes: torch.Tensor,
) -> torch.Tensor:
    type_one_hot = nodes[:, :, :11]
    weighted = type_one_hot * node_mask[:, :, None]
    class_counts = weighted.sum(dim=(0, 1))
    class_errors = (weighted * node_error[:, :, None]).sum(dim=(0, 1))
    present = class_counts > 0
    return (class_errors[present] / class_counts[present]).mean()


def bounds_loss(
    boxes: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    x0, y0, x1, y1 = box_edges(boxes)
    error = F.relu(-x0) + F.relu(-y0) + F.relu(x1 - 1.0) + F.relu(y1 - 1.0)
    return (error * node_mask).sum() / node_mask.sum().clamp_min(1.0)


def compute_loss(output: dict, batch: dict, args: argparse.Namespace) -> tuple:
    boxes = output["boxes"]
    mask = batch["node_mask"][:, :, None]
    box_error = F.smooth_l1_loss(
        boxes,
        batch["target_boxes"],
        reduction="none",
    ).mean(dim=-1)
    box_loss = balanced_type_mean(
        box_error,
        batch["node_mask"],
        batch["nodes"],
    )
    overlap = pairwise_overlap_loss(boxes, batch["node_mask"], batch["floors"])
    contact = topology_contact_loss(boxes, batch["adjacency"])
    coverage = soft_coverage_loss(
        boxes,
        batch["node_mask"],
        batch["floors"],
        batch["volume"],
        batch["nodes"],
    )
    shape = room_shape_loss(
        boxes,
        batch["target_boxes"],
        batch["node_mask"],
        batch["nodes"],
    )
    bounds = bounds_loss(boxes, batch["node_mask"])
    total = (
        args.box_weight * box_loss
        + args.overlap_weight * overlap
        + args.contact_weight * contact
        + args.coverage_weight * coverage
        + args.shape_weight * shape
        + args.bounds_weight * bounds
    )
    return total, {
        "box": float(box_loss.detach()),
        "overlap": float(overlap.detach()),
        "contact": float(contact.detach()),
        "coverage": float(coverage.detach()),
        "shape": float(shape.detach()),
        "bounds": float(bounds.detach()),
    }


def run_epoch(model, loader, device, optimizer, scaler, args) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "box": 0.0,
        "overlap": 0.0,
        "contact": 0.0,
        "coverage": 0.0,
        "shape": 0.0,
        "bounds": 0.0,
    }
    count = 0
    for raw in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw.items()
        }
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["volume"],
                batch["nodes"],
                batch["node_mask"],
                batch["adjacency"],
            )
            loss, parts = compute_loss(output, batch, args)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = int(batch["nodes"].shape[0])
        totals["loss"] += float(loss.detach()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train = JointLayoutDataset("train", max_houses=args.max_train_houses)
    val = JointLayoutDataset("val", max_houses=args.max_val_houses)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_joint,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_joint,
        num_workers=args.num_workers,
    )
    model = JointLayoutPolicy(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, args)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, None, scaler, args)
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "validation_loss": val_metrics["loss"]}
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train={train_metrics} val={val_metrics}")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
