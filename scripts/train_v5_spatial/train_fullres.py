"""Train the native 300 mm graph-voxel V5 model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fullres_dataset import (
    SEMANTIC_CLASSES,
    FullResolutionLayoutDataset,
    collate_fullres,
)
from fullres_model import FullResolutionGraphVoxelModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--spatial-channels", type=int, default=24)
    parser.add_argument("--query-channels", type=int, default=32)
    parser.add_argument("--architecture", choices=("v1", "v2"), default="v1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--max-train-houses", type=int)
    parser.add_argument("--max-val-houses", type=int)
    parser.add_argument(
        "--validation-split",
        choices=("train", "val"),
        default="val",
    )
    parser.add_argument(
        "--train-condition-mode",
        choices=("teacher", "robust", "program"),
        default="robust",
    )
    parser.add_argument(
        "--val-condition-mode",
        choices=("teacher", "robust", "program"),
        default="teacher",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instance-bce-weight", type=float, default=1.0)
    parser.add_argument("--instance-dice-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--building-weight", type=float, default=1.0)
    parser.add_argument("--outside-weight", type=float, default=2.0)
    parser.add_argument("--overlap-weight", type=float, default=1.0)
    parser.add_argument("--topology-weight", type=float, default=4.0)
    parser.add_argument("--existence-weight", type=float, default=2.0)
    parser.add_argument("--area-weight", type=float, default=2.0)
    parser.add_argument("--compactness-weight", type=float, default=0.5)
    parser.add_argument("--height-weight", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def assignment_probabilities(
    output: dict[str, torch.Tensor],
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    instance_logits = output["instance_logits"].masked_fill(
        node_mask[:, :, None, None, None] == 0,
        -20.0,
    )
    assignment_logits = torch.cat(
        (output["empty_logits"][:, None], instance_logits),
        dim=1,
    )
    probabilities = torch.softmax(assignment_logits.float(), dim=1)
    return assignment_logits, probabilities


def assignment_instance_loss(
    output: dict[str, torch.Tensor],
    targets: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assignment_logits, probabilities = assignment_probabilities(
        output,
        node_mask,
    )
    occupied = targets.any(dim=1)
    target_index = targets.argmax(dim=1) + 1
    target_index = torch.where(
        occupied,
        target_index,
        torch.zeros_like(target_index),
    )
    assignment = F.cross_entropy(
        assignment_logits,
        target_index,
    )
    instance_probabilities = probabilities[:, 1:]
    intersection = (instance_probabilities * targets).sum(dim=(2, 3, 4))
    denominator = instance_probabilities.sum(dim=(2, 3, 4)) + targets.sum(
        dim=(2, 3, 4)
    )
    dice_per_node = 1.0 - (2.0 * intersection + 1.0) / (
        denominator + 1.0
    )
    dice = (dice_per_node * node_mask).sum() / node_mask.sum().clamp_min(1.0)
    return assignment, dice


def semantic_logits_from_instances(
    instance_logits: torch.Tensor,
    nodes: torch.Tensor,
    node_mask: torch.Tensor,
    empty_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    if empty_logits is None:
        empty_logits = -torch.amax(instance_logits, dim=1)
    class_logits = [empty_logits]
    types = nodes[:, :, :11]
    for type_index in range(11):
        valid = (types[:, :, type_index] * node_mask).bool()
        selected = instance_logits.masked_fill(
            ~valid[:, :, None, None, None],
            -20.0,
        )
        class_logits.append(torch.logsumexp(selected, dim=1))
    return torch.stack(class_logits, dim=1)


def union_probability(
    output: dict[str, torch.Tensor],
    node_mask: torch.Tensor,
) -> torch.Tensor:
    _, probabilities = assignment_probabilities(output, node_mask)
    return 1.0 - probabilities[:, 0]


def topology_contact_loss(
    probabilities: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    confident = probabilities.float()
    batch, count, x, y, z = confident.shape
    values = confident.permute(0, 1, 4, 2, 3).reshape(
        batch * count,
        1,
        z,
        x,
        y,
    )
    horizontal_neighborhood = F.max_pool3d(
        values,
        kernel_size=(1, 3, 3),
        stride=1,
        padding=(0, 1, 1),
    ).reshape(batch, count, z, x, y).permute(0, 1, 3, 4, 2)
    vertical_neighborhood = F.max_pool3d(
        values,
        kernel_size=(3, 1, 1),
        stride=1,
        padding=(1, 0, 0),
    ).reshape(batch, count, z, x, y).permute(0, 1, 3, 4, 2)
    horizontal_ring = (horizontal_neighborhood - confident).clamp_min(0)
    vertical_ring = (vertical_neighborhood - confident).clamp_min(0)
    horizontal_score = torch.einsum(
        "bnxyz,bmxyz->bnm",
        confident,
        horizontal_ring,
    )
    vertical_score = torch.einsum(
        "bnxyz,bmxyz->bnm",
        confident,
        vertical_ring,
    )
    horizontal_cost = torch.exp(-horizontal_score / 10.0)
    vertical_cost = torch.exp(-vertical_score / 10.0)
    numerator = (
        horizontal_cost * adjacency[:, 0]
        + vertical_cost * adjacency[:, 1]
    ).sum()
    return numerator / adjacency.sum().clamp_min(1.0)


def instance_distribution_losses(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize missing, wrongly sized, fragmented and partial-height rooms."""
    valid = node_mask.float()
    predicted_volume = probabilities.sum(dim=(2, 3, 4))
    target_volume = targets.sum(dim=(2, 3, 4)).clamp_min(1.0)
    volume_ratio = predicted_volume / target_volume
    existence = (
        F.relu(0.25 - volume_ratio).square() * valid
    ).sum() / valid.sum().clamp_min(1.0)

    predicted_floor_area = []
    target_floor_area = []
    height_terms = []
    for floor in range(2):
        z0, z1 = floor * 10, (floor + 1) * 10
        predicted_layers = probabilities[:, :, :, :, z0:z1]
        target_layers = targets[:, :, :, :, z0:z1]
        predicted_footprint = predicted_layers.mean(dim=-1)
        target_footprint = target_layers.amax(dim=-1)
        predicted_floor_area.append(predicted_footprint.sum(dim=(2, 3)))
        target_floor_area.append(target_footprint.sum(dim=(2, 3)))
        layer_variance = predicted_layers.var(dim=-1, unbiased=False)
        height_terms.append(
            (layer_variance * predicted_footprint.detach()).sum(dim=(2, 3))
            / predicted_footprint.detach().sum(dim=(2, 3)).clamp_min(1.0)
        )
    predicted_area = torch.stack(predicted_floor_area, dim=-1)
    target_area = torch.stack(target_floor_area, dim=-1)
    floor_valid = (target_area > 0).float() * valid[:, :, None]
    area = (
        torch.abs(
            torch.log1p(predicted_area) - torch.log1p(target_area)
        )
        * floor_valid
    ).sum() / floor_valid.sum().clamp_min(1.0)
    height = (
        torch.stack(height_terms, dim=-1) * floor_valid
    ).sum() / floor_valid.sum().clamp_min(1.0)

    def boundary_density(values: torch.Tensor) -> torch.Tensor:
        boundary = (
            torch.abs(values[:, :, 1:] - values[:, :, :-1]).sum(
                dim=(2, 3, 4)
            )
            + torch.abs(values[:, :, :, 1:] - values[:, :, :, :-1]).sum(
                dim=(2, 3, 4)
            )
            + torch.abs(values[:, :, :, :, 1:] - values[:, :, :, :, :-1]).sum(
                dim=(2, 3, 4)
            )
        )
        mass = values.sum(dim=(2, 3, 4)).clamp_min(1.0)
        return boundary / mass

    predicted_boundary = boundary_density(probabilities)
    target_boundary = boundary_density(targets)
    compactness = (
        torch.abs(predicted_boundary - target_boundary) * valid
    ).sum() / valid.sum().clamp_min(1.0)
    return existence, area, compactness, height


def compute_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: argparse.Namespace | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["instance_logits"]
    assignment, dice = assignment_instance_loss(
        output,
        batch["instance_targets"],
        batch["node_mask"],
    )
    semantic_logits = semantic_logits_from_instances(
        logits,
        batch["nodes"],
        batch["node_mask"],
        output["empty_logits"],
    )
    semantic = F.cross_entropy(
        semantic_logits,
        batch["semantic_target"],
        ignore_index=255,
    )
    union = union_probability(output, batch["node_mask"])
    stable_union = union.float().clamp(1e-5, 1.0 - 1e-5)
    building_target = batch["building_target"].float()
    building = -(
        building_target * torch.log(stable_union)
        + (1.0 - building_target) * torch.log(1.0 - stable_union)
    ).mean()
    site = batch["volume"][:, 0]
    outside = (union * (1.0 - site)).mean()
    _, assignment_probs = assignment_probabilities(
        output,
        batch["node_mask"],
    )
    instance_probabilities = assignment_probs[:, 1:]
    overlap = torch.zeros((), device=logits.device)
    topology = topology_contact_loss(
        instance_probabilities,
        batch["adjacency"],
    )
    existence, area, compactness, height = instance_distribution_losses(
        instance_probabilities,
        batch["instance_targets"],
        batch["node_mask"],
    )
    weight = {
        "instance_bce": 1.0,
        "instance_dice": 1.0,
        "semantic": 1.0,
        "building": 1.0,
        "outside": 2.0,
        "overlap": 1.0,
        "topology": 4.0,
        "existence": 2.0,
        "area": 2.0,
        "compactness": 0.5,
        "height": 1.0,
    }
    if weights is not None:
        weight = {
            "instance_bce": weights.instance_bce_weight,
            "instance_dice": weights.instance_dice_weight,
            "semantic": weights.semantic_weight,
            "building": weights.building_weight,
            "outside": weights.outside_weight,
            "overlap": weights.overlap_weight,
            "topology": weights.topology_weight,
            "existence": weights.existence_weight,
            "area": weights.area_weight,
            "compactness": weights.compactness_weight,
            "height": weights.height_weight,
        }
    total = (
        weight["instance_bce"] * assignment
        + weight["instance_dice"] * dice
        + weight["semantic"] * semantic
        + weight["building"] * building
        + weight["outside"] * outside
        + weight["overlap"] * overlap
        + weight["topology"] * topology
        + weight["existence"] * existence
        + weight["area"] * area
        + weight["compactness"] * compactness
        + weight["height"] * height
    )
    return total, {
        "instance_bce": float(assignment.detach()),
        "instance_dice": float(dice.detach()),
        "semantic": float(semantic.detach()),
        "building": float(building.detach()),
        "outside": float(outside.detach()),
        "overlap": float(overlap.detach()),
        "topology": float(topology.detach()),
        "existence": float(existence.detach()),
        "area": float(area.detach()),
        "compactness": float(compactness.detach()),
        "height": float(height.detach()),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    accumulation_steps: int,
    loss_weights: argparse.Namespace | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
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
            loss, parts = compute_loss(output, batch, loss_weights)
        if training:
            scaled_loss = loss / accumulation_steps
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
        current_batch = int(batch["volume"].shape[0])
        sample_count += current_batch
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * current_batch
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * current_batch
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.smoke_test:
        args.epochs = 1
        args.max_train_houses = args.max_train_houses or 2
        args.max_val_houses = args.max_val_houses or 1
    device = torch.device(args.device)
    train_set = FullResolutionLayoutDataset(
        "train",
        condition_mode=args.train_condition_mode,
        max_houses=args.max_train_houses,
    )
    val_set = FullResolutionLayoutDataset(
        args.validation_split,
        condition_mode=args.val_condition_mode,
        max_houses=args.max_val_houses,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fullres,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fullres,
        pin_memory=device.type == "cuda",
    )
    model = FullResolutionGraphVoxelModel(
        spatial_channels=args.spatial_channels,
        query_channels=args.query_channels,
        architecture=args.architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda"
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["grid"] = [88, 88, 20]
    config["voxel_mm"] = 300
    config["semantic_classes"] = SEMANTIC_CLASSES
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.accumulation_steps,
            args,
            scaler,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                None,
                args.accumulation_steps,
                args,
                None,
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} "
            f"train={train_metrics['loss']:.5f} "
            f"val={val_metrics['loss']:.5f} "
            f"dice={val_metrics['instance_dice']:.5f} "
            f"topology={val_metrics['topology']:.5f}"
        )
        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation_loss": val_metrics["loss"],
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"cuda_peak_memory_mib={peak_mib:.1f}")


if __name__ == "__main__":
    main()
