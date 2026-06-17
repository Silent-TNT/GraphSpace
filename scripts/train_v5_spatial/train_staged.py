#!/usr/bin/env python3
"""Train the shared seven-stage graph and 3D voxel policy."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from staged_dataset import StagedSpatialDataset, collate_staged
from staged_model import StagedSpatialPolicy


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-train-houses", type=int)
    parser.add_argument("--max-val-houses", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260614)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(2, logits.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))


def compute_loss(output: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    logits = output["mask_logits"]
    target = batch["target_volume"]
    valid = batch["target_valid"]
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    ).mean(dim=(2, 3, 4))
    dice = dice_loss(logits, target)
    mask_loss = ((bce + dice) * valid).sum() / valid.sum().clamp_min(1.0)
    reach_mask = (batch["stage_id"] == 6).float()
    reach_bce = nn.functional.binary_cross_entropy_with_logits(
        output["reachability_logit"],
        batch["reachability"],
        reduction="none",
    )
    reach_loss = (reach_bce * reach_mask).sum() / reach_mask.sum().clamp_min(1.0)
    cut_mask = (batch["stage_id"] == 1).float()
    cut_error = (output["cut_ratio"] - batch["cut_ratio"]).abs()
    cut_loss = (cut_error * cut_mask).sum() / cut_mask.sum().clamp_min(1.0)
    total = mask_loss + 0.25 * reach_loss + cut_loss
    return total, {
        "mask_loss": float(mask_loss.detach()),
        "reach_loss": float(reach_loss.detach()),
        "cut_loss": float(cut_loss.detach()),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: StagedSpatialPolicy,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "mask_loss": 0.0,
        "reach_loss": 0.0,
        "cut_loss": 0.0,
    }
    count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
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
                batch["stage_id"],
            )
            loss, parts = compute_loss(output, batch)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = int(batch["stage_id"].shape[0])
        totals["loss"] += float(loss.detach()) * batch_size
        totals["mask_loss"] += parts["mask_loss"] * batch_size
        totals["reach_loss"] += parts["reach_loss"] * batch_size
        totals["cut_loss"] += parts["cut_loss"] * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    train_data = StagedSpatialDataset(
        "train",
        max_houses=args.max_train_houses,
    )
    val_data = StagedSpatialDataset(
        "val",
        max_houses=args.max_val_houses,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_staged,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_staged,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = StagedSpatialPolicy(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    checkpoint_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                None,
                scaler,
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        checkpoint = {
            "model": model.state_dict(),
            "config": checkpoint_config,
            "epoch": epoch,
            "validation_loss": val_metrics["loss"],
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} "
                f"train={train_metrics['loss']:.4f} "
                f"val={val_metrics['loss']:.4f}"
            )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
