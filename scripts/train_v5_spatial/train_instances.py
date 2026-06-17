#!/usr/bin/env python3
"""Train autoregressive room-instance box placement."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from instance_dataset import InstancePlacementDataset, collate_instances
from instance_model import InstancePlacementPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-train-houses", type=int)
    parser.add_argument("--max-val-houses", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--center-loss-weight", type=float, default=1.0)
    parser.add_argument("--size-loss-weight", type=float, default=2.0)
    parser.add_argument("--balanced-sampling", action="store_true")
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: InstancePlacementPolicy,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    center_loss_weight: float,
    size_loss_weight: float,
) -> dict:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    center_loss_sum = 0.0
    size_loss_sum = 0.0
    error_sum = torch.zeros(4, device=device)
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
                batch["room_index"],
                batch["step_ratio"],
            )
            coordinate_error = torch.nn.functional.smooth_l1_loss(
                output["box"],
                batch["target_box"],
                reduction="none",
            )
            center_loss = coordinate_error[:, :2].mean()
            size_loss = coordinate_error[:, 2:].mean()
            loss = (
                center_loss_weight * center_loss
                + size_loss_weight * size_loss
            )
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = int(batch["room_index"].shape[0])
        loss_sum += float(loss.detach()) * batch_size
        center_loss_sum += float(center_loss.detach()) * batch_size
        size_loss_sum += float(size_loss.detach()) * batch_size
        error_sum += (
            (output["box"] - batch["target_box"]).abs().sum(dim=0)
        )
        count += batch_size
    return {
        "loss": loss_sum / max(count, 1),
        "center_loss": center_loss_sum / max(count, 1),
        "size_loss": size_loss_sum / max(count, 1),
        "mae": (error_sum / max(count, 1)).detach().cpu().tolist(),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_dataset = InstancePlacementDataset(
        "train",
        max_houses=args.max_train_houses,
    )
    validation_dataset = InstancePlacementDataset(
        "val",
        max_houses=args.max_val_houses,
    )
    sampler = None
    if args.balanced_sampling:
        type_counts = np.bincount(
            np.asarray(train_dataset.item_type_ids, dtype=np.int64),
            minlength=11,
        )
        sample_weights = [
            1.0 / max(int(type_counts[type_id]), 1)
            for type_id in train_dataset.item_type_ids
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_instances,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_instances,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = InstancePlacementPolicy(args.base_channels).to(device)
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
    checkpoint_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loader,
            device,
            optimizer,
            scaler,
            args.center_loss_weight,
            args.size_loss_weight,
        )
        with torch.no_grad():
            validation = run_epoch(
                model,
                validation_loader,
                device,
                None,
                scaler,
                args.center_loss_weight,
                args.size_loss_weight,
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
        }
        history.append(record)
        checkpoint = {
            "model": model.state_dict(),
            "config": checkpoint_config,
            "epoch": epoch,
            "validation_loss": validation["loss"],
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if validation["loss"] < best:
            best = validation["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
                f"val={validation['loss']:.6f} "
                f"center={validation['center_loss']:.6f} "
                f"size={validation['size_loss']:.6f} "
                f"mae={validation['mae']}"
            )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
