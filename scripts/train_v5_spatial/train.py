#!/usr/bin/env python3
"""Train the topology-conditioned 3D block-cut policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import SpatialCutDataset, collate_cut_actions
from model import SpatialModalCutPolicy


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "v5_spatial_cut")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-train-houses", type=int)
    parser.add_argument("--max-val-houses", type=int)
    parser.add_argument("--overfit-houses", type=int)
    return parser.parse_args()


def move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(model, loader, device, optimizer, scaler) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {
        "loss": 0.0,
        "axis_accuracy": 0.0,
        "cut_mae": 0.0,
        "side_accuracy": 0.0,
        "partition_mae": 0.0,
    }
    count = 0
    for raw_batch in loader:
        batch = move(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                output = model(
                    batch["volume"],
                    batch["nodes"],
                    batch["active"],
                    batch["adjacency"],
                )
                axis_loss = F.cross_entropy(output["axis_logits"], batch["axis"])
                cut_mask = batch["axis"] != 3
                if cut_mask.any():
                    cut_loss = F.smooth_l1_loss(
                        output["cut_ratio"][cut_mask],
                        batch["cut_ratio"][cut_mask],
                    )
                    cut_mae = (
                        output["cut_ratio"][cut_mask] - batch["cut_ratio"][cut_mask]
                    ).abs().mean()
                else:
                    cut_loss = output["cut_ratio"].sum() * 0.0
                    cut_mae = cut_loss
                side_mask = batch["side_target"] >= 0
                if side_mask.any():
                    side_loss = F.cross_entropy(
                        output["side_logits"][side_mask],
                        batch["side_target"][side_mask],
                    )
                    side_accuracy = (
                        output["side_logits"][side_mask].argmax(dim=1)
                        == batch["side_target"][side_mask]
                    ).float().mean()
                else:
                    side_loss = output["side_logits"].sum() * 0.0
                    side_accuracy = side_loss
                if cut_mask.any():
                    partition_loss = F.smooth_l1_loss(
                        output["left_fraction"][cut_mask],
                        batch["left_fraction"][cut_mask],
                    )
                    partition_mae = (
                        output["left_fraction"][cut_mask]
                        - batch["left_fraction"][cut_mask]
                    ).abs().mean()
                else:
                    partition_loss = output["left_fraction"].sum() * 0.0
                    partition_mae = partition_loss
                loss = (
                    axis_loss
                    + 2.0 * cut_loss
                    + side_loss
                    + partition_loss
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_size = int(batch["axis"].shape[0])
        sums["loss"] += float(loss.detach()) * batch_size
        sums["axis_accuracy"] += float(
            (output["axis_logits"].argmax(dim=1) == batch["axis"]).float().mean()
        ) * batch_size
        sums["cut_mae"] += float(cut_mae.detach()) * batch_size
        sums["side_accuracy"] += float(side_accuracy.detach()) * batch_size
        sums["partition_mae"] += float(partition_mae.detach()) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_set = SpatialCutDataset(
        "train",
        max_houses=args.overfit_houses or args.max_train_houses,
    )
    val_set = (
        train_set
        if args.overfit_houses
        else SpatialCutDataset("val", max_houses=args.max_val_houses)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_cut_actions,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_cut_actions,
    )
    model = SpatialModalCutPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    best = float("inf")
    print(
        f"device={device} train_actions={len(train_set)} "
        f"val_actions={len(val_set)} batch={args.batch_size}"
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, None, scaler)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "schema": "graphspace_v5_spatial_cut_policy_v1",
            "epoch": epoch,
            "model": model.state_dict(),
            "config": vars(args),
            "val": val_metrics,
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"axis_acc={val_metrics['axis_accuracy']:.4f} "
            f"cut_mae={val_metrics['cut_mae']:.4f} "
            f"side_acc={val_metrics['side_accuracy']:.4f} "
            f"partition_mae={val_metrics['partition_mae']:.4f}"
        )


if __name__ == "__main__":
    main()
