#!/usr/bin/env python3
"""Train Phase9 graph-voxel stepwise action policy."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from stepwise_dataset import ACTION_TO_ID, StepwiseActionDataset, collate_stepwise
from stepwise_model import StepwiseActionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--max-train-houses", type=int)
    parser.add_argument("--max-val-houses", type=int)
    parser.add_argument("--validation-split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def balanced_sampler(dataset: StepwiseActionDataset) -> WeightedRandomSampler:
    targets = [int(target) for target in dataset.action_targets]
    counts = {
        name: sum(1 for target in targets if target == value)
        for name, value in ACTION_TO_ID.items()
    }
    id_counts = {
        ACTION_TO_ID[name]: max(count, 1)
        for name, count in counts.items()
    }
    median_count = float(np.median(list(id_counts.values())))
    weights = [
        min((median_count / id_counts[target]) ** 0.5, 4.0)
        for target in targets
    ]
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )


def compute_loss(output: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    action_weight = torch.ones(
        len(ACTION_TO_ID),
        device=output["action_logits"].device,
    )
    action_weight[ACTION_TO_ID["reject"]] = 1.0
    action_weight[ACTION_TO_ID["rollback"]] = 1.5
    action_weight[ACTION_TO_ID["reserve_empty"]] = 1.25
    action = F.cross_entropy(
        output["action_logits"],
        batch["action_target"],
        weight=action_weight,
    )
    accept_raw = F.binary_cross_entropy_with_logits(
        output["accept_logit"],
        batch["accepted_target"],
        reduction="none",
    )
    accept_weight = torch.where(batch["accepted_target"] > 0.5, 1.0, 1.5)
    accept = (accept_raw * accept_weight).mean()

    cut_mask = batch["action_target"] == ACTION_TO_ID["cut"]
    if cut_mask.any():
        axis = F.cross_entropy(
            output["axis_logits"][cut_mask],
            batch["axis_target"][cut_mask],
        )
        cut = F.smooth_l1_loss(output["cut"][cut_mask], batch["cut_target"][cut_mask])
    else:
        axis = output["axis_logits"].sum() * 0.0
        cut = output["cut"].sum() * 0.0

    box_mask = (
        (batch["action_target"] == ACTION_TO_ID["place"])
        | (batch["action_target"] == ACTION_TO_ID["reserve_empty"])
    )
    if box_mask.any():
        box = F.smooth_l1_loss(output["box"][box_mask], batch["box_target"][box_mask])
    else:
        box = output["box"].sum() * 0.0

    node_mask = batch["node_mask"].bool()
    node = F.binary_cross_entropy_with_logits(
        output["node_logits"][node_mask],
        batch["node_target"][node_mask],
    )
    total = action + 0.5 * accept + axis + 2.0 * cut + 2.0 * box + 0.5 * node
    return total, {
        "loss": float(total.detach()),
        "action": float(action.detach()),
        "accept": float(accept.detach()),
        "axis": float(axis.detach()),
        "cut": float(cut.detach()),
        "box": float(box.detach()),
        "node": float(node.detach()),
    }


def quiz_metrics(output: dict, batch: dict) -> dict:
    predicted = output["action_logits"].argmax(dim=-1)
    target = batch["action_target"]
    count = int(target.numel())
    reject_mask = target == ACTION_TO_ID["reject"]
    rollback_mask = target == ACTION_TO_ID["rollback"]
    cut_mask = target == ACTION_TO_ID["cut"]
    node_mask = batch["node_mask"].bool()
    node_pred = output["node_logits"].sigmoid() >= 0.5
    node_target = batch["node_target"] >= 0.5
    metrics = {
        "quiz_action_correct": float((predicted == target).sum()),
        "quiz_count": float(count),
        "quiz_node_correct": float(
            (node_pred[node_mask] == node_target[node_mask]).sum()
        ),
        "quiz_node_count": float(node_mask.sum()),
    }
    metrics["quiz_reject_correct"] = float(
        (predicted[reject_mask] == ACTION_TO_ID["reject"]).sum()
    )
    metrics["quiz_reject_count"] = float(reject_mask.sum())
    metrics["quiz_rollback_correct"] = float(
        (predicted[rollback_mask] == ACTION_TO_ID["rollback"]).sum()
    )
    metrics["quiz_rollback_count"] = float(rollback_mask.sum())
    axis_pred = output["axis_logits"][cut_mask].argmax(dim=-1)
    metrics["quiz_cut_axis_correct"] = float(
        (axis_pred == batch["axis_target"][cut_mask]).sum()
    )
    metrics["quiz_cut_axis_count"] = float(cut_mask.sum())
    accept_pred = output["accept_logit"].sigmoid() >= 0.5
    accept_target = batch["accepted_target"] >= 0.5
    metrics["quiz_accept_correct"] = float((accept_pred == accept_target).sum())
    return metrics


def finalize_metrics(totals: dict, samples: int) -> dict:
    metrics = {
        key: value / max(samples, 1)
        for key, value in totals.items()
        if not key.startswith("quiz_")
    }
    metrics["quiz_action_acc"] = totals.get("quiz_action_correct", 0.0) / max(
        totals.get("quiz_count", 0.0),
        1.0,
    )
    metrics["quiz_accept_acc"] = totals.get("quiz_accept_correct", 0.0) / max(
        totals.get("quiz_count", 0.0),
        1.0,
    )
    metrics["quiz_node_bit_acc"] = totals.get("quiz_node_correct", 0.0) / max(
        totals.get("quiz_node_count", 0.0),
        1.0,
    )
    metrics["quiz_reject_acc"] = totals.get("quiz_reject_correct", 0.0) / max(
        totals.get("quiz_reject_count", 0.0),
        1.0,
    )
    metrics["quiz_rollback_acc"] = totals.get("quiz_rollback_correct", 0.0) / max(
        totals.get("quiz_rollback_count", 0.0),
        1.0,
    )
    metrics["quiz_cut_axis_acc"] = totals.get("quiz_cut_axis_correct", 0.0) / max(
        totals.get("quiz_cut_axis_count", 0.0),
        1.0,
    )
    return metrics


def run_epoch(model, loader, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {}
    samples = 0
    for raw_batch in loader:
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
            loss, parts = compute_loss(output, batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        current = int(batch["volume"].shape[0])
        samples += current
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value) * current
        for key, value in quiz_metrics(output, batch).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return finalize_metrics(totals, samples)


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
        args.batch_size = min(args.batch_size, 2)
    device = torch.device(args.device)
    train_set = StepwiseActionDataset("train", max_houses=args.max_train_houses)
    val_set = StepwiseActionDataset(
        args.validation_split,
        max_houses=args.max_val_houses,
    )
    sampler = balanced_sampler(train_set) if args.balanced_sampling else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_stepwise,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_stepwise,
    )
    model = StepwiseActionPolicy(args.base_channels, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            validation = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train, "validation": validation})
        print(
            f"epoch={epoch:03d} train={train['loss']:.4f} "
            f"val={validation['loss']:.4f} "
            f"quiz_action={validation['quiz_action_acc']:.3f} "
            f"quiz_cut_axis={validation['quiz_cut_axis_acc']:.3f} "
            f"quiz_reject={validation['quiz_reject_acc']:.3f} "
            f"quiz_rollback={validation['quiz_rollback_acc']:.3f}"
        )
        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation_loss": validation["loss"],
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if validation["loss"] < best:
            best = validation["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
