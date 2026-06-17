#!/usr/bin/env python3
"""Train the minimal V5 supervised model locally or in Colab."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import DEFAULT_SPLIT_PATH, make_dataset  # noqa: E402
from losses import compute_losses, compute_metrics  # noqa: E402
from model import V5MinimalNet  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "v5_minimal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument(
        "--overfit-samples",
        type=int,
        help="Use the same first N training samples for train and validation.",
    )
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but this PyTorch build has no CUDA")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def run_epoch(
    model: V5MinimalNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    accumulate_steps: int,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    sample_count = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        batch_size = int(batch["condition"].shape[0])
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled
            ):
                output = model(batch["condition"], batch["site_mask"])
                losses = compute_losses(output, batch)
                scaled_loss = losses["total"] / accumulate_steps
            if training:
                scaler.scale(scaled_loss).backward()
                should_step = (
                    (step + 1) % accumulate_steps == 0
                    or step + 1 == len(loader)
                )
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        metrics = compute_metrics(output, batch)
        values = {
            **{key: float(value.detach().cpu()) for key, value in losses.items()},
            **metrics,
        }
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value * batch_size
        sample_count += batch_size
    return {key: value / max(sample_count, 1) for key, value in sums.items()}


def save_checkpoint(
    path: Path,
    model: V5MinimalNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "graphspace_v5_minimal_checkpoint_v1",
            "epoch": epoch,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 2)
        args.base_channels = min(args.base_channels, 8)
        args.max_train_samples = args.max_train_samples or 2
        args.max_val_samples = args.max_val_samples or 2
    if args.accumulate_steps < 1:
        raise ValueError("--accumulate-steps must be at least 1")
    set_seed(args.seed)
    device = choose_device(args.device)
    amp_enabled = device.type == "cuda"
    pin_memory = device.type == "cuda"
    train_set = make_dataset(
        "train",
        args.split_path,
        max_samples=args.overfit_samples or args.max_train_samples,
    )
    if args.overfit_samples:
        val_set = train_set
    else:
        val_set = make_dataset(
            "val", args.split_path, max_samples=args.max_val_samples
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model = V5MinimalNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    print(
        "device={} train={} val={} batch={} base_channels={} amp={}".format(
            device,
            len(train_set),
            len(val_set),
            args.batch_size,
            args.base_channels,
            amp_enabled,
        )
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            args.accumulate_steps,
            amp_enabled,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                None,
                scaler,
                1,
                amp_enabled,
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        save_checkpoint(
            args.output_dir / "latest.pt",
            model,
            optimizer,
            scaler,
            epoch,
            best_val,
            args,
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_val,
                args,
            )
        if epoch == start_epoch or epoch == args.epochs or epoch % args.log_every == 0:
            print(
                "epoch={:03d} train_loss={:.4f} val_loss={:.4f} "
                "val_class_acc={:.4f} val_count_mae={:.3f}".format(
                    epoch,
                    train_metrics["total"],
                    val_metrics["total"],
                    val_metrics["class_accuracy"],
                    val_metrics["count_mae"],
                )
            )


if __name__ == "__main__":
    main()
