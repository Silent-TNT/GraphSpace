#!/usr/bin/env python3
"""Train the user-condition to functional-program topology model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from program_graph_dataset import ProgramGraphDataset, collate_program_graph
from program_graph_model import ProgramGraphModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-train-houses", type=int)
    parser.add_argument("--max-val-houses", type=int)
    parser.add_argument("--validation-split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_loss(output: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    floor = F.cross_entropy(
        output["floor_logits"].transpose(1, 2),
        batch["floor_target"],
        ignore_index=-1,
    )
    lighting = F.cross_entropy(
        output["lighting_logits"].transpose(1, 2),
        batch["lighting_target"],
        ignore_index=-1,
    )
    valid = batch["node_mask"].bool()
    area = F.smooth_l1_loss(output["area"][valid], batch["area_target"][valid])
    exterior = F.binary_cross_entropy_with_logits(
        output["exterior_logits"][valid],
        batch["exterior_target"][valid],
    )
    relation_target = batch["relation_target"]
    relation_valid = relation_target >= 0
    relation = F.cross_entropy(
        output["relation_logits"][relation_valid],
        relation_target[relation_valid],
        weight=torch.tensor(
            [0.12, 1.0, 1.5],
            device=output["relation_logits"].device,
        ),
    )
    total = floor + 8.0 * area + 0.5 * lighting + 0.5 * exterior + 2.0 * relation
    return total, {
        "loss": float(total.detach()),
        "floor": float(floor.detach()),
        "area": float(area.detach()),
        "lighting": float(lighting.detach()),
        "exterior": float(exterior.detach()),
        "relation": float(relation.detach()),
    }


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
            output = model(batch["node_input"], batch["node_mask"])
            loss, parts = compute_loss(output, batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        current = int(batch["node_input"].shape[0])
        samples += current
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * current
    return {key: value / max(samples, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.smoke_test:
        args.epochs = 1
        args.max_train_houses = args.max_train_houses or 4
        args.max_val_houses = args.max_val_houses or 2
    device = torch.device(args.device)
    train_set = ProgramGraphDataset("train", max_houses=args.max_train_houses)
    val_set = ProgramGraphDataset(
        args.validation_split,
        max_houses=args.max_val_houses,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_program_graph,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_program_graph,
    )
    model = ProgramGraphModel(args.hidden, args.layers).to(device)
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
            f"val={validation['loss']:.4f} floor={validation['floor']:.4f} "
            f"relation={validation['relation']:.4f}"
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
