#!/usr/bin/env python3
"""Evaluate a V5 checkpoint by decoding predicted room instances."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import DEFAULT_PHASE5_DIR, make_dataset  # noqa: E402
from decode_instances import decode_model_output, matched_instance_metrics  # noqa: E402
from model import V5MinimalNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def to_numpy_output(output: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        key: value[0].detach().float().cpu().numpy()
        for key, value in output.items()
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    base_channels = int(checkpoint["config"]["base_channels"])
    model = V5MinimalNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = make_dataset(args.split, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    reports = []
    with torch.no_grad():
        for batch in loader:
            condition = batch["condition"].to(device)
            site_mask = batch["site_mask"].to(device)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                output = model(condition, site_mask)
            decoded = decode_model_output(
                to_numpy_output(output),
                batch["site_mask"][0, 0].numpy(),
                batch["class_instance_counts"][0].numpy(),
            )
            metrics = matched_instance_metrics(
                decoded["floor_instances"],
                batch["instance_grid"][0].numpy()
                if "instance_grid" in batch
                else np.zeros_like(decoded["floor_instances"]),
                batch["class_grid"][0].numpy(),
                decoded["class_grid"],
            )
            metrics["house_id"] = batch["house_id"][0]
            metadata = json.loads(
                (
                    DEFAULT_PHASE5_DIR
                    / "{}.json".format(metrics["house_id"])
                ).read_text(encoding="utf-8")
            )
            metrics["building_instance_count"] = len(
                decoded["building_instances"]
            )
            metrics["expected_building_instance_count"] = int(
                metadata["building_instance_count"]
            )
            metrics["building_count_exact"] = (
                metrics["building_instance_count"]
                == metrics["expected_building_instance_count"]
            )
            metrics["predicted_cross_floor_count"] = sum(
                len(item["floors"]) == 2
                for item in decoded["building_instances"]
            )
            metrics["expected_cross_floor_count"] = int(
                metadata["cross_floor_instance_count"]
            )
            metrics["cross_floor_count_exact"] = (
                metrics["predicted_cross_floor_count"]
                == metrics["expected_cross_floor_count"]
            )
            reports.append(metrics)
    summary = {
        "schema": "graphspace_v5_prediction_decode_eval_v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sample_count": len(reports),
        "mean_count_error": float(
            np.mean([item["count_error"] for item in reports])
        ),
        "mean_matched_iou": float(
            np.mean([item["mean_matched_iou"] for item in reports])
        ),
        "mean_instance_recall_iou50": float(
            np.mean([item["instance_recall_iou50"] for item in reports])
        ),
        "floor_count_exact_rate": float(
            np.mean([item["count_error"] == 0 for item in reports])
        ),
        "building_count_exact_rate": float(
            np.mean([item["building_count_exact"] for item in reports])
        ),
        "cross_floor_count_exact_rate": float(
            np.mean([item["cross_floor_count_exact"] for item in reports])
        ),
        "reports": reports,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
