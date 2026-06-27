#!/usr/bin/env python3
"""Smoke/overfit trainer for functional-group size priors.

This is not a formal V6 generator. It validates a small prediction head for
program-level size information: functional group type/floor/site context ->
rough area, width, depth, and part-count priors. These priors can condition
topology prediction without exposing target group locations.
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
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spatial_modal_infer.config import ROOM_TYPES  # noqa: E402
from scripts.train_v5_spatial.v6_multipart_decoder import (  # noqa: E402
    group_bbox,
    read_json,
    room_floors,
    write_json,
)


DEFAULT_PHASE10 = ROOT / "data" / "phase10_functional_parts" / "samples"
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_size_area_head_smoke"
SITE_NORMALIZER_MM = 26400.0
MAX_PARTS = 8
TYPE_TO_ID = {room_type: index for index, room_type in enumerate(ROOM_TYPES)}


@dataclass(frozen=True)
class SizeSample:
    house_id: str
    group_id: str
    room_type: str
    site: tuple[float, float]
    floors: tuple[int, ...]
    type_index: int
    type_count: int
    group_count: int
    part_count: int
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]
    area_mm2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def rooms_by_group(source: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for room in source.get("rooms", []):
        group_id = str(room.get("functional_id", room["id"]))
        groups.setdefault(group_id, []).append(room)
    return groups


def room_area_mm2(room: dict[str, Any]) -> float:
    if "area" in room:
        return float(room["area"])
    x0, y0, _z0 = [float(value) for value in room["box_min"]]
    x1, y1, _z1 = [float(value) for value in room["box_max"]]
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def load_size_samples(phase10_dir: Path, max_houses: int | None = None) -> list[SizeSample]:
    paths = sorted(Path(phase10_dir).glob("house_*.json"))
    if max_houses is not None:
        paths = paths[:max_houses]
    samples: list[SizeSample] = []
    for path in paths:
        source = read_json(path)
        house_id = str(source["house_id"])
        site = source["metadata"]["building_size"]
        site_xy = (float(site["x"]), float(site["y"]))
        grouped = rooms_by_group(source)
        groups = [group for group in source.get("functional_groups", []) if grouped.get(str(group["functional_id"]))]
        type_counts: dict[str, int] = {}
        for group in groups:
            room_type = str(group["type"])
            type_counts[room_type] = type_counts.get(room_type, 0) + 1
        seen_by_type: dict[str, int] = {}
        for group in groups:
            group_id = str(group["functional_id"])
            room_type = str(group["type"])
            parts = grouped[group_id]
            type_index = seen_by_type.get(room_type, 0)
            seen_by_type[room_type] = type_index + 1
            box_min, box_max = group_bbox(parts)
            floors = tuple(sorted({floor for part in parts for floor in room_floors(part)}))
            samples.append(
                SizeSample(
                    house_id=house_id,
                    group_id=group_id,
                    room_type=room_type,
                    site=site_xy,
                    floors=floors,
                    type_index=type_index,
                    type_count=type_counts[room_type],
                    group_count=len(groups),
                    part_count=len(parts),
                    box_min=tuple(float(value) for value in box_min),
                    box_max=tuple(float(value) for value in box_max),
                    area_mm2=sum(room_area_mm2(part) for part in parts),
                )
            )
    return samples


def size_feature(sample: SizeSample) -> torch.Tensor:
    one_hot = [0.0] * len(ROOM_TYPES)
    one_hot[TYPE_TO_ID.get(sample.room_type, 0)] = 1.0
    site_x, site_y = sample.site
    type_denominator = max(sample.type_count - 1, 1)
    features = one_hot + [
        1.0 if 1 in sample.floors else 0.0,
        1.0 if 2 in sample.floors else 0.0,
        site_x / SITE_NORMALIZER_MM,
        site_y / SITE_NORMALIZER_MM,
        (site_x * site_y) / (SITE_NORMALIZER_MM * SITE_NORMALIZER_MM),
        sample.type_index / type_denominator,
        sample.type_count / 16.0,
        sample.group_count / 64.0,
    ]
    return torch.tensor(features, dtype=torch.float32)


def size_target(sample: SizeSample) -> torch.Tensor:
    site_x, site_y = sample.site
    x0, y0, _z0 = sample.box_min
    x1, y1, _z1 = sample.box_max
    site_area = max(site_x * site_y, 1.0)
    return torch.tensor(
        [
            sample.area_mm2 / site_area,
            max(x1 - x0, 300.0) / max(site_x, 1.0),
            max(y1 - y0, 300.0) / max(site_y, 1.0),
            min(sample.part_count, MAX_PARTS) / MAX_PARTS,
        ],
        dtype=torch.float32,
    )


class SizeDataset(Dataset):
    def __init__(self, samples: list[SizeSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "features": size_feature(sample),
            "target": size_target(sample),
        }


class SizeAreaHead(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def evaluate_predictions(model: SizeAreaHead, samples: list[SizeSample], device: torch.device) -> dict[str, Any]:
    model.eval()
    errors = []
    within_20 = 0
    with torch.no_grad():
        for sample in samples:
            pred = model(size_feature(sample).unsqueeze(0).to(device))[0].cpu()
            target = size_target(sample)
            error = torch.abs(pred - target)
            errors.append(error)
            if target[0] > 0 and torch.abs(pred[0] - target[0]) / target[0] <= 0.2:
                within_20 += 1
    stacked = torch.stack(errors)
    return {
        "group_count": len(samples),
        "area_ratio_mae": float(stacked[:, 0].mean()),
        "width_ratio_mae": float(stacked[:, 1].mean()),
        "depth_ratio_mae": float(stacked[:, 2].mean()),
        "part_count_norm_mae": float(stacked[:, 3].mean()),
        "area_within_20pct_rate": within_20 / max(len(samples), 1),
    }


def predicted_payload(model: SizeAreaHead, samples: list[SizeSample], device: torch.device) -> dict[str, dict[str, Any]]:
    by_house: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for sample in samples:
            pred = model(size_feature(sample).unsqueeze(0).to(device))[0].cpu()
            target = size_target(sample)
            site_area = sample.site[0] * sample.site[1]
            group = {
                "functional_id": sample.group_id,
                "type": sample.room_type,
                "floors": list(sample.floors),
                "predicted": {
                    "area_ratio": float(pred[0]),
                    "width_ratio": float(pred[1]),
                    "depth_ratio": float(pred[2]),
                    "part_count": max(1, min(MAX_PARTS, int(round(float(pred[3]) * MAX_PARTS)))),
                    "area_m2": float(pred[0] * site_area / 1_000_000.0),
                    "width_mm": float(pred[1] * sample.site[0]),
                    "depth_mm": float(pred[2] * sample.site[1]),
                },
                "target": {
                    "area_ratio": float(target[0]),
                    "width_ratio": float(target[1]),
                    "depth_ratio": float(target[2]),
                    "part_count": sample.part_count,
                    "area_m2": float(sample.area_mm2 / 1_000_000.0),
                    "width_mm": float(target[1] * sample.site[0]),
                    "depth_mm": float(target[2] * sample.site[1]),
                },
            }
            house = by_house.setdefault(
                sample.house_id,
                {
                    "schema": "graphspace_v6_size_area_predictions_v1",
                    "source": "learned_size_area_head_smoke",
                    "house_id": sample.house_id,
                    "groups": [],
                },
            )
            house["groups"].append(group)
    return by_house


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    samples = load_size_samples(args.phase10_dir, args.max_houses)
    if not samples:
        raise ValueError("no functional group size samples found")
    dataset = SizeDataset(samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    feature_dim = int(dataset[0]["features"].numel())
    model = SizeAreaHead(feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            pred = model(features)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            metrics = evaluate_predictions(model, samples, device)
            metrics["epoch"] = epoch
            metrics["loss"] = total_loss / max(batches, 1)
            history.append(metrics)
            print(
                f"epoch={epoch:04d} loss={metrics['loss']:.6f} "
                f"area_mae={metrics['area_ratio_mae']:.4f} "
                f"within20={metrics['area_within_20pct_rate']:.3f}"
            )
    final_metrics = evaluate_predictions(model, samples, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "graphspace_v6_size_area_head_smoke_v1",
            "model": model.state_dict(),
            "config": {
                "feature_dim": feature_dim,
                "room_types": ROOM_TYPES,
                "targets": ["area_ratio", "width_ratio", "depth_ratio", "part_count_norm"],
            },
            "source_phase10_dir": str(args.phase10_dir),
        },
        args.output_dir / "size_area_head.pt",
    )
    for house_id, payload in predicted_payload(model, samples, device).items():
        write_json(args.output_dir / "size_predictions" / house_id / "predicted_sizes.json", payload)
    summary = {
        "schema": "graphspace_v6_size_area_head_smoke_summary_v1",
        "purpose": (
            "Interface validation only: predict rough group area/width/depth/part-count "
            "priors from program-level features; not a formal V6 generator."
        ),
        "phase10_dir": str(args.phase10_dir),
        "house_count": len({sample.house_id for sample in samples}),
        "group_count": len(samples),
        "epochs": args.epochs,
        "final_metrics": final_metrics,
        "history": history,
        "outputs": {
            "checkpoint": str(args.output_dir / "size_area_head.pt"),
            "size_predictions": str(args.output_dir / "size_predictions"),
        },
        "formal_v6_training_ready": False,
        "blocking_reason": (
            "This is a small overfit check using Phase10 inferred functional groups. "
            "It predicts size priors but does not yet generate group nodes from raw "
            "user length/width conditions."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
