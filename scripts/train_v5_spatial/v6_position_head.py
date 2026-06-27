#!/usr/bin/env python3
"""Smoke/overfit trainer for coarse functional-group position priors.

This is not a formal V6 decoder. It validates a small prediction head for
program-level rough placement: functional group type/floor/site context ->
normalized group center and a coarse 3x3 site zone. The output is intended to
condition topology generation before geometry search does final P0-safe repair.
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
from scripts.train_v5_spatial.v6_size_area_head import (  # noqa: E402
    DEFAULT_PHASE10,
    SITE_NORMALIZER_MM,
    TYPE_TO_ID,
    rooms_by_group,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "v6_position_head_smoke"
ZONE_COUNT = 9
PRIMARY_POSITION_TYPES = {"living_room", "dining_room", "bedroom"}
CIRCULATION_TYPES = {"entryway", "corridor", "stairs"}


@dataclass(frozen=True)
class PositionSample:
    house_id: str
    group_id: str
    room_type: str
    site: tuple[float, float]
    floors: tuple[int, ...]
    type_index: int
    type_count: int
    group_count: int
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]


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
    parser.add_argument(
        "--primary-position-loss-weight",
        type=float,
        default=2.0,
        help="Loss weight for living/dining/bedroom coarse position targets.",
    )
    parser.add_argument(
        "--circulation-position-loss-weight",
        type=float,
        default=0.75,
        help="Loss weight for entryway/corridor/stairs coarse position targets.",
    )
    return parser.parse_args()


def load_position_samples(phase10_dir: Path, max_houses: int | None = None) -> list[PositionSample]:
    paths = sorted(Path(phase10_dir).glob("house_*.json"))
    if max_houses is not None:
        paths = paths[:max_houses]
    samples: list[PositionSample] = []
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
                PositionSample(
                    house_id=house_id,
                    group_id=group_id,
                    room_type=room_type,
                    site=site_xy,
                    floors=floors,
                    type_index=type_index,
                    type_count=type_counts[room_type],
                    group_count=len(groups),
                    box_min=tuple(float(value) for value in box_min),
                    box_max=tuple(float(value) for value in box_max),
                )
            )
    return samples


def position_feature(sample: PositionSample) -> torch.Tensor:
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


def position_loss_weight(
    sample: PositionSample,
    primary_weight: float = 2.0,
    circulation_weight: float = 0.75,
) -> float:
    if sample.room_type in PRIMARY_POSITION_TYPES:
        return float(primary_weight)
    if sample.room_type in CIRCULATION_TYPES:
        return float(circulation_weight)
    return 1.0


def position_target(sample: PositionSample) -> tuple[torch.Tensor, torch.Tensor]:
    site_x, site_y = sample.site
    x0, y0, _z0 = sample.box_min
    x1, y1, _z1 = sample.box_max
    center_x = min(max(((x0 + x1) * 0.5) / max(site_x, 1.0), 0.0), 1.0)
    center_y = min(max(((y0 + y1) * 0.5) / max(site_y, 1.0), 0.0), 1.0)
    zone_x = min(2, max(0, int(center_x * 3.0)))
    zone_y = min(2, max(0, int(center_y * 3.0)))
    return torch.tensor([center_x, center_y], dtype=torch.float32), torch.tensor(zone_y * 3 + zone_x, dtype=torch.long)


class PositionDataset(Dataset):
    def __init__(
        self,
        samples: list[PositionSample],
        primary_weight: float = 2.0,
        circulation_weight: float = 0.75,
    ) -> None:
        self.samples = samples
        self.primary_weight = primary_weight
        self.circulation_weight = circulation_weight

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        center, zone = position_target(sample)
        return {
            "features": position_feature(sample),
            "center": center,
            "zone": zone,
            "weight": torch.tensor(
                position_loss_weight(sample, self.primary_weight, self.circulation_weight),
                dtype=torch.float32,
            ),
        }


class PositionHead(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.center_head = nn.Sequential(nn.Linear(hidden, 2), nn.Sigmoid())
        self.zone_head = nn.Linear(hidden, ZONE_COUNT)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.center_head(hidden), self.zone_head(hidden)


def evaluate_predictions(model: PositionHead, samples: list[PositionSample], device: torch.device) -> dict[str, Any]:
    model.eval()
    center_errors = []
    primary_errors = []
    circulation_errors = []
    zone_hits = 0
    primary_zone_hits = 0
    primary_count = 0
    within_one_zone = 0
    with torch.no_grad():
        for sample in samples:
            pred_center, pred_zone_logits = model(position_feature(sample).unsqueeze(0).to(device))
            target_center, target_zone = position_target(sample)
            pred_center = pred_center[0].cpu()
            pred_zone = int(torch.argmax(pred_zone_logits[0]).cpu())
            target_zone_int = int(target_zone)
            center_error = torch.abs(pred_center - target_center)
            center_errors.append(center_error)
            if sample.room_type in PRIMARY_POSITION_TYPES:
                primary_errors.append(center_error)
                primary_count += 1
            if sample.room_type in CIRCULATION_TYPES:
                circulation_errors.append(center_error)
            if pred_zone == target_zone_int:
                zone_hits += 1
                if sample.room_type in PRIMARY_POSITION_TYPES:
                    primary_zone_hits += 1
            pred_x, pred_y = pred_zone % 3, pred_zone // 3
            target_x, target_y = target_zone_int % 3, target_zone_int // 3
            if abs(pred_x - target_x) <= 1 and abs(pred_y - target_y) <= 1:
                within_one_zone += 1
    stacked = torch.stack(center_errors)
    primary_stacked = torch.stack(primary_errors) if primary_errors else torch.empty((0, 2))
    circulation_stacked = torch.stack(circulation_errors) if circulation_errors else torch.empty((0, 2))
    return {
        "group_count": len(samples),
        "center_x_mae": float(stacked[:, 0].mean()),
        "center_y_mae": float(stacked[:, 1].mean()),
        "center_l1_mae": float(stacked.sum(dim=1).mean()),
        "primary_group_count": primary_count,
        "primary_center_l1_mae": (
            float(primary_stacked.sum(dim=1).mean()) if len(primary_stacked) else 0.0
        ),
        "primary_zone_accuracy": primary_zone_hits / max(primary_count, 1),
        "circulation_center_l1_mae": (
            float(circulation_stacked.sum(dim=1).mean()) if len(circulation_stacked) else 0.0
        ),
        "zone_accuracy": zone_hits / max(len(samples), 1),
        "zone_within_one_step_rate": within_one_zone / max(len(samples), 1),
    }


def predicted_payload(
    model: PositionHead,
    samples: list[PositionSample],
    device: torch.device,
    primary_weight: float = 2.0,
    circulation_weight: float = 0.75,
) -> dict[str, dict[str, Any]]:
    by_house: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for sample in samples:
            pred_center, pred_zone_logits = model(position_feature(sample).unsqueeze(0).to(device))
            target_center, target_zone = position_target(sample)
            pred_center = pred_center[0].cpu()
            pred_zone = int(torch.argmax(pred_zone_logits[0]).cpu())
            group = {
                "functional_id": sample.group_id,
                "type": sample.room_type,
                "floors": list(sample.floors),
                "position_priority": position_loss_weight(sample, primary_weight, circulation_weight),
                "predicted": {
                    "center_x_ratio": float(pred_center[0]),
                    "center_y_ratio": float(pred_center[1]),
                    "center_x_mm": float(pred_center[0] * sample.site[0]),
                    "center_y_mm": float(pred_center[1] * sample.site[1]),
                    "zone_index": pred_zone,
                    "zone_x": pred_zone % 3,
                    "zone_y": pred_zone // 3,
                },
                "target": {
                    "center_x_ratio": float(target_center[0]),
                    "center_y_ratio": float(target_center[1]),
                    "center_x_mm": float(target_center[0] * sample.site[0]),
                    "center_y_mm": float(target_center[1] * sample.site[1]),
                    "zone_index": int(target_zone),
                    "zone_x": int(target_zone) % 3,
                    "zone_y": int(target_zone) // 3,
                },
            }
            house = by_house.setdefault(
                sample.house_id,
                {
                    "schema": "graphspace_v6_position_predictions_v1",
                    "source": "learned_position_head_smoke",
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
    samples = load_position_samples(args.phase10_dir, args.max_houses)
    if not samples:
        raise ValueError("no functional group position samples found")
    dataset = PositionDataset(
        samples,
        primary_weight=float(args.primary_position_loss_weight),
        circulation_weight=float(args.circulation_position_loss_weight),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    feature_dim = int(dataset[0]["features"].numel())
    model = PositionHead(feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    center_criterion = nn.SmoothL1Loss(reduction="none")
    zone_criterion = nn.CrossEntropyLoss(reduction="none")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            features = batch["features"].to(device)
            center = batch["center"].to(device)
            zone = batch["zone"].to(device)
            weights = batch["weight"].to(device)
            pred_center, pred_zone = model(features)
            center_loss = center_criterion(pred_center, center).mean(dim=1)
            zone_loss = zone_criterion(pred_zone, zone)
            loss = ((center_loss + 0.2 * zone_loss) * weights).sum() / weights.sum().clamp_min(1.0)
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
                f"center_l1={metrics['center_l1_mae']:.4f} "
                f"zone_acc={metrics['zone_accuracy']:.3f}"
            )
    final_metrics = evaluate_predictions(model, samples, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "graphspace_v6_position_head_smoke_v1",
            "model": model.state_dict(),
            "config": {
                "feature_dim": feature_dim,
                "room_types": ROOM_TYPES,
                "targets": ["center_x_ratio", "center_y_ratio", "zone_index_3x3"],
                "primary_position_types": sorted(PRIMARY_POSITION_TYPES),
                "circulation_types": sorted(CIRCULATION_TYPES),
                "primary_position_loss_weight": float(args.primary_position_loss_weight),
                "circulation_position_loss_weight": float(args.circulation_position_loss_weight),
            },
            "source_phase10_dir": str(args.phase10_dir),
        },
        args.output_dir / "position_head.pt",
    )
    for house_id, payload in predicted_payload(
        model,
        samples,
        device,
        primary_weight=float(args.primary_position_loss_weight),
        circulation_weight=float(args.circulation_position_loss_weight),
    ).items():
        write_json(args.output_dir / "position_predictions" / house_id / "predicted_positions.json", payload)
    summary = {
        "schema": "graphspace_v6_position_head_smoke_summary_v1",
        "purpose": (
            "Interface validation only: predict rough group center and 3x3 site "
            "zone priors from program-level features; not a formal V6 generator."
        ),
        "phase10_dir": str(args.phase10_dir),
        "house_count": len({sample.house_id for sample in samples}),
        "group_count": len(samples),
        "epochs": args.epochs,
        "primary_position_loss_weight": float(args.primary_position_loss_weight),
        "circulation_position_loss_weight": float(args.circulation_position_loss_weight),
        "final_metrics": final_metrics,
        "history": history,
        "outputs": {
            "checkpoint": str(args.output_dir / "position_head.pt"),
            "position_predictions": str(args.output_dir / "position_predictions"),
        },
        "formal_v6_training_ready": False,
        "blocking_reason": (
            "This is a small overfit check using Phase10 inferred functional groups. "
            "It predicts coarse position priors but does not yet generate the group "
            "list from raw user length/width conditions."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
