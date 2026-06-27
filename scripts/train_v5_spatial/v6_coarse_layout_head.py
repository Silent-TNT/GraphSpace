#!/usr/bin/env python3
"""Train a coarse functional-group layout head.

This head fills the gap between program/topology generation and the Phase24
multi-part decoder. It predicts a whole functional group's rough bbox
center/size from program-level inputs plus learned size/position priors, without
reading the target bbox as an input feature.
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
    VOXEL_MM,
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
from scripts.train_v5_spatial.v6_topology_learner import (  # noqa: E402
    POSITION_PRIOR_DEFAULT,
    SIZE_PRIOR_DEFAULT,
    read_position_priors,
    read_size_priors,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "v6_coarse_layout_head_smoke"
MAX_PARTS = 8


@dataclass(frozen=True)
class CoarseLayoutSample:
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
    size_prior: tuple[float, float, float, float] = SIZE_PRIOR_DEFAULT
    position_prior: tuple[float, float, float, float] = POSITION_PRIOR_DEFAULT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--size-conditioning-dir",
        type=Path,
        help="Optional directory containing size_predictions/<house_id>/predicted_sizes.json.",
    )
    parser.add_argument(
        "--position-conditioning-dir",
        type=Path,
        help="Optional directory containing position_predictions/<house_id>/predicted_positions.json.",
    )
    return parser.parse_args()


def coarse_layout_feature_from_fields(
    room_type: str,
    floors: tuple[int, ...] | list[int],
    site: tuple[float, float],
    type_index: int,
    type_count: int,
    group_count: int,
    size_prior: tuple[float, float, float, float] = SIZE_PRIOR_DEFAULT,
    position_prior: tuple[float, float, float, float] = POSITION_PRIOR_DEFAULT,
) -> torch.Tensor:
    one_hot = [0.0] * len(ROOM_TYPES)
    one_hot[TYPE_TO_ID.get(room_type, 0)] = 1.0
    site_x, site_y = site
    type_denominator = max(type_count - 1, 1)
    features = one_hot + [
        1.0 if 1 in floors else 0.0,
        1.0 if 2 in floors else 0.0,
        site_x / SITE_NORMALIZER_MM,
        site_y / SITE_NORMALIZER_MM,
        (site_x * site_y) / (SITE_NORMALIZER_MM * SITE_NORMALIZER_MM),
        type_index / type_denominator,
        type_count / 16.0,
        group_count / 64.0,
    ]
    features.extend(float(value) for value in size_prior)
    features.extend(float(value) for value in position_prior)
    return torch.tensor(features, dtype=torch.float32)


def coarse_layout_feature(sample: CoarseLayoutSample) -> torch.Tensor:
    return coarse_layout_feature_from_fields(
        sample.room_type,
        sample.floors,
        sample.site,
        sample.type_index,
        sample.type_count,
        sample.group_count,
        sample.size_prior,
        sample.position_prior,
    )


def coarse_layout_target(sample: CoarseLayoutSample) -> torch.Tensor:
    site_x, site_y = sample.site
    x0, y0, _z0 = sample.box_min
    x1, y1, _z1 = sample.box_max
    width = max(x1 - x0, VOXEL_MM)
    depth = max(y1 - y0, VOXEL_MM)
    return torch.tensor(
        [
            min(max(((x0 + x1) * 0.5) / max(site_x, 1.0), 0.0), 1.0),
            min(max(((y0 + y1) * 0.5) / max(site_y, 1.0), 0.0), 1.0),
            min(max(width / max(site_x, 1.0), 0.0), 1.0),
            min(max(depth / max(site_y, 1.0), 0.0), 1.0),
        ],
        dtype=torch.float32,
    )


def load_coarse_layout_samples(
    phase10_dir: Path,
    max_houses: int | None = None,
    size_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None = None,
    position_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None = None,
) -> list[CoarseLayoutSample]:
    paths = sorted(Path(phase10_dir).glob("house_*.json"))
    if max_houses is not None:
        paths = paths[:max_houses]
    samples: list[CoarseLayoutSample] = []
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
        house_size_priors = (size_priors or {}).get(house_id, {})
        house_position_priors = (position_priors or {}).get(house_id, {})
        for group in groups:
            group_id = str(group["functional_id"])
            room_type = str(group["type"])
            parts = grouped[group_id]
            type_index = seen_by_type.get(room_type, 0)
            seen_by_type[room_type] = type_index + 1
            box_min, box_max = group_bbox(parts)
            floors = tuple(sorted({floor for part in parts for floor in room_floors(part)}))
            samples.append(
                CoarseLayoutSample(
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
                    size_prior=house_size_priors.get(group_id, SIZE_PRIOR_DEFAULT),
                    position_prior=house_position_priors.get(group_id, POSITION_PRIOR_DEFAULT),
                )
            )
    return samples


class CoarseLayoutDataset(Dataset):
    def __init__(self, samples: list[CoarseLayoutSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "features": coarse_layout_feature(sample),
            "target": coarse_layout_target(sample),
        }


class CoarseLayoutHead(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 160) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 4),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def bbox_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pcx, pcy, pw, pd = [float(value) for value in pred]
    tcx, tcy, tw, td = [float(value) for value in target]
    px0, py0, px1, py1 = pcx - pw * 0.5, pcy - pd * 0.5, pcx + pw * 0.5, pcy + pd * 0.5
    tx0, ty0, tx1, ty1 = tcx - tw * 0.5, tcy - td * 0.5, tcx + tw * 0.5, tcy + td * 0.5
    ix0, iy0 = max(px0, tx0), max(py0, ty0)
    ix1, iy1 = min(px1, tx1), min(py1, ty1)
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    pred_area = max(px1 - px0, 0.0) * max(py1 - py0, 0.0)
    target_area = max(tx1 - tx0, 0.0) * max(ty1 - ty0, 0.0)
    return inter / max(pred_area + target_area - inter, 1e-9)


def evaluate_predictions(model: CoarseLayoutHead, samples: list[CoarseLayoutSample], device: torch.device) -> dict[str, Any]:
    model.eval()
    errors = []
    ious = []
    with torch.no_grad():
        for sample in samples:
            pred = model(coarse_layout_feature(sample).unsqueeze(0).to(device))[0].cpu()
            target = coarse_layout_target(sample)
            errors.append(torch.abs(pred - target))
            ious.append(bbox_iou(pred, target))
    stacked = torch.stack(errors)
    return {
        "group_count": len(samples),
        "center_x_mae": float(stacked[:, 0].mean()),
        "center_y_mae": float(stacked[:, 1].mean()),
        "center_l1_mae": float(stacked[:, :2].sum(dim=1).mean()),
        "width_ratio_mae": float(stacked[:, 2].mean()),
        "depth_ratio_mae": float(stacked[:, 3].mean()),
        "bbox_iou_mean": float(sum(ious) / max(len(ious), 1)),
    }


def predicted_payload(model: CoarseLayoutHead, samples: list[CoarseLayoutSample], device: torch.device) -> dict[str, dict[str, Any]]:
    by_house: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for sample in samples:
            pred = model(coarse_layout_feature(sample).unsqueeze(0).to(device))[0].cpu()
            target = coarse_layout_target(sample)
            group = {
                "functional_id": sample.group_id,
                "type": sample.room_type,
                "floors": list(sample.floors),
                "predicted": {
                    "center_x_ratio": float(pred[0]),
                    "center_y_ratio": float(pred[1]),
                    "width_ratio": float(pred[2]),
                    "depth_ratio": float(pred[3]),
                    "center_x_mm": float(pred[0] * sample.site[0]),
                    "center_y_mm": float(pred[1] * sample.site[1]),
                    "width_mm": float(pred[2] * sample.site[0]),
                    "depth_mm": float(pred[3] * sample.site[1]),
                },
                "target": {
                    "center_x_ratio": float(target[0]),
                    "center_y_ratio": float(target[1]),
                    "width_ratio": float(target[2]),
                    "depth_ratio": float(target[3]),
                    "center_x_mm": float(target[0] * sample.site[0]),
                    "center_y_mm": float(target[1] * sample.site[1]),
                    "width_mm": float(target[2] * sample.site[0]),
                    "depth_mm": float(target[3] * sample.site[1]),
                },
            }
            house = by_house.setdefault(
                sample.house_id,
                {
                    "schema": "graphspace_v6_coarse_layout_predictions_v1",
                    "source": "learned_coarse_layout_head",
                    "house_id": sample.house_id,
                    "groups": [],
                },
            )
            house["groups"].append(group)
    return by_house


def load_coarse_layout_head(checkpoint_path: Path, device: torch.device) -> CoarseLayoutHead:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_dim = int(checkpoint["config"]["feature_dim"])
    model = CoarseLayoutHead(feature_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def predict_coarse_layout_ratios(
    model: CoarseLayoutHead,
    device: torch.device,
    room_type: str,
    floors: tuple[int, ...] | list[int],
    site: tuple[float, float],
    type_index: int,
    type_count: int,
    group_count: int,
    size_prior: tuple[float, float, float, float] = SIZE_PRIOR_DEFAULT,
    position_prior: tuple[float, float, float, float] = POSITION_PRIOR_DEFAULT,
) -> tuple[float, float, float, float]:
    feature = coarse_layout_feature_from_fields(
        room_type,
        floors,
        site,
        type_index,
        type_count,
        group_count,
        size_prior,
        position_prior,
    ).unsqueeze(0)
    with torch.no_grad():
        pred = model(feature.to(device))[0].cpu()
    return tuple(float(value) for value in pred)


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    size_priors = read_size_priors(args.size_conditioning_dir)
    position_priors = read_position_priors(args.position_conditioning_dir)
    samples = load_coarse_layout_samples(args.phase10_dir, args.max_houses, size_priors, position_priors)
    if not samples:
        raise ValueError("no functional group coarse layout samples found")
    dataset = CoarseLayoutDataset(samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    feature_dim = int(dataset[0]["features"].numel())
    model = CoarseLayoutHead(feature_dim).to(device)
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
                f"center_l1={metrics['center_l1_mae']:.4f} "
                f"bbox_iou={metrics['bbox_iou_mean']:.3f}"
            )
    final_metrics = evaluate_predictions(model, samples, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "graphspace_v6_coarse_layout_head_v1",
            "model": model.state_dict(),
            "config": {
                "feature_dim": feature_dim,
                "room_types": ROOM_TYPES,
                "targets": ["center_x_ratio", "center_y_ratio", "width_ratio", "depth_ratio"],
                "uses_size_priors": bool(args.size_conditioning_dir),
                "uses_position_priors": bool(args.position_conditioning_dir),
            },
            "source_phase10_dir": str(args.phase10_dir),
        },
        args.output_dir / "coarse_layout_head.pt",
    )
    for house_id, payload in predicted_payload(model, samples, device).items():
        write_json(args.output_dir / "coarse_layout_predictions" / house_id / "predicted_coarse_layout.json", payload)
    summary = {
        "schema": "graphspace_v6_coarse_layout_head_summary_v1",
        "purpose": (
            "Predict whole functional-group coarse bbox center/size before Phase24 "
            "multi-part decoding, so the downstream decoder starts from a learned "
            "global layout instead of a hand-written packer only."
        ),
        "phase10_dir": str(args.phase10_dir),
        "house_count": len({sample.house_id for sample in samples}),
        "group_count": len(samples),
        "epochs": args.epochs,
        "size_conditioning_dir": str(args.size_conditioning_dir) if args.size_conditioning_dir else None,
        "position_conditioning_dir": str(args.position_conditioning_dir) if args.position_conditioning_dir else None,
        "final_metrics": final_metrics,
        "history": history,
        "outputs": {
            "checkpoint": str(args.output_dir / "coarse_layout_head.pt"),
            "coarse_layout_predictions": str(args.output_dir / "coarse_layout_predictions"),
        },
        "formal_v6_training_ready": False,
        "blocking_reason": (
            "This module predicts a coarse layout for an already inferred group list. "
            "It still relies on ProgramPrior/user counts for the group list and on "
            "Phase24 for multipart decoding."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
