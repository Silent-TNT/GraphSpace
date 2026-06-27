#!/usr/bin/env python3
"""Train a lightweight candidate-position scorer for Phase24 coarse placement.

The scorer does not place rooms by itself. It ranks legal 300mm-grid candidate
boxes using whole-floor occupancy and target-neighbor contact features, then the
rule packer can blend this learned score with its existing hard constraints.
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


ROOT = Path(__file__).resolve().parents[2]
for import_dir in (
    ROOT,
    ROOT / "scripts" / "train_v5_spatial",
    ROOT / "scripts" / "spatial_modal_infer",
):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.spatial_modal_infer.config import ROOM_TYPES  # noqa: E402
from scripts.train_v5_spatial.v6_multipart_decoder import (  # noqa: E402
    VOXEL_MM,
    build_target_topology,
    group_bbox,
    read_json,
    room_floors,
    write_json,
)
from scripts.train_v5_spatial.v6_size_area_head import DEFAULT_PHASE10, rooms_by_group  # noqa: E402


TYPE_TO_ID = {room_type: index for index, room_type in enumerate(ROOM_TYPES)}
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_candidate_scorer_smoke"


@dataclass(frozen=True)
class CandidateTrainingExample:
    features: torch.Tensor
    target: float


class CandidateScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--candidates-per-group", type=int, default=64)
    return parser.parse_args()


def overlaps(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> bool:
    return min(box[2], other[2]) > max(box[0], other[0]) and min(box[3], other[3]) > max(box[1], other[1])


def touches(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> bool:
    horizontal_touch = (box[2] == other[0] or other[2] == box[0]) and min(box[3], other[3]) > max(box[1], other[1])
    vertical_touch = (box[3] == other[1] or other[3] == box[1]) and min(box[2], other[2]) > max(box[0], other[0])
    return horizontal_touch or vertical_touch


def grid_gap(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> int:
    dx = max(other[0] - box[2], box[0] - other[2], 0)
    dy = max(other[1] - box[3], box[1] - other[3], 0)
    return dx + dy


def share_floor(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> bool:
    return bool(set(left) & set(right))


def box_iou(box: tuple[int, int, int, int], target: tuple[int, int, int, int]) -> float:
    ix0 = max(box[0], target[0])
    iy0 = max(box[1], target[1])
    ix1 = min(box[2], target[2])
    iy1 = min(box[3], target[3])
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    target_area = max(target[2] - target[0], 0) * max(target[3] - target[1], 0)
    union = area + target_area - inter
    return float(inter / union) if union > 0 else 0.0


def topology_neighbors(topology: dict[str, Any]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = {}
    for edge in topology.get("edges", []):
        left = str(edge["source"])
        right = str(edge["target"])
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    return neighbors


def candidate_feature_vector(
    *,
    room_type: str,
    floors: list[int] | tuple[int, ...],
    candidate: tuple[int, int, int, int],
    preferred_xy: tuple[int, int],
    site_cells: tuple[int, int],
    placed: dict[str, dict[str, Any]],
    neighbors_by_node: dict[str, set[str]],
    node_id: str,
    floor_occupied_cells: dict[int, int],
) -> torch.Tensor:
    sx, sy = site_cells
    wc = max(candidate[2] - candidate[0], 1)
    dc = max(candidate[3] - candidate[1], 1)
    center_x = candidate[0] + wc * 0.5
    center_y = candidate[1] + dc * 0.5
    preferred_dist = (abs(candidate[0] - preferred_xy[0]) + abs(candidate[1] - preferred_xy[1])) / max(sx + sy, 1)
    target_neighbors = neighbors_by_node.get(node_id, set())
    placed_neighbor_count = 0
    touch_count = 0
    gap_sum = 0.0
    for neighbor_id in target_neighbors:
        neighbor = placed.get(neighbor_id)
        if not neighbor or not share_floor(floors, neighbor["floors"]):
            continue
        placed_neighbor_count += 1
        neighbor_box = tuple(int(value) for value in neighbor["box"])
        if touches(candidate, neighbor_box):
            touch_count += 1
        gap_sum += grid_gap(candidate, neighbor_box)
    candidate_area = wc * dc
    floor_ratios_before = []
    floor_ratios_after = []
    floor_capacity = max(sx * sy, 1)
    for floor in floors:
        before = floor_occupied_cells.get(int(floor), 0) / floor_capacity
        after = (floor_occupied_cells.get(int(floor), 0) + candidate_area) / floor_capacity
        floor_ratios_before.append(before)
        floor_ratios_after.append(after)
    one_hot = [0.0] * len(ROOM_TYPES)
    one_hot[TYPE_TO_ID.get(room_type, 0)] = 1.0
    numeric = [
        center_x / max(sx, 1),
        center_y / max(sy, 1),
        wc / max(sx, 1),
        dc / max(sy, 1),
        preferred_dist,
        float(1 in floors),
        float(2 in floors),
        float(len(floors) > 1),
        placed_neighbor_count / max(len(target_neighbors), 1),
        touch_count / max(placed_neighbor_count, 1),
        gap_sum / max((sx + sy) * max(placed_neighbor_count, 1), 1),
        len(target_neighbors) / 12.0,
        sum(floor_ratios_before) / max(len(floor_ratios_before), 1),
        sum(floor_ratios_after) / max(len(floor_ratios_after), 1),
        abs((sum(floor_ratios_after) / max(len(floor_ratios_after), 1)) - 0.82),
    ]
    return torch.tensor(one_hot + numeric, dtype=torch.float32)


def target_score(
    *,
    candidate: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    features: torch.Tensor,
) -> float:
    iou = box_iou(candidate, target_box)
    touch_ratio = float(features[len(ROOM_TYPES) + 9])
    gap_norm = float(features[len(ROOM_TYPES) + 10])
    occupancy_error = float(features[len(ROOM_TYPES) + 14])
    score = iou + 0.18 * touch_ratio - 0.08 * gap_norm - 0.10 * occupancy_error
    return max(0.0, min(1.0, score))


def ordered_group_ids(group_ids: list[str], room_types: dict[str, str], floors: dict[str, list[int]], neighbors: dict[str, set[str]]) -> list[str]:
    priority = {
        "stairs": 0,
        "entryway": 1,
        "living_room": 2,
        "dining_room": 3,
        "kitchen": 4,
        "corridor": 5,
        "bedroom": 6,
        "bathroom": 7,
        "utility": 8,
        "multi_purpose": 9,
        "balcony": 10,
    }
    remaining = set(group_ids)
    ordered: list[str] = []
    while remaining:
        next_id = min(
            remaining,
            key=lambda group_id: (
                -len(neighbors.get(group_id, set()) & set(ordered)),
                min(floors[group_id]),
                priority.get(room_types[group_id], 99),
                -len(neighbors.get(group_id, set())),
                group_id,
            ),
        )
        ordered.append(next_id)
        remaining.remove(next_id)
    return ordered


def sampled_candidates(
    target_box: tuple[int, int, int, int],
    wc: int,
    dc: int,
    sx: int,
    sy: int,
    max_candidates: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    x0, y0, _x1, _y1 = target_box
    max_x = max(sx - wc, 0)
    max_y = max(sy - dc, 0)
    capped_candidates = min(max_candidates, (max_x + 1) * (max_y + 1))
    seeds: set[tuple[int, int]] = {(max(0, min(x0, sx - wc)), max(0, min(y0, sy - dc)))}
    for dx in range(-6, 7, 2):
        for dy in range(-6, 7, 2):
            seeds.add((max(0, min(x0 + dx, sx - wc)), max(0, min(y0 + dy, sy - dc))))
    while len(seeds) < capped_candidates:
        seeds.add((rng.randint(0, max_x), rng.randint(0, max_y)))
    positions = list(seeds)
    rng.shuffle(positions)
    return [(x, y, x + wc, y + dc) for x, y in positions[:capped_candidates]]


def build_examples(phase10_dir: Path, max_houses: int | None, candidates_per_group: int, seed: int) -> list[CandidateTrainingExample]:
    rng = random.Random(seed)
    paths = sorted(Path(phase10_dir).glob("house_*.json"))
    if max_houses is not None:
        paths = paths[:max_houses]
    examples: list[CandidateTrainingExample] = []
    for path in paths:
        source = read_json(path)
        site = source["metadata"]["building_size"]
        sx = int(round(float(site["x"]) / VOXEL_MM))
        sy = int(round(float(site["y"]) / VOXEL_MM))
        grouped = rooms_by_group(source)
        topology = build_target_topology(source)
        neighbors = topology_neighbors(topology)
        group_ids: list[str] = []
        room_types: dict[str, str] = {}
        floors_by_group: dict[str, list[int]] = {}
        target_boxes: dict[str, tuple[int, int, int, int]] = {}
        for group in source.get("functional_groups", []):
            group_id = str(group["functional_id"])
            parts = grouped.get(group_id, [])
            if not parts:
                continue
            box_min, box_max = group_bbox(parts)
            group_ids.append(group_id)
            room_types[group_id] = str(group["type"])
            floors_by_group[group_id] = sorted({floor for part in parts for floor in room_floors(part)})
            target_boxes[group_id] = (
                int(round(float(box_min[0]) / VOXEL_MM)),
                int(round(float(box_min[1]) / VOXEL_MM)),
                int(round(float(box_max[0]) / VOXEL_MM)),
                int(round(float(box_max[1]) / VOXEL_MM)),
            )
        placed: dict[str, dict[str, Any]] = {}
        occupied_by_floor: dict[int, list[tuple[int, int, int, int]]] = {1: [], 2: []}
        floor_occupied_cells = {1: 0, 2: 0}
        for group_id in ordered_group_ids(group_ids, room_types, floors_by_group, neighbors):
            target = target_boxes[group_id]
            wc = max(target[2] - target[0], 1)
            dc = max(target[3] - target[1], 1)
            preferred_xy = (max(0, min(target[0], sx - wc)), max(0, min(target[1], sy - dc)))
            for candidate in sampled_candidates(target, wc, dc, sx, sy, candidates_per_group, rng):
                floors = floors_by_group[group_id]
                if any(any(overlaps(candidate, other) for other in occupied_by_floor[floor]) for floor in floors):
                    continue
                features = candidate_feature_vector(
                    room_type=room_types[group_id],
                    floors=floors,
                    candidate=candidate,
                    preferred_xy=preferred_xy,
                    site_cells=(sx, sy),
                    placed=placed,
                    neighbors_by_node=neighbors,
                    node_id=group_id,
                    floor_occupied_cells=floor_occupied_cells,
                )
                examples.append(CandidateTrainingExample(features, target_score(candidate=candidate, target_box=target, features=features)))
            floors = floors_by_group[group_id]
            placed[group_id] = {"box": target, "floors": floors}
            area = max(target[2] - target[0], 0) * max(target[3] - target[1], 0)
            for floor in floors:
                occupied_by_floor[floor].append(target)
                floor_occupied_cells[floor] += area
    return examples


def split_examples(examples: list[CandidateTrainingExample], seed: int) -> tuple[list[CandidateTrainingExample], list[CandidateTrainingExample]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * 0.85))
    return shuffled[:cut], shuffled[cut:]


def evaluate(model: CandidateScorer, examples: list[CandidateTrainingExample], device: torch.device) -> dict[str, float]:
    if not examples:
        return {"mae": 0.0, "mse": 0.0, "count": 0.0}
    features = torch.stack([example.features for example in examples]).to(device)
    target = torch.tensor([example.target for example in examples], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(features)
    mae = torch.nn.functional.l1_loss(pred, target).item()
    mse = torch.nn.functional.mse_loss(pred, target).item()
    return {"mae": float(mae), "mse": float(mse), "count": float(len(examples))}


def load_candidate_scorer(checkpoint_path: Path, device: torch.device) -> CandidateScorer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CandidateScorer(int(checkpoint["config"]["feature_dim"]), int(checkpoint["config"].get("hidden", 96))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    examples = build_examples(args.phase10_dir, args.max_houses, args.candidates_per_group, args.seed)
    if not examples:
        raise ValueError("no candidate scorer training examples were generated")
    train_examples, val_examples = split_examples(examples, args.seed)
    feature_dim = int(examples[0].features.numel())
    model = CandidateScorer(feature_dim, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(train_examples)
        losses = []
        for start in range(0, len(train_examples), 512):
            batch = train_examples[start : start + 512]
            features = torch.stack([example.features for example in batch]).to(device)
            target = torch.tensor([example.target for example in batch], dtype=torch.float32, device=device)
            pred = model(features)
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 10, 1) == 0:
            metrics = evaluate(model, val_examples, device)
            history.append({"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1), **metrics})
            print(json.dumps(history[-1], ensure_ascii=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_metrics = evaluate(model, val_examples, device)
    checkpoint = {
        "model": model.state_dict(),
        "config": {
            "feature_dim": feature_dim,
            "hidden": int(args.hidden),
            "room_types": ROOM_TYPES,
            "schema": "graphspace_v6_candidate_scorer_v1",
        },
    }
    torch.save(checkpoint, args.output_dir / "candidate_scorer.pt")
    summary = {
        "schema": "graphspace_v6_candidate_scorer_summary_v1",
        "phase10_dir": str(args.phase10_dir),
        "max_houses": args.max_houses,
        "epochs": args.epochs,
        "candidate_count": len(examples),
        "train_count": len(train_examples),
        "validation_count": len(val_examples),
        "feature_dim": feature_dim,
        "final_validation": final_metrics,
        "history": history,
        "checkpoint": str(args.output_dir / "candidate_scorer.pt"),
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
