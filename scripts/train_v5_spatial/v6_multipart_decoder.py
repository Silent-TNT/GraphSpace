#!/usr/bin/env python3
"""Smoke/overfit trainer for a functional-group -> multi-part decoder.

This is not the final V6 decoder. It only validates the interface: a functional
group can carry one or more rectangular parts, those parts can be learned from
Phase10 supervision, exported as standard JSON, and evaluated at group level.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
for import_dir in (
    ROOT,
    ROOT / "scripts" / "spatial_modal_infer",
    ROOT / "scripts" / "data_phase4",
):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.data_phase4.evaluate_candidates import evaluate_candidate  # noqa: E402
from scripts.spatial_modal_infer.config import ROOM_TYPES  # noqa: E402


DEFAULT_PHASE10 = ROOT / "data" / "phase10_functional_parts" / "samples"
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_multipart_smoke"
VOXEL_MM = 300.0
FLOOR_Z = {1: (0.0, 3000.0), 2: (3000.0, 6000.0)}
MAX_PARTS = 8
TYPE_TO_ID = {room_type: index for index, room_type in enumerate(ROOM_TYPES)}


@dataclass
class GroupSample:
    house_id: str
    group_id: str
    room_type: str
    site: tuple[float, float]
    floors: list[int]
    box_min: list[float]
    box_max: list[float]
    target_parts: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=2)
    parser.add_argument("--max-parts", type=int, default=MAX_PARTS)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Existing decoder checkpoint to load for export-only runs.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and export/evaluate houses from --checkpoint.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="During export, reuse existing predicted_json/<house_id>/evaluation.json outputs.",
    )
    parser.add_argument(
        "--topology-conditioning-dir",
        type=Path,
        help=(
            "Optional directory containing topologies/<house_id>/predicted_topology.json "
            "from v6_topology_learner.py. Used only to guide decoding; evaluation still "
            "uses target topology from Phase10 geometry."
        ),
    )
    parser.add_argument(
        "--enable-topology-placement-search",
        action="store_true",
        help=(
            "After initial decoding, try local P0-safe part moves that realize "
            "missing target topology edges."
        ),
    )
    parser.add_argument(
        "--enable-overlap-repair",
        action="store_true",
        help=(
            "Before topology placement search, try local moves that remove "
            "same-floor volume overlaps without changing part sizes."
        ),
    )
    parser.add_argument(
        "--max-topology-move-mm",
        type=float,
        default=1800.0,
        help=(
            "Maximum Manhattan XY distance for topology placement moves. "
            "Use a negative value to disable this limit."
        ),
    )
    parser.add_argument(
        "--expand-topology-search-to-site",
        action="store_true",
        help=(
            "For topology placement only, enumerate local candidates across the "
            "building site instead of restricting moves to the original group bbox."
        ),
    )
    parser.add_argument(
        "--enable-linked-part-placement",
        action="store_true",
        help=(
            "During topology placement search, try moving every part in a functional "
            "group together before falling back to single-part moves."
        ),
    )
    parser.add_argument(
        "--enable-controlled-size-adjustment",
        action="store_true",
        help=(
            "During topology placement search, allow bounded 300mm local width/depth "
            "adjustments when pure movement cannot realize a missing edge."
        ),
    )
    parser.add_argument(
        "--max-size-adjustment-mm",
        type=float,
        default=600.0,
        help="Maximum total width/depth adjustment used by controlled size search.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def room_floors(room: dict[str, Any]) -> list[int]:
    if room.get("floors"):
        return sorted({int(value) for value in room["floors"]})
    z0 = float(room["box_min"][2])
    z1 = float(room["box_max"][2])
    floors = [
        floor
        for floor, (floor_z0, floor_z1) in FLOOR_Z.items()
        if min(z1, floor_z1) - max(z0, floor_z0) > 1e-6
    ]
    return floors or ([2] if z0 >= 3000.0 else [1])


def group_bbox(parts: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    mins = [
        min(float(part["box_min"][axis]) for part in parts)
        for axis in range(3)
    ]
    maxs = [
        max(float(part["box_max"][axis]) for part in parts)
        for axis in range(3)
    ]
    return mins, maxs


def load_group_samples(phase10_dir: Path, max_houses: int | None = None) -> list[GroupSample]:
    paths = sorted(Path(phase10_dir).glob("house_*.json"))
    if max_houses is not None:
        paths = paths[:max_houses]
    samples: list[GroupSample] = []
    for path in paths:
        payload = read_json(path)
        site = payload.get("metadata", {}).get("building_size", {})
        site_xy = (float(site["x"]), float(site["y"]))
        rooms_by_group: dict[str, list[dict[str, Any]]] = {}
        for room in payload.get("rooms", []):
            group_id = str(room.get("functional_id", room["id"]))
            rooms_by_group.setdefault(group_id, []).append(room)
        for group in payload.get("functional_groups", []):
            group_id = str(group["functional_id"])
            parts = sorted(
                rooms_by_group.get(group_id, []),
                key=lambda item: (
                    min(room_floors(item)),
                    float(item["box_min"][0]),
                    float(item["box_min"][1]),
                    str(item["id"]),
                ),
            )
            if not parts:
                continue
            box_min, box_max = group_bbox(parts)
            samples.append(
                GroupSample(
                    house_id=str(payload["house_id"]),
                    group_id=group_id,
                    room_type=str(group["type"]),
                    site=site_xy,
                    floors=[int(value) for value in group.get("floors", room_floors(parts[0]))],
                    box_min=box_min,
                    box_max=box_max,
                    target_parts=parts,
                )
            )
    return samples


def feature_vector(sample: GroupSample, max_parts: int) -> torch.Tensor:
    one_hot = [0.0] * len(ROOM_TYPES)
    one_hot[TYPE_TO_ID.get(sample.room_type, 0)] = 1.0
    x0, y0, z0 = sample.box_min
    x1, y1, z1 = sample.box_max
    site_x, site_y = sample.site
    width = max(x1 - x0, VOXEL_MM)
    depth = max(y1 - y0, VOXEL_MM)
    height = max(z1 - z0, VOXEL_MM)
    floor_flags = [1.0 if floor in sample.floors else 0.0 for floor in (1, 2)]
    geom = [
        x0 / max(site_x, 1.0),
        y0 / max(site_y, 1.0),
        z0 / 6000.0,
        x1 / max(site_x, 1.0),
        y1 / max(site_y, 1.0),
        z1 / 6000.0,
        width / max(site_x, 1.0),
        depth / max(site_y, 1.0),
        height / 6000.0,
        min(len(sample.target_parts), max_parts) / max(max_parts, 1),
    ]
    return torch.tensor(one_hot + floor_flags + geom, dtype=torch.float32)


def target_tensor(sample: GroupSample, max_parts: int) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros((max_parts, 6), dtype=torch.float32)
    mask = torch.zeros((max_parts,), dtype=torch.float32)
    x0, y0, z0 = sample.box_min
    x1, y1, z1 = sample.box_max
    span = [
        max(x1 - x0, VOXEL_MM),
        max(y1 - y0, VOXEL_MM),
        max(z1 - z0, VOXEL_MM),
    ]
    for index, part in enumerate(sample.target_parts[:max_parts]):
        px0, py0, pz0 = [float(value) for value in part["box_min"]]
        px1, py1, pz1 = [float(value) for value in part["box_max"]]
        target[index] = torch.tensor(
            [
                (px0 - x0) / span[0],
                (py0 - y0) / span[1],
                (pz0 - z0) / span[2],
                (px1 - x0) / span[0],
                (py1 - y0) / span[1],
                (pz1 - z0) / span[2],
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        mask[index] = 1.0
    return target, mask


class FunctionalPartDataset(Dataset):
    def __init__(self, samples: list[GroupSample], max_parts: int) -> None:
        self.samples = samples
        self.max_parts = max_parts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        target, mask = target_tensor(sample, self.max_parts)
        return {
            "features": feature_vector(sample, self.max_parts),
            "target": target,
            "mask": mask,
        }


class MultiPartDecoder(nn.Module):
    def __init__(self, feature_dim: int, max_parts: int, hidden: int = 192) -> None:
        super().__init__()
        self.max_parts = max_parts
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, max_parts * 6),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(features.shape[0], self.max_parts, 6)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    part_mask = mask.unsqueeze(-1)
    denominator = part_mask.sum().clamp_min(1.0) * pred.shape[-1]
    return ((pred - target).pow(2) * part_mask).sum() / denominator


def snap(value: float) -> float:
    return round(value / VOXEL_MM) * VOXEL_MM


def floor_z_bounds(floors: list[int]) -> tuple[float, float]:
    floor_set = {int(floor) for floor in floors}
    if floor_set == {1, 2}:
        return 0.0, 6000.0
    if floor_set == {2}:
        return 3000.0, 6000.0
    return 0.0, 3000.0


def boxes_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for axis in range(3):
        left_min = float(left["box_min"][axis])
        left_max = float(left["box_max"][axis])
        right_min = float(right["box_min"][axis])
        right_max = float(right["box_max"][axis])
        if min(left_max, right_max) - max(left_min, right_min) <= 1e-6:
            return False
    return True


def overlap_volume(left: dict[str, Any], right: dict[str, Any]) -> float:
    volume = 1.0
    for axis in range(3):
        volume *= axis_overlap(
            float(left["box_min"][axis]),
            float(left["box_max"][axis]),
            float(right["box_min"][axis]),
            float(right["box_max"][axis]),
        )
    return volume


def overlap_pairs(rooms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for left_index, left in enumerate(rooms):
        for right_index, right in enumerate(rooms[left_index + 1 :], start=left_index + 1):
            if boxes_overlap(left, right):
                pairs.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "left_id": str(left.get("id", "")),
                        "right_id": str(right.get("id", "")),
                        "volume": overlap_volume(left, right),
                    }
                )
    return pairs


def axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def projection_overlap_area(left: dict[str, Any], right: dict[str, Any]) -> float:
    return axis_overlap(
        float(left["box_min"][0]),
        float(left["box_max"][0]),
        float(right["box_min"][0]),
        float(right["box_max"][0]),
    ) * axis_overlap(
        float(left["box_min"][1]),
        float(left["box_max"][1]),
        float(right["box_min"][1]),
        float(right["box_max"][1]),
    )


def face_contact_quality(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx0, ly0, _ = [float(value) for value in left["box_min"]]
    lx1, ly1, _ = [float(value) for value in left["box_max"]]
    rx0, ry0, _ = [float(value) for value in right["box_min"]]
    rx1, ry1, _ = [float(value) for value in right["box_max"]]
    vertical_face = abs(lx1 - rx0) <= 1e-6 or abs(rx1 - lx0) <= 1e-6
    horizontal_face = abs(ly1 - ry0) <= 1e-6 or abs(ry1 - ly0) <= 1e-6
    if vertical_face:
        overlap = axis_overlap(ly0, ly1, ry0, ry1)
        return overlap / max(min(ly1 - ly0, ry1 - ry0), VOXEL_MM)
    if horizontal_face:
        overlap = axis_overlap(lx0, lx1, rx0, rx1)
        return overlap / max(min(lx1 - lx0, rx1 - rx0), VOXEL_MM)
    return 0.0


def room_functional_id(room: dict[str, Any]) -> str:
    for key in ("functional_id", "group_id", "parent_id"):
        value = room.get(key)
        if value:
            return str(value)
    room_id = str(room.get("id", ""))
    if "_part_" in room_id:
        return room_id.split("_part_", 1)[0]
    return room_id


def build_target_topology(source: dict[str, Any]) -> dict[str, Any]:
    """Infer group-level target topology from Phase10 part geometry."""
    rooms = []
    for room in source.get("rooms", []):
        item = dict(room)
        item["functional_id"] = room_functional_id(item)
        item["floors"] = room_floors(item)
        rooms.append(item)
    nodes = []
    seen_nodes = set()
    for group in source.get("functional_groups", []):
        group_id = str(group["functional_id"])
        seen_nodes.add(group_id)
        floors = [int(value) for value in group.get("floors", [])]
        nodes.append(
            {
                "id": group_id,
                "type": str(group["type"]),
                "floor": min(floors) if floors else None,
                "floors": floors,
            }
        )
    for room in rooms:
        group_id = room["functional_id"]
        if group_id in seen_nodes:
            continue
        seen_nodes.add(group_id)
        floors = room_floors(room)
        nodes.append(
            {
                "id": group_id,
                "type": str(room.get("type", "unknown")),
                "floor": min(floors),
                "floors": floors,
            }
        )
    edge_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, left in enumerate(rooms):
        for right in rooms[index + 1 :]:
            left_id = left["functional_id"]
            right_id = right["functional_id"]
            if left_id == right_id:
                continue
            shared_floors = set(room_floors(left)) & set(room_floors(right))
            if shared_floors and face_contact_quality(left, right) > 0:
                source_id, target_id = sorted((left_id, right_id))
                key = (source_id, target_id, "horizontal")
                edge_by_key[key] = {
                    "source": source_id,
                    "target": target_id,
                    "relation": "horizontal",
                }
    edges = sorted(edge_by_key.values(), key=lambda edge: (edge["source"], edge["target"], edge["relation"]))
    required_edges = sorted({tuple(sorted((edge["source"], edge["target"]))) for edge in edges})
    return {
        "schema": "graphspace_v6_phase13_target_topology_v1",
        "nodes": sorted(nodes, key=lambda node: str(node["id"])),
        "edges": edges,
        "required_edges": [list(edge) for edge in required_edges],
        "source": "inferred_from_phase10_group_part_geometry",
    }


def group_neighbor_map(topology: dict[str, Any]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in topology.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        neighbors[source].add(target)
        neighbors[target].add(source)
    return neighbors


def read_conditioning_topologies(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    topologies = {}
    root = path / "topologies" if (path / "topologies").exists() else path
    for topology_path in sorted(root.glob("*/predicted_topology.json")):
        topologies[topology_path.parent.name] = read_json(topology_path)
    return topologies


def topology_contact_score(
    candidate: dict[str, Any],
    occupied: list[dict[str, Any]],
    target_neighbors: set[str] | None,
) -> float:
    if not target_neighbors:
        return 0.0
    score = 0.0
    candidate_floors = set(room_floors(candidate))
    for other in occupied:
        other_id = room_functional_id(other)
        if other_id not in target_neighbors:
            continue
        if candidate_floors & set(room_floors(other)):
            score += face_contact_quality(candidate, other)
        elif projection_overlap_area(candidate, other) > 0:
            score += 1.0
    return score


def overlaps_any(box: dict[str, Any], occupied: list[dict[str, Any]]) -> bool:
    return any(boxes_overlap(box, other) for other in occupied)


def box_center_xy(box: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(box["box_min"][0]) + float(box["box_max"][0])) * 0.5,
        (float(box["box_min"][1]) + float(box["box_max"][1])) * 0.5,
    )


def make_part_box(
    sample: GroupSample,
    index: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    z0: float,
    z1: float,
    floors: list[int] | None = None,
) -> dict[str, Any]:
    part_floors = list(floors) if floors is not None else list(sample.floors)
    return {
        "id": f"{sample.group_id}_part_{index}",
        "functional_id": sample.group_id,
        "type": sample.room_type,
        "floor": min(part_floors),
        "floors": part_floors,
        "box_min": [x0, y0, z0],
        "box_max": [x1, y1, z1],
    }


def search_non_overlapping_box(
    sample: GroupSample,
    index: int,
    preferred: dict[str, Any],
    occupied: list[dict[str, Any]],
    width_cells: int,
    depth_cells: int,
    z0: float,
    z1: float,
    floors: list[int] | None = None,
    target_neighbors: set[str] | None = None,
) -> dict[str, Any] | None:
    gx0 = int(round(float(sample.box_min[0]) / VOXEL_MM))
    gy0 = int(round(float(sample.box_min[1]) / VOXEL_MM))
    gx1 = int(round(float(sample.box_max[0]) / VOXEL_MM))
    gy1 = int(round(float(sample.box_max[1]) / VOXEL_MM))
    width_cells = max(1, min(width_cells, gx1 - gx0))
    depth_cells = max(1, min(depth_cells, gy1 - gy0))
    px, py = box_center_xy(preferred)
    candidates = []
    for x_cell in range(gx0, gx1 - width_cells + 1):
        for y_cell in range(gy0, gy1 - depth_cells + 1):
            candidate = make_part_box(
                sample,
                index,
                x_cell * VOXEL_MM,
                y_cell * VOXEL_MM,
                (x_cell + width_cells) * VOXEL_MM,
                (y_cell + depth_cells) * VOXEL_MM,
                z0,
                z1,
                floors,
            )
            if overlaps_any(candidate, occupied):
                continue
            cx, cy = box_center_xy(candidate)
            topology_score = topology_contact_score(candidate, occupied, target_neighbors)
            candidates.append((-topology_score, abs(cx - px) + abs(cy - py), x_cell, y_cell, candidate))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))[4]


def search_flexible_non_overlapping_box(
    sample: GroupSample,
    index: int,
    preferred: dict[str, Any],
    occupied: list[dict[str, Any]],
    max_width_cells: int,
    max_depth_cells: int,
    z0: float,
    z1: float,
    floors: list[int] | None = None,
    target_neighbors: set[str] | None = None,
) -> dict[str, Any] | None:
    gx0 = int(round(float(sample.box_min[0]) / VOXEL_MM))
    gy0 = int(round(float(sample.box_min[1]) / VOXEL_MM))
    gx1 = int(round(float(sample.box_max[0]) / VOXEL_MM))
    gy1 = int(round(float(sample.box_max[1]) / VOXEL_MM))
    max_width_cells = max(1, min(max_width_cells, gx1 - gx0))
    max_depth_cells = max(1, min(max_depth_cells, gy1 - gy0))
    px, py = box_center_xy(preferred)
    candidates = []
    for width_cells in range(max_width_cells, 0, -1):
        for depth_cells in range(max_depth_cells, 0, -1):
            for x_cell in range(gx0, gx1 - width_cells + 1):
                for y_cell in range(gy0, gy1 - depth_cells + 1):
                    candidate = make_part_box(
                        sample,
                        index,
                        x_cell * VOXEL_MM,
                        y_cell * VOXEL_MM,
                        (x_cell + width_cells) * VOXEL_MM,
                        (y_cell + depth_cells) * VOXEL_MM,
                        z0,
                        z1,
                        floors,
                    )
                    if overlaps_any(candidate, occupied):
                        continue
                    cx, cy = box_center_xy(candidate)
                    area = width_cells * depth_cells
                    distance = abs(cx - px) + abs(cy - py)
                    topology_score = topology_contact_score(candidate, occupied, target_neighbors)
                    candidates.append((-topology_score, -area, distance, x_cell, y_cell, candidate))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0], item[1], item[2], item[3], item[4]))[5]


def fallback_part_cells(sample: GroupSample, part_count: int) -> tuple[int, int]:
    x_cells = max(1, int(round((sample.box_max[0] - sample.box_min[0]) / VOXEL_MM)))
    y_cells = max(1, int(round((sample.box_max[1] - sample.box_min[1]) / VOXEL_MM)))
    if x_cells >= y_cells:
        return max(1, x_cells // max(part_count, 1)), y_cells
    return x_cells, max(1, y_cells // max(part_count, 1))


def target_part_floors(sample: GroupSample, index: int) -> list[int]:
    if index < len(sample.target_parts):
        return room_floors(sample.target_parts[index])
    return list(sample.floors)


def decode_parts(
    sample: GroupSample,
    pred: torch.Tensor,
    max_parts: int,
    use_target_count: bool = True,
    occupied: list[dict[str, Any]] | None = None,
    target_neighbors: set[str] | None = None,
) -> list[dict[str, Any]]:
    part_count = min(len(sample.target_parts), max_parts) if use_target_count else max_parts
    occupied = occupied if occupied is not None else []
    x0, y0, _z0 = sample.box_min
    x1, y1, _z1 = sample.box_max
    span = [max(x1 - x0, VOXEL_MM), max(y1 - y0, VOXEL_MM)]
    decoded = []
    for index in range(part_count):
        part_floors = target_part_floors(sample, index)
        z0, z1 = floor_z_bounds(part_floors)
        values = pred[index].detach().cpu().tolist()
        rx0, ry0, _rz0, rx1, ry1, _rz1 = values
        ax0, ax1 = sorted((rx0, rx1))
        ay0, ay1 = sorted((ry0, ry1))
        px0 = max(x0, min(x1 - VOXEL_MM, snap(x0 + ax0 * span[0])))
        py0 = max(y0, min(y1 - VOXEL_MM, snap(y0 + ay0 * span[1])))
        px1 = min(x1, max(px0 + VOXEL_MM, snap(x0 + ax1 * span[0])))
        py1 = min(y1, max(py0 + VOXEL_MM, snap(y0 + ay1 * span[1])))
        part = make_part_box(sample, index, px0, py0, px1, py1, z0, z1, part_floors)
        placed_boxes = occupied + decoded
        if overlaps_any(part, placed_boxes):
            width_cells = max(1, int(round((px1 - px0) / VOXEL_MM)))
            depth_cells = max(1, int(round((py1 - py0) / VOXEL_MM)))
            part = search_non_overlapping_box(
                sample,
                index,
                part,
                placed_boxes,
                width_cells,
                depth_cells,
                z0,
                z1,
                part_floors,
                target_neighbors,
            )
        if part is None:
            width_cells, depth_cells = fallback_part_cells(sample, part_count)
            part = search_flexible_non_overlapping_box(
                sample,
                index,
                make_part_box(
                    sample,
                    index,
                    x0,
                    y0,
                    x0 + VOXEL_MM,
                    y0 + VOXEL_MM,
                    z0,
                    z1,
                    part_floors,
                ),
                placed_boxes,
                width_cells,
                depth_cells,
                z0,
                z1,
                part_floors,
                target_neighbors,
            )
            if part is not None:
                part["constraint_adjustment"] = "shrunk_to_non_overlapping_candidate"
        if part is None:
            part = make_part_box(sample, index, px0, py0, px1, py1, z0, z1, part_floors)
            part["constraint_error"] = "no_non_overlapping_candidate_found"
        decoded.append(part)
    occupied.extend(decoded)
    return decoded


def part_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_min = [float(value) for value in left["box_min"]]
    left_max = [float(value) for value in left["box_max"]]
    right_min = [float(value) for value in right["box_min"]]
    right_max = [float(value) for value in right["box_max"]]
    intersection = 1.0
    left_volume = 1.0
    right_volume = 1.0
    for axis in range(3):
        overlap = max(0.0, min(left_max[axis], right_max[axis]) - max(left_min[axis], right_min[axis]))
        intersection *= overlap
        left_volume *= max(0.0, left_max[axis] - left_min[axis])
        right_volume *= max(0.0, right_max[axis] - right_min[axis])
    union = left_volume + right_volume - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_predictions(
    model: MultiPartDecoder,
    samples: list[GroupSample],
    max_parts: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    ious = []
    exact_count = 0
    exact_snap = 0
    with torch.no_grad():
        for sample in samples:
            features = feature_vector(sample, max_parts).unsqueeze(0).to(device)
            pred = model(features)[0]
            decoded = decode_parts(sample, pred, max_parts)
            target = sample.target_parts[:max_parts]
            if len(decoded) == len(target):
                exact_count += 1
            for left, right in zip(decoded, target):
                iou = part_iou(left, right)
                ious.append(iou)
                if iou >= 0.999:
                    exact_snap += 1
    return {
        "group_count": len(samples),
        "part_count": sum(min(len(sample.target_parts), max_parts) for sample in samples),
        "exact_part_count_groups": exact_count,
        "mean_part_iou": sum(ious) / max(len(ious), 1),
        "min_part_iou": min(ious) if ious else 0.0,
        "exact_snapped_parts": exact_snap,
        "exact_snapped_part_rate": exact_snap / max(len(ious), 1),
    }


def topology_metrics_from_report(report: dict[str, Any]) -> dict[str, Any]:
    p1 = report.get("p1_spatial_organization", {})
    target = p1.get("target_topology", {})
    return {
        "p1_mode": p1.get("mode"),
        "target_edge_count": target.get("target_edge_count", 0),
        "realized_edge_count": target.get("realized_edge_count", 0),
        "realization_rate": target.get("realization_rate", 0.0),
        "required_edge_count": target.get("required_edge_count", 0),
        "required_realized_edge_count": target.get("required_realized_edge_count", 0),
        "required_realization_rate": target.get("required_realization_rate", 0.0),
    }


def layout_report(
    house_id: str,
    rooms: list[dict[str, Any]],
    counts: dict[str, int],
    site_xy: tuple[float, float],
    topology: dict[str, Any],
) -> dict[str, Any]:
    report, _ = evaluate_candidate(house_id, rooms, counts, site_xy, topology=topology)
    return report


def topology_score(report: dict[str, Any]) -> tuple[int, int]:
    target = report.get("p1_spatial_organization", {}).get("target_topology", {})
    return (
        int(target.get("required_realized_edge_count", 0)),
        int(target.get("realized_edge_count", 0)),
    )


def unrealized_edges(report: dict[str, Any]) -> list[dict[str, Any]]:
    target = report.get("p1_spatial_organization", {}).get("target_topology", {})
    return [
        edge
        for edge in target.get("edges", [])
        if not edge.get("realized_in_dual")
    ]


def replacement_part(room: dict[str, Any], x0: float, y0: float, x1: float, y1: float) -> dict[str, Any]:
    moved = dict(room)
    moved["box_min"] = [float(x0), float(y0), float(room["box_min"][2])]
    moved["box_max"] = [float(x1), float(y1), float(room["box_max"][2])]
    moved["constraint_adjustment"] = "topology_placement_search_move"
    return moved


def translated_part(room: dict[str, Any], dx: float, dy: float, adjustment: str) -> dict[str, Any]:
    moved = dict(room)
    moved["box_min"] = [
        float(room["box_min"][0]) + dx,
        float(room["box_min"][1]) + dy,
        float(room["box_min"][2]),
    ]
    moved["box_max"] = [
        float(room["box_max"][0]) + dx,
        float(room["box_max"][1]) + dy,
        float(room["box_max"][2]),
    ]
    moved["constraint_adjustment"] = adjustment
    return moved


def group_center_xy(parts: list[dict[str, Any]]) -> tuple[float, float]:
    if not parts:
        return (0.0, 0.0)
    centers = [box_center_xy(part) for part in parts]
    return (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers),
    )


def candidate_contact_moves(
    sample: GroupSample,
    moving_part: dict[str, Any],
    fixed_parts: list[dict[str, Any]],
    occupied: list[dict[str, Any]],
    limit: int = 48,
    max_move_mm: float | None = None,
    search_site_xy: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    width_cells = max(1, int(round((moving_part["box_max"][0] - moving_part["box_min"][0]) / VOXEL_MM)))
    depth_cells = max(1, int(round((moving_part["box_max"][1] - moving_part["box_min"][1]) / VOXEL_MM)))
    if search_site_xy is None:
        gx0 = int(round(float(sample.box_min[0]) / VOXEL_MM))
        gy0 = int(round(float(sample.box_min[1]) / VOXEL_MM))
        gx1 = int(round(float(sample.box_max[0]) / VOXEL_MM))
        gy1 = int(round(float(sample.box_max[1]) / VOXEL_MM))
    else:
        gx0 = 0
        gy0 = 0
        gx1 = int(round(float(search_site_xy[0]) / VOXEL_MM))
        gy1 = int(round(float(search_site_xy[1]) / VOXEL_MM))
    if width_cells > gx1 - gx0 or depth_cells > gy1 - gy0:
        return []
    original_center = box_center_xy(moving_part)
    candidates = []
    moving_floors = set(room_floors(moving_part))
    for x_cell in range(gx0, gx1 - width_cells + 1):
        for y_cell in range(gy0, gy1 - depth_cells + 1):
            candidate = replacement_part(
                moving_part,
                x_cell * VOXEL_MM,
                y_cell * VOXEL_MM,
                (x_cell + width_cells) * VOXEL_MM,
                (y_cell + depth_cells) * VOXEL_MM,
            )
            if overlaps_any(candidate, occupied):
                continue
            contact = 0.0
            for fixed in fixed_parts:
                if moving_floors & set(room_floors(fixed)):
                    contact += face_contact_quality(candidate, fixed)
            if contact <= 0.0:
                continue
            center = box_center_xy(candidate)
            move_distance = abs(center[0] - original_center[0]) + abs(center[1] - original_center[1])
            if max_move_mm is not None and move_distance > max_move_mm:
                continue
            candidates.append((-contact, move_distance, x_cell, y_cell, candidate))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [item[4] for item in candidates[:limit]]


def candidate_linked_group_contact_moves(
    sample: GroupSample,
    moving_parts: list[dict[str, Any]],
    fixed_parts: list[dict[str, Any]],
    occupied: list[dict[str, Any]],
    limit: int = 32,
    max_move_mm: float | None = None,
    search_site_xy: tuple[float, float] | None = None,
) -> list[list[dict[str, Any]]]:
    if not moving_parts:
        return []
    min_x = min(float(part["box_min"][0]) for part in moving_parts)
    min_y = min(float(part["box_min"][1]) for part in moving_parts)
    max_x = max(float(part["box_max"][0]) for part in moving_parts)
    max_y = max(float(part["box_max"][1]) for part in moving_parts)
    width_cells = max(1, int(round((max_x - min_x) / VOXEL_MM)))
    depth_cells = max(1, int(round((max_y - min_y) / VOXEL_MM)))
    if search_site_xy is None:
        gx0 = int(round(float(sample.box_min[0]) / VOXEL_MM))
        gy0 = int(round(float(sample.box_min[1]) / VOXEL_MM))
        gx1 = int(round(float(sample.box_max[0]) / VOXEL_MM))
        gy1 = int(round(float(sample.box_max[1]) / VOXEL_MM))
    else:
        gx0 = 0
        gy0 = 0
        gx1 = int(round(float(search_site_xy[0]) / VOXEL_MM))
        gy1 = int(round(float(search_site_xy[1]) / VOXEL_MM))
    if width_cells > gx1 - gx0 or depth_cells > gy1 - gy0:
        return []

    source_x_cell = int(round(min_x / VOXEL_MM))
    source_y_cell = int(round(min_y / VOXEL_MM))
    original_group_center = group_center_xy(moving_parts)
    moving_floors = set().union(*(set(room_floors(part)) for part in moving_parts))
    candidates = []
    for x_cell in range(gx0, gx1 - width_cells + 1):
        for y_cell in range(gy0, gy1 - depth_cells + 1):
            dx = (x_cell - source_x_cell) * VOXEL_MM
            dy = (y_cell - source_y_cell) * VOXEL_MM
            if dx == 0.0 and dy == 0.0:
                continue
            moved_parts = [
                translated_part(part, dx, dy, "linked_part_topology_placement_move")
                for part in moving_parts
            ]
            if any(overlaps_any(part, occupied) for part in moved_parts):
                continue
            contact = 0.0
            for moved in moved_parts:
                for fixed in fixed_parts:
                    if moving_floors & set(room_floors(fixed)):
                        contact += face_contact_quality(moved, fixed)
            if contact <= 0.0:
                continue
            center = group_center_xy(moved_parts)
            move_distance = abs(center[0] - original_group_center[0]) + abs(
                center[1] - original_group_center[1]
            )
            if max_move_mm is not None and move_distance > max_move_mm:
                continue
            candidates.append((-contact, move_distance, x_cell, y_cell, moved_parts))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [item[4] for item in candidates[:limit]]


def candidate_controlled_size_contact_moves(
    sample: GroupSample,
    moving_part: dict[str, Any],
    fixed_parts: list[dict[str, Any]],
    occupied: list[dict[str, Any]],
    limit: int = 64,
    max_move_mm: float | None = None,
    max_size_adjustment_mm: float = 600.0,
    search_site_xy: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    original_width_cells = max(
        1, int(round((moving_part["box_max"][0] - moving_part["box_min"][0]) / VOXEL_MM))
    )
    original_depth_cells = max(
        1, int(round((moving_part["box_max"][1] - moving_part["box_min"][1]) / VOXEL_MM))
    )
    max_adjust_cells = max(0, int(round(max_size_adjustment_mm / VOXEL_MM)))
    if max_adjust_cells <= 0:
        return []
    if search_site_xy is None:
        gx0 = int(round(float(sample.box_min[0]) / VOXEL_MM))
        gy0 = int(round(float(sample.box_min[1]) / VOXEL_MM))
        gx1 = int(round(float(sample.box_max[0]) / VOXEL_MM))
        gy1 = int(round(float(sample.box_max[1]) / VOXEL_MM))
    else:
        gx0 = 0
        gy0 = 0
        gx1 = int(round(float(search_site_xy[0]) / VOXEL_MM))
        gy1 = int(round(float(search_site_xy[1]) / VOXEL_MM))

    original_center = box_center_xy(moving_part)
    moving_floors = set(room_floors(moving_part))
    min_width_cells = max(1, original_width_cells - max_adjust_cells)
    min_depth_cells = max(1, original_depth_cells - max_adjust_cells)
    max_width_cells = min(gx1 - gx0, original_width_cells + max_adjust_cells)
    max_depth_cells = min(gy1 - gy0, original_depth_cells + max_adjust_cells)
    candidates = []
    for width_cells in range(min_width_cells, max_width_cells + 1):
        for depth_cells in range(min_depth_cells, max_depth_cells + 1):
            size_delta = abs(width_cells - original_width_cells) + abs(depth_cells - original_depth_cells)
            if size_delta <= 0 or size_delta > max_adjust_cells:
                continue
            for x_cell in range(gx0, gx1 - width_cells + 1):
                for y_cell in range(gy0, gy1 - depth_cells + 1):
                    candidate = replacement_part(
                        moving_part,
                        x_cell * VOXEL_MM,
                        y_cell * VOXEL_MM,
                        (x_cell + width_cells) * VOXEL_MM,
                        (y_cell + depth_cells) * VOXEL_MM,
                    )
                    candidate["constraint_adjustment"] = "controlled_size_topology_adjustment"
                    if overlaps_any(candidate, occupied):
                        continue
                    contact = 0.0
                    for fixed in fixed_parts:
                        if moving_floors & set(room_floors(fixed)):
                            contact += face_contact_quality(candidate, fixed)
                    if contact <= 0.0:
                        continue
                    center = box_center_xy(candidate)
                    move_distance = abs(center[0] - original_center[0]) + abs(center[1] - original_center[1])
                    if max_move_mm is not None and move_distance > max_move_mm:
                        continue
                    area_cells = width_cells * depth_cells
                    candidates.append(
                        (-contact, size_delta, move_distance, -area_cells, x_cell, y_cell, candidate)
                    )
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]))
    return [item[6] for item in candidates[:limit]]


def candidate_non_overlapping_moves(
    sample: GroupSample,
    moving_part: dict[str, Any],
    occupied: list[dict[str, Any]],
    limit: int = 64,
    max_move_mm: float | None = None,
) -> list[dict[str, Any]]:
    width_cells = max(1, int(round((moving_part["box_max"][0] - moving_part["box_min"][0]) / VOXEL_MM)))
    depth_cells = max(1, int(round((moving_part["box_max"][1] - moving_part["box_min"][1]) / VOXEL_MM)))
    gx0 = int(round(float(sample.box_min[0]) / VOXEL_MM))
    gy0 = int(round(float(sample.box_min[1]) / VOXEL_MM))
    gx1 = int(round(float(sample.box_max[0]) / VOXEL_MM))
    gy1 = int(round(float(sample.box_max[1]) / VOXEL_MM))
    if width_cells > gx1 - gx0 or depth_cells > gy1 - gy0:
        return []
    original_center = box_center_xy(moving_part)
    candidates = []
    for x_cell in range(gx0, gx1 - width_cells + 1):
        for y_cell in range(gy0, gy1 - depth_cells + 1):
            candidate = replacement_part(
                moving_part,
                x_cell * VOXEL_MM,
                y_cell * VOXEL_MM,
                (x_cell + width_cells) * VOXEL_MM,
                (y_cell + depth_cells) * VOXEL_MM,
            )
            if overlaps_any(candidate, occupied):
                continue
            center = box_center_xy(candidate)
            move_distance = abs(center[0] - original_center[0]) + abs(center[1] - original_center[1])
            if max_move_mm is not None and move_distance > max_move_mm:
                continue
            candidates.append((move_distance, x_cell, y_cell, candidate))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in candidates[:limit]]


def candidate_flexible_non_overlapping_moves(
    sample: GroupSample,
    moving_part: dict[str, Any],
    occupied: list[dict[str, Any]],
    limit: int = 64,
    max_move_mm: float | None = None,
) -> list[dict[str, Any]]:
    max_width_cells = max(1, int(round((moving_part["box_max"][0] - moving_part["box_min"][0]) / VOXEL_MM)))
    max_depth_cells = max(1, int(round((moving_part["box_max"][1] - moving_part["box_min"][1]) / VOXEL_MM)))
    gx0 = int(round(float(sample.box_min[0]) / VOXEL_MM))
    gy0 = int(round(float(sample.box_min[1]) / VOXEL_MM))
    gx1 = int(round(float(sample.box_max[0]) / VOXEL_MM))
    gy1 = int(round(float(sample.box_max[1]) / VOXEL_MM))
    max_width_cells = max(1, min(max_width_cells, gx1 - gx0))
    max_depth_cells = max(1, min(max_depth_cells, gy1 - gy0))
    original_center = box_center_xy(moving_part)
    candidates = []
    for width_cells in range(max_width_cells, 0, -1):
        for depth_cells in range(max_depth_cells, 0, -1):
            for x_cell in range(gx0, gx1 - width_cells + 1):
                for y_cell in range(gy0, gy1 - depth_cells + 1):
                    candidate = replacement_part(
                        moving_part,
                        x_cell * VOXEL_MM,
                        y_cell * VOXEL_MM,
                        (x_cell + width_cells) * VOXEL_MM,
                        (y_cell + depth_cells) * VOXEL_MM,
                    )
                    if overlaps_any(candidate, occupied):
                        continue
                    center = box_center_xy(candidate)
                    move_distance = abs(center[0] - original_center[0]) + abs(center[1] - original_center[1])
                    if max_move_mm is not None and move_distance > max_move_mm:
                        continue
                    area_cells = width_cells * depth_cells
                    candidates.append((-area_cells, move_distance, x_cell, y_cell, candidate))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [item[4] for item in candidates[:limit]]


def p0_pass_except_overlap(report: dict[str, Any]) -> bool:
    checks = report.get("p0", {}).get("checks", {})
    if not checks:
        return True
    for key, value in checks.items():
        if key == "no_volume_overlap":
            continue
        if value is False:
            return False
    return True


def repair_overlaps(
    house_id: str,
    decoded_rooms: list[dict[str, Any]],
    samples_by_group: dict[str, GroupSample],
    counts: dict[str, int],
    site_xy: tuple[float, float],
    target_topology: dict[str, Any],
    max_iterations: int = 8,
    max_move_mm: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rooms = [dict(room) for room in decoded_rooms]
    report = layout_report(house_id, rooms, counts, site_xy, target_topology)
    initial_report = report
    initial_pairs = overlap_pairs(rooms)
    accepted_moves = []
    rejected_candidate_count = 0

    for _iteration in range(max_iterations):
        current_pairs = overlap_pairs(rooms)
        if not current_pairs:
            break
        current_overlap_count = len(current_pairs)
        current_score = topology_score(report)
        best: tuple[int, tuple[int, int], float, float, list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None = None
        for pair in current_pairs:
            for moving_index in (int(pair["left_index"]), int(pair["right_index"])):
                moving_part = rooms[moving_index]
                sample = samples_by_group.get(room_functional_id(moving_part))
                if sample is None:
                    continue
                occupied = [room for index, room in enumerate(rooms) if index != moving_index]
                candidates = candidate_non_overlapping_moves(
                    sample,
                    moving_part,
                    occupied,
                    max_move_mm=max_move_mm,
                )
                if not candidates:
                    candidates = candidate_flexible_non_overlapping_moves(
                        sample,
                        moving_part,
                        occupied,
                        max_move_mm=max_move_mm,
                    )
                for candidate in candidates:
                    trial_rooms = [dict(room) for room in rooms]
                    candidate = dict(candidate)
                    candidate["constraint_adjustment"] = "overlap_repair_move"
                    trial_rooms[moving_index] = candidate
                    trial_report = layout_report(house_id, trial_rooms, counts, site_xy, target_topology)
                    if not p0_pass_except_overlap(trial_report):
                        rejected_candidate_count += 1
                        continue
                    trial_overlap_count = len(overlap_pairs(trial_rooms))
                    if trial_overlap_count >= current_overlap_count:
                        rejected_candidate_count += 1
                        continue
                    trial_score = topology_score(trial_report)
                    move_distance = abs(box_center_xy(candidate)[0] - box_center_xy(moving_part)[0]) + abs(
                        box_center_xy(candidate)[1] - box_center_xy(moving_part)[1]
                    )
                    area = (
                        float(candidate["box_max"][0]) - float(candidate["box_min"][0])
                    ) * (
                        float(candidate["box_max"][1]) - float(candidate["box_min"][1])
                    )
                    improvement = current_overlap_count - trial_overlap_count
                    best_key = (improvement, trial_score, area, -move_distance)
                    if best is None or best_key > (best[0], best[1], best[2], -best[3]):
                        best = (
                            improvement,
                            trial_score,
                            area,
                            move_distance,
                            trial_rooms,
                            trial_report,
                            {
                                "moved_part_id": str(moving_part.get("id", "")),
                                "moved_group": room_functional_id(moving_part),
                                "move_distance_mm": int(move_distance),
                                "overlap_count_before": current_overlap_count,
                                "overlap_count_after": trial_overlap_count,
                                "topology_score_before": list(current_score),
                                "topology_score_after": list(trial_score),
                            },
                        )
        if best is None:
            break
        rooms = best[4]
        report = best[5]
        accepted_moves.append(best[6])

    final_pairs = overlap_pairs(rooms)
    final_report = layout_report(house_id, rooms, counts, site_xy, target_topology)
    return rooms, final_report, {
        "enabled": True,
        "initial_p0_pass": bool(initial_report.get("p0", {}).get("pass", False)),
        "final_p0_pass": bool(final_report.get("p0", {}).get("pass", False)),
        "initial_overlap_count": len(initial_pairs),
        "final_overlap_count": len(final_pairs),
        "accepted_move_count": len(accepted_moves),
        "rejected_candidate_count": rejected_candidate_count,
        "initial_overlaps": initial_pairs,
        "final_overlaps": final_pairs,
        "moves": accepted_moves,
    }


def topology_placement_search(
    house_id: str,
    decoded_rooms: list[dict[str, Any]],
    samples_by_group: dict[str, GroupSample],
    counts: dict[str, int],
    site_xy: tuple[float, float],
    target_topology: dict[str, Any],
    max_iterations: int = 6,
    max_move_mm: float | None = 1800.0,
    expand_search_to_site: bool = False,
    enable_linked_part_placement: bool = False,
    enable_controlled_size_adjustment: bool = False,
    max_size_adjustment_mm: float = 600.0,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rooms = [dict(room) for room in decoded_rooms]
    report = layout_report(house_id, rooms, counts, site_xy, target_topology)
    initial_metrics = topology_metrics_from_report(report)
    if not report["p0"]["pass"]:
        return rooms, report, {
            "enabled": True,
            "skipped_reason": "initial_layout_failed_p0",
            "accepted_move_count": 0,
            "expand_search_to_site": expand_search_to_site,
            "initial_topology": initial_metrics,
            "final_topology": initial_metrics,
        }

    accepted_moves = []
    rejected_candidate_count = 0
    for _iteration in range(max_iterations):
        current_score = topology_score(report)
        best: tuple[tuple[int, int], int, int, list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None = None
        missing_edges = unrealized_edges(report)
        if not missing_edges:
            break
        for edge in missing_edges:
            source = str(edge["source"])
            target = str(edge["target"])
            for moving_group, fixed_group in ((source, target), (target, source)):
                sample = samples_by_group.get(moving_group)
                if sample is None:
                    continue
                moving_indices = [
                    index
                    for index, room in enumerate(rooms)
                    if room_functional_id(room) == moving_group
                ]
                fixed_parts = [room for room in rooms if room_functional_id(room) == fixed_group]
                if not moving_indices or not fixed_parts:
                    continue
                if enable_linked_part_placement and len(moving_indices) > 1:
                    moving_index_set = set(moving_indices)
                    occupied = [room for index, room in enumerate(rooms) if index not in moving_index_set]
                    moving_parts = [rooms[index] for index in moving_indices]
                    for moved_parts in candidate_linked_group_contact_moves(
                        sample,
                        moving_parts,
                        fixed_parts,
                        occupied,
                        max_move_mm=max_move_mm,
                        search_site_xy=site_xy if expand_search_to_site else None,
                    ):
                        trial_rooms = [dict(room) for room in rooms]
                        for index, moved_part in zip(moving_indices, moved_parts):
                            trial_rooms[index] = moved_part
                        trial_report = layout_report(house_id, trial_rooms, counts, site_xy, target_topology)
                        if not trial_report["p0"]["pass"]:
                            rejected_candidate_count += 1
                            continue
                        trial_score = topology_score(trial_report)
                        if trial_score <= current_score:
                            rejected_candidate_count += 1
                            continue
                        original_center = group_center_xy(moving_parts)
                        moved_center = group_center_xy(moved_parts)
                        move_distance = int(
                            abs(moved_center[0] - original_center[0])
                            + abs(moved_center[1] - original_center[1])
                        )
                        if best is None or (trial_score, 1, -move_distance) > (best[0], best[1], best[2]):
                            best = (
                                trial_score,
                                1,
                                -move_distance,
                                trial_rooms,
                                trial_report,
                                {
                                    "edge": [source, target],
                                    "move_type": "linked_part_group_move",
                                    "moved_group": moving_group,
                                    "fixed_group": fixed_group,
                                    "moved_part_ids": [str(room["id"]) for room in moving_parts],
                                    "move_distance_mm": move_distance,
                                    "score_before": list(current_score),
                                    "score_after": list(trial_score),
                                },
                            )
                for moving_index in moving_indices:
                    occupied = [room for index, room in enumerate(rooms) if index != moving_index]
                    move_candidates = candidate_contact_moves(
                        sample,
                        rooms[moving_index],
                        fixed_parts,
                        occupied,
                        max_move_mm=max_move_mm,
                        search_site_xy=site_xy if expand_search_to_site else None,
                    )
                    if enable_controlled_size_adjustment:
                        move_candidates.extend(
                            candidate_controlled_size_contact_moves(
                                sample,
                                rooms[moving_index],
                                fixed_parts,
                                occupied,
                                max_move_mm=max_move_mm,
                                max_size_adjustment_mm=max_size_adjustment_mm,
                                search_site_xy=site_xy if expand_search_to_site else None,
                            )
                        )
                    for candidate in move_candidates:
                        trial_rooms = [dict(room) for room in rooms]
                        trial_rooms[moving_index] = candidate
                        trial_report = layout_report(house_id, trial_rooms, counts, site_xy, target_topology)
                        if not trial_report["p0"]["pass"]:
                            rejected_candidate_count += 1
                            continue
                        trial_score = topology_score(trial_report)
                        if trial_score <= current_score:
                            rejected_candidate_count += 1
                            continue
                        move_distance = int(
                            abs(box_center_xy(candidate)[0] - box_center_xy(rooms[moving_index])[0])
                            + abs(box_center_xy(candidate)[1] - box_center_xy(rooms[moving_index])[1])
                        )
                        if best is None or (trial_score, 0, -move_distance) > (best[0], best[1], best[2]):
                            best = (
                                trial_score,
                                0,
                                -move_distance,
                                trial_rooms,
                                trial_report,
                                {
                                    "edge": [source, target],
                                    "move_type": str(candidate.get("constraint_adjustment", "single_part_move")),
                                    "moved_group": moving_group,
                                    "fixed_group": fixed_group,
                                    "moved_part_id": str(rooms[moving_index]["id"]),
                                    "move_distance_mm": move_distance,
                                    "score_before": list(current_score),
                                    "score_after": list(trial_score),
                                },
                            )
        if best is None:
            break
        rooms = best[3]
        report = best[4]
        accepted_moves.append(best[5])

    final_metrics = topology_metrics_from_report(report)
    return rooms, report, {
        "enabled": True,
        "expand_search_to_site": expand_search_to_site,
        "linked_part_placement_enabled": enable_linked_part_placement,
        "controlled_size_adjustment_enabled": enable_controlled_size_adjustment,
        "max_size_adjustment_mm": max_size_adjustment_mm,
        "accepted_move_count": len(accepted_moves),
        "rejected_candidate_count": rejected_candidate_count,
        "initial_topology": initial_metrics,
        "final_topology": final_metrics,
        "moves": accepted_moves,
    }


def export_predicted_houses(
    model: MultiPartDecoder,
    source_paths: list[Path],
    output_dir: Path,
    max_parts: int,
    device: torch.device,
    topology_conditioning: dict[str, dict[str, Any]] | None = None,
    enable_topology_placement_search: bool = False,
    enable_overlap_repair: bool = False,
    max_topology_move_mm: float | None = 1800.0,
    expand_topology_search_to_site: bool = False,
    enable_linked_part_placement: bool = False,
    enable_controlled_size_adjustment: bool = False,
    max_size_adjustment_mm: float = 600.0,
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    reports = []
    samples_by_house: dict[str, list[GroupSample]] = {}
    for sample in load_group_samples_from_paths(source_paths):
        samples_by_house.setdefault(sample.house_id, []).append(sample)
    with torch.no_grad():
        for path in source_paths:
            source = read_json(path)
            house_id = str(source["house_id"])
            house_dir = output_dir / "predicted_json" / house_id
            evaluation_path = house_dir / "evaluation.json"
            if skip_existing and evaluation_path.exists():
                report = read_json(evaluation_path)
                placement_path = house_dir / "topology_placement_search.json"
                overlap_path = house_dir / "overlap_repair.json"
                placement_search = read_json(placement_path) if placement_path.exists() else {}
                overlap_repair = read_json(overlap_path) if overlap_path.exists() else {}
                layout_path = house_dir / "generated_layout.json"
                decoded_rooms = read_json(layout_path).get("rooms", []) if layout_path.exists() else []
                source_topology = house_dir / "conditioning_topology.json"
                conditioning_topology = read_json(source_topology) if source_topology.exists() else {}
                counts = {
                    room_type: 0
                    for room_type in ROOM_TYPES
                }
                for group in source.get("functional_groups", []):
                    counts[str(group["type"])] = counts.get(str(group["type"]), 0) + 1
                counts = {key: value for key, value in counts.items() if value > 0}
                adjusted_parts = sum(
                    1
                    for room in decoded_rooms
                    if room.get("constraint_adjustment") or room.get("constraint_error")
                )
                reports.append(
                    {
                        "house_id": house_id,
                        "group_count": sum(counts.values()),
                        "rectangular_part_count": len(decoded_rooms),
                        "constraint_adjusted_part_count": adjusted_parts,
                        "conditioning_topology_source": conditioning_topology.get("source"),
                        "conditioning_edge_count": len(conditioning_topology.get("edges", [])),
                        "p0_pass": report["p0"]["pass"],
                        "p1_hard_geometry_pass": report["p1_spatial_organization"]["hard_geometry_pass"],
                        "p1_spatial_organization_pass": report["p1_spatial_organization"]["spatial_organization_pass"],
                        "topology": topology_metrics_from_report(report),
                        "overlap_repair": overlap_repair,
                        "topology_placement_search": placement_search,
                        "p2_enabled": report["p2"].get("enabled", True),
                        "reused_existing": True,
                    }
                )
                continue
            decoded_rooms = []
            occupied_rooms: list[dict[str, Any]] = []
            target_topology = build_target_topology(source)
            conditioning_topology = (topology_conditioning or {}).get(house_id, target_topology)
            neighbors = group_neighbor_map(conditioning_topology)
            house_samples_by_group = {
                sample.group_id: sample
                for sample in samples_by_house.get(house_id, [])
            }
            for sample in samples_by_house.get(house_id, []):
                features = feature_vector(sample, max_parts).unsqueeze(0).to(device)
                pred = model(features)[0]
                decoded_rooms.extend(
                    decode_parts(
                        sample,
                        pred,
                        max_parts,
                        occupied=occupied_rooms,
                        target_neighbors=neighbors.get(sample.group_id, set()),
                    )
                )
            site = source["metadata"]["building_size"]
            counts = {
                room_type: 0
                for room_type in ROOM_TYPES
            }
            for group in source.get("functional_groups", []):
                counts[str(group["type"])] = counts.get(str(group["type"]), 0) + 1
            counts = {key: value for key, value in counts.items() if value > 0}
            candidate = {
                "house_id": house_id,
                "metadata": source.get("metadata", {}),
                "rooms": decoded_rooms,
            }
            site_xy = (float(site["x"]), float(site["y"]))
            report = layout_report(house_id, decoded_rooms, counts, site_xy, target_topology)
            overlap_repair = {
                "enabled": False,
                "initial_p0_pass": bool(report.get("p0", {}).get("pass", False)),
                "final_p0_pass": bool(report.get("p0", {}).get("pass", False)),
                "initial_overlap_count": len(overlap_pairs(decoded_rooms)),
                "final_overlap_count": len(overlap_pairs(decoded_rooms)),
                "accepted_move_count": 0,
            }
            if enable_overlap_repair:
                decoded_rooms, report, overlap_repair = repair_overlaps(
                    house_id,
                    decoded_rooms,
                    house_samples_by_group,
                    counts,
                    site_xy,
                    target_topology,
                )
                candidate["rooms"] = decoded_rooms
            placement_search = {
                "enabled": False,
                "initial_topology": topology_metrics_from_report(report),
                "final_topology": topology_metrics_from_report(report),
                "accepted_move_count": 0,
                "expand_search_to_site": expand_topology_search_to_site,
                "linked_part_placement_enabled": enable_linked_part_placement,
                "controlled_size_adjustment_enabled": enable_controlled_size_adjustment,
                "max_size_adjustment_mm": max_size_adjustment_mm,
            }
            if enable_topology_placement_search:
                decoded_rooms, report, placement_search = topology_placement_search(
                    house_id,
                    decoded_rooms,
                    house_samples_by_group,
                    counts,
                    site_xy,
                    target_topology,
                    max_move_mm=max_topology_move_mm,
                    expand_search_to_site=expand_topology_search_to_site,
                    enable_linked_part_placement=enable_linked_part_placement,
                    enable_controlled_size_adjustment=enable_controlled_size_adjustment,
                    max_size_adjustment_mm=max_size_adjustment_mm,
                )
                candidate["rooms"] = decoded_rooms
            write_json(house_dir / "generated_layout.json", candidate)
            write_json(house_dir / "topology.json", target_topology)
            write_json(house_dir / "conditioning_topology.json", conditioning_topology)
            write_json(house_dir / "evaluation.json", report)
            write_json(house_dir / "overlap_repair.json", overlap_repair)
            write_json(house_dir / "topology_placement_search.json", placement_search)
            adjusted_parts = sum(
                1
                for room in decoded_rooms
                if room.get("constraint_adjustment") or room.get("constraint_error")
            )
            reports.append(
                {
                    "house_id": house_id,
                    "group_count": sum(counts.values()),
                    "rectangular_part_count": len(decoded_rooms),
                    "constraint_adjusted_part_count": adjusted_parts,
                    "conditioning_topology_source": conditioning_topology.get("source"),
                    "conditioning_edge_count": len(conditioning_topology.get("edges", [])),
                    "p0_pass": report["p0"]["pass"],
                    "p1_hard_geometry_pass": report["p1_spatial_organization"]["hard_geometry_pass"],
                    "p1_spatial_organization_pass": report["p1_spatial_organization"]["spatial_organization_pass"],
                    "topology": topology_metrics_from_report(report),
                    "overlap_repair": overlap_repair,
                    "topology_placement_search": placement_search,
                    "p2_enabled": report["p2"].get("enabled", True),
                }
            )
    return reports


def export_target_houses(source_paths: list[Path], output_dir: Path) -> list[dict[str, Any]]:
    """Export Phase10 target rooms through the same standard JSON evaluator."""
    reports = []
    for path in source_paths:
        source = read_json(path)
        house_id = str(source["house_id"])
        rooms = []
        for room in source.get("rooms", []):
            item = dict(room)
            item.setdefault("functional_id", item.get("id"))
            item.setdefault("floors", room_floors(item))
            item.setdefault("floor", min(item["floors"]))
            rooms.append(item)
        site = source["metadata"]["building_size"]
        topology = build_target_topology(source)
        counts = {}
        for group in source.get("functional_groups", []):
            room_type = str(group["type"])
            counts[room_type] = counts.get(room_type, 0) + 1
        candidate = {
            "house_id": house_id,
            "metadata": source.get("metadata", {}),
            "rooms": rooms,
        }
        report, _ = evaluate_candidate(
            house_id,
            rooms,
            counts,
            (float(site["x"]), float(site["y"])),
            topology=topology,
        )
        house_dir = output_dir / "target_json" / house_id
        write_json(house_dir / "generated_layout.json", candidate)
        write_json(house_dir / "topology.json", topology)
        write_json(house_dir / "evaluation.json", report)
        reports.append(
            {
                "house_id": house_id,
                "group_count": sum(counts.values()),
                "rectangular_part_count": len(rooms),
                "p0_pass": report["p0"]["pass"],
                "p1_hard_geometry_pass": report["p1_spatial_organization"]["hard_geometry_pass"],
                "p1_spatial_organization_pass": report["p1_spatial_organization"]["spatial_organization_pass"],
                "topology": topology_metrics_from_report(report),
                "p2_enabled": report["p2"].get("enabled", True),
            }
        )
    return reports


def load_group_samples_from_paths(paths: list[Path]) -> list[GroupSample]:
    temp_dir = paths[0].parent if paths else DEFAULT_PHASE10
    all_samples = load_group_samples(temp_dir, max_houses=None)
    wanted = {read_json(path)["house_id"] for path in paths}
    return [sample for sample in all_samples if sample.house_id in wanted]


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source_paths = sorted(Path(args.phase10_dir).glob("house_*.json"))[: args.max_houses]
    samples = load_group_samples_from_paths(source_paths)
    if not samples:
        raise ValueError("no Phase10 functional group samples found")
    if any(len(sample.target_parts) > args.max_parts for sample in samples):
        too_large = max(len(sample.target_parts) for sample in samples)
        raise ValueError(f"max-parts={args.max_parts} is too small; observed {too_large}")
    dataset = FunctionalPartDataset(samples, args.max_parts)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    feature_dim = int(dataset[0]["features"].numel())
    model = MultiPartDecoder(feature_dim, args.max_parts).to(device)
    if args.export_only:
        if args.checkpoint is None:
            raise ValueError("--export-only requires --checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        checkpoint_config = checkpoint.get("config", {})
        if int(checkpoint_config.get("feature_dim", feature_dim)) != feature_dim:
            raise ValueError("checkpoint feature_dim does not match current samples")
        if int(checkpoint_config.get("max_parts", args.max_parts)) != args.max_parts:
            raise ValueError("checkpoint max_parts does not match --max-parts")
        model.load_state_dict(checkpoint["model"])
        optimizer = None
        history = []
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        history = []
    if not args.export_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_batches = 0
            for batch in loader:
                features = batch["features"].to(device)
                target = batch["target"].to(device)
                mask = batch["mask"].to(device)
                pred = model(features)
                loss = masked_mse(pred, target, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                total_batches += 1
            if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
                metrics = evaluate_predictions(model, samples, args.max_parts, device)
                history.append(
                    {
                        "epoch": epoch,
                        "loss": total_loss / max(total_batches, 1),
                        **metrics,
                    }
                )
                print(
                    f"epoch={epoch:04d} loss={history[-1]['loss']:.6f} "
                    f"mean_iou={metrics['mean_part_iou']:.4f} "
                    f"exact_snap={metrics['exact_snapped_part_rate']:.3f}"
                )
    final_metrics = evaluate_predictions(model, samples, args.max_parts, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topology_conditioning = read_conditioning_topologies(args.topology_conditioning_dir)
    max_topology_move_mm = None if args.max_topology_move_mm < 0 else float(args.max_topology_move_mm)
    if not args.export_only:
        torch.save(
            {
                "schema": "graphspace_v6_multipart_decoder_smoke_v1",
                "model": model.state_dict(),
                "config": {
                    "feature_dim": feature_dim,
                    "max_parts": args.max_parts,
                    "room_types": ROOM_TYPES,
                },
                "source_phase10_dir": str(args.phase10_dir),
            },
            args.output_dir / "decoder.pt",
        )
    house_reports = export_predicted_houses(
        model,
        source_paths,
        args.output_dir,
        args.max_parts,
        device,
        topology_conditioning=topology_conditioning,
        enable_topology_placement_search=args.enable_topology_placement_search,
        enable_overlap_repair=args.enable_overlap_repair,
        max_topology_move_mm=max_topology_move_mm,
        expand_topology_search_to_site=args.expand_topology_search_to_site,
        enable_linked_part_placement=args.enable_linked_part_placement,
        enable_controlled_size_adjustment=args.enable_controlled_size_adjustment,
        max_size_adjustment_mm=float(args.max_size_adjustment_mm),
        skip_existing=bool(args.skip_existing),
    )
    target_house_reports = export_target_houses(source_paths, args.output_dir)
    summary = {
        "schema": "graphspace_v6_multipart_smoke_summary_v1",
        "purpose": (
            "Interface validation only: learned functional-group to multi-part "
            "decoder overfits Phase10 inferred groups; not a formal V6 decoder."
        ),
        "phase10_dir": str(args.phase10_dir),
        "house_count": len(source_paths),
        "group_count": len(samples),
        "part_count": sum(len(sample.target_parts) for sample in samples),
        "max_parts": args.max_parts,
        "epochs": args.epochs,
        "topology_conditioning_dir": (
            str(args.topology_conditioning_dir) if args.topology_conditioning_dir else None
        ),
        "topology_conditioned_house_count": len(topology_conditioning),
        "topology_placement_search_enabled": bool(args.enable_topology_placement_search),
        "overlap_repair_enabled": bool(args.enable_overlap_repair),
        "max_topology_move_mm": max_topology_move_mm,
        "expand_topology_search_to_site": bool(args.expand_topology_search_to_site),
        "linked_part_placement_enabled": bool(args.enable_linked_part_placement),
        "controlled_size_adjustment_enabled": bool(args.enable_controlled_size_adjustment),
        "max_size_adjustment_mm": float(args.max_size_adjustment_mm),
        "final_metrics": final_metrics,
        "predicted_house_reports": house_reports,
        "target_house_reports": target_house_reports,
        "history": history,
        "outputs": {
            "checkpoint": str(args.output_dir / "decoder.pt"),
            "predicted_json": str(args.output_dir / "predicted_json"),
            "target_json": str(args.output_dir / "target_json"),
        },
        "formal_v6_training_ready": False,
        "blocking_reason": (
            "Phase10 functional groups are inferred labels and this decoder "
            "is trained only as a small overfit protocol check."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
