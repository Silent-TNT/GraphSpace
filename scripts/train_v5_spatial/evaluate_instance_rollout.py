#!/usr/bin/env python3
"""Autoregressively place individual room boxes on the 300 mm grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from instance_dataset import (
    DEFAULT_DATA_DIR,
    InstancePlacementDataset,
    collate_instances,
    final_coarse_volume,
    read_json,
    room_floors,
)
from instance_model import InstancePlacementPolicy
from staged_dataset import (
    CHANNEL_INDEX,
    StagedSpatialDataset,
    collate_staged,
    load_volumes,
    reduce_2d,
    stage_input,
)
from staged_model import StagedSpatialPolicy


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
VOXEL_MM = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-houses", type=int, default=2)
    parser.add_argument("--staged-checkpoint", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    return parser.parse_args()


def occupied_volume(occupied: np.ndarray) -> np.ndarray:
    output = np.zeros((44, 44, 20), dtype=np.float32)
    for floor in range(2):
        reduced = reduce_2d(occupied[floor])
        output[:, :, floor * 10 : (floor + 1) * 10] = reduced[:, :, None]
    return output


def predict_coarse_volume(
    house_id: str,
    dataset: StagedSpatialDataset,
    model: StagedSpatialPolicy,
    device: torch.device,
) -> np.ndarray:
    sample_path = DEFAULT_DATA_DIR / f"{house_id}.json"
    volumes = load_volumes(sample_path)
    state = stage_input(volumes, 0)
    for stage_id in range(6):
        item_index = dataset.items.index((house_id, stage_id))
        item = dataset[item_index]
        item["volume"] = torch.from_numpy(state.copy())
        batch = collate_staged([item])
        output = model(
            batch["volume"].to(device),
            batch["nodes"].to(device),
            batch["node_mask"].to(device),
            batch["adjacency"].to(device),
            batch["stage_id"].to(device),
        )
        probability = torch.sigmoid(
            output["mask_logits"][0]
        ).float().cpu().numpy()
        prediction = (probability >= 0.5).astype(np.float32)
        if stage_id == 0:
            state[CHANNEL_INDEX["protected_stairs"]] = prediction[0]
        elif stage_id == 1:
            cut_cell = min(
                max(round(float(output["cut_ratio"][0].cpu()) * 20), 1),
                19,
            )
            state[CHANNEL_INDEX["floor_boundary"], :, :, cut_cell] = 1
        elif stage_id == 2:
            site = state[CHANNEL_INDEX["site"]] > 0
            stairs = state[CHANNEL_INDEX["protected_stairs"]] > 0
            building = (probability[0] >= probability[1]) & site
            empty = (probability[1] > probability[0]) & site
            building |= stairs
            empty &= ~stairs
            state[CHANNEL_INDEX["building_envelope"]] = building
            state[CHANNEL_INDEX["explicit_empty"]] = empty
        elif stage_id == 3:
            building = state[CHANNEL_INDEX["building_envelope"]] > 0
            stairs = state[CHANNEL_INDEX["protected_stairs"]] > 0
            state[CHANNEL_INDEX["traffic_reserve"]] = (
                ((prediction[0] > 0) & building) | stairs
            )
        elif stage_id == 4:
            building = state[CHANNEL_INDEX["building_envelope"]] > 0
            traffic = state[CHANNEL_INDEX["traffic_reserve"]] > 0
            state[CHANNEL_INDEX["rigid_functions"]] = (
                (prediction[0] > 0) & building & ~traffic
            )
        elif stage_id == 5:
            building = state[CHANNEL_INDEX["building_envelope"]] > 0
            traffic = state[CHANNEL_INDEX["traffic_reserve"]] > 0
            rigid = state[CHANNEL_INDEX["rigid_functions"]] > 0
            state[CHANNEL_INDEX["service_spaces"]] = (
                (prediction[0] > 0) & building & ~traffic & ~rigid
            )
    return state


def building_from_coarse(
    state: np.ndarray,
    site_cells: list[int],
) -> np.ndarray:
    building = state[CHANNEL_INDEX["building_envelope"]] > 0
    output = np.zeros((2, site_cells[0], site_cells[1]), dtype=bool)
    for floor in range(2):
        floor_mask = building[:, :, floor * 10 : (floor + 1) * 10].any(axis=2)
        expanded = np.repeat(np.repeat(floor_mask, 2, axis=0), 2, axis=1)
        output[floor] = expanded[: site_cells[0], : site_cells[1]]
    return output


def decode_desired_box(
    prediction: np.ndarray,
    site_cells: list[int],
) -> tuple[float, float, int, int]:
    center_x = float(prediction[0]) * site_cells[0]
    center_y = float(prediction[1]) * site_cells[1]
    width = max(1, int(round(float(prediction[2]) * site_cells[0])))
    depth = max(1, int(round(float(prediction[3]) * site_cells[1])))
    return center_x, center_y, width, depth


def candidate_valid(
    box: tuple[int, int, int, int],
    floors: list[int],
    building: np.ndarray,
    occupied: np.ndarray,
) -> bool:
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > building.shape[1] or y1 > building.shape[2]:
        return False
    if x1 <= x0 or y1 <= y0:
        return False
    for floor in floors:
        region = (slice(x0, x1), slice(y0, y1))
        if not building[floor - 1][region].all():
            return False
        if occupied[floor - 1][region].any():
            return False
    return True


def place_nearest_box(
    prediction: np.ndarray,
    site_cells: list[int],
    floors: list[int],
    building: np.ndarray,
    occupied: np.ndarray,
) -> tuple[int, int, int, int] | None:
    center_x, center_y, width, depth = decode_desired_box(
        prediction,
        site_cells,
    )
    widths = sorted(
        {
            max(1, min(site_cells[0], width + delta))
            for delta in range(-4, 5)
        }
    )
    depths = sorted(
        {
            max(1, min(site_cells[1], depth + delta))
            for delta in range(-4, 5)
        }
    )
    best = None
    best_score = float("inf")
    for candidate_width in widths:
        for candidate_depth in depths:
            preferred_x = int(round(center_x - candidate_width / 2))
            preferred_y = int(round(center_y - candidate_depth / 2))
            x_values = sorted(
                range(max(site_cells[0] - candidate_width + 1, 0)),
                key=lambda value: abs(value - preferred_x),
            )
            y_values = sorted(
                range(max(site_cells[1] - candidate_depth + 1, 0)),
                key=lambda value: abs(value - preferred_y),
            )
            for x0 in x_values:
                for y0 in y_values:
                    box = (
                        x0,
                        y0,
                        x0 + candidate_width,
                        y0 + candidate_depth,
                    )
                    if not candidate_valid(
                        box,
                        floors,
                        building,
                        occupied,
                    ):
                        continue
                    box_center_x = x0 + candidate_width / 2
                    box_center_y = y0 + candidate_depth / 2
                    score = (
                        abs(box_center_x - center_x) / max(site_cells[0], 1)
                        + abs(box_center_y - center_y) / max(site_cells[1], 1)
                        + 0.5 * abs(candidate_width - width) / max(site_cells[0], 1)
                        + 0.5 * abs(candidate_depth - depth) / max(site_cells[1], 1)
                    )
                    if score < best_score:
                        best, best_score = box, score
                    if score == 0:
                        return best
                if best is not None and abs(x0 - preferred_x) > 4:
                    break
    return best


def box_face_contact(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    """Return normalized shared face length for two 2D grid boxes."""
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    if lx1 == rx0 or rx1 == lx0:
        overlap = max(0, min(ly1, ry1) - max(ly0, ry0))
        return overlap / max(1, min(ly1 - ly0, ry1 - ry0))
    if ly1 == ry0 or ry1 == ly0:
        overlap = max(0, min(lx1, rx1) - max(lx0, rx0))
        return overlap / max(1, min(lx1 - lx0, rx1 - rx0))
    return 0.0


def box_projection_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    ix = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    iy = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    overlap = ix * iy
    smaller = min(
        (left[2] - left[0]) * (left[3] - left[1]),
        (right[2] - right[0]) * (right[3] - right[1]),
    )
    return overlap / max(smaller, 1)


def box_gap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    dx = max(right[0] - left[2], left[0] - right[2], 0)
    dy = max(right[1] - left[3], left[1] - right[3], 0)
    return dx + dy


def place_topology_box(
    prediction: np.ndarray,
    site_cells: list[int],
    floors: list[int],
    building: np.ndarray,
    occupied: np.ndarray,
    placed_boxes: dict[int, tuple[int, int, int, int]],
    placed_floors: dict[int, list[int]],
    topology_neighbors: list[tuple[int, int, bool]],
    needs_exterior: bool = False,
) -> tuple[tuple[int, int, int, int] | None, dict]:
    """Project a model proposal to legal geometry while realizing graph edges."""
    center_x, center_y, width, depth = decode_desired_box(prediction, site_cells)
    widths = sorted(
        {
            max(1, min(site_cells[0], width + delta))
            for delta in range(-4, 5)
        }
    )
    depths = sorted(
        {
            max(1, min(site_cells[1], depth + delta))
            for delta in range(-4, 5)
        }
    )
    best = None
    best_score = float("inf")
    best_details = {}
    for candidate_width in widths:
        for candidate_depth in depths:
            for x0 in range(site_cells[0] - candidate_width + 1):
                for y0 in range(site_cells[1] - candidate_depth + 1):
                    box = (x0, y0, x0 + candidate_width, y0 + candidate_depth)
                    if not candidate_valid(box, floors, building, occupied):
                        continue
                    box_center_x = x0 + candidate_width / 2
                    box_center_y = y0 + candidate_depth / 2
                    model_cost = (
                        abs(box_center_x - center_x) / max(site_cells[0], 1)
                        + abs(box_center_y - center_y) / max(site_cells[1], 1)
                        + 0.5
                        * abs(candidate_width - width)
                        / max(site_cells[0], 1)
                        + 0.5
                        * abs(candidate_depth - depth)
                        / max(site_cells[1], 1)
                    )
                    topology_cost = 0.0
                    realized = 0
                    considered = 0
                    for neighbor_index, relation, required in topology_neighbors:
                        neighbor_box = placed_boxes.get(neighbor_index)
                        if neighbor_box is None:
                            continue
                        considered += 1
                        if relation == 1:
                            quality = box_projection_overlap(box, neighbor_box)
                        elif set(floors) & set(placed_floors[neighbor_index]):
                            quality = box_face_contact(box, neighbor_box)
                        else:
                            quality = 0.0
                        if quality > 0:
                            realized += 1
                            weight = 3.0 if required else 1.0
                            topology_cost -= weight * (2.8 + 1.2 * quality)
                        else:
                            weight = 3.0 if required else 1.0
                            topology_cost += weight * (
                                2.2 + 0.08 * box_gap(box, neighbor_box)
                            )
                    exterior = (
                        x0 == 0
                        or y0 == 0
                        or box[2] == site_cells[0]
                        or box[3] == site_cells[1]
                    )
                    exterior_cost = -0.5 if needs_exterior and exterior else 0.35 if needs_exterior else 0.0
                    score = model_cost + topology_cost + exterior_cost
                    if score < best_score:
                        best = box
                        best_score = score
                        best_details = {
                            "score": score,
                            "model_cost": model_cost,
                            "topology_cost": topology_cost,
                            "placed_topology_neighbors": considered,
                            "realized_topology_neighbors": realized,
                            "touches_exterior": exterior,
                        }
    return best, best_details


def mark_occupied(
    occupied: np.ndarray,
    box: tuple[int, int, int, int],
    floors: list[int],
) -> None:
    x0, y0, x1, y1 = box
    for floor in floors:
        occupied[floor - 1, x0:x1, y0:y1] = 1


def truth_box(room: dict) -> tuple[int, int, int, int]:
    return (
        int(round(float(room["box_min"][0]) / VOXEL_MM)),
        int(round(float(room["box_min"][1]) / VOXEL_MM)),
        int(round(float(room["box_max"][0]) / VOXEL_MM)),
        int(round(float(room["box_max"][1]) / VOXEL_MM)),
    )


def box_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    ix = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    iy = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = ix * iy
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 1.0


def standard_room(
    index: int,
    node: dict,
    box: tuple[int, int, int, int],
) -> dict:
    floors = []
    if node["floor_1"]:
        floors.append(1)
    if node["floor_2"]:
        floors.append(2)
    x0, y0, x1, y1 = box
    return {
        "id": f"instance_{index:03d}",
        "type": node["type"],
        "floor": floors[0],
        "floors": floors,
        "box_min": [
            x0 * VOXEL_MM,
            y0 * VOXEL_MM,
            (floors[0] - 1) * 3000.0,
        ],
        "box_max": [
            x1 * VOXEL_MM,
            y1 * VOXEL_MM,
            floors[-1] * 3000.0,
        ],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = InstancePlacementPolicy(
        int(checkpoint["config"]["base_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = InstancePlacementDataset(
        args.split,
        max_houses=args.max_houses,
    )
    reports = []
    staged_dataset = None
    staged_model = None
    if args.staged_checkpoint:
        staged_checkpoint = torch.load(
            args.staged_checkpoint,
            map_location=device,
        )
        staged_model = StagedSpatialPolicy(
            int(staged_checkpoint["config"]["base_channels"])
        ).to(device)
        staged_model.load_state_dict(staged_checkpoint["model"])
        staged_model.eval()
        staged_dataset = StagedSpatialDataset(
            args.split,
            max_houses=args.max_houses,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for house_id in dataset.house_ids:
            sample_path = DEFAULT_DATA_DIR / f"{house_id}.json"
            staged = read_json(sample_path)
            processed = read_json(PROCESSED_DIR / f"{house_id}.json")
            rooms = processed["rooms"]
            order = dataset.orders[house_id]
            if staged_model is None:
                with np.load(sample_path.with_suffix(".npz")) as arrays:
                    building = arrays["building_mask"].copy().astype(bool)
                volume = final_coarse_volume(sample_path)
            else:
                volume = predict_coarse_volume(
                    house_id,
                    staged_dataset,
                    staged_model,
                    device,
                )
                volume = np.concatenate(
                    (
                        volume,
                        np.zeros((1, 44, 44, 20), dtype=np.float32),
                    ),
                    axis=0,
                )
                building = building_from_coarse(
                    volume,
                    staged["site_cells"][:2],
                )
            occupied = np.zeros_like(building, dtype=np.uint8)
            predictions: dict[int, tuple[int, int, int, int]] = {}
            for offset, room_index in enumerate(order):
                dataset_index = dataset.items.index((house_id, offset))
                item = dataset[dataset_index]
                volume[8] = occupied_volume(occupied)
                item["volume"] = torch.from_numpy(volume.copy())
                batch = collate_instances([item])
                output = model(
                    batch["volume"].to(device),
                    batch["nodes"].to(device),
                    batch["node_mask"].to(device),
                    batch["adjacency"].to(device),
                    batch["room_index"].to(device),
                    batch["step_ratio"].to(device),
                )
                box = place_nearest_box(
                    output["box"][0].float().cpu().numpy(),
                    staged["site_cells"][:2],
                    room_floors(rooms[room_index]),
                    building,
                    occupied,
                )
                if box is None:
                    continue
                predictions[room_index] = box
                mark_occupied(
                    occupied,
                    box,
                    room_floors(rooms[room_index]),
                )
            ious = [
                box_iou(predictions[index], truth_box(rooms[index]))
                for index in sorted(predictions)
            ]
            predicted_rooms = [
                standard_room(index, staged["graph"]["nodes"][index], box)
                for index, box in sorted(predictions.items())
            ]
            candidate = {
                "house_id": f"{house_id}_instance_rollout",
                "metadata": processed["metadata"],
                "rooms": predicted_rooms,
            }
            (args.output_dir / f"{house_id}.json").write_text(
                json.dumps(candidate, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reports.append(
                {
                    "house_id": house_id,
                    "expected_count": len(rooms),
                    "placed_count": len(predictions),
                    "count_exact": len(predictions) == len(rooms),
                    "mean_box_iou": sum(ious) / max(len(ious), 1),
                    "iou_50_recall": sum(value >= 0.5 for value in ious)
                    / max(len(rooms), 1),
                    "exact_box_count": sum(
                        predictions[index] == truth_box(rooms[index])
                        for index in predictions
                    ),
                    "cross_floor_expected": sum(
                        len(room_floors(room)) == 2 for room in rooms
                    ),
                    "cross_floor_placed": sum(
                        len(room_floors(rooms[index])) == 2
                        for index in predictions
                    ),
                    "overlap_cell_count": 0,
                    "outside_cell_count": 0,
                }
            )
    summary = {
        "checkpoint": str(args.checkpoint),
        "staged_checkpoint": (
            str(args.staged_checkpoint) if args.staged_checkpoint else None
        ),
        "house_count": len(reports),
        "count_exact_count": sum(item["count_exact"] for item in reports),
        "mean_box_iou": sum(item["mean_box_iou"] for item in reports)
        / max(len(reports), 1),
        "mean_iou_50_recall": sum(item["iou_50_recall"] for item in reports)
        / max(len(reports), 1),
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
