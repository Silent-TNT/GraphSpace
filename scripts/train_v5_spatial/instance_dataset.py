"""Autoregressive room-instance placement dataset for V5."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from staged_dataset import (
    DEFAULT_DATA_DIR,
    GRID_SHAPE,
    NODE_DIM,
    VOLUME_CHANNELS,
    graph_arrays,
    load_volumes,
    split_ids,
)
TYPE_TO_ID = {
    "entryway": 0,
    "living_room": 1,
    "dining_room": 2,
    "kitchen": 3,
    "bedroom": 4,
    "bathroom": 5,
    "corridor": 6,
    "stairs": 7,
    "utility": 8,
    "balcony": 9,
    "multi_purpose": 10,
}


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
INSTANCE_VOLUME_CHANNELS = VOLUME_CHANNELS + ["placed_instances"]
TRAFFIC_ORDER = {"stairs": 0, "entryway": 1, "corridor": 2}
RIGID_ORDER = {
    "living_room": 0,
    "dining_room": 1,
    "kitchen": 2,
    "bedroom": 3,
    "bathroom": 4,
    "multi_purpose": 5,
}
SERVICE_ORDER = {"utility": 0, "balcony": 1}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def room_floors(room: dict) -> list[int]:
    if room.get("floors"):
        return sorted(int(value) for value in room["floors"])
    return [int(room.get("floor", 1))]


def room_stage(room_type: str) -> tuple[int, int]:
    if room_type in TRAFFIC_ORDER:
        return 0, TRAFFIC_ORDER[room_type]
    if room_type in RIGID_ORDER:
        return 1, RIGID_ORDER[room_type]
    return 2, SERVICE_ORDER.get(room_type, 99)


def placement_order(rooms: list[dict]) -> list[int]:
    return sorted(
        range(len(rooms)),
        key=lambda index: (
            *room_stage(str(rooms[index]["type"])),
            min(room_floors(rooms[index])),
            -(
                (float(rooms[index]["box_max"][0]) - float(rooms[index]["box_min"][0]))
                * (
                    float(rooms[index]["box_max"][1])
                    - float(rooms[index]["box_min"][1])
                )
            ),
            index,
        ),
    )


def final_coarse_volume(sample_path: Path) -> np.ndarray:
    volumes = load_volumes(sample_path)
    state = np.zeros(
        (len(INSTANCE_VOLUME_CHANNELS), *GRID_SHAPE),
        dtype=np.float32,
    )
    state[0] = volumes["site"]
    state[1] = volumes["building"]
    state[2] = volumes["stairs"]
    state[3] = volumes["empty"]
    state[4] = volumes["traffic"]
    state[5] = volumes["rigid"]
    state[6] = volumes["service"]
    state[7] = volumes["floor"]
    return state


def rasterize_rooms(
    rooms: list[dict],
    room_indices: list[int],
) -> np.ndarray:
    occupied = np.zeros(GRID_SHAPE, dtype=np.float32)
    for index in room_indices:
        room = rooms[index]
        x0 = int(round(float(room["box_min"][0]) / 600.0))
        y0 = int(round(float(room["box_min"][1]) / 600.0))
        x1 = int(np.ceil(float(room["box_max"][0]) / 600.0))
        y1 = int(np.ceil(float(room["box_max"][1]) / 600.0))
        floors = room_floors(room)
        z0, z1 = (min(floors) - 1) * 10, max(floors) * 10
        occupied[x0:x1, y0:y1, z0:z1] = 1.0
    return occupied


def target_box(room: dict, site_cells: list[int]) -> np.ndarray:
    x0 = float(room["box_min"][0]) / 300.0
    y0 = float(room["box_min"][1]) / 300.0
    x1 = float(room["box_max"][0]) / 300.0
    y1 = float(room["box_max"][1]) / 300.0
    return np.asarray(
        [
            ((x0 + x1) * 0.5) / site_cells[0],
            ((y0 + y1) * 0.5) / site_cells[1],
            (x1 - x0) / site_cells[0],
            (y1 - y0) / site_cells[1],
        ],
        dtype=np.float32,
    )


class InstancePlacementDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_dir: Path = DEFAULT_DATA_DIR,
        processed_dir: Path = PROCESSED_DIR,
        split_path: Path = DEFAULT_SPLIT_PATH,
        max_houses: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        ids = split_ids(Path(split_path), split)
        if max_houses is not None:
            ids = ids[:max_houses]
        self.house_ids = ids
        self.items = []
        self.item_type_ids = []
        self.orders: dict[str, list[int]] = {}
        for house_id in ids:
            rooms = read_json(self.processed_dir / f"{house_id}.json")["rooms"]
            order = placement_order(rooms)
            self.orders[house_id] = order
            for offset, room_index in enumerate(order):
                self.items.append((house_id, offset))
                self.item_type_ids.append(
                    TYPE_TO_ID[str(rooms[room_index]["type"])]
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        house_id, offset = self.items[index]
        sample_path = self.data_dir / f"{house_id}.json"
        staged = read_json(sample_path)
        processed = read_json(self.processed_dir / f"{house_id}.json")
        rooms = processed["rooms"]
        order = self.orders[house_id]
        room_index = order[offset]
        volume = final_coarse_volume(sample_path)
        volume[8] = rasterize_rooms(rooms, order[:offset])
        nodes, edge_index, edge_type = graph_arrays(staged["graph"])
        return {
            "house_id": house_id,
            "room_index": torch.tensor(room_index, dtype=torch.long),
            "step_ratio": torch.tensor(
                offset / max(len(order) - 1, 1),
                dtype=torch.float32,
            ),
            "site_cells": torch.tensor(staged["site_cells"][:2]),
            "volume": torch.from_numpy(volume),
            "target_box": torch.from_numpy(
                target_box(rooms[room_index], staged["site_cells"])
            ),
            "nodes": torch.from_numpy(nodes),
            "edge_index": torch.from_numpy(edge_index),
            "edge_type": torch.from_numpy(edge_type),
        }


def collate_instances(items: list[dict]) -> dict:
    max_nodes = max(int(item["nodes"].shape[0]) for item in items)
    batch_size = len(items)
    nodes = torch.zeros(batch_size, max_nodes, NODE_DIM)
    node_mask = torch.zeros(batch_size, max_nodes)
    adjacency = torch.zeros(batch_size, 2, max_nodes, max_nodes)
    for batch_index, item in enumerate(items):
        count = int(item["nodes"].shape[0])
        nodes[batch_index, :count] = item["nodes"]
        node_mask[batch_index, :count] = 1.0
        for edge_offset in range(item["edge_index"].shape[1]):
            source = int(item["edge_index"][0, edge_offset])
            target = int(item["edge_index"][1, edge_offset])
            relation = int(item["edge_type"][edge_offset])
            adjacency[batch_index, relation, source, target] = 1.0
    return {
        "house_id": [item["house_id"] for item in items],
        "room_index": torch.stack([item["room_index"] for item in items]),
        "step_ratio": torch.stack([item["step_ratio"] for item in items]),
        "site_cells": torch.stack([item["site_cells"] for item in items]),
        "volume": torch.stack([item["volume"] for item in items]),
        "target_box": torch.stack([item["target_box"] for item in items]),
        "nodes": nodes,
        "node_mask": node_mask,
        "adjacency": adjacency,
    }
