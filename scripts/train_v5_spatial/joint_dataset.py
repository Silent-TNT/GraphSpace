"""Whole-house dataset for joint room layout prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from instance_dataset import (
    DEFAULT_SPLIT_PATH,
    PROCESSED_DIR,
    final_coarse_volume,
    read_json,
    room_floors,
    target_box,
)
from staged_dataset import DEFAULT_DATA_DIR, NODE_DIM, graph_arrays, split_ids


class JointLayoutDataset(Dataset):
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
        self.house_ids = split_ids(Path(split_path), split)
        if max_houses is not None:
            self.house_ids = self.house_ids[:max_houses]

    def __len__(self) -> int:
        return len(self.house_ids)

    def __getitem__(self, index: int) -> dict:
        house_id = self.house_ids[index]
        sample_path = self.data_dir / f"{house_id}.json"
        staged = read_json(sample_path)
        rooms = read_json(self.processed_dir / f"{house_id}.json")["rooms"]
        nodes, edge_index, edge_type = graph_arrays(staged["graph"])
        site_cells = staged["site_cells"][:2]
        boxes = np.stack([target_box(room, site_cells) for room in rooms])
        floors = np.zeros((len(rooms), 2), dtype=np.float32)
        for room_index, room in enumerate(rooms):
            for floor in room_floors(room):
                floors[room_index, floor - 1] = 1.0
        return {
            "house_id": house_id,
            "volume": torch.from_numpy(final_coarse_volume(sample_path)),
            "nodes": torch.from_numpy(nodes),
            "target_boxes": torch.from_numpy(boxes),
            "floors": torch.from_numpy(floors),
            "edge_index": torch.from_numpy(edge_index),
            "edge_type": torch.from_numpy(edge_type),
        }


def collate_joint(items: list[dict]) -> dict:
    batch_size = len(items)
    max_nodes = max(int(item["nodes"].shape[0]) for item in items)
    nodes = torch.zeros(batch_size, max_nodes, NODE_DIM)
    node_mask = torch.zeros(batch_size, max_nodes)
    target_boxes = torch.zeros(batch_size, max_nodes, 4)
    floors = torch.zeros(batch_size, max_nodes, 2)
    adjacency = torch.zeros(batch_size, 2, max_nodes, max_nodes)
    for batch_index, item in enumerate(items):
        count = int(item["nodes"].shape[0])
        nodes[batch_index, :count] = item["nodes"]
        node_mask[batch_index, :count] = 1.0
        target_boxes[batch_index, :count] = item["target_boxes"]
        floors[batch_index, :count] = item["floors"]
        for edge_offset in range(item["edge_index"].shape[1]):
            source = int(item["edge_index"][0, edge_offset])
            target = int(item["edge_index"][1, edge_offset])
            relation = int(item["edge_type"][edge_offset])
            adjacency[batch_index, relation, source, target] = 1.0
    return {
        "house_id": [item["house_id"] for item in items],
        "volume": torch.stack([item["volume"] for item in items]),
        "nodes": nodes,
        "node_mask": node_mask,
        "target_boxes": target_boxes,
        "floors": floors,
        "adjacency": adjacency,
    }
