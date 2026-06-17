"""Lazy dataset for topology-conditioned 3D cut actions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "phase6_spatial_cut" / "samples"
DEFAULT_SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
GRID_SHAPE = (44, 44, 20)
NODE_DIM = 15


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_ids(path: Path, split: str) -> list[str]:
    payload = read_json(path)
    key = "validation" if split == "val" else split
    return [str(value) for value in payload[key]]


def node_features(nodes: list[dict]) -> np.ndarray:
    features = np.zeros((len(nodes), NODE_DIM), dtype=np.float32)
    for index, node in enumerate(nodes):
        features[index, int(node["type_id"])] = 1.0
        features[index, 11] = float(node["floor_1"])
        features[index, 12] = float(node["floor_2"])
        features[index, 13] = float(node["target_area_ratio"])
        features[index, 14] = 1.0
    return features


def region_volume(site_cells: list[int], region: list[int]) -> np.ndarray:
    volume = np.zeros((3, *GRID_SHAPE), dtype=np.float32)
    sx = min(GRID_SHAPE[0], int(np.ceil(site_cells[0] / 2)))
    sy = min(GRID_SHAPE[1], int(np.ceil(site_cells[1] / 2)))
    volume[0, :sx, :sy, :20] = 1.0
    x0, y0, z0, x1, y1, z1 = region
    x0, y0 = x0 // 2, y0 // 2
    x1, y1 = int(np.ceil(x1 / 2)), int(np.ceil(y1 / 2))
    volume[1, x0:x1, y0:y1, z0:z1] = 1.0
    volume[2, :, :, 10] = 1.0
    return volume


class SpatialCutDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_dir: Path = DEFAULT_DATA_DIR,
        split_path: Path = DEFAULT_SPLIT_PATH,
        max_houses: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        ids = split_ids(Path(split_path), split)
        if max_houses is not None:
            ids = ids[:max_houses]
        self.samples = []
        for house_id in ids:
            payload = read_json(self.data_dir / f"{house_id}.json")
            for action_index in range(len(payload["actions"])):
                self.samples.append((house_id, action_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        house_id, action_index = self.samples[index]
        payload = read_json(self.data_dir / f"{house_id}.json")
        graph = payload["graph"]
        action = payload["actions"][action_index]
        nodes = node_features(graph["nodes"])
        active = np.zeros(len(nodes), dtype=np.float32)
        active[action["room_indices"]] = 1.0
        side_target = np.full(len(nodes), -1, dtype=np.int64)
        side_target[action.get("left_indices", [])] = 0
        side_target[action.get("right_indices", [])] = 1
        edges = np.asarray(graph["edges"], dtype=np.int64)
        if edges.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_type = np.zeros((0,), dtype=np.int64)
        else:
            edge_index = edges[:, :2].T
            edge_type = edges[:, 2]
        return {
            "house_id": house_id,
            "volume": torch.from_numpy(
                region_volume(payload["site_cells"], action["region"])
            ),
            "nodes": torch.from_numpy(nodes),
            "active": torch.from_numpy(active),
            "side_target": torch.from_numpy(side_target),
            "edge_index": torch.from_numpy(edge_index),
            "edge_type": torch.from_numpy(edge_type),
            "axis": torch.tensor(int(action["axis"]), dtype=torch.long),
            "cut_ratio": torch.tensor(
                float(action["cut_ratio"]), dtype=torch.float32
            ),
            "left_fraction": torch.tensor(
                float(action.get("left_fraction", 0.5)), dtype=torch.float32
            ),
        }


def collate_cut_actions(items: list[dict]) -> dict:
    node_counts = [int(item["nodes"].shape[0]) for item in items]
    max_nodes = max(node_counts)
    batch_size = len(items)
    nodes = torch.zeros(batch_size, max_nodes, NODE_DIM)
    active = torch.zeros(batch_size, max_nodes)
    side_target = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    adjacency = torch.zeros(batch_size, 2, max_nodes, max_nodes)
    for batch_index, item in enumerate(items):
        count = node_counts[batch_index]
        nodes[batch_index, :count] = item["nodes"]
        active[batch_index, :count] = item["active"]
        side_target[batch_index, :count] = item["side_target"]
        for edge_offset in range(item["edge_index"].shape[1]):
            src = int(item["edge_index"][0, edge_offset])
            dst = int(item["edge_index"][1, edge_offset])
            rel = int(item["edge_type"][edge_offset])
            adjacency[batch_index, rel, src, dst] = 1.0
    return {
        "house_id": [item["house_id"] for item in items],
        "volume": torch.stack([item["volume"] for item in items]),
        "nodes": nodes,
        "active": active,
        "side_target": side_target,
        "adjacency": adjacency,
        "axis": torch.stack([item["axis"] for item in items]),
        "cut_ratio": torch.stack([item["cut_ratio"] for item in items]),
        "left_fraction": torch.stack(
            [item["left_fraction"] for item in items]
        ),
    }
