"""User-condition dataset for learning room programs and topology."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from build_staged_supervision import LIGHTING_TO_ID, ROOM_TYPES, TYPE_TO_ID
from staged_dataset import split_ids


ROOT = Path(__file__).resolve().parents[2]
PHASE2_DIR = ROOT / "data" / "phase2_v5" / "samples"
PHASE7_DIR = ROOT / "data" / "phase7_staged_spatial" / "samples"
SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
NODE_INPUT_DIM = 19


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_features(house_id: str, instance_index: int) -> np.ndarray:
    digest = hashlib.sha256(
        f"{house_id}:{instance_index}".encode("utf-8")
    ).digest()
    values = np.frombuffer(digest[:4], dtype=np.uint8).astype(np.float32)
    return values / 127.5 - 1.0


def floor_class(node: dict) -> int:
    floor_1 = bool(node["floor_1"])
    floor_2 = bool(node["floor_2"])
    if floor_1 and floor_2:
        return 2
    return 0 if floor_1 else 1


class ProgramGraphDataset(Dataset):
    """Predict program attributes and relations without target geometry input."""

    def __init__(
        self,
        split: str,
        phase2_dir: Path = PHASE2_DIR,
        phase7_dir: Path = PHASE7_DIR,
        split_path: Path = SPLIT_PATH,
        max_houses: int | None = None,
    ) -> None:
        self.phase2_dir = Path(phase2_dir)
        self.phase7_dir = Path(phase7_dir)
        self.house_ids = split_ids(Path(split_path), split)
        if max_houses is not None:
            self.house_ids = self.house_ids[:max_houses]

    def __len__(self) -> int:
        return len(self.house_ids)

    def __getitem__(self, index: int) -> dict:
        house_id = self.house_ids[index]
        phase2 = read_json(self.phase2_dir / f"{house_id}.json")
        graph = read_json(self.phase7_dir / f"{house_id}.json")["graph"]
        counts = Counter(str(node["type"]) for node in graph["nodes"])
        offsets: dict[str, int] = defaultdict(int)
        site_x, site_y = (float(value) for value in phase2["site_size_mm"])
        node_input = np.zeros((len(graph["nodes"]), NODE_INPUT_DIM), np.float32)
        floor_target = np.zeros(len(graph["nodes"]), np.int64)
        area_target = np.zeros(len(graph["nodes"]), np.float32)
        lighting_target = np.zeros(len(graph["nodes"]), np.int64)
        exterior_target = np.zeros((len(graph["nodes"]), 4), np.float32)
        side_offsets = {"W": 0, "E": 1, "S": 2, "N": 3}
        for node_index, node in enumerate(graph["nodes"]):
            room_type = str(node["type"])
            type_id = TYPE_TO_ID[room_type]
            ordinal = offsets[room_type]
            offsets[room_type] += 1
            node_input[node_index, type_id] = 1.0
            node_input[node_index, 11] = ordinal / max(counts[room_type] - 1, 1)
            node_input[node_index, 12] = counts[room_type] / 12.0
            node_input[node_index, 13] = site_x / 26400.0
            node_input[node_index, 14] = site_y / 26400.0
            node_input[node_index, 15:19] = seed_features(house_id, node_index)
            floor_target[node_index] = floor_class(node)
            area_target[node_index] = float(node["target_area_ratio"])
            lighting_target[node_index] = LIGHTING_TO_ID.get(
                str(node.get("lighting_access", "none")),
                0,
            )
            for side in node.get("exterior_sides", []):
                exterior_target[node_index, side_offsets[side]] = 1.0

        relation_target = np.zeros(
            (len(graph["nodes"]), len(graph["nodes"])),
            np.int64,
        )
        for left, right, relation in graph["edges"]:
            relation_target[int(left), int(right)] = int(relation) + 1
        np.fill_diagonal(relation_target, -1)
        return {
            "house_id": house_id,
            "node_input": torch.from_numpy(node_input),
            "floor_target": torch.from_numpy(floor_target),
            "area_target": torch.from_numpy(area_target),
            "lighting_target": torch.from_numpy(lighting_target),
            "exterior_target": torch.from_numpy(exterior_target),
            "relation_target": torch.from_numpy(relation_target),
        }


def collate_program_graph(items: list[dict]) -> dict:
    batch_size = len(items)
    max_nodes = max(int(item["node_input"].shape[0]) for item in items)
    node_input = torch.zeros(batch_size, max_nodes, NODE_INPUT_DIM)
    node_mask = torch.zeros(batch_size, max_nodes)
    floor_target = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    area_target = torch.zeros(batch_size, max_nodes)
    lighting_target = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    exterior_target = torch.zeros(batch_size, max_nodes, 4)
    relation_target = torch.full(
        (batch_size, max_nodes, max_nodes),
        -1,
        dtype=torch.long,
    )
    for batch_index, item in enumerate(items):
        count = int(item["node_input"].shape[0])
        node_input[batch_index, :count] = item["node_input"]
        node_mask[batch_index, :count] = 1.0
        floor_target[batch_index, :count] = item["floor_target"]
        area_target[batch_index, :count] = item["area_target"]
        lighting_target[batch_index, :count] = item["lighting_target"]
        exterior_target[batch_index, :count] = item["exterior_target"]
        relation_target[batch_index, :count, :count] = item["relation_target"]
    return {
        "house_id": [item["house_id"] for item in items],
        "node_input": node_input,
        "node_mask": node_mask,
        "floor_target": floor_target,
        "area_target": area_target,
        "lighting_target": lighting_target,
        "exterior_target": exterior_target,
        "relation_target": relation_target,
    }
