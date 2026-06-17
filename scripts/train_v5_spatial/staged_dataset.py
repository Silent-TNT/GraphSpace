"""Lazy dataset for teacher-forced staged V5 spatial generation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "phase7_staged_spatial" / "samples"
DEFAULT_SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
GRID_SHAPE = (44, 44, 20)
NODE_DIM = 22
STAGE_COUNT = 7
VOLUME_CHANNELS = [
    "site",
    "building_envelope",
    "protected_stairs",
    "explicit_empty",
    "traffic_reserve",
    "rigid_functions",
    "service_spaces",
    "floor_boundary",
]
CHANNEL_INDEX = {name: index for index, name in enumerate(VOLUME_CHANNELS)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_ids(path: Path, split: str) -> list[str]:
    payload = read_json(path)
    key = "validation" if split == "val" else split
    return [str(value) for value in payload[key]]


def reduce_2d(mask: np.ndarray) -> np.ndarray:
    """Reduce a 300 mm mask to the 600 mm training grid by max pooling."""
    width, depth = mask.shape[-2:]
    pad_x, pad_y = width % 2, depth % 2
    padded = np.pad(mask, ((0, pad_x), (0, pad_y)), mode="constant")
    reduced = padded.reshape(
        padded.shape[0] // 2,
        2,
        padded.shape[1] // 2,
        2,
    ).max(axis=(1, 3))
    output = np.zeros(GRID_SHAPE[:2], dtype=np.float32)
    sx = min(output.shape[0], reduced.shape[0])
    sy = min(output.shape[1], reduced.shape[1])
    output[:sx, :sy] = reduced[:sx, :sy]
    return output


def floor_volume(mask: np.ndarray) -> np.ndarray:
    output = np.zeros(GRID_SHAPE, dtype=np.float32)
    for floor in range(2):
        reduced = reduce_2d(mask[floor])
        output[:, :, floor * 10 : (floor + 1) * 10] = reduced[:, :, None]
    return output


def floor_boundary_volume() -> np.ndarray:
    output = np.zeros(GRID_SHAPE, dtype=np.float32)
    output[:, :, 10] = 1.0
    return output


def load_volumes(sample_path: Path) -> dict[str, np.ndarray]:
    with np.load(sample_path.with_suffix(".npz")) as arrays:
        site_2d = reduce_2d(arrays["site_mask"])
        return {
            "site": np.repeat(site_2d[:, :, None], GRID_SHAPE[2], axis=2),
            "building": floor_volume(arrays["building_mask"]),
            "stairs": floor_volume(arrays["stair_mask"]),
            "empty": floor_volume(arrays["empty_mask"]),
            "traffic": floor_volume(arrays["traffic_mask"]),
            "rigid": floor_volume(arrays["rigid_mask"]),
            "service": floor_volume(arrays["service_mask"]),
            "floor": floor_boundary_volume(),
        }


def stage_input(volumes: dict[str, np.ndarray], stage_id: int) -> np.ndarray:
    """Return state before the requested action, without its target."""
    state = np.zeros((len(VOLUME_CHANNELS), *GRID_SHAPE), dtype=np.float32)
    state[CHANNEL_INDEX["site"]] = volumes["site"]
    if stage_id >= 1:
        state[CHANNEL_INDEX["protected_stairs"]] = volumes["stairs"]
    if stage_id >= 2:
        state[CHANNEL_INDEX["floor_boundary"]] = volumes["floor"]
    if stage_id >= 3:
        state[CHANNEL_INDEX["building_envelope"]] = volumes["building"]
        state[CHANNEL_INDEX["explicit_empty"]] = volumes["empty"]
    if stage_id >= 4:
        state[CHANNEL_INDEX["traffic_reserve"]] = volumes["traffic"]
    if stage_id >= 5:
        state[CHANNEL_INDEX["rigid_functions"]] = volumes["rigid"]
    if stage_id >= 6:
        state[CHANNEL_INDEX["service_spaces"]] = volumes["service"]
    return state


def stage_target(
    volumes: dict[str, np.ndarray],
    stage_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((2, *GRID_SHAPE), dtype=np.float32)
    valid = np.zeros(2, dtype=np.float32)
    if stage_id == 0:
        target[0], valid[0] = volumes["stairs"], 1
    elif stage_id == 1:
        pass
    elif stage_id == 2:
        target[0], target[1] = volumes["building"], volumes["empty"]
        valid[:] = 1
    elif stage_id == 3:
        target[0], valid[0] = volumes["traffic"], 1
    elif stage_id == 4:
        target[0], valid[0] = volumes["rigid"], 1
    elif stage_id == 5:
        target[0], valid[0] = volumes["service"], 1
    return target, valid


def node_features(nodes: list[dict]) -> np.ndarray:
    features = np.zeros((len(nodes), NODE_DIM), dtype=np.float32)
    side_offsets = {"W": 14, "E": 15, "S": 16, "N": 17}
    for index, node in enumerate(nodes):
        features[index, int(node["type_id"])] = 1.0
        features[index, 11] = float(node["floor_1"])
        features[index, 12] = float(node["floor_2"])
        features[index, 13] = float(node["target_area_ratio"])
        for side in node.get("exterior_sides", []):
            features[index, side_offsets[side]] = 1.0
        lighting_id = min(max(int(node.get("lighting_id", 0)), 0), 2)
        features[index, 18 + lighting_id] = 1.0
        features[index, 21] = float(node.get("lighting_priority", 0)) / 10.0
    return features


def graph_arrays(graph: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = node_features(graph["nodes"])
    edges = np.asarray(graph["edges"], dtype=np.int64)
    if edges.size == 0:
        return (
            nodes,
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return nodes, edges[:, :2].T, edges[:, 2]


class StagedSpatialDataset(Dataset):
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
        self.items = [
            (house_id, stage_id)
            for house_id in ids
            for stage_id in range(STAGE_COUNT)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        house_id, stage_id = self.items[index]
        sample_path = self.data_dir / f"{house_id}.json"
        payload = read_json(sample_path)
        volumes = load_volumes(sample_path)
        target, target_valid = stage_target(volumes, stage_id)
        nodes, edge_index, edge_type = graph_arrays(payload["graph"])
        reachability = payload["actions"][6]["oracle_report"][
            "all_required_reachable"
        ]
        return {
            "house_id": house_id,
            "stage_id": torch.tensor(stage_id, dtype=torch.long),
            "volume": torch.from_numpy(stage_input(volumes, stage_id)),
            "target_volume": torch.from_numpy(target),
            "target_valid": torch.from_numpy(target_valid),
            "reachability": torch.tensor(float(reachability)),
            "cut_ratio": torch.tensor(0.5, dtype=torch.float32),
            "nodes": torch.from_numpy(nodes),
            "edge_index": torch.from_numpy(edge_index),
            "edge_type": torch.from_numpy(edge_type),
        }


def collate_staged(items: list[dict]) -> dict:
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
        "stage_id": torch.stack([item["stage_id"] for item in items]),
        "volume": torch.stack([item["volume"] for item in items]),
        "target_volume": torch.stack(
            [item["target_volume"] for item in items]
        ),
        "target_valid": torch.stack([item["target_valid"] for item in items]),
        "reachability": torch.stack([item["reachability"] for item in items]),
        "cut_ratio": torch.stack([item["cut_ratio"] for item in items]),
        "nodes": nodes,
        "node_mask": node_mask,
        "adjacency": adjacency,
    }


def staged_volume(sample_path: Path, stage_id: int) -> np.ndarray:
    """Backward-compatible helper used by tests and visual probes."""
    return stage_input(load_volumes(sample_path), stage_id)
