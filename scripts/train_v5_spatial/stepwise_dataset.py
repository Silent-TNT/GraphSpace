"""Lazy dataset for Phase9 stepwise spatial action supervision."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from staged_dataset import NODE_DIM, graph_arrays, split_ids
from stepwise_decision import ActionKind, StepAction, StepwiseDecisionEnvironment


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "phase9_stepwise_spatial" / "samples"
DEFAULT_SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
GRID_X = 88
GRID_Y = 88
GRID_Z = 20
STEPWISE_VOLUME_CHANNELS = [
    "site",
    "assigned",
    "reserved_empty",
    "active_region",
    "open_regions",
    "floor_boundary",
    "x_coord",
    "y_coord",
    "z_coord",
]
ACTION_TO_ID = {
    "reject": 0,
    "cut": 1,
    "place": 2,
    "reserve_empty": 3,
    "rollback": 4,
}
ID_TO_ACTION = {value: key for key, value in ACTION_TO_ID.items()}


def action_name_for_record(record: dict) -> str:
    return "reject" if not record["accepted"] else record["kind"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def record_to_action(record: dict) -> StepAction:
    return StepAction(
        kind=ActionKind(record["kind"]),
        region_id=record.get("region_id"),
        axis=record.get("axis"),
        cut=record.get("cut"),
        left_node_ids=tuple(record.get("left_node_ids", [])),
        right_node_ids=tuple(record.get("right_node_ids", [])),
        node_ids=tuple(record.get("node_ids", [])),
        bounds=tuple(record["bounds"]) if record.get("bounds") else None,
        source_region_ids=tuple(record.get("source_region_ids", [])),
        target_action_index=record.get("target_action_index"),
        reason=record.get("reason", ""),
    )


def fill_box(volume: np.ndarray, bounds: tuple[int, int, int, int, int, int], value: float) -> None:
    x0, y0, z0, x1, y1, z1 = bounds
    volume[x0:x1, y0:y1, z0:z1] = value


def state_volume(
    env: StepwiseDecisionEnvironment,
    site_cells: list[int],
    current: dict,
) -> np.ndarray:
    volume = np.zeros(
        (len(STEPWISE_VOLUME_CHANNELS), GRID_X, GRID_Y, GRID_Z),
        dtype=np.float32,
    )
    sx, sy, sz = (int(site_cells[0]), int(site_cells[1]), int(site_cells[2]))
    volume[0, :sx, :sy, :sz] = 1.0
    for boxes in env.state.assignments.values():
        for bounds in boxes:
            fill_box(volume[1], bounds, 1.0)
    for bounds in env.state.empty_regions:
        fill_box(volume[2], bounds, 1.0)
    region_id = current.get("region_id")
    if region_id in env.state.regions:
        fill_box(volume[3], env.state.regions[region_id].bounds, 1.0)
    for region in env.state.regions.values():
        fill_box(volume[4], region.bounds, 1.0)
    volume[5, :sx, :sy, 10] = 1.0

    x = np.linspace(-1.0, 1.0, GRID_X, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, GRID_Y, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, GRID_Z, dtype=np.float32)
    volume[6] = x[:, None, None]
    volume[7] = y[None, :, None]
    volume[8] = z[None, None, :]
    return volume


def normalized_bounds(bounds: list[int] | tuple[int, ...] | None) -> np.ndarray:
    if bounds is None:
        return np.zeros(6, dtype=np.float32)
    scale = np.asarray([GRID_X, GRID_Y, GRID_Z, GRID_X, GRID_Y, GRID_Z], dtype=np.float32)
    return np.asarray(bounds, dtype=np.float32) / scale


def normalized_cut(record: dict) -> float:
    if record.get("cut") is None or record.get("axis") is None:
        return 0.0
    axis = int(record["axis"])
    bounds = record.get("bounds")
    if not bounds:
        return float(record["cut"]) / [GRID_X, GRID_Y, GRID_Z][axis]
    extent = max(int(bounds[axis + 3]) - int(bounds[axis]), 1)
    return (float(record["cut"]) - float(bounds[axis])) / float(extent)


def replay_until(payload: dict, action_index: int) -> StepwiseDecisionEnvironment:
    env = StepwiseDecisionEnvironment(
        site_bounds=(0, 0, 0, *payload["site_cells"]),
        node_ids=tuple(range(len(payload["rooms"]))),
    )
    for record in payload["actions"][:action_index]:
        if not record["accepted"]:
            continue
        result = env.apply(record_to_action(record))
        if not result.accepted:
            raise ValueError(
                f"{payload['house_id']}: replay failed before action "
                f"{action_index}: {result.issues}"
            )
    return env


class StepwiseActionDataset(Dataset):
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
        self.items = []
        self.action_targets = []
        for house_id in ids:
            payload = read_json(self.data_dir / f"{house_id}.json")
            for action_index, record in enumerate(payload["actions"]):
                self.items.append((house_id, action_index))
                self.action_targets.append(ACTION_TO_ID[action_name_for_record(record)])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        house_id, action_index = self.items[index]
        payload = read_json(self.data_dir / f"{house_id}.json")
        record = payload["actions"][action_index]
        env = replay_until(payload, action_index)
        nodes, edge_index, edge_type = graph_arrays(payload["graph"])
        action_name = action_name_for_record(record)
        node_target = np.zeros(len(nodes), dtype=np.float32)
        if record.get("kind") == "cut" and record.get("accepted"):
            target_node_ids = record.get("left_node_ids", [])
        else:
            target_node_ids = record.get("node_ids", [])
        for node_id in target_node_ids:
            if 0 <= int(node_id) < len(node_target):
                node_target[int(node_id)] = 1.0
        return {
            "house_id": house_id,
            "action_index": torch.tensor(action_index, dtype=torch.long),
            "volume": torch.from_numpy(
                state_volume(env, payload["site_cells"], record)
            ),
            "nodes": torch.from_numpy(nodes),
            "edge_index": torch.from_numpy(edge_index),
            "edge_type": torch.from_numpy(edge_type),
            "action_target": torch.tensor(ACTION_TO_ID[action_name], dtype=torch.long),
            "accepted_target": torch.tensor(float(record["accepted"]), dtype=torch.float32),
            "axis_target": torch.tensor(int(record.get("axis", -1)), dtype=torch.long),
            "cut_target": torch.tensor(normalized_cut(record), dtype=torch.float32),
            "box_target": torch.from_numpy(normalized_bounds(record.get("bounds"))),
            "node_target": torch.from_numpy(node_target),
        }


def collate_stepwise(items: list[dict]) -> dict:
    batch_size = len(items)
    max_nodes = max(int(item["nodes"].shape[0]) for item in items)
    nodes = torch.zeros(batch_size, max_nodes, NODE_DIM)
    node_mask = torch.zeros(batch_size, max_nodes)
    node_target = torch.zeros(batch_size, max_nodes)
    adjacency = torch.zeros(batch_size, 2, max_nodes, max_nodes)
    for batch_index, item in enumerate(items):
        count = int(item["nodes"].shape[0])
        nodes[batch_index, :count] = item["nodes"]
        node_mask[batch_index, :count] = 1.0
        node_target[batch_index, :count] = item["node_target"]
        for edge_offset in range(item["edge_index"].shape[1]):
            source = int(item["edge_index"][0, edge_offset])
            target = int(item["edge_index"][1, edge_offset])
            relation = int(item["edge_type"][edge_offset])
            adjacency[batch_index, relation, source, target] = 1.0
    return {
        "house_id": [item["house_id"] for item in items],
        "action_index": torch.stack([item["action_index"] for item in items]),
        "volume": torch.stack([item["volume"] for item in items]),
        "nodes": nodes,
        "node_mask": node_mask,
        "adjacency": adjacency,
        "action_target": torch.stack([item["action_target"] for item in items]),
        "accepted_target": torch.stack([item["accepted_target"] for item in items]),
        "axis_target": torch.stack([item["axis_target"] for item in items]),
        "cut_target": torch.stack([item["cut_target"] for item in items]),
        "box_target": torch.stack([item["box_target"] for item in items]),
        "node_target": node_target,
    }
