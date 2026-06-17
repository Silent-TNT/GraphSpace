"""Lazy 300 mm multimodal dataset for the final V5 training route."""
from __future__ import annotations

import json
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from program_prior import ProgramPrior
from staged_dataset import NODE_DIM, graph_arrays, node_features, split_ids


ROOT = Path(__file__).resolve().parents[2]
PHASE2_DIR = ROOT / "data" / "phase2_v5" / "samples"
PHASE7_DIR = ROOT / "data" / "phase7_staged_spatial" / "samples"
SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
PROGRAM_PRIOR_PATH = (
    ROOT / "data" / "phase8_program_prior" / "program_prior.json"
)
GRID_X = 88
GRID_Y = 88
GRID_Z = 20
FLOOR_CELLS = 10
INPUT_CHANNELS = 8
SEMANTIC_CLASSES = 12


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_input_volume(site_mask: np.ndarray) -> np.ndarray:
    """Build inference-available 3D conditions without target-layout leakage."""
    site = site_mask.astype(np.float32)
    volume = np.zeros((INPUT_CHANNELS, GRID_X, GRID_Y, GRID_Z), np.float32)
    volume[0] = site[:, :, None]
    volume[1, :, :, :FLOOR_CELLS] = site[:, :, None]
    volume[2, :, :, FLOOR_CELLS:] = site[:, :, None]
    volume[3, :, :, FLOOR_CELLS - 1 : FLOOR_CELLS + 1] = site[:, :, None]

    x = np.linspace(-1.0, 1.0, GRID_X, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, GRID_Y, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, GRID_Z, dtype=np.float32)
    volume[4] = x[:, None, None]
    volume[5] = y[None, :, None]
    volume[6] = z[None, None, :]
    volume[7, :, :, :FLOOR_CELLS] = 1.0
    volume[7, :, :, FLOOR_CELLS:] = -1.0
    return volume


def expand_floor_grid(grid: np.ndarray) -> np.ndarray:
    output = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=grid.dtype)
    output[:, :, :FLOOR_CELLS] = grid[0, :, :, None]
    output[:, :, FLOOR_CELLS:] = grid[1, :, :, None]
    return output


def build_instance_targets(
    instance_grid: np.ndarray,
    instance_count: int,
) -> np.ndarray:
    targets = np.zeros(
        (instance_count, GRID_X, GRID_Y, GRID_Z),
        dtype=np.uint8,
    )
    for instance_offset in range(instance_count):
        instance_id = instance_offset + 1
        for floor in range(2):
            z0 = floor * FLOOR_CELLS
            z1 = z0 + FLOOR_CELLS
            mask = instance_grid[floor] == instance_id
            targets[instance_offset, :, :, z0:z1] = mask[:, :, None]
    return targets


def _condition_seed(house_id: str) -> int:
    digest = hashlib.sha256(house_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _floor_signature(floors: list[int]) -> str:
    values = {int(value) for value in floors}
    return "1&2" if values == {1, 2} else str(min(values))


def _floor_distance(left: str, right: str) -> int:
    left_set = {1, 2} if left == "1&2" else {int(left)}
    right_set = {1, 2} if right == "1&2" else {int(right)}
    return len(left_set.symmetric_difference(right_set))


def inference_condition_graph(
    phase2_json: dict,
    prior: ProgramPrior,
    house_id: str,
) -> tuple[dict, list[int]]:
    """Build the same kind of uncertain graph available during generation."""
    table = list(phase2_json["instance_table"])
    counts = Counter(str(item["type"]) for item in table)
    site_x, site_y = (float(value) for value in phase2_json["site_size_mm"])
    graph, _positions, topology_nodes, edge_types, evidence = (
        prior.build_topology(
            dict(counts),
            site_x,
            site_y,
            seed=_condition_seed(house_id),
            exclude_house_id=house_id,
        )
    )
    conditions = evidence.get("node_conditions", {})
    target_by_type: dict[str, list[int]] = defaultdict(list)
    for target_index, record in enumerate(table):
        target_by_type[str(record["type"])].append(target_index)

    target_order = []
    nodes = []
    node_lookup = {}
    used: set[int] = set()
    for node_index, (node_id, room_type, floor) in enumerate(topology_nodes):
        node_lookup[node_id] = node_index
        candidates = [
            index
            for index in target_by_type[room_type]
            if index not in used
        ]
        if not candidates:
            raise ValueError(f"{house_id}: no target for generated node {node_id}")
        condition = conditions.get(node_id, {})
        target_index = min(
            candidates,
            key=lambda index: (
                _floor_distance(
                    str(floor),
                    _floor_signature(table[index]["floors"]),
                ),
                index,
            ),
        )
        used.add(target_index)
        target_order.append(target_index)
        floor_text = str(floor)
        lighting_access = str(condition.get("lighting_access", "none"))
        nodes.append(
            {
                "instance_token": node_id,
                "type": room_type,
                "type_id": int(
                    next(
                        offset
                        for offset, name in enumerate(
                            (
                                "entryway",
                                "living_room",
                                "dining_room",
                                "kitchen",
                                "bedroom",
                                "bathroom",
                                "corridor",
                                "stairs",
                                "utility",
                                "balcony",
                                "multi_purpose",
                            )
                        )
                        if name == room_type
                    )
                ),
                "floor_1": int(floor_text in {"1", "1&2"}),
                "floor_2": int(floor_text in {"2", "1&2"}),
                "target_area_ratio": float(
                    condition.get("area_ratio", 0.04)
                ),
                "exterior_sides": [],
                "lighting_access": lighting_access,
                "lighting_id": {
                    "none": 0,
                    "indirect": 1,
                    "direct": 2,
                }.get(lighting_access, 0),
                "lighting_priority": int(
                    condition.get("lighting_priority", 0)
                ),
            }
        )
    edges = []
    for left, right in graph.edges:
        relation_name = edge_types.get(
            (left, right),
            edge_types.get((right, left), "horizontal"),
        )
        relation = 1 if relation_name == "vertical" else 0
        left_index, right_index = node_lookup[left], node_lookup[right]
        edges.extend(
            (
                [left_index, right_index, relation],
                [right_index, left_index, relation],
            )
        )
    return {"nodes": nodes, "edges": edges}, target_order


def robust_teacher_graph(graph: dict, house_id: str) -> dict:
    """Perturb a target-compatible graph without replacing its design logic."""
    seed = _condition_seed(house_id)
    rng = random.Random(seed)
    nodes = []
    for node in graph["nodes"]:
        copied = dict(node)
        copied["target_area_ratio"] = max(
            0.002,
            float(node["target_area_ratio"]) * rng.uniform(0.85, 1.15),
        )
        copied["exterior_sides"] = [
            side
            for side in node.get("exterior_sides", [])
            if rng.random() >= 0.35
        ]
        nodes.append(copied)

    unique = {}
    for left, right, relation in graph["edges"]:
        pair = tuple(sorted((int(left), int(right))))
        if pair[0] != pair[1]:
            unique[pair] = int(relation)
    retained = {
        pair: relation
        for pair, relation in unique.items()
        if rng.random() >= 0.08
    }
    candidates = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            pair = (left, right)
            if pair in retained:
                continue
            same_floor = (
                bool(nodes[left]["floor_1"]) and bool(nodes[right]["floor_1"])
            ) or (
                bool(nodes[left]["floor_2"]) and bool(nodes[right]["floor_2"])
            )
            if same_floor:
                candidates.append(pair)
    rng.shuffle(candidates)
    add_count = min(
        max(1, round(len(unique) * 0.04)),
        len(candidates),
    )
    for pair in candidates[:add_count]:
        retained[pair] = 0
    edges = []
    for (left, right), relation in sorted(retained.items()):
        edges.extend(([left, right, relation], [right, left, relation]))
    return {"nodes": nodes, "edges": edges}


class FullResolutionLayoutDataset(Dataset):
    """One whole house per sample at the native 300 mm resolution."""

    def __init__(
        self,
        split: str,
        phase2_dir: Path = PHASE2_DIR,
        phase7_dir: Path = PHASE7_DIR,
        split_path: Path = SPLIT_PATH,
        program_prior_path: Path = PROGRAM_PRIOR_PATH,
        condition_mode: str = "teacher",
        max_houses: int | None = None,
    ) -> None:
        self.phase2_dir = Path(phase2_dir)
        self.phase7_dir = Path(phase7_dir)
        self.program_prior = ProgramPrior(program_prior_path)
        if condition_mode not in {"teacher", "robust", "program"}:
            raise ValueError(f"unknown condition_mode: {condition_mode}")
        self.condition_mode = condition_mode
        self.house_ids = split_ids(Path(split_path), split)
        if max_houses is not None:
            self.house_ids = self.house_ids[:max_houses]

    def __len__(self) -> int:
        return len(self.house_ids)

    def __getitem__(self, index: int) -> dict:
        house_id = self.house_ids[index]
        phase2_json = read_json(self.phase2_dir / f"{house_id}.json")
        phase7_json = read_json(self.phase7_dir / f"{house_id}.json")
        target_order = list(range(len(phase2_json["instance_table"])))
        if self.condition_mode == "program":
            condition_graph, target_order = inference_condition_graph(
                phase2_json,
                self.program_prior,
                house_id,
            )
        elif self.condition_mode == "robust":
            condition_graph = robust_teacher_graph(
                phase7_json["graph"],
                house_id,
            )
        else:
            condition_graph = phase7_json["graph"]
        nodes, edge_index, edge_type = graph_arrays(condition_graph)
        with np.load(self.phase2_dir / f"{house_id}.npz") as arrays:
            site_mask = arrays["site_mask"].copy()
            semantic = expand_floor_grid(arrays["class_grid"])
            building = expand_floor_grid(arrays["building_mask"])
            empty = expand_floor_grid(arrays["empty_inside_mask"])
            instance_targets = build_instance_targets(
                arrays["instance_grid"],
                len(phase2_json["instance_table"]),
            )
            instance_targets = instance_targets[target_order]
        if len(nodes) != len(instance_targets):
            raise ValueError(
                f"{house_id}: graph nodes={len(nodes)} "
                f"instance targets={len(instance_targets)}"
            )
        return {
            "house_id": house_id,
            "volume": torch.from_numpy(build_input_volume(site_mask)),
            "nodes": torch.from_numpy(nodes),
            "edge_index": torch.from_numpy(edge_index),
            "edge_type": torch.from_numpy(edge_type),
            "instance_targets": torch.from_numpy(instance_targets),
            "semantic_target": torch.from_numpy(semantic.astype(np.int64)),
            "building_target": torch.from_numpy(building.astype(np.float32)),
            "empty_target": torch.from_numpy(empty.astype(np.float32)),
        }


def collate_fullres(items: list[dict]) -> dict:
    batch_size = len(items)
    max_nodes = max(int(item["nodes"].shape[0]) for item in items)
    nodes = torch.zeros(batch_size, max_nodes, NODE_DIM)
    node_mask = torch.zeros(batch_size, max_nodes)
    instance_targets = torch.zeros(
        batch_size,
        max_nodes,
        GRID_X,
        GRID_Y,
        GRID_Z,
    )
    adjacency = torch.zeros(batch_size, 2, max_nodes, max_nodes)
    for batch_index, item in enumerate(items):
        count = int(item["nodes"].shape[0])
        nodes[batch_index, :count] = item["nodes"]
        node_mask[batch_index, :count] = 1.0
        instance_targets[batch_index, :count] = item[
            "instance_targets"
        ].float()
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
        "adjacency": adjacency,
        "instance_targets": instance_targets,
        "semantic_target": torch.stack(
            [item["semantic_target"] for item in items]
        ),
        "building_target": torch.stack(
            [item["building_target"] for item in items]
        ),
        "empty_target": torch.stack(
            [item["empty_target"] for item in items]
        ),
    }
