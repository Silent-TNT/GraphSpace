#!/usr/bin/env python3
"""Learned user-condition to whole heterogeneous topology graph.

This module generates the complete graph skeleton used by the Phase24 bridge:
functional room nodes, learned node attributes, floor containment edges and
target adjacency edges. It intentionally does not call ProgramPrior for node
completion or topology construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "train_v5_spatial"
for import_dir in (ROOT, SCRIPT_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from program_graph_dataset import NODE_INPUT_DIM, seed_features  # noqa: E402
from program_graph_model import ProgramGraphModel  # noqa: E402
from build_staged_supervision import ROOM_TYPES, TYPE_TO_ID  # noqa: E402


DEFAULT_PRIOR = ROOT / "data" / "phase8_program_prior" / "program_prior.json"
MAX_COUNT = 12
COUNT_INPUT_DIM = 2 + len(ROOM_TYPES) * 2 + 4
LIGHTING_BY_ID = {0: "none", 1: "indirect", 2: "direct"}
FLOOR_BY_CLASS = {0: ("1", [1]), 1: ("2", [2]), 2: ("1&2", [1, 2])}
RELATION_BY_CLASS = {1: "horizontal", 2: "vertical"}
EXTERIOR_SIDES = ["W", "E", "S", "N"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train-counts")
    train.add_argument("--program-prior", type=Path, default=DEFAULT_PRIOR)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--hidden", type=int, default=128)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=20260625)
    train.add_argument("--smoke-test", action="store_true")

    infer = sub.add_parser("generate")
    infer.add_argument("--count-checkpoint", type=Path, required=True)
    infer.add_argument("--program-graph-checkpoint", type=Path, required=True)
    infer.add_argument("--site-x", type=float, required=True)
    infer.add_argument("--site-y", type=float, required=True)
    infer.add_argument("--rooms-json")
    infer.add_argument("--rooms-file", type=Path)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--edge-threshold", type=float)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def site_seed_features(site_x: float, site_y: float, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"{site_x:.1f}:{site_y:.1f}:{seed}".encode("utf-8")).digest()
    values = np.frombuffer(digest[:4], dtype=np.uint8).astype(np.float32)
    return values / 127.5 - 1.0


def count_input(
    site_x: float,
    site_y: float,
    explicit_counts: dict[str, int] | None,
    seed: int,
) -> np.ndarray:
    explicit_counts = explicit_counts or {}
    vector = np.zeros(COUNT_INPUT_DIM, np.float32)
    vector[0] = float(site_x) / 26400.0
    vector[1] = float(site_y) / 26400.0
    offset = 2
    for index, room_type in enumerate(ROOM_TYPES):
        if room_type in explicit_counts:
            vector[offset + index] = min(float(explicit_counts[room_type]), MAX_COUNT) / MAX_COUNT
            vector[offset + len(ROOM_TYPES) + index] = 1.0
    vector[-4:] = site_seed_features(site_x, site_y, seed)
    return vector


class CountDataset(Dataset):
    def __init__(self, program_prior: Path, split: str = "train") -> None:
        payload = read_json(program_prior)
        houses = list(payload["houses"])
        random.Random(20260625).shuffle(houses)
        cut = max(1, int(round(len(houses) * 0.8)))
        self.houses = houses[:cut] if split == "train" else houses[cut:]

    def __len__(self) -> int:
        return len(self.houses)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        house = self.houses[index]
        source_counts = (
            house.get("functional_group_counts")
            or house.get("room_counts")
            or house.get("raw_part_counts")
            or {}
        )
        counts = {str(key): int(value) for key, value in source_counts.items()}
        explicit = {}
        rng = random.Random(f"{house['house_id']}:count-mask")
        for room_type, value in counts.items():
            if rng.random() < 0.35:
                explicit[room_type] = value
        target = np.array([min(counts.get(room_type, 0), MAX_COUNT) for room_type in ROOM_TYPES], np.float32)
        return {
            "input": torch.from_numpy(count_input(float(house["site_x"]), float(house["site_y"]), explicit, index)),
            "target": torch.from_numpy(target),
        }


class CountGenerator(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(COUNT_INPUT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(ROOM_TYPES)),
            nn.Softplus(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def train_counts(args: argparse.Namespace) -> None:
    if args.smoke_test:
        args.epochs = 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_loader = DataLoader(CountDataset(args.program_prior, "train"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CountDataset(args.program_prior, "val"), batch_size=args.batch_size, shuffle=False)
    model = CountGenerator(args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best = float("inf")
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["program_prior"] = str(config["program_prior"])
    config["output_dir"] = str(config["output_dir"])
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch}
        for name, loader, training in (("train", train_loader, True), ("validation", val_loader, False)):
            model.train(training)
            total = 0.0
            samples = 0
            exact = 0
            with torch.set_grad_enabled(training):
                for batch in loader:
                    features = batch["input"].to(device)
                    target = batch["target"].to(device)
                    pred = model(features)
                    loss = F.smooth_l1_loss(pred, target)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    total += float(loss.detach()) * int(features.shape[0])
                    samples += int(features.shape[0])
                    exact += int((torch.round(pred).clamp(0, MAX_COUNT) == target).all(dim=1).sum().detach())
            row[name] = {"loss": total / max(samples, 1), "exact_house_rate": exact / max(samples, 1)}
        history.append(row)
        print(f"epoch={epoch:03d} train={row['train']['loss']:.4f} val={row['validation']['loss']:.4f}")
        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation_loss": row["validation"]["loss"],
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if row["validation"]["loss"] < best:
            best = row["validation"]["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")
        write_json(args.output_dir / "history.json", {"history": history})


def load_count_generator(path: Path, device: torch.device) -> CountGenerator:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = CountGenerator(int(config.get("hidden", 128))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_program_graph(path: Path, device: torch.device) -> tuple[ProgramGraphModel, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = ProgramGraphModel(
        hidden=int(config.get("hidden", 128)),
        layers=int(config.get("layers", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    threshold = 0.75
    return model, threshold


def normalize_counts(predicted: torch.Tensor, explicit_counts: dict[str, int]) -> dict[str, int]:
    values = torch.round(predicted).clamp(0, MAX_COUNT).detach().cpu().tolist()
    counts = {room_type: int(values[index]) for index, room_type in enumerate(ROOM_TYPES)}
    counts.update({str(key): int(value) for key, value in explicit_counts.items()})
    counts["dining_room"] = max(1, counts.get("dining_room", 0))
    counts["stairs"] = max(1, counts.get("stairs", 0))
    if counts.get("living_room", 0) <= 0:
        counts["living_room"] = 1
    return {key: value for key, value in counts.items() if value > 0}


def node_input_from_counts(
    counts: dict[str, int],
    site_x: float,
    site_y: float,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    rows = []
    records = []
    for room_type in ROOM_TYPES:
        count = int(counts.get(room_type, 0))
        for ordinal in range(count):
            row = np.zeros(NODE_INPUT_DIM, np.float32)
            row[TYPE_TO_ID[room_type]] = 1.0
            row[11] = ordinal / max(count - 1, 1)
            row[12] = count / 12.0
            row[13] = site_x / 26400.0
            row[14] = site_y / 26400.0
            row[15:19] = seed_features(f"user:{site_x}:{site_y}:{seed}:{room_type}", ordinal)
            rows.append(row)
            records.append({"id": f"{room_type}_{ordinal}", "type": room_type, "ordinal": ordinal})
    if not rows:
        raise ValueError("learned count generator produced no nodes")
    return torch.from_numpy(np.stack(rows, axis=0)), records


def generate_whole_graph(
    count_checkpoint: Path,
    program_graph_checkpoint: Path,
    site_x: float,
    site_y: float,
    explicit_counts: dict[str, int] | None = None,
    seed: int = 42,
    edge_threshold: float | None = None,
    device: torch.device | None = None,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    device = device or torch.device("cpu")
    explicit_counts = explicit_counts or {}
    count_model = load_count_generator(count_checkpoint, device)
    program_model, default_threshold = load_program_graph(program_graph_checkpoint, device)
    threshold = float(default_threshold if edge_threshold is None else edge_threshold)
    features = torch.from_numpy(count_input(site_x, site_y, explicit_counts, seed)).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_counts = count_model(features)[0]
    counts = normalize_counts(predicted_counts, explicit_counts)
    node_input, node_records = node_input_from_counts(counts, site_x, site_y, seed)
    node_batch = node_input.unsqueeze(0).to(device)
    node_mask = torch.ones(1, node_batch.shape[1], device=device)
    with torch.no_grad():
        output = program_model(node_batch, node_mask)
    floor_pred = output["floor_logits"][0].argmax(dim=-1).detach().cpu()
    area_pred = output["area"][0].detach().cpu()
    lighting_pred = output["lighting_logits"][0].argmax(dim=-1).detach().cpu()
    exterior_pred = output["exterior_logits"][0].sigmoid().detach().cpu()
    relation_probs = output["relation_logits"][0].softmax(dim=-1).detach().cpu()

    room_nodes = []
    node_conditions = {}
    for index, record in enumerate(node_records):
        floor_text, floors = FLOOR_BY_CLASS[int(floor_pred[index].item())]
        if record["type"] == "stairs":
            floor_text, floors = "1&2", [1, 2]
        exterior_sides = [
            side for side_index, side in enumerate(EXTERIOR_SIDES)
            if float(exterior_pred[index, side_index]) >= 0.5
        ]
        node = {
            "id": record["id"],
            "node_type": "room_instance",
            "type": record["type"],
            "floor": floor_text,
            "floors": floors,
            "area_ratio": float(area_pred[index].item()),
            "lighting_access": LIGHTING_BY_ID[int(lighting_pred[index].item())],
            "exterior_sides": exterior_sides,
            "position": [0.5, 0.5],
        }
        room_nodes.append(node)
        node_conditions[record["id"]] = {
            "area_ratio": node["area_ratio"],
            "lighting_access": node["lighting_access"],
            "exterior_sides": exterior_sides,
        }

    geometric_edges = []
    guidance_candidates = []
    for left in range(len(room_nodes)):
        for right in range(left + 1, len(room_nodes)):
            score = float(1.0 - relation_probs[left, right, 0].item())
            if score < threshold:
                continue
            relation_class = int(relation_probs[left, right, 1:].argmax().item() + 1)
            edge = {
                "source": room_nodes[left]["id"],
                "target": room_nodes[right]["id"],
                "relation": RELATION_BY_CLASS[relation_class],
                "probability": score,
            }
            geometric_edges.append({**edge, "edge_type": "geometric_contact_observed"})
            if room_nodes[left]["type"] != room_nodes[right]["type"]:
                guidance_candidates.append(edge)
    guidance_budget = min(len(guidance_candidates), max(len(room_nodes) - 1, int(round(len(room_nodes) * 1.5))))
    topology_edges = sorted(guidance_candidates, key=lambda edge: float(edge["probability"]), reverse=True)[
        :guidance_budget
    ]
    required_edges = sorted({tuple(sorted((edge["source"], edge["target"]))) for edge in topology_edges})
    topology = {
        "schema": "graphspace_learned_whole_heterogeneous_topology_v1",
        "seed": seed,
        "source": "learned_whole_heterogeneous_graph",
        "site": {"x": float(site_x), "y": float(site_y), "z": 6000.0},
        "nodes": room_nodes,
        "edges": topology_edges,
        "geometric_contact_observed": geometric_edges,
        "required_edges": [list(edge) for edge in required_edges],
        "heterogeneous_nodes": [
            {"id": "user_house", "node_type": "house"},
            {"id": "floor_1", "node_type": "floor", "floor": 1},
            {"id": "floor_2", "node_type": "floor", "floor": 2},
            *room_nodes,
        ],
        "heterogeneous_edges": [
            {"source": "user_house", "target": "floor_1", "edge_type": "contains"},
            {"source": "user_house", "target": "floor_2", "edge_type": "contains"},
            *[
                {"source": f"floor_{floor}", "target": node["id"], "edge_type": "contains"}
                for node in room_nodes
                for floor in node["floors"]
            ],
            *[
                {**edge, "edge_type": "guidance_relation"}
                for edge in topology_edges
            ],
            *geometric_edges,
        ],
        "evidence": {
            "source": "learned_whole_heterogeneous_graph",
            "count_checkpoint": str(count_checkpoint),
            "program_graph_checkpoint": str(program_graph_checkpoint),
            "edge_threshold": threshold,
            "explicit_counts": explicit_counts,
            "node_conditions": node_conditions,
            "required_edges": [list(edge) for edge in required_edges],
            "guidance_edge_semantics": (
                "edges are sparse learned guidance relations used by the generator; "
                "same-type geometric contacts are retained separately as observations and "
                "are not used as functional guidance edges"
            ),
            "geometric_contact_observed_count": len(geometric_edges),
            "guidance_candidate_count": len(guidance_candidates),
            "guidance_budget": guidance_budget,
        },
    }
    metadata = {
        "predicted_counts": counts,
        "raw_count_prediction": {
            room_type: float(predicted_counts[index].detach().cpu().item())
            for index, room_type in enumerate(ROOM_TYPES)
        },
        "edge_threshold": threshold,
        "geometric_contact_observed_count": len(geometric_edges),
        "guidance_edge_count": len(topology_edges),
        "guidance_budget": guidance_budget,
    }
    return topology, counts, metadata


def explicit_counts_from_args(args: argparse.Namespace) -> dict[str, int]:
    if args.rooms_file:
        payload = read_json(args.rooms_file)
    elif args.rooms_json:
        payload = json.loads(args.rooms_json)
    else:
        payload = {}
    return {str(key): int(value) for key, value in payload.items() if int(value) >= 0}


def generate_command(args: argparse.Namespace) -> None:
    topology, counts, metadata = generate_whole_graph(
        args.count_checkpoint,
        args.program_graph_checkpoint,
        float(args.site_x),
        float(args.site_y),
        explicit_counts_from_args(args),
        int(args.seed),
        args.edge_threshold,
        torch.device(args.device),
    )
    write_json(args.output, {"topology": topology, "counts": counts, "metadata": metadata})
    print(json.dumps({"output": str(args.output), **metadata}, indent=2))


def main() -> None:
    args = parse_args()
    if args.cmd == "train-counts":
        train_counts(args)
    elif args.cmd == "generate":
        generate_command(args)


if __name__ == "__main__":
    main()
