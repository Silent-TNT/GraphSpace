#!/usr/bin/env python3
"""Train voxel-conditioned access topology from generated/ground-truth massing.

This model learns access-vs-blocked relation labels from 3D pair voxel tensors.
The labels are weak supervision derived from the user-defined transition
function set; inference uses the trained model, not rule-time edge typing.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOM_TYPES = [
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
]
TYPE_TO_ID = {room_type: index for index, room_type in enumerate(ROOM_TYPES)}
PASSAGE_TYPES = {
    "entryway",
    "living_room",
    "dining_room",
    "kitchen",
    "corridor",
    "multi_purpose",
}
VERTICAL_TYPES = {"stairs"}
GRID = (32, 32, 8)
TYPE_COLORS = {
    "entryway": "#d9a441",
    "living_room": "#4c78a8",
    "dining_room": "#72b7b2",
    "kitchen": "#f58518",
    "bedroom": "#54a24b",
    "bathroom": "#b279a2",
    "corridor": "#8c8c8c",
    "stairs": "#e45756",
    "utility": "#9d755d",
    "balcony": "#76b7b2",
    "multi_purpose": "#59a14f",
}
TOL = 1e-6


@dataclass(frozen=True)
class GroupBox:
    group_id: str
    room_type: str
    floors: tuple[int, ...]
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]
    site: tuple[float, float, float]


@dataclass(frozen=True)
class PairRecord:
    house_id: str
    left: GroupBox
    right: GroupBox
    label: int
    contact_area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--phase10-dir", type=Path, default=Path("data/phase10_functional_parts/samples"))
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=96)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--max-houses", type=int)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=20260626)
    infer = sub.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--case-dir", action="append", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def room_floors(room: dict[str, Any]) -> tuple[int, ...]:
    if room.get("floors"):
        return tuple(sorted({int(value) for value in room["floors"]}))
    return (int(room.get("floor", 1)),)


def groups_from_rooms(rooms: list[dict[str, Any]], site: tuple[float, float, float]) -> list[GroupBox]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for room in rooms:
        grouped.setdefault(str(room.get("functional_id", room["id"])), []).append(room)
    groups = []
    for group_id, parts in grouped.items():
        mins = tuple(min(float(part["box_min"][axis]) for part in parts) for axis in range(3))
        maxs = tuple(max(float(part["box_max"][axis]) for part in parts) for axis in range(3))
        floors = tuple(sorted({floor for part in parts for floor in room_floors(part)}))
        groups.append(
            GroupBox(
                group_id=group_id,
                room_type=str(parts[0].get("type", "unknown")),
                floors=floors,
                box_min=mins,
                box_max=maxs,
                site=site,
            )
        )
    return sorted(groups, key=lambda item: item.group_id)


def groups_from_phase10(path: Path) -> tuple[str, list[GroupBox]]:
    payload = read_json(path)
    size = payload.get("metadata", {}).get("building_size", {})
    site = (float(size.get("x", 1.0)), float(size.get("y", 1.0)), float(size.get("z", 6000.0)))
    return str(payload["house_id"]), groups_from_rooms(payload.get("rooms", []), site)


def groups_from_case(case_dir: Path) -> tuple[str, list[GroupBox]]:
    payload = read_json(case_dir / "generated_layout.json")
    size = payload.get("metadata", {}).get("building_size", {})
    site = (float(size.get("x", 1.0)), float(size.get("y", 1.0)), float(size.get("z", 6000.0)))
    return str(payload.get("house_id", case_dir.name)), groups_from_rooms(payload.get("rooms", []), site)


def overlap_length(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def face_contact_area(left: GroupBox, right: GroupBox) -> float:
    lx0, ly0, lz0 = left.box_min
    lx1, ly1, lz1 = left.box_max
    rx0, ry0, rz0 = right.box_min
    rx1, ry1, rz1 = right.box_max
    z_overlap = overlap_length(lz0, lz1, rz0, rz1)
    if z_overlap <= TOL:
        return 0.0
    if abs(lx1 - rx0) <= TOL or abs(rx1 - lx0) <= TOL:
        return overlap_length(ly0, ly1, ry0, ry1) * z_overlap
    if abs(ly1 - ry0) <= TOL or abs(ry1 - ly0) <= TOL:
        return overlap_length(lx0, lx1, rx0, rx1) * z_overlap
    return 0.0


def weak_access_label(left: GroupBox, right: GroupBox) -> int:
    if left.room_type in VERTICAL_TYPES or right.room_type in VERTICAL_TYPES:
        other_type = right.room_type if left.room_type in VERTICAL_TYPES else left.room_type
        return 1 if other_type in PASSAGE_TYPES else 0
    return 1 if left.room_type in PASSAGE_TYPES or right.room_type in PASSAGE_TYPES else 0


def pair_records_for_groups(house_id: str, groups: list[GroupBox]) -> list[PairRecord]:
    records = []
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            area = face_contact_area(left, right)
            if area <= TOL:
                continue
            records.append(PairRecord(house_id, left, right, weak_access_label(left, right), area))
    return records


def voxel_bounds(box: GroupBox) -> tuple[int, int, int, int, int, int]:
    gx, gy, gz = GRID
    sx, sy, sz = box.site
    x0 = max(0, min(gx, int(math.floor(box.box_min[0] / sx * gx))))
    x1 = max(x0 + 1, min(gx, int(math.ceil(box.box_max[0] / sx * gx))))
    y0 = max(0, min(gy, int(math.floor(box.box_min[1] / sy * gy))))
    y1 = max(y0 + 1, min(gy, int(math.ceil(box.box_max[1] / sy * gy))))
    z0 = max(0, min(gz, int(math.floor(box.box_min[2] / sz * gz))))
    z1 = max(z0 + 1, min(gz, int(math.ceil(box.box_max[2] / sz * gz))))
    return x0, x1, y0, y1, z0, z1


def pair_voxel(record: PairRecord) -> np.ndarray:
    gx, gy, gz = GRID
    voxel = np.zeros((2, gz, gy, gx), dtype=np.float32)
    for channel, box in enumerate((record.left, record.right)):
        x0, x1, y0, y1, z0, z1 = voxel_bounds(box)
        voxel[channel, z0:z1, y0:y1, x0:x1] = 1.0
    return voxel


def type_one_hot(room_type: str) -> list[float]:
    result = [0.0] * len(ROOM_TYPES)
    if room_type in TYPE_TO_ID:
        result[TYPE_TO_ID[room_type]] = 1.0
    return result


def pair_features(record: PairRecord) -> np.ndarray:
    left = record.left
    right = record.right
    sx, sy, sz = left.site
    lc = [(left.box_min[i] + left.box_max[i]) * 0.5 for i in range(3)]
    rc = [(right.box_min[i] + right.box_max[i]) * 0.5 for i in range(3)]
    dims_l = [max(left.box_max[i] - left.box_min[i], 1.0) for i in range(3)]
    dims_r = [max(right.box_max[i] - right.box_min[i], 1.0) for i in range(3)]
    feature = (
        type_one_hot(left.room_type)
        + type_one_hot(right.room_type)
        + [
            1.0 if set(left.floors) & set(right.floors) else 0.0,
            record.contact_area / max(sx * sy, 1.0),
            abs(lc[0] - rc[0]) / max(sx, 1.0),
            abs(lc[1] - rc[1]) / max(sy, 1.0),
            abs(lc[2] - rc[2]) / max(sz, 1.0),
            dims_l[0] / max(sx, 1.0),
            dims_l[1] / max(sy, 1.0),
            dims_l[2] / max(sz, 1.0),
            dims_r[0] / max(sx, 1.0),
            dims_r[1] / max(sy, 1.0),
            dims_r[2] / max(sz, 1.0),
        ]
    )
    return np.asarray(feature, dtype=np.float32)


class AccessPairDataset(Dataset):
    def __init__(self, records: list[PairRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        return {
            "voxel": torch.from_numpy(pair_voxel(record)),
            "features": torch.from_numpy(pair_features(record)),
            "label": torch.tensor(record.label, dtype=torch.long),
        }


class VoxelAccessModel(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.voxel_encoder = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(16 + feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, voxel: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        encoded = self.voxel_encoder(voxel)
        return self.head(torch.cat([encoded, features], dim=1))


def split_records(records: list[PairRecord], seed: int) -> tuple[list[PairRecord], list[PairRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(round(len(shuffled) * 0.8)))
    return shuffled[:cut], shuffled[cut:]


def metrics_for(model: VoxelAccessModel, records: list[PairRecord], device: torch.device, batch_size: int) -> dict[str, Any]:
    loader = DataLoader(AccessPairDataset(records), batch_size=batch_size, shuffle=False)
    tp = fp = tn = fn = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["voxel"].to(device), batch["features"].to(device))
            pred = logits.argmax(dim=1).cpu()
            label = batch["label"]
            tp += int(((pred == 1) & (label == 1)).sum())
            fp += int(((pred == 1) & (label == 0)).sum())
            tn += int(((pred == 0) & (label == 0)).sum())
            fn += int(((pred == 0) & (label == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_training_records(phase10_dir: Path, max_houses: int | None) -> list[PairRecord]:
    records = []
    paths = sorted(phase10_dir.glob("house_*.json"))
    if max_houses is not None:
        paths = paths[:max_houses]
    for path in paths:
        house_id, groups = groups_from_phase10(path)
        records.extend(pair_records_for_groups(house_id, groups))
    return records


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    records = load_training_records(args.phase10_dir, args.max_houses)
    if not records:
        raise ValueError("no contacting pair records found")
    train_records, val_records = split_records(records, args.seed)
    feature_dim = int(pair_features(records[0]).shape[0])
    model = VoxelAccessModel(feature_dim).to(device)
    labels = torch.tensor([record.label for record in train_records], dtype=torch.long)
    neg = int((labels == 0).sum())
    pos = int((labels == 1).sum())
    weights = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = DataLoader(AccessPairDataset(train_records), batch_size=args.batch_size, shuffle=True)
        total = 0.0
        seen = 0
        for batch in loader:
            logits = model(batch["voxel"].to(device), batch["features"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(batch["label"].shape[0])
            seen += int(batch["label"].shape[0])
        row = {
            "epoch": epoch,
            "loss": total / max(seen, 1),
            "validation": metrics_for(model, val_records, device, args.batch_size),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['loss']:.4f} "
            f"val_f1={row['validation']['f1']:.4f} "
            f"precision={row['validation']['precision']:.4f} "
            f"recall={row['validation']['recall']:.4f}"
        )
    checkpoint = {
        "schema": "graphspace_voxel_access_topology_model_v1",
        "model": model.state_dict(),
        "config": {
            "feature_dim": feature_dim,
            "grid": GRID,
            "room_types": ROOM_TYPES,
            "passage_types": sorted(PASSAGE_TYPES),
            "vertical_types": sorted(VERTICAL_TYPES),
        },
    }
    torch.save(checkpoint, args.output_dir / "voxel_access_topology_model.pt")
    summary = {
        "schema": "graphspace_voxel_access_training_summary_v1",
        "phase10_dir": str(args.phase10_dir),
        "pair_count": len(records),
        "train_pair_count": len(train_records),
        "validation_pair_count": len(val_records),
        "positive_access_count": sum(record.label for record in records),
        "blocked_count": sum(1 - record.label for record in records),
        "epochs": args.epochs,
        "final_validation": history[-1]["validation"],
        "history": history,
        "outputs": {"checkpoint": str(args.output_dir / "voxel_access_topology_model.pt")},
        "label_note": "Weak supervision from user-defined passage functions; inference is model-predicted from 3D pair voxels.",
    }
    write_json(args.output_dir / "summary.json", summary)


def load_model(path: Path, device: torch.device) -> VoxelAccessModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = VoxelAccessModel(int(config["feature_dim"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def contact_voxels(record: PairRecord) -> list[list[int]]:
    left = pair_voxel(record)[0] > 0.5
    right = pair_voxel(record)[1] > 0.5
    coords: set[tuple[int, int, int]] = set()
    for dz, dy, dx in ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)):
        shifted = np.zeros_like(right)
        z_src = slice(max(0, -dz), right.shape[0] - max(0, dz))
        y_src = slice(max(0, -dy), right.shape[1] - max(0, dy))
        x_src = slice(max(0, -dx), right.shape[2] - max(0, dx))
        z_dst = slice(max(0, dz), right.shape[0] - max(0, -dz))
        y_dst = slice(max(0, dy), right.shape[1] - max(0, -dy))
        x_dst = slice(max(0, dx), right.shape[2] - max(0, -dx))
        shifted[z_dst, y_dst, x_dst] = right[z_src, y_src, x_src]
        for z, y, x in np.argwhere(left & shifted):
            coords.add((int(x), int(y), int(z)))
    return [[x, y, z] for x, y, z in sorted(coords)[:64]]


def predict_case(model: VoxelAccessModel, case_dir: Path, device: torch.device) -> dict[str, Any]:
    house_id, groups = groups_from_case(case_dir)
    records = pair_records_for_groups(house_id, groups)
    nodes = [
        {
            "id": group.group_id,
            "type": group.room_type,
            "floors": list(group.floors),
            "box_min": list(group.box_min),
            "box_max": list(group.box_max),
            "center": [(group.box_min[0] + group.box_max[0]) * 0.5, (group.box_min[1] + group.box_max[1]) * 0.5],
            "node_type": "room_instance",
        }
        for group in groups
    ]
    edges = []
    model.eval()
    with torch.no_grad():
        for record in records:
            voxel = torch.from_numpy(pair_voxel(record)).unsqueeze(0).to(device)
            features = torch.from_numpy(pair_features(record)).unsqueeze(0).to(device)
            prob = torch.softmax(model(voxel, features), dim=1)[0].cpu().tolist()
            predicted = int(prob[1] >= prob[0])
            edges.append(
                {
                    "source": record.left.group_id,
                    "target": record.right.group_id,
                    "edge_type": "voxel_access_relation" if predicted else "voxel_blocked_direct_access",
                    "access_probability": float(prob[1]),
                    "blocked_probability": float(prob[0]),
                    "contact_area": record.contact_area,
                    "relation_voxels": contact_voxels(record),
                }
            )
    return {
        "schema": "graphspace_voxel_access_topology_prediction_v1",
        "source_case": str(case_dir),
        "house_id": house_id,
        "grid": {"x": GRID[0], "y": GRID[1], "z": GRID[2]},
        "nodes": nodes,
        "edges": edges,
        "metrics": {
            "contact_pair_count": len(edges),
            "predicted_access_count": sum(edge["edge_type"] == "voxel_access_relation" for edge in edges),
            "predicted_blocked_count": sum(edge["edge_type"] == "voxel_blocked_direct_access" for edge in edges),
            "predicted_stairs_bedroom_access_count": sum(
                edge["edge_type"] == "voxel_access_relation"
                and {
                    next(node["type"] for node in nodes if node["id"] == edge["source"]),
                    next(node["type"] for node in nodes if node["id"] == edge["target"]),
                }
                == {"stairs", "bedroom"}
                for edge in edges
            ),
        },
    }


def render(topology: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    graph = nx.Graph()
    nodes = {str(node["id"]): node for node in topology["nodes"]}
    for node_id in nodes:
        graph.add_node(node_id)
    access = [(edge["source"], edge["target"]) for edge in topology["edges"] if edge["edge_type"] == "voxel_access_relation"]
    blocked = [
        (edge["source"], edge["target"])
        for edge in topology["edges"]
        if edge["edge_type"] == "voxel_blocked_direct_access"
    ]
    for edge in [*access, *blocked]:
        graph.add_edge(*edge)
    pos = {
        node_id: (
            float(node["center"][0]) / 3000.0,
            float(node["center"][1]) / 3000.0 + (2.5 if 2 in node.get("floors", []) and 1 not in node.get("floors", []) else 0.0),
        )
        for node_id, node in nodes.items()
    }
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f"Voxel-predicted access topology | {Path(str(topology['source_case'])).name}", fontsize=14)
    ax.axis("off")
    nx.draw_networkx_edges(graph, pos, edgelist=blocked, edge_color="#dc2626", width=1.7, alpha=0.65, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=access, edge_color="#111827", width=2.5, alpha=0.90, ax=ax)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=[TYPE_COLORS.get(nodes[node_id]["type"], "#8f8f8f") for node_id in graph.nodes],
        node_size=950,
        edgecolors="#ffffff",
        linewidths=1.5,
        ax=ax,
    )
    labels = {
        node_id: f"{nodes[node_id]['type'].replace('_', ' ')}\n{node_id.split('_')[-1]}"
        for node_id in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)
    m = topology["metrics"]
    ax.text(
        0.01,
        0.01,
        (
            "black=model voxel_access_relation  red=model voxel_blocked_direct_access\n"
            f"access={m['predicted_access_count']}  blocked={m['predicted_blocked_count']}  "
            f"stairs-bedroom access={m['predicted_stairs_bedroom_access_count']}  grid={topology['grid']}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    outputs = []
    for case_dir in args.case_dir:
        topology = predict_case(model, case_dir, device)
        json_path = args.output_dir / f"{case_dir.name}_voxel_access_topology.json"
        png_path = args.output_dir / f"{case_dir.name}_voxel_access_topology.png"
        write_json(json_path, topology)
        render(topology, png_path)
        outputs.append({"json": str(json_path), "png": str(png_path), "metrics": topology["metrics"]})
    print(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer":
        infer(args)


if __name__ == "__main__":
    main()
