#!/usr/bin/env python3
"""Smoke/overfit trainer for learned group-level topology edges.

This is not a formal V6 topology generator. It validates that Phase10
functional groups can be converted into pairwise training samples, learned by a
small classifier, exported as target/predicted topology JSON, and scored with
P1-style edge precision/recall.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spatial_modal_infer.config import ROOM_TYPES  # noqa: E402
from scripts.train_v5_spatial.v6_multipart_decoder import (  # noqa: E402
    build_target_topology,
    group_bbox,
    read_json,
    room_floors,
    write_json,
)


DEFAULT_PHASE10 = ROOT / "data" / "phase10_functional_parts" / "samples"
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_topology_learner_smoke"
VOXEL_MM = 300.0
SITE_NORMALIZER_MM = 26400.0
TYPE_TO_ID = {room_type: index for index, room_type in enumerate(ROOM_TYPES)}
FEATURE_MODES = ("full", "program_only", "program_size", "program_size_position")
SIZE_PRIOR_DEFAULT = (0.0, 0.0, 0.0, 0.0)
POSITION_PRIOR_DEFAULT = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TopologyNode:
    house_id: str
    node_id: str
    room_type: str
    site: tuple[float, float]
    floors: tuple[int, ...]
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]
    size_prior: tuple[float, float, float, float] = SIZE_PRIOR_DEFAULT
    position_prior: tuple[float, float, float, float] = POSITION_PRIOR_DEFAULT


@dataclass(frozen=True)
class PairSample:
    house_id: str
    source: str
    target: str
    left: TopologyNode
    right: TopologyNode
    label: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="full",
        help=(
            "full keeps target group-box geometry; program_only removes target "
            "box and center-distance features; program_size additionally uses "
            "predicted size priors from v6_size_area_head.py; "
            "program_size_position also uses coarse center/zone priors."
        ),
    )
    parser.add_argument(
        "--size-conditioning-dir",
        type=Path,
        help=(
            "Optional directory containing size_predictions/<house_id>/predicted_sizes.json "
            "from v6_size_area_head.py. Required for feature-mode program_size."
        ),
    )
    parser.add_argument(
        "--position-conditioning-dir",
        type=Path,
        help=(
            "Optional directory containing position_predictions/<house_id>/predicted_positions.json "
            "from v6_position_head.py. Required for feature-mode program_size_position."
        ),
    )
    return parser.parse_args()


def node_feature(node: TopologyNode, feature_mode: str = "full") -> list[float]:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    one_hot = [0.0] * len(ROOM_TYPES)
    one_hot[TYPE_TO_ID.get(node.room_type, 0)] = 1.0
    site_x, site_y = node.site
    program = one_hot + [
        1.0 if 1 in node.floors else 0.0,
        1.0 if 2 in node.floors else 0.0,
        site_x / SITE_NORMALIZER_MM,
        site_y / SITE_NORMALIZER_MM,
        (site_x * site_y) / (SITE_NORMALIZER_MM * SITE_NORMALIZER_MM),
    ]
    if feature_mode == "program_only":
        return program
    if feature_mode == "program_size":
        return program + list(node.size_prior)
    if feature_mode == "program_size_position":
        return program + list(node.size_prior) + list(node.position_prior)
    x0, y0, z0 = node.box_min
    x1, y1, z1 = node.box_max
    return program + [
        x0 / max(site_x, 1.0),
        y0 / max(site_y, 1.0),
        z0 / 6000.0,
        x1 / max(site_x, 1.0),
        y1 / max(site_y, 1.0),
        z1 / 6000.0,
        max(x1 - x0, VOXEL_MM) / max(site_x, 1.0),
        max(y1 - y0, VOXEL_MM) / max(site_y, 1.0),
        max(z1 - z0, VOXEL_MM) / 6000.0,
    ]


def pair_feature(sample: PairSample, feature_mode: str = "full") -> torch.Tensor:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    left = sample.left
    right = sample.right
    relation = [
        1.0 if set(left.floors) & set(right.floors) else 0.0,
    ]
    if feature_mode == "full":
        lx = (left.box_min[0] + left.box_max[0]) * 0.5 / max(left.site[0], 1.0)
        ly = (left.box_min[1] + left.box_max[1]) * 0.5 / max(left.site[1], 1.0)
        rx = (right.box_min[0] + right.box_max[0]) * 0.5 / max(right.site[0], 1.0)
        ry = (right.box_min[1] + right.box_max[1]) * 0.5 / max(right.site[1], 1.0)
        relation.extend([abs(lx - rx), abs(ly - ry)])
    elif feature_mode == "program_size":
        relation.extend(
            [
                abs(left.size_prior[0] - right.size_prior[0]),
                abs(left.size_prior[1] - right.size_prior[1]),
                abs(left.size_prior[2] - right.size_prior[2]),
                abs(left.size_prior[3] - right.size_prior[3]),
            ]
        )
    elif feature_mode == "program_size_position":
        relation.extend(
            [
                abs(left.size_prior[0] - right.size_prior[0]),
                abs(left.size_prior[1] - right.size_prior[1]),
                abs(left.size_prior[2] - right.size_prior[2]),
                abs(left.size_prior[3] - right.size_prior[3]),
                abs(left.position_prior[0] - right.position_prior[0]),
                abs(left.position_prior[1] - right.position_prior[1]),
                1.0 if int(round(left.position_prior[2] * 2.0)) == int(round(right.position_prior[2] * 2.0)) else 0.0,
                1.0 if int(round(left.position_prior[3] * 2.0)) == int(round(right.position_prior[3] * 2.0)) else 0.0,
            ]
        )
    return torch.tensor(
        node_feature(left, feature_mode) + node_feature(right, feature_mode) + relation,
        dtype=torch.float32,
    )


def rooms_by_group(source: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for room in source.get("rooms", []):
        group_id = str(room.get("functional_id", room["id"]))
        groups.setdefault(group_id, []).append(room)
    return groups


def read_size_priors(path: Path | None) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    if path is None:
        return {}
    priors: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    root = Path(path) / "size_predictions"
    for size_path in sorted(root.glob("*/predicted_sizes.json")):
        payload = read_json(size_path)
        house_id = str(payload.get("house_id", size_path.parent.name))
        group_priors = {}
        for group in payload.get("groups", []):
            predicted = group.get("predicted", {})
            group_priors[str(group["functional_id"])] = (
                float(predicted.get("area_ratio", 0.0)),
                float(predicted.get("width_ratio", 0.0)),
                float(predicted.get("depth_ratio", 0.0)),
                float(predicted.get("part_count", 0.0)) / 8.0,
            )
        priors[house_id] = group_priors
    return priors


def read_position_priors(path: Path | None) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    if path is None:
        return {}
    priors: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    root = Path(path) / "position_predictions"
    for position_path in sorted(root.glob("*/predicted_positions.json")):
        payload = read_json(position_path)
        house_id = str(payload.get("house_id", position_path.parent.name))
        group_priors = {}
        for group in payload.get("groups", []):
            predicted = group.get("predicted", {})
            group_priors[str(group["functional_id"])] = (
                float(predicted.get("center_x_ratio", 0.0)),
                float(predicted.get("center_y_ratio", 0.0)),
                float(predicted.get("zone_x", 0.0)) / 2.0,
                float(predicted.get("zone_y", 0.0)) / 2.0,
            )
        priors[house_id] = group_priors
    return priors


def load_house_pair_samples(
    path: Path,
    size_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None = None,
    position_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None = None,
) -> tuple[dict[str, Any], list[PairSample]]:
    source = read_json(path)
    house_id = str(source["house_id"])
    site = source["metadata"]["building_size"]
    site_xy = (float(site["x"]), float(site["y"]))
    grouped = rooms_by_group(source)
    topology = build_target_topology(source)
    target_edges = {
        tuple(sorted((str(edge["source"]), str(edge["target"]))))
        for edge in topology.get("edges", [])
        if edge.get("relation", "horizontal") == "horizontal"
    }
    nodes = []
    house_size_priors = (size_priors or {}).get(house_id, {})
    house_position_priors = (position_priors or {}).get(house_id, {})
    for group in source.get("functional_groups", []):
        group_id = str(group["functional_id"])
        parts = grouped.get(group_id, [])
        if not parts:
            continue
        box_min, box_max = group_bbox(parts)
        floors = tuple(sorted({floor for part in parts for floor in room_floors(part)}))
        nodes.append(
            TopologyNode(
                house_id=house_id,
                node_id=group_id,
                room_type=str(group["type"]),
                site=site_xy,
                floors=floors,
                box_min=tuple(float(value) for value in box_min),
                box_max=tuple(float(value) for value in box_max),
                size_prior=house_size_priors.get(group_id, SIZE_PRIOR_DEFAULT),
                position_prior=house_position_priors.get(group_id, POSITION_PRIOR_DEFAULT),
            )
        )
    pairs = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            source_id, target_id = sorted((left.node_id, right.node_id))
            label = 1.0 if (source_id, target_id) in target_edges else 0.0
            pairs.append(
                PairSample(
                    house_id=house_id,
                    source=source_id,
                    target=target_id,
                    left=left,
                    right=right,
                    label=label,
                )
            )
    return topology, pairs


def load_pair_samples(
    phase10_dir: Path,
    max_houses: int | None,
    size_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None = None,
    position_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None = None,
) -> tuple[list[Path], dict[str, dict[str, Any]], list[PairSample]]:
    paths = sorted(Path(phase10_dir).glob("house_*.json"))
    if max_houses is not None:
        paths = paths[:max_houses]
    topologies = {}
    pairs = []
    for path in paths:
        topology, house_pairs = load_house_pair_samples(path, size_priors, position_priors)
        topologies[str(read_json(path)["house_id"])] = topology
        pairs.extend(house_pairs)
    return paths, topologies, pairs


class TopologyPairDataset(Dataset):
    def __init__(self, samples: list[PairSample], feature_mode: str = "full") -> None:
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"unknown feature_mode: {feature_mode}")
        self.samples = samples
        self.feature_mode = feature_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "features": pair_feature(sample, self.feature_mode),
            "label": torch.tensor([sample.label], dtype=torch.float32),
        }


class TopologyEdgeClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def edge_metrics(probabilities: list[float], labels: list[float], threshold: float = 0.5) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for probability, label in zip(probabilities, labels):
        pred = probability >= threshold
        truth = label >= 0.5
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(
    model: TopologyEdgeClassifier,
    samples: list[PairSample],
    device: torch.device,
    feature_mode: str,
) -> dict[str, Any]:
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for sample in samples:
            logits = model(pair_feature(sample, feature_mode).unsqueeze(0).to(device))
            probabilities.append(float(torch.sigmoid(logits)[0, 0].cpu()))
            labels.append(sample.label)
    positives = sum(1 for label in labels if label >= 0.5)
    negatives = len(labels) - positives
    return {
        "pair_count": len(samples),
        "positive_edge_count": positives,
        "negative_pair_count": negatives,
        **edge_metrics(probabilities, labels),
    }


def predicted_topologies(
    model: TopologyEdgeClassifier,
    samples: list[PairSample],
    target_topologies: dict[str, dict[str, Any]],
    device: torch.device,
    feature_mode: str,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    edges_by_house: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for sample in samples:
            probability = float(
                torch.sigmoid(model(pair_feature(sample, feature_mode).unsqueeze(0).to(device)))[0, 0].cpu()
            )
            if probability < threshold:
                continue
            edges_by_house.setdefault(sample.house_id, []).append(
                {
                    "source": sample.source,
                    "target": sample.target,
                    "relation": "horizontal",
                    "probability": probability,
                }
            )
    output = {}
    for house_id, target in target_topologies.items():
        edges = sorted(
            edges_by_house.get(house_id, []),
            key=lambda edge: (edge["source"], edge["target"]),
        )
        required = sorted({tuple(sorted((edge["source"], edge["target"]))) for edge in edges})
        output[house_id] = {
            "schema": "graphspace_v6_phase14_predicted_topology_v1",
            "nodes": target.get("nodes", []),
            "edges": edges,
            "required_edges": [list(edge) for edge in required],
            "source": "learned_pairwise_topology_classifier_smoke",
        }
    return output


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    size_priors = read_size_priors(args.size_conditioning_dir)
    position_priors = read_position_priors(args.position_conditioning_dir)
    if args.feature_mode == "program_size" and not size_priors:
        raise ValueError("feature-mode program_size requires --size-conditioning-dir")
    if args.feature_mode == "program_size_position" and (not size_priors or not position_priors):
        raise ValueError("feature-mode program_size_position requires --size-conditioning-dir and --position-conditioning-dir")
    source_paths, target_topologies, pairs = load_pair_samples(
        args.phase10_dir,
        args.max_houses,
        size_priors,
        position_priors,
    )
    if not pairs:
        raise ValueError("no topology pair samples found")
    dataset = TopologyPairDataset(pairs, args.feature_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    feature_dim = int(dataset[0]["features"].numel())
    model = TopologyEdgeClassifier(feature_dim).to(device)
    positive_count = sum(1 for sample in pairs if sample.label >= 0.5)
    negative_count = len(pairs) - positive_count
    pos_weight = torch.tensor([negative_count / max(positive_count, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            metrics = evaluate_model(model, pairs, device, args.feature_mode)
            metrics["epoch"] = epoch
            metrics["loss"] = total_loss / max(total_batches, 1)
            history.append(metrics)
            print(
                f"epoch={epoch:04d} loss={metrics['loss']:.6f} "
                f"f1={metrics['f1']:.4f} recall={metrics['recall']:.4f} "
                f"precision={metrics['precision']:.4f}"
            )
    final_metrics = evaluate_model(model, pairs, device, args.feature_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "graphspace_v6_topology_learner_smoke_v1",
            "model": model.state_dict(),
            "config": {
                "feature_dim": feature_dim,
                "feature_mode": args.feature_mode,
                "size_conditioning_dir": str(args.size_conditioning_dir) if args.size_conditioning_dir else None,
                "position_conditioning_dir": (
                    str(args.position_conditioning_dir) if args.position_conditioning_dir else None
                ),
                "room_types": ROOM_TYPES,
                "threshold": 0.5,
            },
            "source_phase10_dir": str(args.phase10_dir),
        },
        args.output_dir / "topology_learner.pt",
    )
    predicted = predicted_topologies(model, pairs, target_topologies, device, args.feature_mode)
    for house_id, topology in target_topologies.items():
        house_dir = args.output_dir / "topologies" / house_id
        write_json(house_dir / "target_topology.json", topology)
        write_json(house_dir / "predicted_topology.json", predicted[house_id])
    summary = {
        "schema": "graphspace_v6_topology_learner_smoke_summary_v1",
        "purpose": (
            "Interface validation only: learn group-level adjacency edges from "
            "Phase10 inferred functional groups; not a formal V6 topology generator."
        ),
        "phase10_dir": str(args.phase10_dir),
        "house_count": len(source_paths),
        "pair_count": len(pairs),
        "positive_edge_count": positive_count,
        "negative_pair_count": negative_count,
        "epochs": args.epochs,
        "feature_mode": args.feature_mode,
        "size_conditioning_dir": str(args.size_conditioning_dir) if args.size_conditioning_dir else None,
        "position_conditioning_dir": str(args.position_conditioning_dir) if args.position_conditioning_dir else None,
        "size_conditioned_house_count": len(size_priors),
        "position_conditioned_house_count": len(position_priors),
        "final_metrics": final_metrics,
        "history": history,
        "outputs": {
            "checkpoint": str(args.output_dir / "topology_learner.pt"),
            "topologies": str(args.output_dir / "topologies"),
        },
        "formal_v6_training_ready": False,
        "blocking_reason": (
            "This is a small overfit check and uses Phase10 inferred groups. "
            "In full mode it also uses target group boxes; in program_only mode "
            "it removes those target geometry features; in program_size mode it "
            "uses predicted size priors but still depends on known functional "
            "groups rather than generating topology from raw user length/width "
            "conditions alone."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
