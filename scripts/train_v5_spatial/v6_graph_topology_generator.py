#!/usr/bin/env python3
"""Graph-level topology generator smoke for V6 functional groups.

This is still a small overfit/interface validation. Unlike the Phase14 pairwise
classifier, decoding happens at whole-house graph level: the model predicts an
edge budget, selects the strongest candidate edges under that budget, and then
adds the minimum high-scoring edges needed to keep the functional graph
connected.
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spatial_modal_infer.config import ROOM_TYPES  # noqa: E402
from scripts.train_v5_spatial.v6_multipart_decoder import read_json, write_json  # noqa: E402
from scripts.train_v5_spatial.v6_topology_learner import (  # noqa: E402
    DEFAULT_PHASE10,
    SITE_NORMALIZER_MM,
    PairSample,
    TopologyNode,
    edge_metrics,
    load_house_pair_samples,
    node_feature,
    read_position_priors,
    read_size_priors,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "v6_graph_topology_generator_smoke"
FEATURE_MODES = ("program_only", "program_size", "program_size_position")


@dataclass(frozen=True)
class GraphTopologySample:
    house_id: str
    target_topology: dict[str, Any]
    nodes: list[TopologyNode]
    pairs: list[PairSample]

    @property
    def candidate_count(self) -> int:
        return len(self.pairs)

    @property
    def target_edge_count(self) -> int:
        return sum(1 for pair in self.pairs if pair.label >= 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase10-dir", type=Path, default=DEFAULT_PHASE10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="program_size")
    parser.add_argument(
        "--size-conditioning-dir",
        type=Path,
        help="Directory containing size_predictions/<house_id>/predicted_sizes.json.",
    )
    parser.add_argument(
        "--position-conditioning-dir",
        type=Path,
        help="Directory containing position_predictions/<house_id>/predicted_positions.json.",
    )
    parser.add_argument(
        "--no-connectivity",
        action="store_true",
        help="Disable decode-time connectivity repair. Kept only for ablation.",
    )
    return parser.parse_args()


def ordered_nodes_from_pairs(pairs: list[PairSample]) -> list[TopologyNode]:
    nodes: dict[str, TopologyNode] = {}
    for pair in pairs:
        nodes[pair.left.node_id] = pair.left
        nodes[pair.right.node_id] = pair.right
    return [nodes[node_id] for node_id in sorted(nodes)]


def graph_pair_relation(pair: PairSample, feature_mode: str) -> list[float]:
    relation = [1.0 if set(pair.left.floors) & set(pair.right.floors) else 0.0]
    if feature_mode == "program_size":
        relation.extend(
            [
                abs(pair.left.size_prior[0] - pair.right.size_prior[0]),
                abs(pair.left.size_prior[1] - pair.right.size_prior[1]),
                abs(pair.left.size_prior[2] - pair.right.size_prior[2]),
                abs(pair.left.size_prior[3] - pair.right.size_prior[3]),
            ]
        )
    elif feature_mode == "program_size_position":
        relation.extend(
            [
                abs(pair.left.size_prior[0] - pair.right.size_prior[0]),
                abs(pair.left.size_prior[1] - pair.right.size_prior[1]),
                abs(pair.left.size_prior[2] - pair.right.size_prior[2]),
                abs(pair.left.size_prior[3] - pair.right.size_prior[3]),
                abs(pair.left.position_prior[0] - pair.right.position_prior[0]),
                abs(pair.left.position_prior[1] - pair.right.position_prior[1]),
                1.0
                if int(round(pair.left.position_prior[2] * 2.0))
                == int(round(pair.right.position_prior[2] * 2.0))
                else 0.0,
                1.0
                if int(round(pair.left.position_prior[3] * 2.0))
                == int(round(pair.right.position_prior[3] * 2.0))
                else 0.0,
            ]
        )
    return relation


def load_graph_samples(
    phase10_dir: Path,
    max_houses: int | None,
    size_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None,
    position_priors: dict[str, dict[str, tuple[float, float, float, float]]] | None = None,
) -> list[GraphTopologySample]:
    paths = sorted(Path(phase10_dir).glob("house_*.json"))
    if max_houses is not None:
        paths = paths[:max_houses]
    samples: list[GraphTopologySample] = []
    for path in paths:
        target_topology, pairs = load_house_pair_samples(path, size_priors, position_priors)
        if not pairs:
            continue
        house_id = str(read_json(path)["house_id"])
        samples.append(
            GraphTopologySample(
                house_id=house_id,
                target_topology=target_topology,
                nodes=ordered_nodes_from_pairs(pairs),
                pairs=pairs,
            )
        )
    return samples


class GraphTopologyGenerator(nn.Module):
    def __init__(self, node_dim: int, relation_dim: int, hidden: int = 160) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 3 + relation_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.edge_count_head = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        pair_indices: torch.Tensor,
        relation_features: torch.Tensor,
        graph_stats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.node_encoder(node_features)
        left = encoded[pair_indices[:, 0]]
        right = encoded[pair_indices[:, 1]]
        edge_features = torch.cat([left, right, torch.abs(left - right), relation_features], dim=1)
        logits = self.edge_head(edge_features).squeeze(1)
        pooled = encoded.mean(dim=0)
        edge_ratio = self.edge_count_head(torch.cat([pooled, graph_stats], dim=0)).squeeze(0)
        return logits, edge_ratio


def tensors_for_graph(
    sample: GraphTopologySample,
    feature_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_ids = [node.node_id for node in sample.nodes]
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    node_features = torch.stack(
        [torch.tensor(node_feature(node, feature_mode), dtype=torch.float32) for node in sample.nodes]
    ).to(device)
    pair_indices = torch.tensor(
        [[node_index[pair.source], node_index[pair.target]] for pair in sample.pairs],
        dtype=torch.long,
        device=device,
    )
    relations = torch.tensor(
        [graph_pair_relation(pair, feature_mode) for pair in sample.pairs],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor([pair.label for pair in sample.pairs], dtype=torch.float32, device=device)
    graph_stats = torch.tensor(
        [
            len(sample.nodes) / 64.0,
            sample.candidate_count / 1024.0,
            sample.target_edge_count / max(sample.candidate_count, 1),
        ],
        dtype=torch.float32,
        device=device,
    )
    target_edge_ratio = torch.tensor(
        sample.target_edge_count / max(sample.candidate_count, 1),
        dtype=torch.float32,
        device=device,
    )
    return node_features, pair_indices, relations, labels, graph_stats, target_edge_ratio


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        self.parent[right_root] = left_root
        return True

    def component_count(self) -> int:
        return len({self.find(item) for item in self.parent})


def decode_graph_edges(
    pairs: list[PairSample],
    probabilities: list[float],
    edge_ratio: float,
    node_ids: list[str],
    enforce_connectivity: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    candidate_count = len(pairs)
    if candidate_count == 0:
        return [], {"predicted_edge_budget": 0, "candidate_count": 0, "connectivity_added_edges": 0}
    budget = int(round(edge_ratio * candidate_count))
    if enforce_connectivity and len(node_ids) > 1:
        budget = max(budget, len(node_ids) - 1)
    budget = max(0, min(candidate_count, budget))
    ranked = sorted(range(candidate_count), key=lambda index: probabilities[index], reverse=True)
    selected = set(ranked[:budget])
    connectivity_added = 0
    if enforce_connectivity and len(node_ids) > 1:
        union_find = UnionFind(node_ids)
        for index in selected:
            union_find.union(pairs[index].source, pairs[index].target)
        for index in ranked:
            if union_find.component_count() <= 1:
                break
            if index in selected:
                continue
            if union_find.union(pairs[index].source, pairs[index].target):
                selected.add(index)
                connectivity_added += 1
    selected_indices = sorted(selected, key=lambda index: (pairs[index].source, pairs[index].target))
    return selected_indices, {
        "predicted_edge_budget": budget,
        "candidate_count": candidate_count,
        "selected_edge_count": len(selected_indices),
        "connectivity_added_edges": connectivity_added,
        "connected_after_decode": _is_connected(node_ids, [pairs[index] for index in selected_indices]),
    }


def _is_connected(node_ids: list[str], selected_pairs: list[PairSample]) -> bool:
    if len(node_ids) <= 1:
        return True
    union_find = UnionFind(node_ids)
    for pair in selected_pairs:
        union_find.union(pair.source, pair.target)
    return union_find.component_count() == 1


def evaluate_selected_edges(sample: GraphTopologySample, selected_indices: list[int]) -> dict[str, Any]:
    selected = set(selected_indices)
    probabilities = [1.0 if index in selected else 0.0 for index in range(len(sample.pairs))]
    labels = [pair.label for pair in sample.pairs]
    metrics = edge_metrics(probabilities, labels, threshold=0.5)
    metrics.update(
        {
            "target_edge_count": sample.target_edge_count,
            "predicted_edge_count": len(selected_indices),
            "edge_count_abs_error": abs(len(selected_indices) - sample.target_edge_count),
        }
    )
    return metrics


def predict_graph(
    model: GraphTopologyGenerator,
    sample: GraphTopologySample,
    feature_mode: str,
    device: torch.device,
    enforce_connectivity: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        node_features, pair_indices, relations, _labels, graph_stats, _target_ratio = tensors_for_graph(
            sample, feature_mode, device
        )
        logits, edge_ratio = model(node_features, pair_indices, relations, graph_stats)
        probabilities = torch.sigmoid(logits).detach().cpu().tolist()
        decoded_ratio = float(edge_ratio.detach().cpu())
    node_ids = [node.node_id for node in sample.nodes]
    selected_indices, decode_info = decode_graph_edges(
        sample.pairs,
        probabilities,
        decoded_ratio,
        node_ids,
        enforce_connectivity=enforce_connectivity,
    )
    edges = [
        {
            "source": sample.pairs[index].source,
            "target": sample.pairs[index].target,
            "relation": "horizontal",
            "probability": probabilities[index],
        }
        for index in selected_indices
    ]
    required = sorted({tuple(sorted((edge["source"], edge["target"]))) for edge in edges})
    topology = {
        "schema": "graphspace_v6_graph_topology_generator_v1",
        "nodes": sample.target_topology.get("nodes", []),
        "edges": edges,
        "required_edges": [list(edge) for edge in required],
        "source": "learned_graph_level_topology_generator_smoke",
        "graph_decode": {
            **decode_info,
            "predicted_edge_ratio": decoded_ratio,
            "enforce_connectivity": enforce_connectivity,
        },
    }
    return topology, evaluate_selected_edges(sample, selected_indices)


def load_graph_topology_generator(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[GraphTopologyGenerator, str, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = GraphTopologyGenerator(
        node_dim=int(config["node_dim"]),
        relation_dim=int(config["relation_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    feature_mode = str(config.get("feature_mode", "program_size"))
    enforce_connectivity = bool(config.get("enforce_connectivity", True))
    return model, feature_mode, enforce_connectivity


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    size_priors = read_size_priors(args.size_conditioning_dir)
    position_priors = read_position_priors(args.position_conditioning_dir)
    if args.feature_mode == "program_size" and not size_priors:
        raise ValueError("feature-mode program_size requires --size-conditioning-dir")
    if args.feature_mode == "program_size_position" and (not size_priors or not position_priors):
        raise ValueError(
            "feature-mode program_size_position requires --size-conditioning-dir and --position-conditioning-dir"
        )
    samples = load_graph_samples(args.phase10_dir, args.max_houses, size_priors, position_priors)
    if not samples:
        raise ValueError("no graph topology samples found")
    feature_mode = args.feature_mode
    first = samples[0]
    node_dim = len(node_feature(first.nodes[0], feature_mode))
    relation_dim = len(graph_pair_relation(first.pairs[0], feature_mode))
    model = GraphTopologyGenerator(node_dim=node_dim, relation_dim=relation_dim).to(device)
    positive_count = sum(sample.target_edge_count for sample in samples)
    pair_count = sum(sample.candidate_count for sample in samples)
    negative_count = pair_count - positive_count
    pos_weight = torch.tensor(negative_count / max(positive_count, 1), dtype=torch.float32, device=device)
    edge_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    count_criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(samples)
        total_loss = 0.0
        for sample in samples:
            node_features, pair_indices, relations, labels, graph_stats, target_ratio = tensors_for_graph(
                sample, feature_mode, device
            )
            logits, edge_ratio = model(node_features, pair_indices, relations, graph_stats)
            edge_loss = edge_criterion(logits, labels)
            count_loss = count_criterion(edge_ratio, target_ratio)
            loss = edge_loss + 2.0 * count_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            metrics = evaluate_model(model, samples, feature_mode, device, not args.no_connectivity)
            metrics["epoch"] = epoch
            metrics["loss"] = total_loss / max(len(samples), 1)
            history.append(metrics)
            print(
                f"epoch={epoch:04d} loss={metrics['loss']:.6f} "
                f"f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} edge_mae={metrics['mean_edge_count_abs_error']:.2f}"
            )
    final_metrics = evaluate_model(model, samples, feature_mode, device, not args.no_connectivity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "graphspace_v6_graph_topology_generator_smoke_v1",
            "model": model.state_dict(),
            "config": {
                "node_dim": node_dim,
                "relation_dim": relation_dim,
                "feature_mode": feature_mode,
                "room_types": ROOM_TYPES,
                "size_conditioning_dir": str(args.size_conditioning_dir) if args.size_conditioning_dir else None,
                "position_conditioning_dir": (
                    str(args.position_conditioning_dir) if args.position_conditioning_dir else None
                ),
                "enforce_connectivity": not args.no_connectivity,
            },
            "source_phase10_dir": str(args.phase10_dir),
        },
        args.output_dir / "graph_topology_generator.pt",
    )
    per_house = {}
    for sample in samples:
        predicted, metrics = predict_graph(model, sample, feature_mode, device, not args.no_connectivity)
        house_dir = args.output_dir / "topologies" / sample.house_id
        write_json(house_dir / "target_topology.json", sample.target_topology)
        write_json(house_dir / "predicted_topology.json", predicted)
        per_house[sample.house_id] = {
            **metrics,
            "graph_decode": predicted["graph_decode"],
        }
    summary = {
        "schema": "graphspace_v6_graph_topology_generator_summary_v1",
        "purpose": (
            "Interface validation only: generate a whole-house sparse functional "
            "topology graph from program/size features, with graph-level edge "
            "budget and connectivity decode."
        ),
        "phase10_dir": str(args.phase10_dir),
        "house_count": len(samples),
        "pair_count": pair_count,
        "positive_edge_count": positive_count,
        "negative_pair_count": negative_count,
        "epochs": args.epochs,
        "feature_mode": feature_mode,
        "size_conditioning_dir": str(args.size_conditioning_dir) if args.size_conditioning_dir else None,
        "position_conditioning_dir": str(args.position_conditioning_dir) if args.position_conditioning_dir else None,
        "size_conditioned_house_count": len(size_priors),
        "position_conditioned_house_count": len(position_priors),
        "final_metrics": final_metrics,
        "per_house": per_house,
        "history": history,
        "outputs": {
            "checkpoint": str(args.output_dir / "graph_topology_generator.pt"),
            "topologies": str(args.output_dir / "topologies"),
        },
        "formal_v6_training_ready": False,
        "blocking_reason": (
            "This is a graph-level topology smoke on two Phase10 inferred houses. "
            "It still assumes the functional group list is known; a formal V6 "
            "generator must learn group completion, coarse zoning, and topology "
            "from raw user conditions before long training."
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def evaluate_model(
    model: GraphTopologyGenerator,
    samples: list[GraphTopologySample],
    feature_mode: str,
    device: torch.device,
    enforce_connectivity: bool,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    edge_errors = []
    predicted_counts = []
    target_counts = []
    connected_count = 0
    for sample in samples:
        predicted, metrics = predict_graph(model, sample, feature_mode, device, enforce_connectivity)
        tp += int(metrics["true_positive"])
        fp += int(metrics["false_positive"])
        tn += int(metrics["true_negative"])
        fn += int(metrics["false_negative"])
        edge_errors.append(float(metrics["edge_count_abs_error"]))
        predicted_counts.append(int(metrics["predicted_edge_count"]))
        target_counts.append(int(metrics["target_edge_count"]))
        if predicted["graph_decode"]["connected_after_decode"]:
            connected_count += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "house_count": len(samples),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_edge_count_abs_error": sum(edge_errors) / max(len(edge_errors), 1),
        "predicted_edge_counts": predicted_counts,
        "target_edge_counts": target_counts,
        "connected_house_count": connected_count,
    }


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
