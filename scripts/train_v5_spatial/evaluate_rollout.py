#!/usr/bin/env python3
"""Roll out a trained policy into recursive 3D block cuts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import node_features, region_volume
from model import SpatialModalCutPolicy


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "phase6_spatial_cut" / "samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--houses", nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-depth", type=int, default=64)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def graph_tensors(graph: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = torch.from_numpy(node_features(graph["nodes"])).unsqueeze(0).to(device)
    count = nodes.shape[1]
    adjacency = torch.zeros(1, 2, count, count, device=device)
    for src, dst, relation in graph["edges"]:
        adjacency[0, int(relation), int(src), int(dst)] = 1.0
    return nodes, adjacency


def predict_action(
    model: SpatialModalCutPolicy,
    nodes: torch.Tensor,
    adjacency: torch.Tensor,
    site_cells: list[int],
    region: list[int],
    room_indices: list[int],
    device: torch.device,
) -> tuple[int, float, list[int], list[int]]:
    active = torch.zeros(1, nodes.shape[1], device=device)
    active[0, room_indices] = 1.0
    volume = torch.from_numpy(region_volume(site_cells, region)).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast(
        device.type, enabled=device.type == "cuda"
    ):
        output = model(volume, nodes, active, adjacency)
    axis = int(output["axis_logits"].argmax(dim=1).item())
    ratio = float(output["cut_ratio"].item())
    right_probability = torch.softmax(
        output["side_logits"][0], dim=1
    )[:, 1].cpu().numpy()
    left_fraction = float(output["left_fraction"].item())
    left_count = int(round(left_fraction * len(room_indices)))
    left_count = max(1, min(len(room_indices) - 1, left_count))
    ordered = sorted(room_indices, key=lambda index: float(right_probability[index]))
    left = ordered[:left_count]
    right = ordered[left_count:]
    return axis, ratio, left, right


def rollout_house(
    model: SpatialModalCutPolicy,
    payload: dict,
    device: torch.device,
    max_depth: int,
) -> dict:
    site_cells = payload["site_cells"]
    graph = payload["graph"]
    nodes, adjacency = graph_tensors(graph, device)
    leaves = []
    cuts = []

    def visit(room_indices: list[int], region: list[int], depth: int) -> None:
        if len(room_indices) <= 1 or depth >= max_depth:
            leaves.append({"room_indices": room_indices, "region": region})
            return
        axis, ratio, left, right = predict_action(
            model,
            nodes,
            adjacency,
            site_cells,
            region,
            room_indices,
            device,
        )
        if axis == 3 or not left or not right:
            leaves.append({"room_indices": room_indices, "region": region})
            return
        extent = region[axis + 3] - region[axis]
        cut = region[axis] + int(round(ratio * extent))
        cut = max(region[axis] + 1, min(region[axis + 3] - 1, cut))
        left_region = list(region)
        right_region = list(region)
        left_region[axis + 3] = cut
        right_region[axis] = cut
        cuts.append(
            {
                "axis": axis,
                "cut_cell": cut,
                "cut_ratio": ratio,
                "room_count": len(room_indices),
            }
        )
        visit(left, left_region, depth + 1)
        visit(right, right_region, depth + 1)

    visit(
        list(range(len(graph["nodes"]))),
        [0, 0, 0, site_cells[0], site_cells[1], 20],
        0,
    )
    resolved = sum(len(leaf["room_indices"]) == 1 for leaf in leaves)
    return {
        "house_id": payload["house_id"],
        "room_count": len(graph["nodes"]),
        "resolved_room_count": resolved,
        "resolution_rate": resolved / max(len(graph["nodes"]), 1),
        "leaf_count": len(leaves),
        "unresolved_groups": [
            leaf for leaf in leaves if len(leaf["room_indices"]) > 1
        ],
        "cuts": cuts,
        "leaves": leaves,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SpatialModalCutPolicy().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    reports = [
        rollout_house(
            model,
            read_json(DATA_DIR / f"{house_id}.json"),
            device,
            args.max_depth,
        )
        for house_id in args.houses
    ]
    summary = {
        "schema": "graphspace_v5_spatial_cut_rollout_v1",
        "checkpoint": str(args.checkpoint),
        "house_count": len(reports),
        "mean_resolution_rate": float(
            np.mean([item["resolution_rate"] for item in reports])
        ),
        "fully_resolved_count": sum(
            item["resolved_room_count"] == item["room_count"] for item in reports
        ),
        "reports": reports,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        "houses={} fully_resolved={} mean_resolution_rate={:.4f}".format(
            summary["house_count"],
            summary["fully_resolved_count"],
            summary["mean_resolution_rate"],
        )
    )


if __name__ == "__main__":
    main()
