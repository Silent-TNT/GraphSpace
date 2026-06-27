#!/usr/bin/env python3
"""Render learned heterogeneous topology graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx


TYPE_COLORS = {
    "entryway": "#d9a441",
    "living_room": "#4c78a8",
    "dining_room": "#72b7b2",
    "kitchen": "#f58518",
    "bedroom": "#54a24b",
    "bathroom": "#b279a2",
    "corridor": "#bab0ac",
    "stairs": "#e45756",
    "utility": "#9d755d",
    "balcony": "#76b7b2",
    "multi_purpose": "#59a14f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/learned_hetero_topology_eval_val_full/graphs"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--house-id", action="append", dest="house_ids")
    parser.add_argument(
        "--input-file",
        action="append",
        dest="input_files",
        type=Path,
        help="Optional learned/bridge topology JSON file. Can be passed more than once.",
    )
    parser.add_argument("--count", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def short_label(node: dict[str, Any]) -> str:
    node_type = node.get("node_type")
    if node_type == "floor":
        return str(node["id"]).replace("_", " ")
    if node_type == "house":
        return "house"
    room_type = str(node.get("type", "room"))
    suffix = str(node["id"]).split("_")[-1]
    return f"{room_type.replace('_', ' ')}\n{suffix}"


def node_floor(node: dict[str, Any]) -> int:
    floors = node.get("floors") or []
    if 1 in floors and 2 in floors:
        return 3
    if 2 in floors:
        return 2
    return 1


def positions_for(payload: dict[str, Any]) -> dict[str, tuple[float, float]]:
    rooms = [node for node in payload["nodes"] if node.get("node_type") == "room_instance"]
    by_floor: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for node in rooms:
        by_floor[node_floor(node)].append(node)
    positions: dict[str, tuple[float, float]] = {
        str(payload["house_id"]): (0.0, 3.6),
        "floor_1": (-1.5, 2.8),
        "floor_2": (1.5, 2.8),
    }
    floor_y = {1: 0.4, 2: 1.55, 3: 2.15}
    for floor, floor_nodes in by_floor.items():
        floor_nodes = sorted(floor_nodes, key=lambda item: (str(item.get("type")), str(item["id"])))
        count = max(len(floor_nodes), 1)
        for index, node in enumerate(floor_nodes):
            x = (index - (count - 1) / 2.0) * 0.75
            positions[str(node["id"])] = (x, floor_y[floor])
    return positions


def render(payload: dict[str, Any], output_path: Path) -> None:
    payload = normalize_payload(payload, output_path.stem)
    graph = nx.Graph()
    contains_edges = []
    topology_edges = []
    observed_edges = []
    nodes_by_id = {str(node["id"]): node for node in payload["nodes"]}
    for node_id, node in nodes_by_id.items():
        graph.add_node(node_id)
    for edge in payload["edges"]:
        source = str(edge["source"])
        target = str(edge["target"])
        edge_type = str(edge.get("edge_type", ""))
        if source not in graph or target not in graph:
            continue
        graph.add_edge(source, target)
        if edge_type == "contains":
            contains_edges.append((source, target))
        elif edge_type in {"target_adjacent", "guidance_relation"}:
            topology_edges.append((source, target))
        elif edge_type == "geometric_contact_observed":
            observed_edges.append((source, target))

    pos = positions_for(payload)
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.set_title(
        (
            f"Learned heterogeneous topology | {payload['house_id']} | "
            f"threshold={payload.get('edge_threshold', 'n/a')}"
        ),
        fontsize=14,
        pad=16,
    )
    axis.axis("off")

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=contains_edges,
        edge_color="#c7c7c7",
        width=1.0,
        alpha=0.55,
        ax=axis,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=observed_edges,
        edge_color="#d8d8d8",
        width=0.75,
        alpha=0.30,
        ax=axis,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=topology_edges,
        edge_color="#333333",
        width=1.8,
        alpha=0.82,
        ax=axis,
    )

    node_colors = []
    node_sizes = []
    for node_id in graph.nodes:
        node = nodes_by_id[node_id]
        if node.get("node_type") == "house":
            node_colors.append("#111111")
            node_sizes.append(1400)
        elif node.get("node_type") == "floor":
            node_colors.append("#6b7280")
            node_sizes.append(1200)
        else:
            node_colors.append(TYPE_COLORS.get(str(node.get("type")), "#8f8f8f"))
            node_sizes.append(850)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#ffffff",
        linewidths=1.5,
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node_id: short_label(nodes_by_id[node_id]) for node_id in graph.nodes},
        font_size=7,
        font_color="#111111",
        ax=axis,
    )

    metrics = payload.get("metrics", {})
    metric_text = (
        f"valid={metrics.get('graph_valid')}  "
        f"connected={metrics.get('connected')}  "
        f"semantic={metrics.get('semantic_checks', {}).get('pass')}  "
        f"edges={metrics.get('predicted_edge_count')}/{metrics.get('target_edge_count')}  "
        f"F1={float(metrics.get('edge_f1', 0.0)):.3f}"
    )
    axis.text(
        0.01,
        0.01,
        metric_text,
        transform=axis.transAxes,
        fontsize=10,
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def normalize_payload(payload: dict[str, Any], fallback_house_id: str) -> dict[str, Any]:
    if "heterogeneous_nodes" in payload:
        return {
            **payload,
            "house_id": payload.get("house_id", fallback_house_id),
            "nodes": payload["heterogeneous_nodes"],
            "edges": payload.get("heterogeneous_edges", []),
        }
    if any(node.get("node_type") for node in payload.get("nodes", [])):
        return {"house_id": payload.get("house_id", fallback_house_id), **payload}

    room_nodes = []
    contains_edges = [
        {"source": fallback_house_id, "target": "floor_1", "edge_type": "contains"},
        {"source": fallback_house_id, "target": "floor_2", "edge_type": "contains"},
    ]
    for node in payload.get("nodes", []):
        normalized = {
            **node,
            "node_type": "room_instance",
            "floors": node.get("floors") or ([2] if str(node.get("floor")) == "2" else [1]),
        }
        room_nodes.append(normalized)
        for floor in normalized["floors"]:
            contains_edges.append(
                {"source": f"floor_{int(floor)}", "target": normalized["id"], "edge_type": "contains"}
            )
    topology_edges = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "edge_type": "guidance_relation",
            "relation": edge.get("relation", "horizontal"),
        }
        for edge in payload.get("edges", [])
    ]
    observed_edges = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "edge_type": "geometric_contact_observed",
            "relation": edge.get("relation", "horizontal"),
        }
        for edge in payload.get("geometric_contact_observed", [])
    ]
    return {
        "house_id": fallback_house_id,
        "edge_threshold": payload.get("evidence", {}).get("edge_threshold"),
        "nodes": [
            {"id": fallback_house_id, "node_type": "house"},
            {"id": "floor_1", "node_type": "floor", "floor": 1},
            {"id": "floor_2", "node_type": "floor", "floor": 2},
            *room_nodes,
        ],
        "edges": [*contains_edges, *observed_edges, *topology_edges],
        "metrics": {
            "predicted_edge_count": len(topology_edges),
            "target_edge_count": len(topology_edges),
        },
    }


def main() -> None:
    args = parse_args()
    if args.input_files:
        outputs = []
        for input_file in args.input_files[: args.count]:
            payload = read_json(input_file)
            house_id = input_file.parent.name
            output_path = args.output_dir / f"{house_id}_hetero_topology.png"
            render(payload, output_path)
            outputs.append(str(output_path))
        print(json.dumps({"outputs": outputs}, indent=2))
        return
    if args.house_ids:
        house_ids = args.house_ids
    else:
        house_ids = [
            path.parent.name
            for path in sorted(args.input_root.glob("*/learned_topology.json"))[: args.count]
        ]
    outputs = []
    for house_id in house_ids[: args.count]:
        payload_path = args.input_root / house_id / "learned_topology.json"
        payload = read_json(payload_path)
        output_path = args.output_dir / f"{house_id}_learned_topology.png"
        render(payload, output_path)
        outputs.append(str(output_path))
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
