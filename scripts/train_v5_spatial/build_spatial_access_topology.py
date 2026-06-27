#!/usr/bin/env python3
"""Build and render typed spatial-access topology from generated massing.

This separates geometric contact from passable access. Contact is a physical
observation; access is only granted through user-approved transition functions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx


PASSAGE_TYPES = {
    "entryway",
    "living_room",
    "dining_room",
    "kitchen",
    "corridor",
    "multi_purpose",
}
VERTICAL_TYPES = {"stairs"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        action="append",
        type=Path,
        required=True,
        help="Phase24 bridge output directory containing generated_layout.json and topology.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def overlap_length(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def face_contact_area(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx0, ly0, lz0 = [float(value) for value in left["box_min"]]
    lx1, ly1, lz1 = [float(value) for value in left["box_max"]]
    rx0, ry0, rz0 = [float(value) for value in right["box_min"]]
    rx1, ry1, rz1 = [float(value) for value in right["box_max"]]
    z_overlap = overlap_length(lz0, lz1, rz0, rz1)
    if z_overlap <= TOL:
        return 0.0
    if abs(lx1 - rx0) <= TOL or abs(rx1 - lx0) <= TOL:
        return overlap_length(ly0, ly1, ry0, ry1) * z_overlap
    if abs(ly1 - ry0) <= TOL or abs(ry1 - ly0) <= TOL:
        return overlap_length(lx0, lx1, rx0, rx1) * z_overlap
    return 0.0


def group_rooms(layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for room in layout.get("rooms", []):
        group_id = str(room.get("functional_id", room["id"]))
        grouped.setdefault(group_id, []).append(room)
    groups: dict[str, dict[str, Any]] = {}
    for group_id, rooms in grouped.items():
        mins = [min(float(room["box_min"][axis]) for room in rooms) for axis in range(3)]
        maxs = [max(float(room["box_max"][axis]) for room in rooms) for axis in range(3)]
        room_type = str(rooms[0].get("type", "unknown"))
        floors = sorted({int(floor) for room in rooms for floor in room.get("floors", [room.get("floor", 1)])})
        groups[group_id] = {
            "id": group_id,
            "type": room_type,
            "node_type": "room_instance",
            "floors": floors,
            "box_min": mins,
            "box_max": maxs,
            "center": [(mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5],
            "passage_role": "passage" if room_type in PASSAGE_TYPES else "vertical" if room_type in VERTICAL_TYPES else "served",
        }
    return groups


def classify_contact(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, str]:
    left_type = str(left["type"])
    right_type = str(right["type"])
    if left_type in VERTICAL_TYPES or right_type in VERTICAL_TYPES:
        other_type = right_type if left_type in VERTICAL_TYPES else left_type
        if other_type in PASSAGE_TYPES:
            return "access_relation", "vertical_transfer_to_passage"
        return "blocked_direct_access", "stairs_requires_passage_transition"
    if left_type in PASSAGE_TYPES or right_type in PASSAGE_TYPES:
        return "access_relation", "passage_to_space"
    return "blocked_direct_access", "served_spaces_require_passage_transition"


def build_spatial_topology(case_dir: Path) -> dict[str, Any]:
    layout = read_json(case_dir / "generated_layout.json")
    learned_topology = read_json(case_dir / "topology.json")
    groups = group_rooms(layout)
    group_ids = sorted(groups)
    contact_edges = []
    access_edges = []
    blocked_edges = []
    for left_index, left_id in enumerate(group_ids):
        for right_id in group_ids[left_index + 1 :]:
            left = groups[left_id]
            right = groups[right_id]
            area = face_contact_area(left, right)
            if area <= TOL:
                continue
            edge_type, reason = classify_contact(left, right)
            edge = {
                "source": left_id,
                "target": right_id,
                "relation": "horizontal_contact",
                "edge_type": edge_type,
                "contact_area": area,
                "access_reason": reason,
            }
            contact_edges.append({**edge, "edge_type": "contact_observed"})
            if edge_type == "access_relation":
                access_edges.append(edge)
            else:
                blocked_edges.append(edge)
    learned_edges = [
        {
            "source": str(edge["source"]),
            "target": str(edge["target"]),
            "edge_type": "learned_guidance_input",
            "relation": str(edge.get("relation", "horizontal")),
        }
        for edge in learned_topology.get("edges", [])
    ]
    topology = {
        "schema": "graphspace_spatial_access_topology_v1",
        "source_case": str(case_dir),
        "house_id": layout.get("house_id", case_dir.name),
        "passage_types": sorted(PASSAGE_TYPES),
        "vertical_types": sorted(VERTICAL_TYPES),
        "nodes": [groups[group_id] for group_id in group_ids],
        "access_edges": access_edges,
        "blocked_direct_access_edges": blocked_edges,
        "contact_observed_edges": contact_edges,
        "learned_guidance_input_edges": learned_edges,
        "heterogeneous_nodes": [
            {"id": layout.get("house_id", case_dir.name), "node_type": "house"},
            {"id": "floor_1", "node_type": "floor", "floor": 1},
            {"id": "floor_2", "node_type": "floor", "floor": 2},
            *[groups[group_id] for group_id in group_ids],
        ],
        "heterogeneous_edges": [
            {"source": layout.get("house_id", case_dir.name), "target": "floor_1", "edge_type": "contains"},
            {"source": layout.get("house_id", case_dir.name), "target": "floor_2", "edge_type": "contains"},
            *[
                {"source": f"floor_{floor}", "target": group_id, "edge_type": "contains"}
                for group_id in group_ids
                for floor in groups[group_id]["floors"]
            ],
            *access_edges,
            *blocked_edges,
            *contact_edges,
            *learned_edges,
        ],
        "metrics": {
            "node_count": len(group_ids),
            "contact_observed_count": len(contact_edges),
            "access_relation_count": len(access_edges),
            "blocked_direct_access_count": len(blocked_edges),
            "blocked_stairs_bedroom_count": sum(
                1
                for edge in blocked_edges
                if {groups[edge["source"]]["type"], groups[edge["target"]]["type"]} == {"stairs", "bedroom"}
            ),
        },
    }
    return topology


def node_label(node: dict[str, Any]) -> str:
    suffix = str(node["id"]).split("_")[-1]
    return f"{str(node['type']).replace('_', ' ')}\n{suffix}"


def render(topology: dict[str, Any], output_path: Path) -> None:
    graph = nx.Graph()
    nodes = {str(node["id"]): node for node in topology["nodes"]}
    for node_id in nodes:
        graph.add_node(node_id)
    access = [(edge["source"], edge["target"]) for edge in topology["access_edges"]]
    blocked = [(edge["source"], edge["target"]) for edge in topology["blocked_direct_access_edges"]]
    learned = [(edge["source"], edge["target"]) for edge in topology["learned_guidance_input_edges"]]
    contact = [
        (edge["source"], edge["target"])
        for edge in topology["contact_observed_edges"]
        if (edge["source"], edge["target"]) not in access and (edge["source"], edge["target"]) not in blocked
    ]
    for edge in [*access, *blocked, *learned, *contact]:
        if edge[0] in graph and edge[1] in graph:
            graph.add_edge(*edge)
    pos = {
        node_id: (
            float(node["center"][0]) / 3000.0,
            float(node["center"][1]) / 3000.0 + (2.5 if 2 in node.get("floors", []) and 1 not in node.get("floors", []) else 0.0),
        )
        for node_id, node in nodes.items()
    }
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.set_title(f"Spatial access topology | {Path(str(topology['source_case'])).name}", fontsize=14, pad=14)
    axis.axis("off")
    nx.draw_networkx_edges(graph, pos, edgelist=learned, edge_color="#d1d5db", width=1.0, style="dashed", alpha=0.55, ax=axis)
    nx.draw_networkx_edges(graph, pos, edgelist=contact, edge_color="#c7c7c7", width=0.8, alpha=0.35, ax=axis)
    nx.draw_networkx_edges(graph, pos, edgelist=blocked, edge_color="#dc2626", width=2.0, alpha=0.80, ax=axis)
    nx.draw_networkx_edges(graph, pos, edgelist=access, edge_color="#1f2937", width=2.4, alpha=0.90, ax=axis)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=[TYPE_COLORS.get(nodes[node_id]["type"], "#8f8f8f") for node_id in graph.nodes],
        node_size=[1100 if nodes[node_id]["type"] in PASSAGE_TYPES else 950 for node_id in graph.nodes],
        edgecolors="#ffffff",
        linewidths=1.6,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, pos, labels={node_id: node_label(nodes[node_id]) for node_id in graph.nodes}, font_size=7, ax=axis)
    metrics = topology["metrics"]
    text = (
        "black=access_relation  red=blocked_direct_access  gray=dashed=learned guidance input\n"
        f"access={metrics['access_relation_count']}  blocked={metrics['blocked_direct_access_count']}  "
        f"blocked stairs-bedroom={metrics['blocked_stairs_bedroom_count']}  contact={metrics['contact_observed_count']}"
    )
    axis.text(
        0.01,
        0.01,
        text,
        transform=axis.transAxes,
        fontsize=10,
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    outputs = []
    for case_dir in args.case_dir:
        topology = build_spatial_topology(case_dir)
        json_path = args.output_dir / f"{case_dir.name}_spatial_access_topology.json"
        png_path = args.output_dir / f"{case_dir.name}_spatial_access_topology.png"
        write_json(json_path, topology)
        render(topology, png_path)
        outputs.append({"json": str(json_path), "png": str(png_path), "metrics": topology["metrics"]})
    print(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
