#!/usr/bin/env python3
"""Render topology, floor plans and 3D views for a user-condition generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spatial_modal_infer.visualize import (
    plot_3d_layout,
    plot_3d_layout_static,
    plot_floor_plan,
    plot_topology_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    request = read_json(args.input_dir / "request.json")
    topology = read_json(args.input_dir / "topology.json")
    candidate = read_json(args.input_dir / "generated_layout.json")
    graph = nx.Graph()
    positions = {}
    edge_types = {}
    for node in topology["nodes"]:
        graph.add_node(node["id"], type=node["type"], floor=node["floor"])
        positions[node["id"]] = node["position"]
    for edge in topology["edges"]:
        graph.add_edge(edge["source"], edge["target"])
        edge_types[(edge["source"], edge["target"])] = edge["relation"]
        edge_types[(edge["target"], edge["source"])] = edge["relation"]
    rooms = candidate["rooms"]
    site_x, site_y = request["site_x"], request["site_y"]
    seed = topology["seed"]
    site_label = f"{site_x:,.0f} × {site_y:,.0f} mm"
    topology_figure = plot_topology_graph(
        graph,
        positions,
        edge_types,
        title=f"Generated functional topology | site={site_label} | seed={seed}",
        show_node_ids=False,
    )
    topology_figure.savefig(
        args.input_dir / "topology.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(topology_figure)
    for floor in (1, 2):
        figure = plot_floor_plan(
            rooms,
            floor,
            site_x,
            site_y,
            title=f"Generated floor {floor} | site={site_label} | seed={seed}",
        )
        figure.savefig(
            args.input_dir / f"floor_{floor}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)
    figure_3d = plot_3d_layout_static(
        rooms,
        site_x,
        site_y,
        title=f"Generated 3D functional blocks | site={site_label} | seed={seed}",
        site_z=6000,
    )
    figure_3d.savefig(
        args.input_dir / "layout_3d.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure_3d)
    interactive_3d = plot_3d_layout(
        rooms,
        site_x,
        site_y,
        title=f"Generated 3D functional blocks | site={site_label} | seed={seed}",
        site_z=6000,
        show_topology=False,
    )
    interactive_3d.write_html(
        str(args.input_dir / "layout_3d_interactive.html"),
        include_plotlyjs=True,
        config={
            "scrollZoom": True,
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["resetCameraLastSave3d"],
        },
    )
    print(args.input_dir)


if __name__ == "__main__":
    main()
