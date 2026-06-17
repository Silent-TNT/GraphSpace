#!/usr/bin/env python3
"""Render exact 300 mm full-resolution generation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap

from generate_from_user_conditions import ROOT


ROOM_COLORS = {
    "entryway": "#d9b38c",
    "living_room": "#ef8a62",
    "dining_room": "#fdb863",
    "kitchen": "#b2abd2",
    "bedroom": "#67a9cf",
    "bathroom": "#80cdc1",
    "corridor": "#d8daeb",
    "stairs": "#5e3c99",
    "utility": "#a6dba0",
    "balcony": "#c7eae5",
    "multi_purpose": "#f6e8c3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cuboid_mesh(
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
) -> tuple:
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 6, 3, 3]
    j = [1, 2, 4, 2, 5, 3, 5, 6, 6, 7, 7, 4]
    k = [2, 3, 5, 5, 4, 7, 6, 7, 2, 3, 4, 0]
    return x, y, z, i, j, k


def vertical_runs(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    runs = []
    for x, y in np.argwhere(mask.any(axis=2)):
        values = mask[x, y]
        start = None
        for z, occupied in enumerate(np.r_[values, False]):
            if occupied and start is None:
                start = z
            elif not occupied and start is not None:
                runs.append((int(x), int(y), int(start), int(z)))
                start = None
    return runs


def main() -> None:
    args = parse_args()
    summary = read_json(args.input_dir / "summary.json")
    topology = read_json(args.input_dir / "topology.json")
    model_graph = read_json(args.input_dir / "model_graph.json")
    with np.load(args.input_dir / "assignment_grid.npz") as arrays:
        assignments = arrays["assignments"]
        floor_grids = arrays["floor_instance_grid"]
    x0, y0, x1, y1 = summary["canvas_placement"]
    site_x, site_y, _ = summary["site_mm"]
    room_types = [node["type"] for node in model_graph["nodes"]]
    colors = ["#ffffff"] + [ROOM_COLORS[value] for value in room_types]

    for floor in (1, 2):
        grid = floor_grids[floor - 1, x0:x1, y0:y1].T
        figure, axis = plt.subplots(figsize=(10, 8))
        axis.imshow(
            grid,
            origin="lower",
            interpolation="nearest",
            cmap=ListedColormap(colors),
            vmin=0,
            vmax=max(len(colors) - 1, 1),
            extent=(0, site_x, 0, site_y),
        )
        axis.set_title(
            f"300 mm instance partition - floor {floor}\n"
            f"site={site_x:,.0f} x {site_y:,.0f} mm | seed={summary['seed']}"
        )
        axis.set_xlabel("X (mm)")
        axis.set_ylabel("Y (mm)")
        axis.set_aspect("equal")
        figure.tight_layout()
        figure.savefig(args.input_dir / f"floor_{floor}.png", dpi=180)
        plt.close(figure)

    graph = nx.Graph()
    positions = {}
    for node in topology["nodes"]:
        graph.add_node(node["id"], type=node["type"], floor=node["floor"])
        positions[node["id"]] = node["position"]
    for edge in topology["edges"]:
        graph.add_edge(edge["source"], edge["target"])
    figure, axis = plt.subplots(figsize=(12, 9))
    node_colors = [
        ROOM_COLORS[graph.nodes[node]["type"]] for node in graph.nodes
    ]
    nx.draw_networkx(
        graph,
        positions,
        ax=axis,
        node_color=node_colors,
        node_size=700,
        font_size=7,
        labels={
            node: f"{graph.nodes[node]['type']}\nF{graph.nodes[node]['floor']}"
            for node in graph.nodes
        },
    )
    axis.set_title(
        f"Functional topology | site={site_x:,.0f} x {site_y:,.0f} mm "
        f"| seed={summary['seed']}"
    )
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.input_dir / "topology.png", dpi=180)
    plt.close(figure)

    plot = go.Figure()
    for instance_id, room_type in enumerate(room_types, start=1):
        mask = assignments == instance_id
        first = True
        for x, y, z_start, z_end in vertical_runs(mask):
            mesh = cuboid_mesh(
                (x - x0) * 300,
                (y - y0) * 300,
                z_start * 300,
                (x - x0 + 1) * 300,
                (y - y0 + 1) * 300,
                z_end * 300,
            )
            plot.add_trace(
                go.Mesh3d(
                    x=mesh[0],
                    y=mesh[1],
                    z=mesh[2],
                    i=mesh[3],
                    j=mesh[4],
                    k=mesh[5],
                    color=ROOM_COLORS[room_type],
                    opacity=0.82,
                    name=room_type,
                    legendgroup=room_type,
                    showlegend=first,
                    hovertext=f"{room_type} | instance {instance_id}",
                    hoverinfo="text",
                )
            )
            first = False
    plot.update_layout(
        title=(
            f"Exact 300 mm generated voxels | site={site_x:,.0f} x "
            f"{site_y:,.0f} mm | seed={summary['seed']}"
        ),
        scene={
            "xaxis_title": "X (mm)",
            "yaxis_title": "Y (mm)",
            "zaxis_title": "Z (mm)",
            "aspectmode": "manual",
            "aspectratio": {
                "x": site_x / max(site_x, site_y, 6000),
                "y": site_y / max(site_x, site_y, 6000),
                "z": 6000 / max(site_x, site_y, 6000),
            },
        },
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    plot.write_html(
        str(args.input_dir / "layout_3d_interactive.html"),
        include_plotlyjs=True,
        config={
            "scrollZoom": True,
            "responsive": True,
            "displaylogo": False,
        },
    )
    plot.write_image(
        str(args.input_dir / "layout_3d.png"),
        width=1400,
        height=900,
        scale=1,
    )
    print(args.input_dir)


if __name__ == "__main__":
    main()
