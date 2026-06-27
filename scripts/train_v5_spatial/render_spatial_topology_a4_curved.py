"""Render a portrait A4 curved spatial topology diagram."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


FLOOR_GAP = 16.0

TYPE_COLORS = {
    # Dataset / Rhino layer palette from scripts/spatial_modal_infer/config.py
    "entryway": "#808080",
    "living_room": "#FF8000",
    "dining_room": "#FFFF00",
    "kitchen": "#00FF00",
    "bedroom": "#0000FF",
    "bathroom": "#FF0000",
    "corridor": "#B0B0FF",
    "stairs": "#A000FF",
    "utility": "#3CB371",
    "balcony": "#00FFFF",
    "multi_purpose": "#FFC0CB",
}

LABELS = {
    "entryway": "entryway",
    "living_room": "living room",
    "dining_room": "dining room",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "corridor": "corridor",
    "stairs": "stairs",
    "utility": "utility",
    "balcony": "balcony",
    "multi_purpose": "multi purpose",
}


def load_topology(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def site_size(nodes: list[dict[str, Any]]) -> tuple[float, float]:
    return (
        max(float(node["box_max"][0]) for node in nodes),
        max(float(node["box_max"][1]) for node in nodes),
    )


def node_xy(node: dict[str, Any], grid: dict[str, int], site: tuple[float, float]) -> tuple[float, float]:
    sx, sy = site
    gx, gy = float(grid["x"]), float(grid["y"])
    x = (float(node["box_min"][0]) + float(node["box_max"][0])) * 0.5 / sx * gx
    y = (float(node["box_min"][1]) + float(node["box_max"][1])) * 0.5 / sy * gy
    return x, y


def node_floor_z(node: dict[str, Any]) -> float:
    floors = set(int(value) for value in node.get("floors", []))
    if floors == {2}:
        return FLOOR_GAP
    return 0.0


def floor_z(floor: int) -> float:
    return FLOOR_GAP if floor == 2 else 0.0


def project(x: float, y: float, z: float) -> tuple[float, float]:
    # Controlled 2D axonometric projection. Z is intentionally amplified to
    # keep the two floors visually separated on a portrait sheet.
    px = (x - y) * 0.82
    py = -(x + y) * 0.34 + z * 1.95
    return px, py


def floor_polygon(grid: dict[str, int], z: float) -> list[tuple[float, float]]:
    gx, gy = float(grid["x"]), float(grid["y"])
    corners = [(0.0, 0.0, z), (gx, 0.0, z), (gx, gy, z), (0.0, gy, z)]
    return [project(*corner) for corner in corners]


def scale_points(points: dict[str, tuple[float, float]], polygons: list[list[tuple[float, float]]]) -> tuple[dict[str, tuple[float, float]], list[list[tuple[float, float]]]]:
    all_points = list(points.values()) + [point for polygon in polygons for point in polygon]
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    scale = min(0.82 / width, 0.76 / height)

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x = 0.43 + (point[0] - (min_x + max_x) * 0.5) * scale
        y = 0.52 + (point[1] - (min_y + max_y) * 0.5) * scale
        return x, y

    return (
        {key: transform(value) for key, value in points.items()},
        [[transform(point) for point in polygon] for polygon in polygons],
    )


def contrast_text_color(hex_color: str) -> str:
    raw = hex_color.lstrip("#")
    if len(raw) != 6:
        return "black"
    r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 150 else "white"


def draw_curve(ax: Any, p0: tuple[float, float], p1: tuple[float, float], *, color: str, linewidth: float, alpha: float, linestyle: str, zorder: int, curve_index: int) -> None:
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    nx = -dy / length
    ny = dx / length
    direction = -1.0 if curve_index % 2 else 1.0
    strength = min(0.046, 0.016 + length * 0.065)
    cx = (x0 + x1) * 0.5 + nx * strength * direction
    cy = (y0 + y1) * 0.5 + ny * strength * direction
    path = MplPath([(x0, y0), (cx, cy), (x1, y1)], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )


def visual_node_records(nodes: dict[str, dict[str, Any]], grid: dict[str, int], site: tuple[float, float]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        x, y = node_xy(node, grid, site)
        floors = sorted(set(int(value) for value in node.get("floors", [])))
        if str(node.get("type")) == "stairs" and set(floors) == {1, 2}:
            for floor in (1, 2):
                records.append(
                    {
                        "visual_id": f"{node_id}@floor{floor}",
                        "node_id": node_id,
                        "node": node,
                        "floor": floor,
                        "raw_point": project(x, y, floor_z(floor)),
                        "is_stair_duplicate": True,
                    }
                )
        else:
            floor = 2 if floors == [2] else 1
            records.append(
                {
                    "visual_id": node_id,
                    "node_id": node_id,
                    "node": node,
                    "floor": floor,
                    "raw_point": project(x, y, node_floor_z(node)),
                    "is_stair_duplicate": False,
                }
            )
    return records


def edge_visual_endpoint(node_id: str, other_id: str, nodes: dict[str, dict[str, Any]]) -> str:
    node = nodes[node_id]
    if str(node.get("type")) != "stairs" or set(int(value) for value in node.get("floors", [])) != {1, 2}:
        return node_id
    other_floors = set(int(value) for value in nodes[other_id].get("floors", []))
    if other_floors == {2}:
        return f"{node_id}@floor2"
    if other_floors == {1}:
        return f"{node_id}@floor1"
    return f"{node_id}@floor1"


def draw(topology: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.6,
    })

    nodes = {str(node["id"]): node for node in topology["nodes"]}
    grid = topology["grid"]
    site = site_size(topology["nodes"])

    visual_records = visual_node_records(nodes, grid, site)
    raw_points = {record["visual_id"]: record["raw_point"] for record in visual_records}
    raw_polygons = [floor_polygon(grid, 0.0), floor_polygon(grid, FLOOR_GAP)]
    points, polygons = scale_points(raw_points, raw_polygons)

    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("#fbfbfa")
    ax.set_facecolor("#fbfbfa")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.075, 0.955, "SPATIAL ACCESS TOPOLOGY", fontsize=10.5, ha="left", va="top", color="#1a1a1a")
    ax.text(
        0.075,
        0.941,
        Path(str(topology["source_case"])).name.replace("phase24_whole_hetero_learned_guidance_", ""),
        fontsize=7.0,
        ha="left",
        va="top",
        color="#8a8a8a",
    )

    for polygon, label in zip(polygons, ("floor 1", "floor 2")):
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                fill=True,
                facecolor="#f7f7f5",
                edgecolor="#dededb",
                linewidth=0.72,
                alpha=0.95,
                zorder=0,
            )
        )
        cx = sum(point[0] for point in polygon) / len(polygon)
        cy = max(point[1] for point in polygon)
        ax.text(cx, cy + 0.010, label.upper(), color="#b0b0aa", fontsize=7.0, ha="center", va="bottom")

    blocked_index = 0
    for edge in topology["edges"]:
        source_id = str(edge["source"])
        target_id = str(edge["target"])
        if source_id not in nodes or target_id not in nodes:
            continue
        source_visual_id = edge_visual_endpoint(source_id, target_id, nodes)
        target_visual_id = edge_visual_endpoint(target_id, source_id, nodes)
        if source_visual_id not in points or target_visual_id not in points:
            continue
        if edge["edge_type"] != "voxel_blocked_direct_access":
            continue
        draw_curve(
            ax,
            points[source_visual_id],
            points[target_visual_id],
            color="#d84a4a",
            linestyle="--",
            linewidth=0.72,
            alpha=0.24,
            zorder=1,
            curve_index=blocked_index,
        )
        blocked_index += 1

    access_index = 0
    for edge in topology["edges"]:
        source_id = str(edge["source"])
        target_id = str(edge["target"])
        if source_id not in nodes or target_id not in nodes:
            continue
        source_visual_id = edge_visual_endpoint(source_id, target_id, nodes)
        target_visual_id = edge_visual_endpoint(target_id, source_id, nodes)
        if source_visual_id not in points or target_visual_id not in points:
            continue
        if edge["edge_type"] != "voxel_access_relation":
            continue
        draw_curve(
            ax,
            points[source_visual_id],
            points[target_visual_id],
            color="#252525",
            linestyle="-",
            linewidth=1.45,
            alpha=0.74,
            zorder=2,
            curve_index=access_index,
        )
        access_index += 1

    for node_id, node in nodes.items():
        if str(node.get("type")) != "stairs" or set(int(value) for value in node.get("floors", [])) != {1, 2}:
            continue
        lower_id = f"{node_id}@floor1"
        upper_id = f"{node_id}@floor2"
        if lower_id in points and upper_id in points:
            draw_curve(
                ax,
                points[lower_id],
                points[upper_id],
                color="#8f8f8b",
                linestyle="-",
                linewidth=0.78,
                alpha=0.48,
                zorder=1,
                curve_index=access_index,
            )
            access_index += 1

    type_seen: Counter[str] = Counter()
    occupied_label_slots: defaultdict[tuple[int, int], int] = defaultdict(int)
    visual_type_indices: dict[str, int] = {}
    for node_id, node in nodes.items():
        room_type = str(node["type"])
        visual_type_indices[node_id] = type_seen[room_type]
        type_seen[room_type] += 1

    for record in visual_records:
        node_id = record["node_id"]
        node = record["node"]
        room_type = str(node["type"])
        type_index = visual_type_indices[node_id]
        color = TYPE_COLORS.get(room_type, "#bab0ac")
        x, y = points[record["visual_id"]]
        marker_size = 178 if room_type in {"entryway", "stairs", "corridor"} else 150
        ax.scatter(
            [x],
            [y],
            s=marker_size,
            c=color,
            edgecolors="#202020",
            linewidths=0.8,
            zorder=5,
        )
        slot = (round(x * 30), round(y * 42))
        collision_shift = occupied_label_slots[slot] * 0.014
        occupied_label_slots[slot] += 1
        label_dx = -0.014 if x > 0.43 else 0.014
        label_dy = 0.016 + collision_shift
        label = f"{LABELS.get(room_type, room_type)} {type_index + 1}"
        if record["is_stair_duplicate"]:
            label = f"{label} / floor {record['floor']}"
        ax.text(
            x + label_dx,
            y + label_dy,
            label,
            fontsize=6.7,
            color="#303030",
            ha="right" if label_dx < 0 else "left",
            va="bottom",
            zorder=6,
            bbox={"facecolor": "#fbfbfa", "edgecolor": "none", "alpha": 0.76, "pad": 0.45},
        )

    legend_handles: list[Any] = [
        Line2D([0], [0], color="#252525", lw=1.45, label="access"),
        Line2D([0], [0], color="#d84a4a", lw=0.8, linestyle="--", alpha=0.55, label="blocked"),
    ]

    type_counts = Counter(str(node["type"]) for node in topology["nodes"])
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TYPE_COLORS.get(room_type, "#bab0ac"),
            markeredgecolor="#202020",
            markeredgewidth=0.7,
            markersize=6.0,
            label=f"{LABELS.get(room_type, room_type.replace('_', ' '))} x{count}",
        )
        for room_type, count in sorted(type_counts.items())
    )
    unified_legend = ax.legend(
        handles=legend_handles,
        title="LEGEND",
        loc="center left",
        bbox_to_anchor=(0.748, 0.52),
        frameon=False,
        fontsize=6.8,
        title_fontsize=7.3,
        labelspacing=0.35,
        handletextpad=0.65,
        handlelength=2.5,
    )
    ax.add_artist(unified_legend)

    metrics = topology.get("metrics", {})
    ax.text(
        0.075,
        0.050,
        (
            "A4 portrait / 2.5D topology / curved access relations / no room size shown\n"
            f"access={metrics.get('predicted_access_count', '?')}  "
            f"blocked={metrics.get('predicted_blocked_count', '?')}  "
            f"stairs-bedroom access={metrics.get('predicted_stairs_bedroom_access_count', '?')}"
        ),
        fontsize=6.8,
        color="#969696",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-json", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in args.topology_json:
        topology = load_topology(path)
        output_path = args.output_dir / f"{path.stem}_a4_curved.png"
        draw(topology, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
