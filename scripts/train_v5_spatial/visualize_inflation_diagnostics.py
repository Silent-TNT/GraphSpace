#!/usr/bin/env python3
"""Render diagnostic floor plans for cell-level inflation outputs.

The figures are intentionally diagnostic instead of presentation-oriented:
they show function occupancy, missing target-topology edges, realized edge
counts, and the largest area deficits recorded by the inflation report.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_v5_spatial.v6_multipart_decoder import VOXEL_MM, room_floors, room_functional_id  # noqa: E402


DEFAULT_INPUT = ROOT / "outputs" / "v6_topology_spatial_global_inflation_overfit_8" / "predicted_json"
DEFAULT_OUTPUT = ROOT / "outputs" / "v6_topology_spatial_global_inflation_overfit_8" / "diagnostic_plans"

TYPE_COLORS = {
    "entryway": "#7f8c8d",
    "living_room": "#f39c12",
    "dining_room": "#e67e22",
    "kitchen": "#d35400",
    "bedroom": "#3498db",
    "bathroom": "#1abc9c",
    "corridor": "#95a5a6",
    "stairs": "#8e44ad",
    "utility": "#16a085",
    "balcony": "#2ecc71",
    "multi_purpose": "#9b59b6",
}
DEFAULT_COLOR = "#bdc3c7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-houses", type=int)
    parser.add_argument("--houses", nargs="*", help="Optional house ids to render.")
    parser.add_argument("--title", default="Topology contact diagnostics")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def site_cells(layout: dict[str, Any]) -> tuple[int, int]:
    size = layout.get("metadata", {}).get("building_size", {})
    return int(round(float(size["x"]) / VOXEL_MM)), int(round(float(size["y"]) / VOXEL_MM))


def cell_box(room: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(round(float(room["box_min"][0]) / VOXEL_MM)),
        int(round(float(room["box_min"][1]) / VOXEL_MM)),
        int(round(float(room["box_max"][0]) / VOXEL_MM)),
        int(round(float(room["box_max"][1]) / VOXEL_MM)),
    )


def collect_cells(
    rooms: list[dict[str, Any]],
) -> tuple[dict[int, dict[tuple[int, int], str]], dict[str, str], dict[tuple[str, int], set[tuple[int, int]]]]:
    floor_grid: dict[int, dict[tuple[int, int], str]] = {1: {}, 2: {}}
    group_types: dict[str, str] = {}
    group_floor_cells: dict[tuple[str, int], set[tuple[int, int]]] = defaultdict(set)
    for room in rooms:
        group_id = room_functional_id(room)
        group_types.setdefault(group_id, str(room.get("type", "unknown")))
        x0, y0, x1, y1 = cell_box(room)
        for floor in room_floors(room):
            for x in range(x0, x1):
                for y in range(y0, y1):
                    floor_grid.setdefault(floor, {})[(x, y)] = group_id
                    group_floor_cells[(group_id, floor)].add((x, y))
    return floor_grid, group_types, group_floor_cells


def centroids(group_floor_cells: dict[tuple[str, int], set[tuple[int, int]]]) -> dict[tuple[str, int], tuple[float, float]]:
    output = {}
    for key, cells in group_floor_cells.items():
        if not cells:
            continue
        output[key] = (
            sum(cell[0] + 0.5 for cell in cells) / len(cells),
            sum(cell[1] + 0.5 for cell in cells) / len(cells),
        )
    return output


def target_edges(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    return list(evaluation.get("p1_spatial_organization", {}).get("target_topology", {}).get("edges", []))


def missing_edges(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    return [edge for edge in target_edges(evaluation) if not edge.get("realized_in_dual")]


def group_floor_for_edge(
    source: str,
    target: str,
    group_floor_cells: dict[tuple[str, int], set[tuple[int, int]]],
) -> int | None:
    for floor in (1, 2):
        if group_floor_cells.get((source, floor)) and group_floor_cells.get((target, floor)):
            return floor
    return None


def top_deficits(report: dict[str, Any], limit: int = 12) -> list[tuple[str, int, int, int]]:
    targets = report.get("group_floor_target_cells", {})
    counts = report.get("group_floor_cell_counts", {})
    rows = []
    for key, target in targets.items():
        actual = int(counts.get(key, 0))
        target_value = int(target)
        deficit = max(0, target_value - actual)
        if deficit:
            rows.append((str(key), actual, target_value, deficit))
    rows.sort(key=lambda item: (-item[3], item[0]))
    return rows[:limit]


def draw_floor(
    axis: Any,
    floor: int,
    floor_grid: dict[int, dict[tuple[int, int], str]],
    group_types: dict[str, str],
    group_floor_cells: dict[tuple[str, int], set[tuple[int, int]]],
    centers: dict[tuple[str, int], tuple[float, float]],
    missing: list[dict[str, Any]],
    sx: int,
    sy: int,
) -> None:
    axis.set_title(f"Floor {floor}", fontsize=12)
    axis.set_xlim(0, sx)
    axis.set_ylim(0, sy)
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_xticks([])
    axis.set_yticks([])
    axis.add_patch(Rectangle((0, 0), sx, sy, fill=False, edgecolor="#111111", linewidth=1.5))

    for (x, y), group_id in sorted(floor_grid.get(floor, {}).items()):
        room_type = group_types.get(group_id, "unknown")
        axis.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                facecolor=TYPE_COLORS.get(room_type, DEFAULT_COLOR),
                edgecolor="white",
                linewidth=0.12,
                alpha=0.86,
            )
        )

    for (group_id, cell_floor), cells in sorted(group_floor_cells.items()):
        if cell_floor != floor or len(cells) < 10:
            continue
        cx, cy = centers[(group_id, floor)]
        label = f"{group_types.get(group_id, '?')}\n{group_id}"
        axis.text(cx, cy, label, ha="center", va="center", fontsize=5, color="#111111")

    for edge in missing:
        source = str(edge["source"])
        target = str(edge["target"])
        edge_floor = group_floor_for_edge(source, target, group_floor_cells)
        if edge_floor != floor:
            continue
        if (source, floor) not in centers or (target, floor) not in centers:
            continue
        sx0, sy0 = centers[(source, floor)]
        tx0, ty0 = centers[(target, floor)]
        axis.plot([sx0, tx0], [sy0, ty0], color="#e74c3c", linewidth=1.0, linestyle="--", alpha=0.8)
        mx, my = (sx0 + tx0) / 2, (sy0 + ty0) / 2
        axis.text(mx, my, "missing", color="#c0392b", fontsize=5, ha="center", va="center")


def draw_deficit_panel(axis: Any, evaluation: dict[str, Any], report: dict[str, Any]) -> None:
    axis.axis("off")
    metrics = evaluation.get("p1_spatial_organization", {}).get("target_topology", {})
    realized = metrics.get("realized_edge_count")
    target = metrics.get("target_edge_count")
    missing_count = len(missing_edges(evaluation))
    lines = [
        "Diagnostics",
        f"P0: {evaluation.get('p0', {}).get('pass')}",
        f"Target topology: {realized}/{target}",
        f"Missing edges: {missing_count}",
        "",
        "Largest area deficits",
    ]
    for key, actual, target_value, deficit in top_deficits(report):
        lines.append(f"{key}: {actual}/{target_value}, -{deficit}")
    axis.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", fontsize=9, family="monospace")


def render_house(house_dir: Path, output_dir: Path, title: str = "Topology contact diagnostics") -> Path:
    layout = read_json(house_dir / "generated_layout.json")
    evaluation = read_json(house_dir / "evaluation.json")
    report_path = house_dir / "global_inflation_report.json"
    report = read_json(report_path) if report_path.exists() else {}
    sx, sy = site_cells(layout)
    floor_grid, group_types, group_floor_cells = collect_cells(layout.get("rooms", []))
    centers = centroids(group_floor_cells)
    missing = missing_edges(evaluation)

    figure, axes = plt.subplots(1, 3, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1, 0.65]})
    for index, floor in enumerate((1, 2)):
        draw_floor(axes[index], floor, floor_grid, group_types, group_floor_cells, centers, missing, sx, sy)
    draw_deficit_panel(axes[2], evaluation, report)
    legend_types = sorted({group_types.get(group, "unknown") for group in group_types})
    handles = [Patch(facecolor=TYPE_COLORS.get(room_type, DEFAULT_COLOR), label=room_type) for room_type in legend_types]
    figure.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    figure.suptitle(f"{house_dir.name} | {title}", fontsize=14)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{house_dir.name}_diagnostic.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def make_contact_sheet(image_paths: list[Path], output: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return
    thumb_w = 560
    thumb_h = 260
    thumbs = []
    for path, image in zip(image_paths, images):
        image.thumbnail((thumb_w, thumb_h))
        canvas = Image.new("RGB", (thumb_w, thumb_h + 28), "white")
        x = (thumb_w - image.width) // 2
        canvas.paste(image, (x, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_h + 6), path.stem.replace("_diagnostic", ""), fill=(20, 20, 20))
        thumbs.append(canvas)
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 28)), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + 28)
        sheet.paste(thumb, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    args = parse_args()
    house_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if args.houses:
        wanted = set(args.houses)
        house_dirs = [path for path in house_dirs if path.name in wanted]
    if args.max_houses is not None:
        house_dirs = house_dirs[: args.max_houses]
    outputs = [render_house(house_dir, args.output_dir, args.title) for house_dir in house_dirs]
    make_contact_sheet(outputs, args.output_dir / "all_houses_contact_sheet.png")
    print(json.dumps({"output_dir": str(args.output_dir), "image_count": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
