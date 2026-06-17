#!/usr/bin/env python3
"""Render representative latest-weight validation rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ROOT / "outputs" / "v5_full_374_47_validation"
DEFAULT_OUTPUT = ROOT / "outputs" / "v5_latest_weight_visual_review"
ROOM_COLORS = {
    "entryway": "#f4a261",
    "living_room": "#e9c46a",
    "dining_room": "#f6bd60",
    "kitchen": "#e76f51",
    "bedroom": "#8ecae6",
    "bathroom": "#90be6d",
    "corridor": "#b8b8b8",
    "stairs": "#9b5de5",
    "utility": "#6c757d",
    "balcony": "#52b788",
    "multi_purpose": "#f28482",
}
ROOM_LABELS = {
    "entryway": "Entry",
    "living_room": "Living",
    "dining_room": "Dining",
    "kitchen": "Kitchen",
    "bedroom": "Bedroom",
    "bathroom": "Bath",
    "corridor": "Corridor",
    "stairs": "Stairs",
    "utility": "Utility",
    "balcony": "Balcony",
    "multi_purpose": "Multi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--houses",
        nargs="*",
        default=["house_375", "house_1232", "house_1077", "house_1964"],
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def room_on_floor(room: dict, floor: int) -> bool:
    floors = room.get("floors", [room.get("floor", 1)])
    return floor in [int(value) for value in floors]


def draw_floor(
    axis: plt.Axes,
    rooms: list[dict],
    floor: int,
    site_x: float,
    site_y: float,
    title: str,
) -> None:
    axis.add_patch(
        Rectangle(
            (0, 0),
            site_x,
            site_y,
            facecolor="#f4f4f4",
            edgecolor="#111111",
            linewidth=1.8,
        )
    )
    for room in rooms:
        if not room_on_floor(room, floor):
            continue
        room_type = str(room["type"])
        x0, y0 = (float(value) for value in room["box_min"][:2])
        x1, y1 = (float(value) for value in room["box_max"][:2])
        width, depth = x1 - x0, y1 - y0
        axis.add_patch(
            Rectangle(
                (x0, y0),
                width,
                depth,
                facecolor=ROOM_COLORS.get(room_type, "#dddddd"),
                edgecolor="#202020",
                linewidth=0.9,
                alpha=0.9,
            )
        )
        if width >= 1200 and depth >= 900:
            axis.text(
                x0 + width / 2,
                y0 + depth / 2,
                ROOM_LABELS.get(room_type, room_type),
                ha="center",
                va="center",
                fontsize=6,
                clip_on=True,
            )
    axis.set_xlim(0, site_x)
    axis.set_ylim(0, site_y)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontsize=10, weight="bold")
    axis.tick_params(labelsize=6)
    axis.grid(color="#ffffff", linewidth=0.5)


def report_maps(candidate_dir: Path) -> tuple[dict, dict]:
    rollout = read_json(candidate_dir / "summary.json")
    unified = read_json(candidate_dir / "unified_evaluation.json")
    return (
        {item["house_id"]: item for item in rollout["reports"]},
        {item["candidate_id"]: item for item in unified["reports"]},
    )


def render_case(
    house_id: str,
    candidate_dir: Path,
    output_dir: Path,
    rollout: dict,
    evaluation: dict,
) -> Path:
    truth = read_json(ROOT / "data" / "processed" / f"{house_id}.json")
    prediction = read_json(candidate_dir / f"{house_id}.json")
    building = truth["metadata"]["building_size"]
    site_x, site_y = float(building["x"]), float(building["y"])
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for column, floor in enumerate((1, 2)):
        draw_floor(
            axes[0, column],
            truth["rooms"],
            floor,
            site_x,
            site_y,
            f"Dataset truth - Floor {floor}",
        )
        draw_floor(
            axes[1, column],
            prediction["rooms"],
            floor,
            site_x,
            site_y,
            f"Latest weights - Floor {floor}",
        )
    p0 = evaluation["p0"]["pass"]
    p1 = evaluation["p1_spatial_organization"]["spatial_organization_pass"]
    p2 = evaluation["p2"]["quality_gate_pass"]
    figure.suptitle(
        (
            f"{house_id} | rooms {rollout['placed_count']}/"
            f"{rollout['expected_count']} | box IoU {rollout['mean_box_iou']:.3f}"
            f" | P0={'pass' if p0 else 'fail'}"
            f" P1={'pass' if p1 else 'fail'}"
            f" P2={'pass' if p2 else 'fail'}"
        ),
        fontsize=13,
        weight="bold",
    )
    handles = [
        Patch(facecolor=color, edgecolor="#202020", label=ROOM_LABELS[key])
        for key, color in ROOM_COLORS.items()
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{house_id}_latest_comparison.png"
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def main() -> None:
    args = parse_args()
    rollout_map, evaluation_map = report_maps(args.candidate_dir)
    manifest = []
    for house_id in args.houses:
        output = render_case(
            house_id,
            args.candidate_dir,
            args.output_dir,
            rollout_map[house_id],
            evaluation_map[house_id],
        )
        manifest.append(
            {
                "house_id": house_id,
                "image": str(output),
                "rollout": rollout_map[house_id],
                "p0": evaluation_map[house_id]["p0"]["pass"],
                "p1": evaluation_map[house_id][
                    "p1_spatial_organization"
                ]["spatial_organization_pass"],
                "p2": evaluation_map[house_id]["p2"]["quality_gate_pass"],
            }
        )
        print(output)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
