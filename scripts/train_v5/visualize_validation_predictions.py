#!/usr/bin/env python3
"""Render validation truth/prediction floor-plan comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "outputs" / "v5_standard_pipeline_val_best"
DEFAULT_OUTPUT = ROOT / "outputs" / "v5_validation_visual_review"

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
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--houses",
        nargs="*",
        help="House IDs to render. Defaults to representative validation cases.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def room_on_floor(room: dict, floor: int) -> bool:
    floors = room.get("floors")
    if floors:
        return floor in [int(value) for value in floors]
    return int(room.get("floor", 1)) == floor


def draw_floor(
    ax: plt.Axes,
    rooms: list[dict],
    floor: int,
    site_x: float,
    site_y: float,
    title: str,
) -> None:
    ax.add_patch(
        Rectangle(
            (0, 0),
            site_x,
            site_y,
            facecolor="#f2f2f2",
            edgecolor="#222222",
            linewidth=1.8,
            zorder=0,
        )
    )
    for room in rooms:
        if not room_on_floor(room, floor):
            continue
        room_type = room["type"]
        x0, y0 = map(float, room["box_min"][:2])
        x1, y1 = map(float, room["box_max"][:2])
        width, depth = x1 - x0, y1 - y0
        ax.add_patch(
            Rectangle(
                (x0, y0),
                width,
                depth,
                facecolor=ROOM_COLORS.get(room_type, "#dddddd"),
                edgecolor="#202020",
                linewidth=1.0,
                alpha=0.88,
                zorder=1,
            )
        )
        label = ROOM_LABELS.get(room_type, room_type)
        if width >= 900 and depth >= 600:
            ax.text(
                x0 + width / 2,
                y0 + depth / 2,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color="#111111",
                clip_on=True,
                zorder=2,
            )
    ax.set_xlim(0, site_x)
    ax.set_ylim(0, site_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("X (mm)", fontsize=7)
    ax.set_ylabel("Y (mm)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(color="#ffffff", linewidth=0.4, alpha=0.7)


def render_case(
    house_id: str,
    report: dict,
    prediction: dict,
    truth: dict,
    output_dir: Path,
) -> Path:
    building = truth["metadata"]["building_size"]
    site_x = float(building["x"])
    site_y = float(building["y"])
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for column, floor in enumerate((1, 2)):
        draw_floor(
            axes[0, column],
            truth["rooms"],
            floor,
            site_x,
            site_y,
            "Truth - Floor {}".format(floor),
        )
        draw_floor(
            axes[1, column],
            prediction["rooms"],
            floor,
            site_x,
            site_y,
            "V5 prediction - Floor {}".format(floor),
        )
    figure.suptitle(
        (
            "{} | P0={} P1={} P2={} | rectangle coverage={:.3f}"
        ).format(
            house_id,
            "pass" if report["p0_pass"] else "fail",
            "pass" if report["p1_spatial_organization_pass"] else "fail",
            "pass" if report["p2_quality_gate_pass"] else "fail",
            report["mean_rectangle_instance_coverage"],
        ),
        fontsize=13,
        weight="bold",
    )
    handles = [
        Patch(facecolor=color, edgecolor="#202020", label=ROOM_LABELS[room_type])
        for room_type, color in ROOM_COLORS.items()
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "{}_comparison.png".format(house_id)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    summary = read_json(args.eval_dir / "summary.json")
    reports = {item["house_id"]: item for item in summary["reports"]}
    houses = args.houses or [
        "house_1232",
        "house_1809",
        "house_9",
        "house_1964",
    ]
    manifest = []
    for house_id in houses:
        report = reports[house_id]
        prediction = read_json(args.eval_dir / "candidates" / f"{house_id}.json")
        truth = read_json(ROOT / "data" / "processed" / f"{house_id}.json")
        path = render_case(
            house_id,
            report,
            prediction,
            truth,
            args.output_dir,
        )
        manifest.append(
            {
                "house_id": house_id,
                "image": str(path),
                "p0_pass": report["p0_pass"],
                "p1_pass": report["p1_spatial_organization_pass"],
                "p2_pass": report["p2_quality_gate_pass"],
                "rectangle_coverage": report[
                    "mean_rectangle_instance_coverage"
                ],
            }
        )
        print(path)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
