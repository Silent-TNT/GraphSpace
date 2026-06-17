#!/usr/bin/env python3
"""Run checkpoint -> instances -> standard JSON -> unified evaluator."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for import_dir in (SCRIPT_DIR, ROOT):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from dataset import DEFAULT_PHASE2_DIR, make_dataset  # noqa: E402
from decode_instances import decode_model_output  # noqa: E402
from export_standard_json import (  # noqa: E402
    building_instances_to_rooms,
    standard_candidate_payload,
)
from model import V5MinimalNet  # noqa: E402
from scripts.data_phase4.evaluate_candidates import evaluate_candidate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "v5_standard_pipeline",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def numpy_output(output: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        key: value[0].detach().float().cpu().numpy()
        for key, value in output.items()
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model = V5MinimalNet(
        base_channels=int(checkpoint["config"]["base_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = make_dataset(args.split, max_samples=args.max_samples)
    reports = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False):
            house_id = batch["house_id"][0]
            canvas_metadata = read_json(DEFAULT_PHASE2_DIR / "{}.json".format(house_id))
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                output = model(
                    batch["condition"].to(device),
                    batch["site_mask"].to(device),
                )
            decoded = decode_model_output(
                numpy_output(output),
                batch["site_mask"][0, 0].numpy(),
                batch["class_instance_counts"][0].numpy(),
            )
            rooms, shape_diagnostics = building_instances_to_rooms(
                decoded["building_instances"], canvas_metadata, house_id
            )
            source = read_json(ROOT / "data" / "processed" / "{}.json".format(house_id))
            requested = dict(Counter(room["type"] for room in source["rooms"]))
            site = tuple(float(value) for value in canvas_metadata["site_size_mm"])
            evaluation, _ = evaluate_candidate(
                "{}_prediction".format(house_id),
                rooms,
                requested,
                site,
            )
            candidate = standard_candidate_payload(
                "{}_prediction".format(house_id), rooms, canvas_metadata
            )
            candidate_dir = args.output_dir / "candidates"
            write_json(candidate_dir / "{}.json".format(house_id), candidate)
            minimum_fill = min(
                (
                    item["instance_coverage_by_rectangle"]
                    for item in shape_diagnostics
                    if item["instance_cells"] > 0
                ),
                default=1.0,
            )
            mean_fill = float(
                np.mean(
                    [
                        item["bounding_box_fill_ratio"]
                        if "bounding_box_fill_ratio" in item
                        else item["instance_coverage_by_rectangle"]
                        for item in shape_diagnostics
                        if item["instance_cells"] > 0
                    ]
                )
            )
            reports.append(
                {
                    "house_id": house_id,
                    "room_count": len(rooms),
                    "p0_pass": bool(evaluation["p0"]["pass"]),
                    "p1_hard_geometry_pass": bool(
                        evaluation["p1_spatial_organization"]["hard_geometry_pass"]
                    ),
                    "p1_spatial_organization_pass": bool(
                        evaluation["p1_spatial_organization"][
                            "spatial_organization_pass"
                        ]
                    ),
                    "p2_quality_gate_pass": bool(
                        evaluation["p2"]["quality_gate_pass"]
                    ),
                    "instance_round_trip_pass": bool(
                        evaluation["instance_recovery"]["pass"]
                    ),
                    "eligible_for_diversity": bool(
                        evaluation["eligible_for_diversity"]
                    ),
                    "mean_rectangle_instance_coverage": mean_fill,
                    "minimum_rectangle_instance_coverage": minimum_fill,
                    "shape_diagnostics": shape_diagnostics,
                    "evaluation": evaluation,
                }
            )
    summary = {
        "schema": "graphspace_v5_standard_pipeline_eval_v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sample_count": len(reports),
        "p0_pass_count": sum(item["p0_pass"] for item in reports),
        "p1_hard_geometry_pass_count": sum(
            item["p1_hard_geometry_pass"] for item in reports
        ),
        "p1_spatial_organization_pass_count": sum(
            item["p1_spatial_organization_pass"] for item in reports
        ),
        "p2_quality_gate_pass_count": sum(
            item["p2_quality_gate_pass"] for item in reports
        ),
        "instance_round_trip_pass_count": sum(
            item["instance_round_trip_pass"] for item in reports
        ),
        "eligible_for_diversity_count": sum(
            item["eligible_for_diversity"] for item in reports
        ),
        "mean_rectangle_instance_coverage": float(
            np.mean([item["mean_rectangle_instance_coverage"] for item in reports])
        ),
        "minimum_rectangle_instance_coverage": min(
            item["minimum_rectangle_instance_coverage"] for item in reports
        ),
        "reports": reports,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        "samples={} p0={} p1_hard={} p1_full={} p2={} round_trip={} eligible={}".format(
            summary["sample_count"],
            summary["p0_pass_count"],
            summary["p1_hard_geometry_pass_count"],
            summary["p1_spatial_organization_pass_count"],
            summary["p2_quality_gate_pass_count"],
            summary["instance_round_trip_pass_count"],
            summary["eligible_for_diversity_count"],
        )
    )
    print(
        "rectangle_coverage_mean={:.4f} rectangle_coverage_min={:.4f}".format(
            summary["mean_rectangle_instance_coverage"],
            summary["minimum_rectangle_instance_coverage"],
        )
    )
    print("wrote={}".format(args.output_dir / "summary.json"))


if __name__ == "__main__":
    main()
