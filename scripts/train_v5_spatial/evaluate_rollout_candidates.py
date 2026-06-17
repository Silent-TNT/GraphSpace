#!/usr/bin/env python3
"""Run the unified P0/P1/P2 evaluator over rollout candidate JSON files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_phase4.evaluate_candidates import evaluate_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    reports = []
    for candidate_path in sorted(args.candidate_dir.glob("house_*.json")):
        if candidate_path.stem.endswith(("_request", "_evaluation")):
            continue
        house_id = candidate_path.stem
        truth_path = ROOT / "data" / "processed" / f"{house_id}.json"
        if not truth_path.exists():
            continue
        candidate = read_json(candidate_path)
        truth = read_json(truth_path)
        building = truth["metadata"]["building_size"]
        requested = dict(Counter(room["type"] for room in truth["rooms"]))
        report, _ = evaluate_candidate(
            house_id,
            candidate["rooms"],
            requested,
            (float(building["x"]), float(building["y"])),
        )
        reports.append(report)
    summary = {
        "candidate_count": len(reports),
        "p0_pass_count": sum(report["p0"]["pass"] for report in reports),
        "p1_hard_geometry_pass_count": sum(
            report["p1_spatial_organization"]["hard_geometry_pass"]
            for report in reports
        ),
        "p1_spatial_organization_pass_count": sum(
            report["p1_spatial_organization"]["spatial_organization_pass"]
            for report in reports
        ),
        "p2_quality_gate_pass_count": sum(
            report["p2"]["quality_gate_pass"] for report in reports
        ),
        "instance_recovery_pass_count": sum(
            report["instance_recovery"]["pass"] for report in reports
        ),
        "eligible_count": sum(
            report["eligible_for_diversity"] for report in reports
        ),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "reports"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
