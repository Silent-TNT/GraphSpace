#!/usr/bin/env python3
"""Validate the unified evaluator on source truth and the fixed 2026-06-11 baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE4_DIR = ROOT / "scripts" / "data_phase4"
if str(PHASE4_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE4_DIR))

from evaluate_candidates import evaluate_candidate, summarize_candidate_set, write_json


def validate_truth() -> dict:
    counts = {
        "total": 0,
        "p0_pass": 0,
        "p1_hard_geometry_pass": 0,
        "p1_spatial_organization_pass": 0,
        "p2_quality_gate_pass": 0,
        "instance_recovery_pass": 0,
        "eligible_for_diversity": 0,
    }
    for path in sorted((ROOT / "data" / "processed").glob("house_*.json")):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        building = data["metadata"]["building_size"]
        report, _ = evaluate_candidate(
            path.stem,
            data["rooms"],
            data["metadata"]["stats"],
            (building["x"], building["y"]),
        )
        counts["total"] += 1
        counts["p0_pass"] += int(report["p0"]["pass"])
        counts["p1_hard_geometry_pass"] += int(
            report["p1_spatial_organization"]["hard_geometry_pass"]
        )
        counts["p1_spatial_organization_pass"] += int(
            report["p1_spatial_organization"]["spatial_organization_pass"]
        )
        counts["p2_quality_gate_pass"] += int(report["p2"]["quality_gate_pass"])
        counts["instance_recovery_pass"] += int(
            report["instance_recovery"]["pass"]
        )
        counts["eligible_for_diversity"] += int(
            report["eligible_for_diversity"]
        )
    return counts


def validate_baseline(output_dir: Path) -> list[dict]:
    base = ROOT / "outputs" / "route_baseline" / "2026-06-11"
    rows = []
    for case in ("small", "medium", "large"):
        for route in ("v4", "block_cut", "zoned"):
            request_path = base / case / "seed_11" / route / "input.json"
            if not request_path.exists():
                continue
            with request_path.open(encoding="utf-8") as handle:
                request = json.load(handle)
            candidates = []
            for seed in (11, 42, 123):
                rooms_path = base / case / "seed_{}".format(seed) / route / "rooms.json"
                with rooms_path.open(encoding="utf-8") as handle:
                    rooms = json.load(handle)
                candidates.append({
                    "candidate_id": "{}_seed_{}".format(route, seed),
                    "rooms": rooms,
                })
            summary = summarize_candidate_set(
                candidates,
                request["room_counts"],
                (request["site"]["x_mm"], request["site"]["y_mm"]),
            )
            write_json(output_dir / "{}_{}_3seeds.json".format(case, route), summary)
            rows.append({
                "case": case,
                "route": route,
                "candidate_count": len(candidates),
                "eligible_candidate_count": summary["eligible_candidate_count"],
                "pair_count": summary["diversity"]["pair_count"],
                "structure_cluster_count": summary["diversity"]["structure_cluster_count"],
                "category_counts": summary["diversity"]["category_counts"],
                "diversity_pass": summary["diversity"]["pass"],
            })
    return rows


def main() -> None:
    output_dir = ROOT / "data" / "phase4_evaluation"
    report = {
        "schema": "graphspace_unified_evaluator_validation_v1",
        "truth_dataset": validate_truth(),
        "fixed_baseline_3seeds": validate_baseline(output_dir),
        "notes": [
            "Three seeds cannot satisfy the final at-least-four-eligible gate.",
            "P1 evaluates functional-block adjacency and cross-floor organization.",
            "P2 is a distribution-aware quality gate, not an all-room hard rule.",
        ],
    }
    write_json(output_dir / "validation_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
