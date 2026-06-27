#!/usr/bin/env python3
"""Unified evaluation for validity, block organization and diversity."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for import_dir in (
    ROOT,
    ROOT / "scripts" / "data_phase2",
    ROOT / "scripts" / "data_phase3",
    ROOT / "scripts" / "train_v5_spatial",
):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.data_phase1.run_phase1 import (
    extract_relations,
    p1_spatial_organization_report,
)
from build_v5_representation import decode_instances, encode_house, evaluate_round_trip
from evaluate_structural_diversity import compare_layouts, layout_signature
from analyze_topology_dual import build_report as build_topology_dual_report


MODULUS_MM = 300.0
TOL = 1e-6
ROOM_TYPES = (
    "entryway",
    "living_room",
    "dining_room",
    "kitchen",
    "bedroom",
    "bathroom",
    "corridor",
    "stairs",
    "utility",
    "balcony",
    "multi_purpose",
)
ROOM_RULES = {
    "entryway": {"area": 6.0, "min_w": 1.8, "max_aspect": 3.0},
    "living_room": {"area": 34.0, "min_w": 3.6, "max_aspect": 2.6, "needs_exterior": True},
    "dining_room": {"area": 16.0, "min_w": 2.7, "max_aspect": 3.0},
    "kitchen": {"area": 12.0, "min_w": 2.4, "max_aspect": 3.2, "needs_exterior": True},
    "bedroom": {"area": 15.0, "min_w": 2.7, "max_aspect": 2.8, "needs_exterior": True},
    "bathroom": {"area": 5.5, "min_w": 1.5, "max_aspect": 3.2},
    "corridor": {"area": 9.0, "min_w": 1.2, "max_aspect": 8.0},
    "stairs": {"area": 9.0, "min_w": 2.4, "max_aspect": 2.2},
    "utility": {"area": 6.0, "min_w": 1.8, "max_aspect": 3.5},
    "balcony": {"area": 7.0, "min_w": 1.5, "max_aspect": 5.0, "needs_exterior": True},
    "multi_purpose": {"area": 18.0, "min_w": 2.7, "max_aspect": 3.2, "needs_exterior": True},
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def infer_floors(room: dict) -> list[int]:
    values = room.get("floors")
    if isinstance(values, list) and values:
        return sorted({int(value) for value in values})
    z0 = float(room["box_min"][2])
    z1 = float(room["box_max"][2])
    if z0 <= TOL and z1 >= 6000.0 - TOL:
        return [1, 2]
    return [2] if z0 >= 3000.0 - TOL else [1]


def normalize_rooms(rooms: list[Any]) -> list[dict]:
    result = []
    for index, room in enumerate(rooms):
        data = room.to_json() if hasattr(room, "to_json") else dict(room)
        room_type = str(data.get("type", data.get("room_type", "unknown")))
        room_id = str(data.get("id", data.get("room_id", "{}_{}".format(room_type, index))))
        functional_id = next(
            (
                str(data[key])
                for key in ("functional_id", "group_id", "parent_id")
                if data.get(key)
            ),
            room_id.split("_part_", 1)[0] if "_part_" in room_id else room_id,
        )
        box_min = [float(value) for value in data["box_min"]]
        box_max = [float(value) for value in data["box_max"]]
        normalized = {
            "id": room_id,
            "functional_id": functional_id,
            "type": room_type,
            "box_min": box_min,
            "box_max": box_max,
            "auto_added": bool(data.get("auto_added", False)),
        }
        normalized["floors"] = infer_floors({**data, **normalized})
        normalized["floor"] = int(data.get("floor") or normalized["floors"][0])
        result.append(normalized)
    return result


def _aligned(value: float) -> bool:
    return abs(value / MODULUS_MM - round(value / MODULUS_MM)) <= TOL


def _axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _volume_overlap(a: dict, b: dict) -> float:
    return math.prod(
        _axis_overlap(a["box_min"][axis], a["box_max"][axis],
                      b["box_min"][axis], b["box_max"][axis])
        for axis in range(3)
    )


def _touches_exterior(room: dict, site_x: float, site_y: float) -> bool:
    x0, y0, _ = room["box_min"]
    x1, y1, _ = room["box_max"]
    return any((
        abs(x0) <= TOL,
        abs(y0) <= TOL,
        abs(x1 - site_x) <= TOL,
        abs(y1 - site_y) <= TOL,
    ))


def evaluate_p0(
    rooms: list[dict], requested: dict[str, int], site: tuple[float, float]
) -> dict:
    site_x, site_y = map(float, site)
    raw_part_counts = Counter(room["type"] for room in rooms)
    group_types: dict[str, str] = {}
    group_part_counts: Counter[str] = Counter()
    mixed_type_groups: dict[str, set[str]] = {}
    for room in rooms:
        group_id = str(room.get("functional_id", room["id"]))
        group_part_counts[group_id] += 1
        if group_id in group_types and group_types[group_id] != room["type"]:
            mixed_type_groups.setdefault(group_id, {group_types[group_id]}).add(
                room["type"]
            )
        group_types.setdefault(group_id, room["type"])
    counts = Counter(group_types.values())
    all_types = sorted(set(requested) | set(counts))
    count_delta = {
        room_type: counts.get(room_type, 0) - int(requested.get(room_type, 0))
        for room_type in all_types
    }
    missing = {key: -value for key, value in count_delta.items() if value < 0}
    extra = {key: value for key, value in count_delta.items() if value > 0}
    unknown = [room["id"] for room in rooms if room["type"] not in ROOM_TYPES]
    invalid = []
    outside = []
    modulus = []
    overlaps = []
    floor_errors = []

    for room in rooms:
        mins, maxs = room["box_min"], room["box_max"]
        dims = [maxs[index] - mins[index] for index in range(3)]
        if any(value <= TOL for value in dims):
            invalid.append(room["id"])
        if (
            mins[0] < -TOL or mins[1] < -TOL or mins[2] < -TOL
            or maxs[0] > site_x + TOL or maxs[1] > site_y + TOL
            or maxs[2] > 6000.0 + TOL
        ):
            outside.append(room["id"])
        if any(not _aligned(value) for value in mins + maxs):
            modulus.append(room["id"])
        floors = infer_floors(room)
        expected_z = (0.0, 6000.0) if floors == [1, 2] else (
            (0.0, 3000.0) if floors == [1] else (3000.0, 6000.0)
        )
        if abs(mins[2] - expected_z[0]) > TOL or abs(maxs[2] - expected_z[1]) > TOL:
            floor_errors.append(room["id"])

    for index, room_a in enumerate(rooms):
        for room_b in rooms[index + 1:]:
            overlap = _volume_overlap(room_a, room_b)
            if overlap > TOL:
                overlaps.append({
                    "a": room_a["id"],
                    "b": room_b["id"],
                    "volume_mm3": overlap,
                })

    checks = {
        "requested_counts_match": not missing and not extra,
        "dining_room_present": counts.get("dining_room", 0) >= 1,
        "known_room_types": not unknown,
        "functional_groups_have_single_type": not mixed_type_groups,
        "positive_geometry": not invalid,
        "inside_site": not outside,
        "modulus_300mm": not modulus,
        "no_volume_overlap": not overlaps,
        "two_floor_z_convention": not floor_errors,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "requested_counts": dict(requested),
        "generated_counts": dict(sorted(counts.items())),
        "count_delta": count_delta,
        "details": {
            "missing_counts": missing,
            "extra_counts": extra,
            "raw_part_counts": dict(sorted(raw_part_counts.items())),
            "multipart_functional_groups": {
                key: value
                for key, value in sorted(group_part_counts.items())
                if value > 1
            },
            "mixed_type_functional_groups": {
                key: sorted(value) for key, value in mixed_type_groups.items()
            },
            "unknown_type_room_ids": unknown,
            "invalid_room_ids": invalid,
            "out_of_bounds_room_ids": outside,
            "modulus_room_ids": modulus,
            "overlap_pairs": overlaps,
            "floor_z_error_room_ids": floor_errors,
        },
    }


def evaluate_p2(rooms: list[dict], site: tuple[float, float]) -> dict:
    site_x, site_y = map(float, site)
    area_fail = []
    width_fail = []
    aspect_fail = []
    exterior_fail = []
    for room in rooms:
        rules = ROOM_RULES.get(room["type"])
        if not rules:
            continue
        dx = room["box_max"][0] - room["box_min"][0]
        dy = room["box_max"][1] - room["box_min"][1]
        if dx <= TOL or dy <= TOL:
            continue
        area = dx * dy / 1_000_000.0
        width = min(dx, dy) / 1000.0
        aspect = max(dx, dy) / min(dx, dy)
        if area < rules["area"] * 0.65:
            area_fail.append(room["id"])
        if width < rules["min_w"] - TOL:
            width_fail.append(room["id"])
        if aspect > rules["max_aspect"] + TOL:
            aspect_fail.append(room["id"])
        if rules.get("needs_exterior") and not _touches_exterior(room, site_x, site_y):
            exterior_fail.append(room["id"])
    checks = {
        "area_thresholds": not area_fail,
        "minimum_widths": not width_fail,
        "aspect_ratios": not aspect_fail,
        "required_exterior_contact": not exterior_fail,
    }
    room_count = max(len(rooms), 1)
    exterior_eligible = sum(
        bool(ROOM_RULES.get(room["type"], {}).get("needs_exterior"))
        for room in rooms
    )
    pass_rates = {
        "area": 1.0 - len(area_fail) / room_count,
        "minimum_width": 1.0 - len(width_fail) / room_count,
        "aspect_ratio": 1.0 - len(aspect_fail) / room_count,
        "exterior_contact": (
            1.0 - len(exterior_fail) / exterior_eligible
            if exterior_eligible else 1.0
        ),
    }
    # These gates are intentionally distribution-aware. The former all-room
    # rule rejected every one of the 468 accepted source houses.
    quality_gate_thresholds = {
        "area": 0.55,
        "minimum_width": 0.75,
        "aspect_ratio": 0.90,
        "exterior_contact": 0.55,
    }
    quality_gate_checks = {
        key: pass_rates[key] >= threshold
        for key, threshold in quality_gate_thresholds.items()
    }
    return {
        "all_rooms_pass": all(checks.values()),
        "checks": checks,
        "pass_rates": pass_rates,
        "quality_gate_thresholds": quality_gate_thresholds,
        "quality_gate_checks": quality_gate_checks,
        "quality_gate_pass": all(quality_gate_checks.values()),
        "details": {
            "area_fail_room_ids": area_fail,
            "width_fail_room_ids": width_fail,
            "aspect_fail_room_ids": aspect_fail,
            "exterior_fail_room_ids": exterior_fail,
        },
    }


def candidate_payload(
    candidate_id: str,
    rooms: list[dict],
    site: tuple[float, float],
    requested: dict[str, int],
) -> dict:
    return {
        "house_id": candidate_id,
        "metadata": {
            "building_size": {"x": float(site[0]), "y": float(site[1]), "z": 6000.0},
            "stats": dict(requested),
        },
        "rooms": rooms,
    }


def p1_topology_realization_report(topology: dict, layout: dict) -> dict:
    """Evaluate P1 as target heterogeneous topology realization in final geometry."""
    topology_report = build_topology_dual_report(topology, layout)
    target_report = topology_report["target_vs_realized"]
    target_count = int(target_report["target_edge_count"])
    realized_count = int(target_report["realized_edge_count"])
    required_count = int(target_report["required_edge_count"])
    required_realized = int(target_report["required_realized_edge_count"])
    all_target_realized = target_count > 0 and realized_count == target_count
    all_required_realized = required_realized == required_count
    checks = {
        "target_topology_provided": target_count > 0,
        "all_target_edges_realized": all_target_realized,
        "all_required_edges_realized": all_required_realized,
    }
    return {
        "mode": "target_topology_realization",
        "checks": checks,
        "hard_geometry_pass": all_required_realized,
        "spatial_organization_pass": all(checks.values()),
        "target_topology": target_report,
        "realized_planar_dual": topology_report["realized_planar_dual"],
        "voxel_assignment": topology_report["voxel_assignment"],
        "scope": [
            "P1 compares the generated heterogeneous topology against the final planar dual graph.",
            "It does not use fixed residential adjacency rules when target topology is provided.",
            "Door positions and pedestrian paths remain outside the current output scope.",
        ],
    }


def disabled_p2_report() -> dict:
    return {
        "enabled": False,
        "quality_gate_pass": False,
        "all_rooms_pass": None,
        "checks": {},
        "pass_rates": {},
        "quality_gate_thresholds": {},
        "quality_gate_checks": {},
        "details": {
            "disabled_reason": (
                "P2 geometry/function quality is temporarily disabled because the "
                "current thresholds are not a reliable training or generation gate."
            )
        },
    }


def evaluate_candidate(
    candidate_id: str,
    rooms: list[Any],
    requested: dict[str, int],
    site: tuple[float, float],
    topology: dict | None = None,
) -> tuple[dict, dict[str, np.ndarray] | None]:
    normalized = normalize_rooms(rooms)
    p0 = evaluate_p0(normalized, requested, site)
    payload = candidate_payload(candidate_id, normalized, site, requested)
    if topology is None:
        relations = extract_relations(normalized, float(site[0]), float(site[1]))
        p1 = p1_spatial_organization_report(normalized, relations)
        p1["mode"] = "legacy_fixed_rule_proxy"
    else:
        p1 = p1_topology_realization_report(topology, payload)
    p2 = disabled_p2_report()
    instance_report: dict[str, Any]
    arrays = None
    try:
        arrays, metadata = encode_house(payload)
        round_trip = evaluate_round_trip(payload, arrays, metadata)
        decoded = decode_instances(arrays, metadata)
        instance_report = {
            "pass": bool(round_trip["exact"]),
            "round_trip": round_trip,
            "decoded_instance_count": len(decoded),
            "signature": layout_signature(arrays),
        }
    except Exception as exc:
        instance_report = {
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    eligible = bool(p0["pass"] and p1["hard_geometry_pass"] and instance_report["pass"])
    report = {
        "candidate_id": candidate_id,
        "site": {"x_mm": float(site[0]), "y_mm": float(site[1]), "z_mm": 6000.0},
        "requested_counts": dict(requested),
        "room_count": len(normalized),
        "p0": p0,
        "p1_spatial_organization": p1,
        "p2": p2,
        "instance_recovery": instance_report,
        "eligible_for_diversity": eligible,
        "overall": {
            "hard_valid": bool(p0["pass"] and p1["hard_geometry_pass"]),
            "quality_gate_pass": None,
            "spatial_organization_pass": bool(
                p1["spatial_organization_pass"]
            ),
            "instance_recovery_pass": bool(instance_report["pass"]),
            "p2_enabled": False,
        },
    }
    return report, arrays


def summarize_candidate_set(
    candidates: list[dict],
    requested: dict[str, int],
    site: tuple[float, float],
) -> dict:
    reports = []
    arrays_by_id = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        report, arrays = evaluate_candidate(
            candidate_id, candidate["rooms"], requested, site
        )
        reports.append(report)
        if arrays is not None and report["eligible_for_diversity"]:
            arrays_by_id[candidate_id] = arrays

    pairs = []
    categories = Counter()
    for left, right in itertools.combinations(sorted(arrays_by_id), 2):
        result = compare_layouts(arrays_by_id[left], arrays_by_id[right])
        categories[result["category"]] += 1
        pairs.append({"candidate_a": left, "candidate_b": right, **result})

    pair_lookup = {
        tuple(sorted((item["candidate_a"], item["candidate_b"]))): item
        for item in pairs
    }
    structure_clusters: list[list[str]] = []
    for candidate_id in sorted(arrays_by_id):
        assigned = False
        for cluster in structure_clusters:
            representative = cluster[0]
            pair = pair_lookup.get(tuple(sorted((candidate_id, representative))))
            if pair is None or pair["geometry_distance"] < 0.20:
                cluster.append(candidate_id)
                assigned = True
                break
        if not assigned:
            structure_clusters.append([candidate_id])

    eligible_count = len(arrays_by_id)
    pair_count = len(pairs)
    substantial = categories.get("substantial_structure", 0)
    weak = categories.get("label_only", 0) + categories.get("near_duplicate", 0)
    diversity_checks = {
        "at_least_4_eligible": eligible_count >= 4,
        "at_least_3_structure_clusters": len(structure_clusters) >= 3,
        "substantial_pair_ratio_at_least_50pct": pair_count > 0 and substantial / pair_count >= 0.5,
        "label_only_plus_near_duplicate_at_most_25pct": pair_count > 0 and weak / pair_count <= 0.25,
    }
    return {
        "schema": "graphspace_candidate_evaluation_v1",
        "site": {"x_mm": float(site[0]), "y_mm": float(site[1]), "z_mm": 6000.0},
        "requested_counts": dict(requested),
        "candidate_count": len(candidates),
        "eligible_candidate_count": eligible_count,
        "candidate_reports": reports,
        "diversity": {
            "pair_count": pair_count,
            "category_counts": dict(categories),
            "structure_cluster_threshold": 0.20,
            "structure_cluster_count": len(structure_clusters),
            "structure_clusters": structure_clusters,
            "checks": diversity_checks,
            "pass": all(diversity_checks.values()),
            "pairs": pairs,
        },
    }


def read_candidate_file(path: Path) -> dict:
    payload = load_json(path)
    rooms = payload if isinstance(payload, list) else payload.get("rooms")
    if not isinstance(rooms, list):
        raise ValueError("{} does not contain a room list".format(path))
    candidate_id = path.stem
    if candidate_id in {"rooms", "layout", "candidate"}:
        candidate_id = "{}_{}".format(path.parent.parent.name, path.parent.name)
    return {"candidate_id": candidate_id, "rooms": rooms}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = load_json(args.request)
    site_value = request.get("site", request)
    site = (
        float(site_value.get("x_mm", site_value.get("x"))),
        float(site_value.get("y_mm", site_value.get("y"))),
    )
    requested = request.get("room_counts", request.get("rooms", {}))
    candidates = [read_candidate_file(path) for path in args.candidates]
    summary = summarize_candidate_set(candidates, requested, site)
    write_json(args.output, summary)
    print("Candidates: {}".format(summary["candidate_count"]))
    print("Eligible: {}".format(summary["eligible_candidate_count"]))
    print("Diversity pass: {}".format(summary["diversity"]["pass"]))
    print("Wrote: {}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
