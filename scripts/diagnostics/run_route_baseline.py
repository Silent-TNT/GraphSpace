#!/usr/bin/env python3
"""Run comparable fixed-case diagnostics for the three GraphSpace generators."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INFER_DIR = ROOT / "scripts" / "spatial_modal_infer"
SANDBOX_DIR = ROOT / "notebooks" / "sandbox"
for import_dir in (INFER_DIR, SANDBOX_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from block_cut_generator import (  # noqa: E402
    ROOM_RULES,
    BlockCutConfig,
    generate_block_cut_layout,
    write_layout_package,
)
from layout import build_user_request  # noqa: E402
from pipeline import generate_user_layout, load_model  # noqa: E402
from zoned_layout_generator import (  # noqa: E402
    generate_zoned_layout,
    write_zoned_layout_package,
)


TEST_CASES = {
    "small": {
        "site": (12000, 12000),
        "rooms": {
            "entryway": 1,
            "living_room": 1,
            "dining_room": 1,
            "bedroom": 2,
            "bathroom": 1,
            "corridor": 1,
            "stairs": 1,
        },
    },
    "medium": {
        "site": (18000, 15000),
        "rooms": {
            "entryway": 1,
            "living_room": 1,
            "dining_room": 1,
            "kitchen": 1,
            "bedroom": 3,
            "bathroom": 2,
            "corridor": 2,
            "stairs": 1,
            "balcony": 1,
        },
    },
    "large": {
        "site": (22000, 18000),
        "rooms": {
            "entryway": 1,
            "living_room": 1,
            "dining_room": 1,
            "kitchen": 1,
            "bedroom": 5,
            "bathroom": 3,
            "corridor": 2,
            "stairs": 1,
            "utility": 1,
            "balcony": 2,
            "multi_purpose": 1,
        },
    },
}

DEFAULT_SEEDS = (11, 42, 123)
ROUTES = ("v4", "block_cut", "zoned")
MODULUS_MM = 300.0
TOL = 1e-6


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=_json_default)
    return path


def normalize_rooms(rooms: list[Any]) -> list[dict]:
    normalized = []
    for index, room in enumerate(rooms):
        if hasattr(room, "to_json"):
            data = room.to_json()
        else:
            data = dict(room)
        room_type = data.get("type", data.get("room_type", "unknown"))
        box_min = [float(v) for v in data["box_min"]]
        box_max = [float(v) for v in data["box_max"]]
        normalized.append(
            {
                "id": str(data.get("id", data.get("room_id", f"{room_type}_{index}"))),
                "type": str(room_type),
                "floor": data.get("floor"),
                "box_min": box_min,
                "box_max": box_max,
                "auto_added": bool(data.get("auto_added", False)),
            }
        )
    return normalized


def _axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _volume_overlap(a: dict, b: dict) -> float:
    return math.prod(
        _axis_overlap(a["box_min"][axis], a["box_max"][axis], b["box_min"][axis], b["box_max"][axis])
        for axis in range(3)
    )


def _aligned(value: float) -> bool:
    return abs(value / MODULUS_MM - round(value / MODULUS_MM)) <= TOL


def _touches_exterior(room: dict, site_x: float, site_y: float) -> bool:
    x0, y0, _ = room["box_min"]
    x1, y1, _ = room["box_max"]
    return (
        abs(x0) <= TOL
        or abs(y0) <= TOL
        or abs(x1 - site_x) <= TOL
        or abs(y1 - site_y) <= TOL
    )


def evaluate_rooms(rooms: list[dict], requested: dict[str, int], site: tuple[int, int]) -> dict:
    site_x, site_y = map(float, site)
    counts = Counter(room["type"] for room in rooms if not room.get("auto_added"))
    count_delta = {
        room_type: counts.get(room_type, 0) - int(target)
        for room_type, target in requested.items()
    }
    missing = {key: -value for key, value in count_delta.items() if value < 0}
    extra = {key: value for key, value in count_delta.items() if value > 0}

    invalid_ids = []
    out_of_bounds_ids = []
    modulus_ids = []
    area_fail_ids = []
    width_fail_ids = []
    aspect_fail_ids = []
    exterior_fail_ids = []
    overlap_pairs = []

    for room in rooms:
        room_id = room["id"]
        mins = room["box_min"]
        maxs = room["box_max"]
        dims = [maxs[i] - mins[i] for i in range(3)]
        if any(dim <= TOL for dim in dims):
            invalid_ids.append(room_id)
        if (
            mins[0] < -TOL
            or mins[1] < -TOL
            or mins[2] < -TOL
            or maxs[0] > site_x + TOL
            or maxs[1] > site_y + TOL
            or maxs[2] > 6000.0 + TOL
        ):
            out_of_bounds_ids.append(room_id)
        if any(not _aligned(value) for value in mins + maxs):
            modulus_ids.append(room_id)

        rules = ROOM_RULES.get(room["type"])
        if not rules or dims[0] <= TOL or dims[1] <= TOL:
            continue
        width_m = min(dims[0], dims[1]) / 1000.0
        area_m2 = dims[0] * dims[1] / 1_000_000.0
        aspect = max(dims[0], dims[1]) / min(dims[0], dims[1])
        if area_m2 < float(rules["area"]) * 0.65:
            area_fail_ids.append(room_id)
        if width_m < float(rules["min_w"]) - TOL:
            width_fail_ids.append(room_id)
        if aspect > float(rules["max_aspect"]) + TOL:
            aspect_fail_ids.append(room_id)
        if rules.get("needs_exterior") and not _touches_exterior(room, site_x, site_y):
            exterior_fail_ids.append(room_id)

    for index, room_a in enumerate(rooms):
        for room_b in rooms[index + 1 :]:
            overlap = _volume_overlap(room_a, room_b)
            if overlap > TOL:
                overlap_pairs.append(
                    {
                        "a": room_a["id"],
                        "b": room_b["id"],
                        "volume_mm3": round(overlap, 3),
                    }
                )

    failures = []
    if missing:
        failures.append("room_count_missing")
    if extra:
        failures.append("room_count_extra")
    if counts.get("dining_room", 0) < 1:
        failures.append("required_dining_room_missing")
    if invalid_ids:
        failures.append("invalid_room_geometry")
    if out_of_bounds_ids:
        failures.append("site_boundary_violation")
    if modulus_ids:
        failures.append("modulus_violation")
    if overlap_pairs:
        failures.append("room_overlap")
    if area_fail_ids:
        failures.append("room_area_below_threshold")
    if width_fail_ids:
        failures.append("room_width_below_threshold")
    if aspect_fail_ids:
        failures.append("room_aspect_above_threshold")
    if exterior_fail_ids:
        failures.append("exterior_contact_missing")

    p0_pass = not any(
        label
        in {
            "room_count_missing",
            "required_dining_room_missing",
            "invalid_room_geometry",
            "site_boundary_violation",
            "modulus_violation",
            "room_overlap",
        }
        for label in failures
    )
    return {
        "requested_counts": requested,
        "generated_counts": dict(sorted(counts.items())),
        "count_delta": count_delta,
        "missing_counts": missing,
        "extra_counts": extra,
        "room_count_match": not missing and not extra,
        "p0_pass": p0_pass,
        "failure_labels": failures,
        "details": {
            "invalid_room_ids": invalid_ids,
            "out_of_bounds_room_ids": out_of_bounds_ids,
            "modulus_room_ids": modulus_ids,
            "overlap_pairs": overlap_pairs,
            "area_fail_room_ids": area_fail_ids,
            "width_fail_room_ids": width_fail_ids,
            "aspect_fail_room_ids": aspect_fail_ids,
            "exterior_fail_room_ids": exterior_fail_ids,
        },
    }


def _graph_record(graph, edge_types: dict) -> dict:
    return {
        "nodes": [
            {"id": node_id, **{key: value for key, value in attrs.items()}}
            for node_id, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "relation": edge_types.get((source, target), edge_types.get((target, source), "unknown")),
            }
            for source, target in graph.edges()
        ],
    }


def run_v4(
    case_dir: Path,
    case: dict,
    seed: int,
    model,
    device,
    weights: Path,
    sample_k: int,
    save_images: bool,
) -> tuple[list[dict], dict]:
    request = build_user_request(*case["site"], case["rooms"])
    result = generate_user_layout(
        request,
        model,
        device,
        seed=seed,
        sample_k=sample_k,
        display_style="regions",
    )
    rooms = normalize_rooms(result["display_rooms"])
    np.save(case_dir / "voxels.npy", result["pred"])
    if result.get("floor_layers"):
        np.savez(case_dir / "floor_layers.npz", **{str(k): v for k, v in result["floor_layers"].items()})
    write_json(case_dir / "program_graph.json", _graph_record(result["graph"], result["edge_types"]))
    write_json(
        case_dir / "candidate_scores.json",
        [
            {
                "decode_mode": candidate[0],
                "non_empty_voxels": candidate[2],
                "score": candidate[3],
                "metrics": candidate[4],
            }
            for candidate in result["candidates"]
        ],
    )
    if save_images:
        from visualize import save_layout_figures

        save_layout_figures(result, request, case_dir / "figures", weights_path=weights)
    metadata = {
        "weights": str(weights.resolve()),
        "device": str(device),
        "decode_mode": result["decode_mode"],
        "display_source": result["display_source"],
        "non_empty_voxels": result["n_occ"],
        "quality_score": result["quality_score"],
        "quality_metrics": result["quality_metrics"],
        "sample_k": sample_k,
    }
    return rooms, metadata


def _config_for_case(case: dict, fill_massing: bool) -> BlockCutConfig:
    site_x, site_y = case["site"]
    return BlockCutConfig(
        length_m=site_x / 1000.0,
        width_m=site_y / 1000.0,
        floors=2,
        floor_height_m=3.0,
        fill_massing=fill_massing,
    )


def run_block_cut(
    case_dir: Path,
    case: dict,
    seed: int,
    candidates: int,
    save_images: bool,
) -> tuple[list[dict], dict]:
    result = generate_block_cut_layout(
        room_counts=case["rooms"],
        config=_config_for_case(case, fill_massing=True),
        seed=seed,
        candidates=candidates,
    )
    write_layout_package(
        result,
        case["rooms"],
        case_dir,
        prefix="layout",
        save_images=save_images,
    )
    return normalize_rooms(result.rooms), {
        "generator_score": result.score,
        "score_detail": result.score_detail,
        "cut_record": result.cuts,
        "candidates": candidates,
        "non_empty_voxels": int((result.voxels > 0).sum()),
    }


def run_zoned(
    case_dir: Path,
    case: dict,
    seed: int,
    candidates: int,
    save_images: bool,
) -> tuple[list[dict], dict]:
    result = generate_zoned_layout(
        room_counts=case["rooms"],
        config=_config_for_case(case, fill_massing=False),
        seed=seed,
        candidates=candidates,
    )
    write_zoned_layout_package(
        result,
        case["rooms"],
        case_dir,
        prefix="layout",
        save_images=save_images,
    )
    return normalize_rooms(result.rooms), {
        "generator_score": result.score,
        "score_detail": result.score_detail,
        "placement_record": result.cuts,
        "candidates": candidates,
        "non_empty_voxels": int((result.voxels > 0).sum()),
    }


def run_one(
    route: str,
    case_name: str,
    case: dict,
    seed: int,
    out_dir: Path,
    model,
    device,
    args,
) -> dict:
    run_dir = out_dir / case_name / f"seed_{seed}" / route
    run_dir.mkdir(parents=True, exist_ok=True)
    input_record = {
        "route": route,
        "case": case_name,
        "site": {"x_mm": case["site"][0], "y_mm": case["site"][1], "z_mm": 6000},
        "room_counts": case["rooms"],
        "seed": seed,
    }
    write_json(run_dir / "input.json", input_record)
    started = time.perf_counter()
    try:
        if route == "v4":
            rooms, route_metadata = run_v4(
                run_dir,
                case,
                seed,
                model,
                device,
                args.weights,
                args.sample_k,
                not args.no_images,
            )
        elif route == "block_cut":
            rooms, route_metadata = run_block_cut(
                run_dir, case, seed, args.block_candidates, not args.no_images
            )
        else:
            rooms, route_metadata = run_zoned(
                run_dir, case, seed, args.zoned_candidates, not args.no_images
            )
        elapsed = time.perf_counter() - started
        evaluation = evaluate_rooms(rooms, case["rooms"], case["site"])
        write_json(run_dir / "rooms.json", rooms)
        report = {
            **input_record,
            "status": "completed",
            "elapsed_seconds": round(elapsed, 3),
            "route_metadata": route_metadata,
            "evaluation": evaluation,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        error_text = traceback.format_exc()
        (run_dir / "error.txt").write_text(error_text, encoding="utf-8")
        report = {
            **input_record,
            "status": "error",
            "elapsed_seconds": round(elapsed, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    write_json(run_dir / "report.json", report)
    return report


def parse_csv_values(value: str, allowed: set[str] | None = None) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if allowed is not None:
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise argparse.ArgumentTypeError(f"Unknown values: {', '.join(unknown)}")
    return values


def write_summary(out_dir: Path, reports: list[dict], manifest: dict) -> None:
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "summary.json", reports)
    fieldnames = [
        "case",
        "seed",
        "route",
        "status",
        "elapsed_seconds",
        "p0_pass",
        "room_count_match",
        "failure_labels",
        "error_type",
        "error",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            evaluation = report.get("evaluation", {})
            writer.writerow(
                {
                    "case": report["case"],
                    "seed": report["seed"],
                    "route": report["route"],
                    "status": report["status"],
                    "elapsed_seconds": report["elapsed_seconds"],
                    "p0_pass": evaluation.get("p0_pass"),
                    "room_count_match": evaluation.get("room_count_match"),
                    "failure_labels": "|".join(evaluation.get("failure_labels", [])),
                    "error_type": report.get("error_type", ""),
                    "error": report.get("error", ""),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="small,medium,large",
        help="Comma-separated cases: small,medium,large",
    )
    parser.add_argument("--seeds", default="11,42,123", help="Comma-separated integer seeds")
    parser.add_argument(
        "--routes",
        default="v4,block_cut,zoned",
        help="Comma-separated routes: v4,block_cut,zoned",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "weights" / "spatial_modal_cvae_v4_88x88x24.pth",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "route_baseline",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-k", type=int, default=4)
    parser.add_argument("--block-candidates", type=int, default=128)
    parser.add_argument("--zoned-candidates", type=int, default=64)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose existing report has status=completed.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        case_names = parse_csv_values(args.cases, set(TEST_CASES))
        routes = parse_csv_values(args.routes, set(ROUTES))
        seeds = [int(value) for value in parse_csv_values(args.seeds)]
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    if "v4" in routes and not args.weights.exists():
        parser.error(f"V4 weights do not exist: {args.weights}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = device = None
    if "v4" in routes:
        model, device = load_model(args.weights, args.device)

    manifest = {
        "cases": {name: TEST_CASES[name] for name in case_names},
        "seeds": seeds,
        "routes": routes,
        "weights": str(args.weights.resolve()) if "v4" in routes else None,
        "device": str(device) if device is not None else None,
        "parameters": {
            "sample_k": args.sample_k,
            "block_candidates": args.block_candidates,
            "zoned_candidates": args.zoned_candidates,
            "save_images": not args.no_images,
        },
    }
    reports = []
    for case_name in case_names:
        for seed in seeds:
            for route in routes:
                report_path = args.out_dir / case_name / f"seed_{seed}" / route / "report.json"
                if args.resume and report_path.exists():
                    with report_path.open(encoding="utf-8") as handle:
                        existing = json.load(handle)
                    if existing.get("status") == "completed":
                        (report_path.parent / "error.txt").unlink(missing_ok=True)
                        print(f"[skip] case={case_name} seed={seed} route={route}", flush=True)
                        reports.append(existing)
                        write_summary(args.out_dir, reports, manifest)
                        continue
                print(f"[run] case={case_name} seed={seed} route={route}", flush=True)
                report = run_one(
                    route,
                    case_name,
                    TEST_CASES[case_name],
                    seed,
                    args.out_dir,
                    model,
                    device,
                    args,
                )
                reports.append(report)
                print(
                    f"[{report['status']}] elapsed={report['elapsed_seconds']}s "
                    f"failures={report.get('evaluation', {}).get('failure_labels', [])}",
                    flush=True,
                )
                write_summary(args.out_dir, reports, manifest)
    return 0 if all(report["status"] == "completed" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
