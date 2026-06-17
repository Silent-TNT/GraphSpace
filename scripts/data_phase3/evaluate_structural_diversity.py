#!/usr/bin/env python3
"""Measure whether layouts differ structurally or only by room labels."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_DIR = ROOT / "data" / "phase2_v5" / "samples"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "phase3_diversity"
OUTSIDE_CLASS = 255
EMPTY_CLASS = 0

CLASS_NAMES = (
    "empty",
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
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
CIRCULATION_IDS = {
    CLASS_TO_ID["entryway"],
    CLASS_TO_ID["corridor"],
    CLASS_TO_ID["stairs"],
}
ZONE_BY_CLASS = {
    CLASS_TO_ID["entryway"]: 1,
    CLASS_TO_ID["living_room"]: 1,
    CLASS_TO_ID["dining_room"]: 1,
    CLASS_TO_ID["kitchen"]: 1,
    CLASS_TO_ID["bedroom"]: 2,
    CLASS_TO_ID["bathroom"]: 2,
    CLASS_TO_ID["corridor"]: 3,
    CLASS_TO_ID["stairs"]: 3,
    CLASS_TO_ID["utility"]: 4,
    CLASS_TO_ID["balcony"]: 4,
    CLASS_TO_ID["multi_purpose"]: 2,
}
STRUCTURE_WEIGHTS = {
    "footprint": 0.30,
    "partition": 0.25,
    "circulation": 0.15,
    "floor_overlap": 0.10,
    "cross_floor": 0.10,
    "functional_zone": 0.10,
}
GEOMETRY_WEIGHTS = {
    "footprint": 0.40,
    "partition": 0.35,
    "floor_overlap": 0.15,
    "cross_floor": 0.10,
}
ORGANIZATION_WEIGHTS = {
    "circulation": 0.60,
    "functional_zone": 0.40,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_sample(sample_dir: Path, house_id: str) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(sample_dir / "{}.npz".format(house_id)) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    with (sample_dir / "{}.json".format(house_id)).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return arrays, metadata


def iou_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return 1.0 - int(np.logical_and(a, b).sum()) / union


def mismatch_distance(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0.0
    return float(np.logical_and(a != b, valid).sum()) / valid_count


def internal_partition_edges(instance_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return internal room-boundary edges independent of instance ID values."""
    grid = np.asarray(instance_grid)
    left, right = grid[:, :-1, :], grid[:, 1:, :]
    bottom, top = grid[:, :, :-1], grid[:, :, 1:]
    x_edges = (left > 0) & (right > 0) & (left != right)
    y_edges = (bottom > 0) & (top > 0) & (bottom != top)
    return x_edges, y_edges


def partition_distance(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay = internal_partition_edges(a)
    bx, by = internal_partition_edges(b)
    return 0.5 * (iou_distance(ax, bx) + iou_distance(ay, by))


def circulation_mask(class_grid: np.ndarray) -> np.ndarray:
    return np.isin(class_grid, list(CIRCULATION_IDS))


def zone_grid(class_grid: np.ndarray) -> np.ndarray:
    result = np.zeros(class_grid.shape, dtype=np.uint8)
    for class_id, zone_id in ZONE_BY_CLASS.items():
        result[class_grid == class_id] = zone_id
    return result


def component_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    count = 0
    for start in np.argwhere(mask):
        start_tuple = tuple(int(value) for value in start)
        if visited[start_tuple]:
            continue
        count += 1
        queue = deque([start_tuple])
        visited[start_tuple] = True
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < mask.shape[0]
                    and 0 <= ny < mask.shape[1]
                    and mask[nx, ny]
                    and not visited[nx, ny]
                ):
                    visited[nx, ny] = True
                    queue.append((nx, ny))
    return count


def layout_signature(arrays: dict[str, np.ndarray]) -> dict:
    building = arrays["building_mask"].astype(bool)
    class_grid = arrays["class_grid"]
    instance_grid = arrays["instance_grid"]
    circulation = circulation_mask(class_grid)
    x_edges, y_edges = internal_partition_edges(instance_grid)
    return {
        "occupied_cells": [int(building[floor].sum()) for floor in range(2)],
        "building_components": [
            component_count(building[floor]) for floor in range(2)
        ],
        "internal_partition_edges": [
            int(x_edges[floor].sum() + y_edges[floor].sum()) for floor in range(2)
        ],
        "circulation_cells": [
            int(circulation[floor].sum()) for floor in range(2)
        ],
        "floor_overlap_cells": int(arrays["floor_overlap_mask"].sum()),
        "cross_floor_cells": int(arrays["cross_floor_mask"][0].sum()),
        "double_height_void_cells": int(
            arrays["double_height_void_mask"][0].sum()
        ),
    }


def compare_layouts(
    arrays_a: dict[str, np.ndarray],
    arrays_b: dict[str, np.ndarray],
) -> dict:
    if not np.array_equal(arrays_a["site_mask"], arrays_b["site_mask"]):
        raise ValueError("Structural diversity requires the same site boundary")

    footprint = float(np.mean([
        iou_distance(arrays_a["building_mask"][floor], arrays_b["building_mask"][floor])
        for floor in range(2)
    ]))
    partition = partition_distance(
        arrays_a["instance_grid"], arrays_b["instance_grid"]
    )
    circulation = iou_distance(
        circulation_mask(arrays_a["class_grid"]),
        circulation_mask(arrays_b["class_grid"]),
    )
    floor_overlap = iou_distance(
        arrays_a["floor_overlap_mask"], arrays_b["floor_overlap_mask"]
    )
    cross_floor = iou_distance(
        arrays_a["cross_floor_mask"], arrays_b["cross_floor_mask"]
    )
    occupied_union = (
        (arrays_a["building_mask"] == 1) | (arrays_b["building_mask"] == 1)
    )
    functional_zone = mismatch_distance(
        zone_grid(arrays_a["class_grid"]),
        zone_grid(arrays_b["class_grid"]),
        occupied_union,
    )
    semantic_class = mismatch_distance(
        arrays_a["class_grid"],
        arrays_b["class_grid"],
        occupied_union,
    )
    components = {
        "footprint": footprint,
        "partition": partition,
        "circulation": circulation,
        "floor_overlap": floor_overlap,
        "cross_floor": cross_floor,
        "functional_zone": functional_zone,
    }
    structural = sum(
        STRUCTURE_WEIGHTS[name] * value for name, value in components.items()
    )
    geometry = sum(
        GEOMETRY_WEIGHTS[name] * components[name] for name in GEOMETRY_WEIGHTS
    )
    organization = sum(
        ORGANIZATION_WEIGHTS[name] * components[name]
        for name in ORGANIZATION_WEIGHTS
    )
    if geometry < 0.03 and semantic_class >= 0.02:
        category = "label_only"
    elif geometry < 0.08 and structural < 0.15:
        category = "near_duplicate"
    elif geometry < 0.20 or structural < 0.30:
        category = "moderate_structure"
    else:
        category = "substantial_structure"
    return {
        "structural_distance": structural,
        "geometry_distance": geometry,
        "organization_distance": organization,
        "semantic_class_distance": semantic_class,
        "category": category,
        "components": components,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def calibrate_same_site_pairs(sample_dir: Path) -> dict:
    by_site: dict[tuple[float, float], list[str]] = defaultdict(list)
    metadata_by_id = {}
    for metadata_path in sorted(sample_dir.glob("house_*.json")):
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        house_id = metadata_path.stem
        metadata_by_id[house_id] = metadata
        by_site[tuple(metadata["site_size_mm"])].append(house_id)

    pairs = []
    distances = []
    geometry_distances = []
    organization_distances = []
    semantic_distances = []
    category_counts = Counter()
    cache = {}
    for site, house_ids in sorted(by_site.items()):
        if len(house_ids) < 2:
            continue
        for house_a, house_b in itertools.combinations(sorted(house_ids), 2):
            if house_a not in cache:
                cache[house_a] = load_sample(sample_dir, house_a)[0]
            if house_b not in cache:
                cache[house_b] = load_sample(sample_dir, house_b)[0]
            result = compare_layouts(cache[house_a], cache[house_b])
            distances.append(result["structural_distance"])
            geometry_distances.append(result["geometry_distance"])
            organization_distances.append(result["organization_distance"])
            semantic_distances.append(result["semantic_class_distance"])
            category_counts[result["category"]] += 1
            pairs.append({
                "site_size_mm": list(site),
                "house_a": house_a,
                "house_b": house_b,
                **result,
            })
    pairs.sort(key=lambda item: item["structural_distance"])
    return {
        "same_site_group_count": sum(len(ids) > 1 for ids in by_site.values()),
        "pair_count": len(pairs),
        "category_counts": dict(category_counts),
        "structural_distance_percentiles": {
            "min": min(distances) if distances else None,
            "p10": percentile(distances, 10),
            "p25": percentile(distances, 25),
            "median": percentile(distances, 50),
            "p75": percentile(distances, 75),
            "p90": percentile(distances, 90),
            "max": max(distances) if distances else None,
        },
        "geometry_distance_percentiles": {
            "min": min(geometry_distances) if geometry_distances else None,
            "p10": percentile(geometry_distances, 10),
            "p25": percentile(geometry_distances, 25),
            "median": percentile(geometry_distances, 50),
            "p75": percentile(geometry_distances, 75),
            "p90": percentile(geometry_distances, 90),
            "max": max(geometry_distances) if geometry_distances else None,
        },
        "organization_distance_percentiles": {
            "min": min(organization_distances) if organization_distances else None,
            "median": percentile(organization_distances, 50),
            "max": max(organization_distances) if organization_distances else None,
        },
        "semantic_distance_percentiles": {
            "min": min(semantic_distances) if semantic_distances else None,
            "median": percentile(semantic_distances, 50),
            "max": max(semantic_distances) if semantic_distances else None,
        },
        "lowest_pairs": pairs[:10],
        "highest_pairs": pairs[-10:],
        "pairs": pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--house-a")
    parser.add_argument("--house-b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = args.sample_dir.resolve()
    output_dir = args.output_dir.resolve()
    if bool(args.house_a) != bool(args.house_b):
        raise SystemExit("--house-a and --house-b must be provided together")
    if args.house_a:
        arrays_a, _ = load_sample(sample_dir, args.house_a)
        arrays_b, _ = load_sample(sample_dir, args.house_b)
        result = compare_layouts(arrays_a, arrays_b)
        result["house_a"] = args.house_a
        result["house_b"] = args.house_b
        result["signature_a"] = layout_signature(arrays_a)
        result["signature_b"] = layout_signature(arrays_b)
        write_json(output_dir / "pair_report.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    report = calibrate_same_site_pairs(sample_dir)
    report["metric_version"] = "graphspace_structural_diversity_v1"
    report["weights"] = STRUCTURE_WEIGHTS
    report["geometry_weights"] = GEOMETRY_WEIGHTS
    report["organization_weights"] = ORGANIZATION_WEIGHTS
    report["category_thresholds"] = {
        "label_only": "geometry < 0.03 and semantic >= 0.02",
        "near_duplicate": "geometry < 0.08 and structural < 0.15",
        "moderate_structure": "geometry < 0.20 or structural < 0.30",
        "substantial_structure": "geometry >= 0.20 and structural >= 0.30",
    }
    write_json(output_dir / "calibration.json", report)
    print("Same-site pairs: {}".format(report["pair_count"]))
    print("Categories: {}".format(report["category_counts"]))
    print("Wrote: {}".format(output_dir / "calibration.json"))


if __name__ == "__main__":
    main()
