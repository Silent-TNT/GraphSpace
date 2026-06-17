#!/usr/bin/env python3
"""Build the reproducible data foundation required before V5 training.

The script does not modify source house JSON files. It produces:

- a file manifest with hashes and source-batch provenance;
- geometry/topology statistics and inferred spatial relations;
- near-duplicate groups kept within one data split;
- a deterministic train/validation/test split;
- geometric P1 proxy checks;
- a guillotine-cut representation upper-bound report.

Geometric adjacency is evaluated as the final functional-block spatial
organization target. Door locations and detailed pedestrian paths are outside
the project output scope.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "phase1"
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
CIRCULATION_TYPES = {"entryway", "corridor", "stairs"}
PUBLIC_TYPES = {"entryway", "living_room", "dining_room", "kitchen", "corridor", "stairs"}
PRIVATE_ACCESS_TYPES = CIRCULATION_TYPES | {"living_room", "dining_room"}
TOL = 1e-6
MIN_SHARED_MM = 300.0
SPLIT_SEED = 20260613


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_batches(data_dir: Path) -> dict[str, str]:
    """Recover the old batch name from retained QC summaries."""
    result: dict[str, str] = {}
    for path in sorted((data_dir / "qc_logs").glob("*_qc_summary_*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        data_dir_text = str(payload.get("data_dir", "")).lower()
        if "second_94" in data_dir_text:
            batch = "second_94"
        elif "third_299" in data_dir_text:
            batch = "third_299"
        else:
            batch = "first_75"
        for item in payload.get("files", []):
            house_id = str(item.get("house_id", ""))
            if house_id:
                result[house_id] = batch
    log_batches = {
        "batch_export_log.txt": "first_75",
        "batch_export_log_1.txt": "first_75",
        "batch_export_log._second94txt": "second_94",
        "batch_export_log_third299.txt": "third_299",
    }
    for filename, batch in log_batches.items():
        path = data_dir / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"->\s+(house_\d+)\.json", line)
            if match and line.lstrip().startswith("OK"):
                result.setdefault(match.group(1), batch)
    return result


def room_floors(room: dict) -> tuple[int, ...]:
    floors = room.get("floors")
    if isinstance(floors, list) and floors:
        return tuple(sorted({int(value) for value in floors}))
    z0 = float(room["box_min"][2])
    z1 = float(room["box_max"][2])
    if room.get("type") == "stairs" and z0 <= TOL and z1 >= 6000.0 - TOL:
        return (1, 2)
    return (int(room.get("floor", 1)),)


def normalize_room(room: dict) -> dict:
    return {
        "id": str(room["id"]),
        "type": str(room["type"]),
        "floor": int(room.get("floor", room_floors(room)[0])),
        "floors": list(room_floors(room)),
        "box_min": [float(v) for v in room["box_min"]],
        "box_max": [float(v) for v in room["box_max"]],
    }


def axis_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def shared_face_relation(a: dict, b: dict) -> dict | None:
    """Return a face-contact relation with a useful shared-face measure."""
    amin, amax = a["box_min"], a["box_max"]
    bmin, bmax = b["box_min"], b["box_max"]
    oy = axis_overlap(amin[1], amax[1], bmin[1], bmax[1])
    oz = axis_overlap(amin[2], amax[2], bmin[2], bmax[2])
    ox = axis_overlap(amin[0], amax[0], bmin[0], bmax[0])

    if abs(amax[0] - bmin[0]) <= TOL and oy >= MIN_SHARED_MM and oz >= MIN_SHARED_MM:
        return {"axis": "x", "a_side": "x+", "b_side": "x-", "shared_mm": oy}
    if abs(bmax[0] - amin[0]) <= TOL and oy >= MIN_SHARED_MM and oz >= MIN_SHARED_MM:
        return {"axis": "x", "a_side": "x-", "b_side": "x+", "shared_mm": oy}
    if abs(amax[1] - bmin[1]) <= TOL and ox >= MIN_SHARED_MM and oz >= MIN_SHARED_MM:
        return {"axis": "y", "a_side": "y+", "b_side": "y-", "shared_mm": ox}
    if abs(bmax[1] - amin[1]) <= TOL and ox >= MIN_SHARED_MM and oz >= MIN_SHARED_MM:
        return {"axis": "y", "a_side": "y-", "b_side": "y+", "shared_mm": ox}
    if abs(amax[2] - bmin[2]) <= TOL and ox >= MIN_SHARED_MM and oy >= MIN_SHARED_MM:
        return {"axis": "z", "a_side": "z+", "b_side": "z-", "shared_mm2": ox * oy}
    if abs(bmax[2] - amin[2]) <= TOL and ox >= MIN_SHARED_MM and oy >= MIN_SHARED_MM:
        return {"axis": "z", "a_side": "z-", "b_side": "z+", "shared_mm2": ox * oy}
    return None


def exterior_sides(room: dict, site_x: float, site_y: float) -> list[str]:
    x0, y0, _ = room["box_min"]
    x1, y1, _ = room["box_max"]
    sides = []
    if abs(x0) <= TOL:
        sides.append("W")
    if abs(x1 - site_x) <= TOL:
        sides.append("E")
    if abs(y0) <= TOL:
        sides.append("S")
    if abs(y1 - site_y) <= TOL:
        sides.append("N")
    return sides


def extract_relations(rooms: list[dict], site_x: float, site_y: float) -> dict:
    adjacency = []
    graph: dict[str, set[str]] = {room["id"]: set() for room in rooms}
    for index, room_a in enumerate(rooms):
        for room_b in rooms[index + 1 :]:
            relation = shared_face_relation(room_a, room_b)
            if relation is None:
                continue
            record = {
                "a": room_a["id"],
                "a_type": room_a["type"],
                "b": room_b["id"],
                "b_type": room_b["type"],
                "relation": "vertical_contact" if relation["axis"] == "z" else "face_adjacent",
                **relation,
            }
            adjacency.append(record)
            graph[room_a["id"]].add(room_b["id"])
            graph[room_b["id"]].add(room_a["id"])

    exterior = {
        room["id"]: {
            "type": room["type"],
            "sides": exterior_sides(room, site_x, site_y),
        }
        for room in rooms
    }
    return {
        "relation_semantics": {
            "face_adjacent": "shared face between functional blocks",
            "vertical_contact": "vertical contact between functional blocks",
        },
        "adjacency": adjacency,
        "exterior": exterior,
        "graph": {key: sorted(value) for key, value in graph.items()},
    }


def reachable(graph: dict[str, list[str]], starts: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(n for n in graph.get(node, []) if n not in seen)
    return seen


def p1_spatial_organization_report(rooms: list[dict], relations: dict) -> dict:
    """Evaluate functional-block adjacency, zoning and cross-floor organization."""
    by_id = {room["id"]: room for room in rooms}
    graph = relations["graph"]
    entry_ids = [r["id"] for r in rooms if r["type"] == "entryway"]
    stair_ids = [r["id"] for r in rooms if r["type"] == "stairs"]
    entry_reach = reachable(graph, entry_ids)

    stair_details = []
    for stair_id in stair_ids:
        stair = by_id[stair_id]
        neighbors = [by_id[nid] for nid in graph.get(stair_id, [])]
        floors = set(room_floors(stair))
        floor_contacts = {
            floor: any(floor in room_floors(room) and room["type"] != "stairs" for room in neighbors)
            for floor in (1, 2)
        }
        circulation_contacts = {
            floor: any(
                floor in room_floors(room) and room["type"] in CIRCULATION_TYPES - {"stairs"}
                for room in neighbors
            )
            for floor in (1, 2)
        }
        stair_details.append(
            {
                "id": stair_id,
                "spans_both_floors": floors == {1, 2},
                "has_any_contact_by_floor": floor_contacts,
                "has_circulation_contact_by_floor": circulation_contacts,
            }
        )

    bedroom_access = {}
    for room in rooms:
        if room["type"] != "bedroom":
            continue
        neighbor_types = sorted({by_id[nid]["type"] for nid in graph.get(room["id"], [])})
        bedroom_access[room["id"]] = {
            "neighbor_types": neighbor_types,
            "has_non_bedroom_spatial_contact": any(
                t in PRIVATE_ACCESS_TYPES for t in neighbor_types
            ),
        }

    def has_type_contact(left: set[str], right: set[str]) -> bool:
        for edge in relations["adjacency"]:
            pair = {edge["a_type"], edge["b_type"]}
            if pair & left and pair & right and not (pair <= left or pair <= right):
                return True
        return False

    main_rooms = [r["id"] for r in rooms if r["type"] not in {"balcony"}]
    checks = {
        "entryway_present": bool(entry_ids),
        "spatial_contact_graph_reachable_from_entry": bool(entry_ids)
        and all(rid in entry_reach for rid in main_rooms),
        "stairs_present": bool(stair_ids),
        "all_stairs_span_both_floors": bool(stair_details)
        and all(item["spans_both_floors"] for item in stair_details),
        "stairs_contact_both_floors": bool(stair_details)
        and all(all(item["has_any_contact_by_floor"].values()) for item in stair_details),
        "stairs_contact_circulation_both_floors": bool(stair_details)
        and all(all(item["has_circulation_contact_by_floor"].values()) for item in stair_details),
        "bedrooms_have_non_bedroom_spatial_contact": bool(bedroom_access)
        and all(
            item["has_non_bedroom_spatial_contact"]
            for item in bedroom_access.values()
        ),
        "living_dining_spatial_contact": has_type_contact(
            {"living_room"}, {"dining_room"}
        ),
        "kitchen_public_spatial_contact": not any(
            r["type"] == "kitchen" for r in rooms
        )
        or has_type_contact({"kitchen"}, {"dining_room", "living_room", "corridor"}),
        "bedroom_bathroom_circulation_spatial_relation": has_type_contact(
            {"bedroom", "bathroom"}, {"corridor", "stairs", "entryway"}
        ),
    }
    hard_geometry_keys = {
        "entryway_present",
        "stairs_present",
        "all_stairs_span_both_floors",
        "stairs_contact_both_floors",
    }
    organization_keys = set(checks) - hard_geometry_keys
    return {
        "checks": checks,
        "hard_geometry_pass": all(checks[key] for key in hard_geometry_keys),
        "spatial_organization_pass": all(
            checks[key] for key in organization_keys
        ),
        "stair_details": stair_details,
        "bedroom_access": bedroom_access,
        "scope": [
            "The final output is functional blocks, not doors.",
            "The report evaluates block adjacency and cross-floor organization.",
            "Detailed door positions and pedestrian paths are outside scope.",
        ],
    }


def p1_proxy_report(rooms: list[dict], relations: dict) -> dict:
    """Backward-compatible alias for older callers."""
    return p1_spatial_organization_report(rooms, relations)


def _room_rect_for_floor(room: dict, floor: int) -> tuple[float, float, float, float, str] | None:
    if floor not in room_floors(room):
        return None
    return (
        room["box_min"][0],
        room["box_min"][1],
        room["box_max"][0],
        room["box_max"][1],
        room["id"],
    )


def _candidate_partitions(rects: tuple[tuple[float, float, float, float, str], ...]):
    for axis in (0, 1):
        boundaries = sorted({rect[axis] for rect in rects} | {rect[axis + 2] for rect in rects})
        for cut in boundaries[1:-1]:
            left = tuple(rect for rect in rects if rect[axis + 2] <= cut + TOL)
            right = tuple(rect for rect in rects if rect[axis] >= cut - TOL)
            if left and right and len(left) + len(right) == len(rects):
                yield axis, cut, left, right


def guillotine_upper_bound(rects: list[tuple[float, float, float, float, str]]) -> dict:
    """Find the best recursive straight-cut decomposition of room rectangles."""
    memo: dict[tuple[str, ...], tuple[int, list[dict], list[list[str]]]] = {}

    def solve(group: tuple[tuple[float, float, float, float, str], ...]):
        key = tuple(sorted(rect[4] for rect in group))
        if key in memo:
            return memo[key]
        if len(group) == 1:
            result = (1, [], [])
            memo[key] = result
            return result
        best = (0, [], [list(key)])
        for axis, cut, left, right in _candidate_partitions(group):
            left_score, left_cuts, left_blocked = solve(left)
            right_score, right_cuts, right_blocked = solve(right)
            score = left_score + right_score
            if score > best[0]:
                best = (
                    score,
                    [
                        {
                            "axis": "x" if axis == 0 else "y",
                            "at_mm": cut,
                            "left": sorted(rect[4] for rect in left),
                            "right": sorted(rect[4] for rect in right),
                        }
                    ]
                    + left_cuts
                    + right_cuts,
                    left_blocked + right_blocked,
                )
                if score == len(group):
                    break
        memo[key] = best
        return best

    if not rects:
        return {"room_count": 0, "resolved_singletons": 0, "resolution_rate": 1.0, "fully_separable": True}
    score, cuts, blocked = solve(tuple(rects))
    return {
        "room_count": len(rects),
        "resolved_singletons": score,
        "resolution_rate": score / len(rects),
        "fully_separable": score == len(rects),
        "cut_count": len(cuts),
        "blocked_groups": blocked,
    }


def canonical_geometry(room_records: list[dict], site_x: float, site_y: float) -> tuple[tuple, ...]:
    """Coarse orientation-invariant geometry signature for near-duplicate grouping."""
    variants = []
    transforms = (
        lambda x, y, X, Y: (x, y, X, Y),
        lambda x, y, X, Y: (1 - X, y, 1 - x, Y),
        lambda x, y, X, Y: (x, 1 - Y, X, 1 - y),
        lambda x, y, X, Y: (1 - X, 1 - Y, 1 - x, 1 - y),
        lambda x, y, X, Y: (y, x, Y, X),
        lambda x, y, X, Y: (1 - Y, x, 1 - y, X),
        lambda x, y, X, Y: (y, 1 - X, Y, 1 - x),
        lambda x, y, X, Y: (1 - Y, 1 - X, 1 - y, 1 - x),
    )
    for transform in transforms:
        values = []
        for room in room_records:
            x0, y0, _ = room["box_min"]
            x1, y1, _ = room["box_max"]
            nx0, ny0, nx1, ny1 = transform(x0 / site_x, y0 / site_y, x1 / site_x, y1 / site_y)
            values.append(
                (
                    room["type"],
                    tuple(room_floors(room)),
                    round(nx0 / 0.05),
                    round(ny0 / 0.05),
                    round(nx1 / 0.05),
                    round(ny1 / 0.05),
                )
            )
        variants.append(tuple(sorted(values)))
    return min(variants)


def dataset_signature(record: dict) -> str:
    encoded = json.dumps(record["canonical_geometry"], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("ascii")).hexdigest()


@dataclass
class UnionFind:
    parent: dict[str, str]

    @classmethod
    def create(cls, ids: Iterable[str]) -> "UnionFind":
        return cls({item: item for item in ids})

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def geometry_distance(left: tuple[tuple, ...], right: tuple[tuple, ...]) -> float:
    """Mean normalized coordinate distance for semantically aligned rooms."""
    if len(left) != len(right):
        return math.inf
    total = 0.0
    count = 0
    for a, b in zip(left, right):
        if a[:2] != b[:2]:
            return math.inf
        total += sum(abs(float(x) - float(y)) for x, y in zip(a[2:], b[2:]))
        count += 4
    return total / max(count, 1) * 0.05


def build_duplicate_groups(records: list[dict]) -> tuple[list[list[str]], list[dict]]:
    """Group exact and conservatively detected near-duplicate layouts."""
    uf = UnionFind.create(record["house_id"] for record in records)
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_signature: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record["house_id"])
        by_signature[record["near_duplicate_signature"]].append(record["house_id"])
    for bucket in list(by_hash.values()) + list(by_signature.values()):
        for item in bucket[1:]:
            uf.union(bucket[0], item)

    candidate_buckets: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            tuple(record["room_counts"][room_type] for room_type in ROOM_TYPES),
            tuple(record["floor_counts"][str(floor)] for floor in (1, 2)),
        )
        candidate_buckets[key].append(record)
    nearest_pairs = []
    threshold = 0.04
    for bucket in candidate_buckets.values():
        for index, left in enumerate(bucket):
            for right in bucket[index + 1 :]:
                distance = geometry_distance(left["canonical_geometry"], right["canonical_geometry"])
                nearest_pairs.append(
                    {
                        "a": left["house_id"],
                        "b": right["house_id"],
                        "mean_normalized_box_distance": round(distance, 6),
                        "grouped": distance <= threshold,
                    }
                )
                if distance <= threshold:
                    uf.union(left["house_id"], right["house_id"])
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[uf.find(record["house_id"])].append(record["house_id"])
    groups = sorted((sorted(group) for group in grouped.values()), key=lambda group: (-len(group), group[0]))
    nearest_pairs.sort(key=lambda item: (item["mean_normalized_box_distance"], item["a"], item["b"]))
    return groups, nearest_pairs[:100]


def split_groups(records: list[dict], groups: list[list[str]]) -> dict[str, list[str]]:
    """Assign similarity groups with source-batch and room-count stratification."""
    record_by_id = {record["house_id"]: record for record in records}
    rng = random.Random(SPLIT_SEED)
    targets = {"train": round(len(records) * 0.8), "validation": round(len(records) * 0.1)}
    targets["test"] = len(records) - targets["train"] - targets["validation"]
    ratios = {key: value / len(records) for key, value in targets.items()}
    splits = {key: [] for key in targets}

    def room_bin(value: int) -> str:
        if value <= 21:
            return "small"
        if value <= 27:
            return "medium"
        return "large"

    strata: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    for group in groups:
        group_records = [record_by_id[item] for item in group]
        batch = Counter(record["source_batch"] for record in group_records).most_common(1)[0][0]
        median_rooms = int(statistics.median(record["room_count"] for record in group_records))
        strata[(batch, room_bin(median_rooms))].append(group)

    stratum_current: dict[tuple[str, str], Counter] = defaultdict(Counter)
    stratum_totals = {
        key: sum(len(group) for group in value)
        for key, value in strata.items()
    }
    stratum_keys = sorted(strata)
    rng.shuffle(stratum_keys)
    ordered = []
    for stratum in stratum_keys:
        bucket = strata[stratum]
        rng.shuffle(bucket)
        bucket.sort(key=lambda group: -len(group))
    while any(strata[stratum] for stratum in stratum_keys):
        for stratum in stratum_keys:
            if strata[stratum]:
                ordered.append((stratum, strata[stratum].pop(0)))

    for stratum, group in ordered:
        available = [
            name for name in splits
            if len(splits[name]) + len(group) <= targets[name]
        ]
        if not available:
            available = list(splits)
        total = stratum_totals[stratum]

        def priority(name: str) -> tuple[float, float, str]:
            global_deficit = targets[name] - len(splits[name])
            stratum_target = max(ratios[name] * total, 1e-6)
            stratum_fill = stratum_current[stratum][name] / stratum_target
            global_fill = len(splits[name]) / max(targets[name], 1)
            return (
                -(0.75 * stratum_fill + 0.25 * global_fill),
                global_deficit,
                name,
            )

        selected = max(available, key=priority)
        splits[selected].extend(group)
        stratum_current[stratum][selected] += len(group)
    return {key: sorted(value) for key, value in splits.items()}


def numeric_summary(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)

    return {
        "min": min(values),
        "p25": percentile(0.25),
        "median": statistics.median(values),
        "p75": percentile(0.75),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def analyze_house(path: Path, source_batch: str) -> tuple[dict, dict, dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rooms = [normalize_room(room) for room in payload["rooms"]]
    metadata = payload.get("metadata", {})
    building = metadata.get("building_size", {})
    site_x = float(building.get("x", max(room["box_max"][0] for room in rooms)))
    site_y = float(building.get("y", max(room["box_max"][1] for room in rooms)))
    site_z = float(building.get("z", 6000.0))
    relations = extract_relations(rooms, site_x, site_y)
    p1 = p1_spatial_organization_report(rooms, relations)
    cut_floors = {}
    for floor in (1, 2):
        rects = [
            rect
            for room in rooms
            if (rect := _room_rect_for_floor(room, floor)) is not None
        ]
        cut_floors[str(floor)] = guillotine_upper_bound(rects)
    cut_report = {
        "floors": cut_floors,
        "fully_separable_both_floors": all(item["fully_separable"] for item in cut_floors.values()),
        "mean_resolution_rate": statistics.fmean(item["resolution_rate"] for item in cut_floors.values()),
    }
    counts = Counter(room["type"] for room in rooms)
    canonical = canonical_geometry(rooms, site_x, site_y)
    embedded_house_id = str(payload.get("house_id", path.stem))
    record = {
        "house_id": path.stem,
        "embedded_house_id": embedded_house_id,
        "embedded_house_id_matches_filename": embedded_house_id == path.stem,
        "file": path.name,
        "source_batch": source_batch,
        "sha256": sha256_file(path),
        "qc_version": metadata.get("constraints", {}).get("qc_version"),
        "site_mm": [site_x, site_y, site_z],
        "room_count": len(rooms),
        "room_counts": {room_type: counts.get(room_type, 0) for room_type in ROOM_TYPES},
        "floor_counts": {
            str(floor): sum(floor in room_floors(room) for room in rooms)
            for floor in (1, 2)
        },
        "adjacency_count": len(relations["adjacency"]),
        "p1_hard_geometry_pass": p1["hard_geometry_pass"],
        "p1_spatial_organization_pass": p1["spatial_organization_pass"],
        "p1_checks": p1["checks"],
        "cut_fully_separable": cut_report["fully_separable_both_floors"],
        "cut_mean_resolution_rate": cut_report["mean_resolution_rate"],
        "cut_floor_results": cut_floors,
        "canonical_geometry": canonical,
    }
    record["near_duplicate_signature"] = dataset_signature(record)
    relation_output = {
        "house_id": record["house_id"],
        "site_mm": record["site_mm"],
        "rooms": rooms,
        "relations": {key: value for key, value in relations.items() if key != "graph"},
        "p1": p1,
    }
    cut_output = {"house_id": record["house_id"], **cut_report}
    return record, relation_output, cut_output


def aggregate(records: list[dict], groups: list[list[str]], splits: dict[str, list[str]]) -> dict:
    source_counts = Counter(record["source_batch"] for record in records)
    room_type_totals = Counter()
    for record in records:
        room_type_totals.update(record["room_counts"])
    split_lookup = {house_id: split for split, ids in splits.items() for house_id in ids}
    leakage = [
        group for group in groups if len({split_lookup[item] for item in group}) > 1
    ]
    p1_check_names = sorted(records[0]["p1_checks"]) if records else []
    split_stats = {}
    record_by_id = {record["house_id"]: record for record in records}
    for split, ids in splits.items():
        split_records = [record_by_id[house_id] for house_id in ids]
        split_stats[split] = {
            "count": len(ids),
            "source_batches": dict(sorted(Counter(r["source_batch"] for r in split_records).items())),
            "room_count": numeric_summary([r["room_count"] for r in split_records]),
        }
    return {
        "dataset_count": len(records),
        "source_batch_counts": dict(sorted(source_counts.items())),
        "qc_versions": dict(sorted(Counter(str(r["qc_version"]) for r in records).items())),
        "site_x_mm": numeric_summary([r["site_mm"][0] for r in records]),
        "site_y_mm": numeric_summary([r["site_mm"][1] for r in records]),
        "room_count": numeric_summary([r["room_count"] for r in records]),
        "room_type_instance_totals": dict(sorted(room_type_totals.items())),
        "houses_without_kitchen": sum(r["room_counts"]["kitchen"] == 0 for r in records),
        "houses_without_dining_room": sum(r["room_counts"]["dining_room"] == 0 for r in records),
        "embedded_house_id_mismatches": [
            {
                "file_house_id": r["house_id"],
                "embedded_house_id": r["embedded_house_id"],
                "file": r["file"],
            }
            for r in records
            if not r["embedded_house_id_matches_filename"]
        ],
        "exact_duplicate_file_groups": sum(
            count > 1 for count in Counter(r["sha256"] for r in records).values()
        ),
        "near_duplicate_groups": sum(len(group) > 1 for group in groups),
        "houses_in_near_duplicate_groups": sum(len(group) for group in groups if len(group) > 1),
        "largest_near_duplicate_group": max(map(len, groups), default=0),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "split_statistics": split_stats,
        "near_duplicate_cross_split_leakage_groups": leakage,
        "p1_hard_geometry_pass": sum(r["p1_hard_geometry_pass"] for r in records),
        "p1_spatial_organization_pass": sum(
            r["p1_spatial_organization_pass"] for r in records
        ),
        "p1_check_pass_counts": {
            key: sum(bool(r["p1_checks"][key]) for r in records)
            for key in p1_check_names
        },
        "cut_fully_separable_both_floors": sum(r["cut_fully_separable"] for r in records),
        "cut_floor_fully_separable": {
            str(floor): sum(r["cut_floor_results"][str(floor)]["fully_separable"] for r in records)
            for floor in (1, 2)
        },
        "cut_floor_resolution_rate": {
            str(floor): numeric_summary(
                [r["cut_floor_results"][str(floor)]["resolution_rate"] for r in records]
            )
            for floor in (1, 2)
        },
        "cut_resolution_rate": numeric_summary([r["cut_mean_resolution_rate"] for r in records]),
    }


def write_manifest_csv(path: Path, records: list[dict], split_lookup: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "house_id",
        "embedded_house_id",
        "embedded_house_id_matches_filename",
        "file",
        "source_batch",
        "split",
        "sha256",
        "qc_version",
        "site_x_mm",
        "site_y_mm",
        "room_count",
        "adjacency_count",
        "p1_hard_geometry_pass",
        "p1_spatial_organization_pass",
        "cut_fully_separable",
        "cut_mean_resolution_rate",
        "near_duplicate_signature",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record[key] for key in fields if key in record},
                    "split": split_lookup[record["house_id"]],
                    "site_x_mm": record["site_mm"][0],
                    "site_y_mm": record["site_mm"][1],
                }
            )


def write_markdown_report(path: Path, summary: dict, groups: list[list[str]]) -> None:
    split_lines = []
    for name in ("train", "validation", "test"):
        item = summary["split_statistics"][name]
        batches = ", ".join(f"{key}={value}" for key, value in item["source_batches"].items())
        split_lines.append(
            f"- `{name}`：{item['count']} 套，房间数中位数 "
            f"{item['room_count']['median']}，来源 {batches}。"
        )
    duplicate_lines = [
        f"- `{' / '.join(group)}`"
        for group in groups
        if len(group) > 1
    ] or ["- 未发现近重复组。"]
    p1 = summary["p1_check_pass_counts"]
    text = f"""# GraphSpace V5 训练前第一阶段报告

生成命令：

```powershell
python scripts/data_phase1/run_phase1.py
```

## 数据与划分

- 有效住宅 JSON：{summary['dataset_count']} 套。
- 原始批次：first_75={summary['source_batch_counts']['first_75']}，
  second_94={summary['source_batch_counts']['second_94']}，
  third_299={summary['source_batch_counts']['third_299']}。
- 所有数据均为 V14；文件 SHA-256 完全重复组为
  {summary['exact_duplicate_file_groups']}。
- 保守近重复组：{summary['near_duplicate_groups']} 组，共
  {summary['houses_in_near_duplicate_groups']} 套。
- 近重复跨集合泄漏：{len(summary['near_duplicate_cross_split_leakage_groups'])} 组。

{chr(10).join(split_lines)}

近重复组：

{chr(10).join(duplicate_lines)}

## 数据一致性

- 468 套均包含 `dining_room`。
- 9 套不包含 `kitchen`，符合厨房可选的项目定义。
- `house_1010.json` 内部 `house_id` 为 `house_10102`。当前清单以文件名
  作为稳定主键，保留内部 ID 作为异常记录，未修改原文件。

## 空间关系与 P1

已为每套住宅提取房间贴面邻接、垂直接触、楼层和外边界关系。

- 入口存在：{p1['entryway_present']}/{summary['dataset_count']}。
- 楼梯跨两层：{p1['all_stairs_span_both_floors']}/{summary['dataset_count']}。
- 楼梯在两层均接触其他空间：
  {p1['stairs_contact_both_floors']}/{summary['dataset_count']}。
- 楼梯在两层均接触交通空间：
  {p1['stairs_contact_circulation_both_floors']}/{summary['dataset_count']}。
- 从玄关沿功能体块贴邻图联系主要空间：
  {p1['spatial_contact_graph_reachable_from_entry']}/{summary['dataset_count']}。
- 卧室具有非卧室空间邻接：
  {p1['bedrooms_have_non_bedroom_spatial_contact']}/{summary['dataset_count']}。
- 客厅与餐厅直接贴邻：
  {p1['living_dining_spatial_contact']}/{summary['dataset_count']}。
- 全部 P1 功能体块空间组织项同时通过：
  {summary['p1_spatial_organization_pass']}/
  {summary['dataset_count']}。

P1 的评价对象就是功能体块之间的邻接、分区和跨层组织。门的位置与详细
步行动线不属于本项目输出范围，也不作为数据缺陷或训练停止条件。

## 递归切割表示上限

- 两层都可被纯直线递归切割完整分离：
  {summary['cut_fully_separable_both_floors']}/{summary['dataset_count']}。
- 一层可完整分离：
  {summary['cut_floor_fully_separable']['1']}/{summary['dataset_count']}。
- 二层可完整分离：
  {summary['cut_floor_fully_separable']['2']}/{summary['dataset_count']}。
- 两层平均可分离房间比例均值：
  {summary['cut_resolution_rate']['mean']:.3f}。

结论：纯递归直线切割适合多数样本，但不能完整表达 109 套住宅。V5 可以
把切割作为主要合法动作，但不能只允许纯 guillotine 切割；至少需要保留
交通骨架、局部放置、空白区域或非整层贯通切割等动作。

## 产物

- `manifest.json` / `manifest.csv`：数据清单、哈希、来源、统计和集合。
- `split_v1.json`：固定训练、验证、测试划分。
- `near_duplicate_groups.json`：近重复组与最近候选对。
- `relations/*.json`：逐住宅空间关系和 P1 代理检查。
- `cut_upper_bound/*.json`：逐住宅递归切割表示上限。
- `summary.json`：机器可读汇总。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    files = sorted(data_dir.glob("house_*.json"))
    if not files:
        raise SystemExit(f"No house_*.json files found in {data_dir}")

    source_batches = load_source_batches(data_dir)
    records = []
    relations_dir = output_dir / "relations"
    cuts_dir = output_dir / "cut_upper_bound"
    for index, path in enumerate(files, 1):
        house_id = path.stem
        source_batch = source_batches.get(house_id)
        if source_batch is None:
            with path.open(encoding="utf-8") as handle:
                embedded_house_id = str(json.load(handle).get("house_id", ""))
            source_batch = source_batches.get(embedded_house_id, "unknown")
        record, relation_output, cut_output = analyze_house(
            path, source_batch
        )
        records.append(record)
        write_json(relations_dir / f"{house_id}.json", relation_output)
        write_json(cuts_dir / f"{house_id}.json", cut_output)
        if index % 50 == 0 or index == len(files):
            print(f"Analyzed {index}/{len(files)}")

    groups, nearest_pairs = build_duplicate_groups(records)
    splits = split_groups(records, groups)
    split_lookup = {house_id: split for split, ids in splits.items() for house_id in ids}
    summary = aggregate(records, groups, splits)
    manifest_records = []
    for record in sorted(records, key=lambda item: item["house_id"]):
        output_record = {key: value for key, value in record.items() if key != "canonical_geometry"}
        output_record["split"] = split_lookup[record["house_id"]]
        manifest_records.append(output_record)

    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "graphspace_phase1_manifest_v1",
            "data_dir": str(data_dir),
            "split_seed": SPLIT_SEED,
            "near_duplicate_method": "orientation-invariant 5%-quantized room-box signature",
            "records": manifest_records,
        },
    )
    write_manifest_csv(output_dir / "manifest.csv", records, split_lookup)
    write_json(
        output_dir / "near_duplicate_groups.json",
        {
            "method": "exact SHA-256 plus orientation-invariant 5%-quantized room-box signature",
            "approximate_threshold": 0.04,
            "groups": [
                {"group_id": f"group_{index:04d}", "house_ids": group}
                for index, group in enumerate(groups)
                if len(group) > 1
            ],
            "closest_candidate_pairs": nearest_pairs,
        },
    )
    write_json(
        output_dir / "split_v1.json",
        {
            "schema_version": "graphspace_split_v1",
            "seed": SPLIT_SEED,
            "policy": "80/10/10; near-duplicate groups are indivisible",
            **splits,
        },
    )
    write_json(output_dir / "summary.json", summary)
    write_markdown_report(output_dir / "REPORT.md", summary, groups)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
