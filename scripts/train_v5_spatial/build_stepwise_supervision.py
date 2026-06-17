#!/usr/bin/env python3
"""Build mixed stepwise action supervision for V5 spatial decoding."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from stepwise_decision import ActionKind, StepAction, StepwiseDecisionEnvironment


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
PHASE2_DIR = ROOT / "data" / "phase2_v5" / "samples"
PHASE7_DIR = ROOT / "data" / "phase7_staged_spatial" / "samples"
SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
OUTPUT_DIR = ROOT / "data" / "phase9_stepwise_spatial"
VOXEL_MM = 300.0
GRID_Z = 20
FLOOR_CELLS = 10
TRAFFIC_TYPES = {"entryway", "corridor", "stairs"}
RIGID_ORDER = {
    "living_room": 0,
    "dining_room": 1,
    "kitchen": 2,
    "bedroom": 3,
    "bathroom": 4,
    "multi_purpose": 5,
}
SERVICE_TYPES = {"utility", "balcony"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--phase2-dir", type=Path, default=PHASE2_DIR)
    parser.add_argument("--phase7-dir", type=Path, default=PHASE7_DIR)
    parser.add_argument("--split-path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_cell(value: float) -> int:
    cell = round(float(value) / VOXEL_MM)
    if abs(cell * VOXEL_MM - float(value)) > 1e-4:
        raise ValueError(f"value is not on 300mm grid: {value}")
    return int(cell)


def room_floors(room: dict) -> tuple[int, ...]:
    if room.get("floors"):
        return tuple(sorted(int(value) for value in room["floors"]))
    return (int(room.get("floor", 1)),)


def bounds_for_room(room: dict) -> tuple[int, int, int, int, int, int]:
    mins = [exact_cell(value) for value in room["box_min"]]
    maxs = [exact_cell(value) for value in room["box_max"]]
    return tuple(mins + maxs)


def load_rooms(processed_path: Path) -> list[dict]:
    payload = read_json(processed_path)
    rooms = []
    for index, room in enumerate(payload["rooms"]):
        rooms.append(
            {
                "index": index,
                "source_id": str(room["id"]),
                "type": str(room["type"]),
                "floors": room_floors(room),
                "bounds": bounds_for_room(room),
            }
        )
    return rooms


def site_cells(processed_path: Path) -> tuple[int, int, int]:
    payload = read_json(processed_path)
    building = payload["metadata"]["building_size"]
    return (
        exact_cell(building["x"]),
        exact_cell(building["y"]),
        GRID_Z,
    )


def region_id_for_bounds(
    env: StepwiseDecisionEnvironment,
    bounds: tuple[int, int, int, int, int, int],
) -> str:
    matches = [
        region.id
        for region in env.state.regions.values()
        if (
            region.bounds[0] <= bounds[0]
            and region.bounds[1] <= bounds[1]
            and region.bounds[2] <= bounds[2]
            and bounds[3] <= region.bounds[3]
            and bounds[4] <= region.bounds[4]
            and bounds[5] <= region.bounds[5]
        )
    ]
    if not matches:
        raise ValueError(f"no active region contains bounds {bounds}")
    return sorted(matches, key=len)[0]


def action_to_record(action: StepAction, result, phase: str) -> dict:
    record = {
        "phase": phase,
        "kind": action.kind.value,
        "accepted": bool(result.accepted),
        "issues": list(result.issues),
        "rollback_available": bool(result.rollback_available),
    }
    for name in (
        "region_id",
        "axis",
        "cut",
        "target_action_index",
        "reason",
    ):
        value = getattr(action, name)
        if value is not None and value != "":
            record[name] = value
    for name in (
        "left_node_ids",
        "right_node_ids",
        "node_ids",
        "bounds",
        "source_region_ids",
    ):
        value = getattr(action, name)
        if value:
            record[name] = list(value)
    if result.action_index is not None:
        record["action_index"] = result.action_index
    return record


def apply_recorded(
    env: StepwiseDecisionEnvironment,
    action: StepAction,
    phase: str,
    records: list[dict],
) -> None:
    result = env.apply(action)
    records.append(action_to_record(action, result, phase))
    if not result.accepted:
        raise ValueError(f"oracle action rejected during {phase}: {result.issues}")


def apply_negative_attempt(
    env: StepwiseDecisionEnvironment,
    action: StepAction,
    phase: str,
    records: list[dict],
) -> None:
    result = env.apply(action)
    records.append(action_to_record(action, result, phase))
    if result.accepted:
        rollback = StepAction(
            kind=ActionKind.ROLLBACK,
            reason="undo accepted synthetic negative attempt",
        )
        rollback_result = env.apply(rollback)
        records.append(action_to_record(rollback, rollback_result, "rollback"))


def apply_probe_then_rollback(
    env: StepwiseDecisionEnvironment,
    action: StepAction,
    phase: str,
    records: list[dict],
) -> None:
    result = env.apply(action)
    records.append(action_to_record(action, result, phase))
    if not result.accepted:
        raise ValueError(f"probe action rejected: {result.issues}")
    rollback = StepAction(
        kind=ActionKind.ROLLBACK,
        reason="rollback exploratory accepted action before final trace",
    )
    rollback_result = env.apply(rollback)
    records.append(action_to_record(rollback, rollback_result, "rollback"))
    if not rollback_result.accepted:
        raise ValueError(f"probe rollback rejected: {rollback_result.issues}")


def room_sort_key(room: dict) -> tuple:
    if room["type"] == "stairs":
        group = 0
    elif room["type"] in TRAFFIC_TYPES:
        group = 1
    elif room["type"] in RIGID_ORDER:
        group = 2
    elif room["type"] in SERVICE_TYPES:
        group = 3
    else:
        group = 4
    return (
        group,
        min(room["floors"]),
        RIGID_ORDER.get(room["type"], 99),
        room["bounds"][0],
        room["bounds"][1],
        room["index"],
    )


def floor_region_ids(env: StepwiseDecisionEnvironment) -> dict[int, str]:
    output = {}
    for region in env.state.regions.values():
        if region.bounds[2] == 0 and region.bounds[5] == FLOOR_CELLS:
            output[1] = region.id
        elif region.bounds[2] == FLOOR_CELLS and region.bounds[5] == GRID_Z:
            output[2] = region.id
    if set(output) != {1, 2}:
        raise ValueError(f"missing floor regions: {output}")
    return output


def child_region_for_bounds(
    env: StepwiseDecisionEnvironment,
    bounds: tuple[int, int, int, int, int, int],
    node_ids: tuple[int, ...],
) -> str:
    candidates = [
        region.id
        for region in env.state.regions.values()
        if region.bounds == bounds and set(region.node_ids) == set(node_ids)
    ]
    if not candidates:
        raise ValueError(f"missing child region for bounds={bounds} nodes={node_ids}")
    return candidates[0]


def axis_overlap(
    a: tuple[int, int, int, int, int, int],
    b: tuple[int, int, int, int, int, int],
    axis: int,
) -> int:
    return max(0, min(a[axis + 3], b[axis + 3]) - max(a[axis], b[axis]))


def valid_cuts(
    room_indices: tuple[int, ...],
    rooms_by_index: dict[int, dict],
    region_bounds: tuple[int, int, int, int, int, int],
) -> list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
    candidates = []
    for axis in (0, 1):
        positions = sorted(
            {
                rooms_by_index[index]["bounds"][axis]
                for index in room_indices
            }
            | {
                rooms_by_index[index]["bounds"][axis + 3]
                for index in room_indices
            }
        )
        for cut in positions:
            if cut <= region_bounds[axis] or cut >= region_bounds[axis + 3]:
                continue
            left, right = [], []
            valid = True
            for index in room_indices:
                bounds = rooms_by_index[index]["bounds"]
                if bounds[axis + 3] <= cut:
                    left.append(index)
                elif bounds[axis] >= cut:
                    right.append(index)
                else:
                    valid = False
                    break
            if valid and left and right:
                candidates.append((axis, cut, tuple(left), tuple(right)))
    return candidates


def choose_cut(
    room_indices: tuple[int, ...],
    rooms_by_index: dict[int, dict],
    region_bounds: tuple[int, int, int, int, int, int],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]] | None:
    candidates = valid_cuts(room_indices, rooms_by_index, region_bounds)
    if not candidates:
        return None

    def score(item: tuple[int, int, tuple[int, ...], tuple[int, ...]]) -> tuple:
        axis, cut, left, right = item
        balance = min(len(left), len(right))
        imbalance = abs(len(left) - len(right))
        position = (cut - region_bounds[axis]) / max(
            region_bounds[axis + 3] - region_bounds[axis],
            1,
        )
        # Prefer balanced cuts near the region center; use Y before X as a
        # deterministic tie breaker for common row-like residential partitions.
        return balance, -imbalance, -abs(position - 0.5), axis

    return max(candidates, key=score)


def place_room(
    env: StepwiseDecisionEnvironment,
    room: dict,
    region_id: str,
    records: list[dict],
    phase: str,
) -> None:
    apply_recorded(
        env,
        StepAction(
            kind=ActionKind.PLACE,
            region_id=region_id,
            node_ids=(room["index"],),
            bounds=room["bounds"],
            reason=f"place {room['type']} after no further clean cut",
        ),
        phase,
        records,
    )


def recursive_cut_or_place(
    env: StepwiseDecisionEnvironment,
    rooms_by_index: dict[int, dict],
    room_indices: tuple[int, ...],
    region_id: str,
    records: list[dict],
    depth: int = 0,
) -> None:
    active = tuple(
        index
        for index in room_indices
        if index in env.state.regions[region_id].node_ids
    )
    if not active:
        return
    if len(active) == 1:
        place_room(
            env,
            rooms_by_index[active[0]],
            region_id,
            records,
            "place_leaf_room",
        )
        return

    region_bounds = env.state.regions[region_id].bounds
    selected = choose_cut(active, rooms_by_index, region_bounds)
    if selected is None:
        for index in sorted(active, key=lambda value: room_sort_key(rooms_by_index[value])):
            place_room(
                env,
                rooms_by_index[index],
                region_id,
                records,
                "place_uncuttable_room",
            )
        return

    axis, cut, left, right = selected
    left_bounds = list(region_bounds)
    right_bounds = list(region_bounds)
    left_bounds[axis + 3] = cut
    right_bounds[axis] = cut
    apply_recorded(
        env,
        StepAction(
            kind=ActionKind.CUT,
            region_id=region_id,
            axis=axis,
            cut=cut,
            left_node_ids=left,
            right_node_ids=right,
            reason=f"clean guillotine split at depth {depth}",
        ),
        "xy_clean_cut",
        records,
    )
    left_id = child_region_for_bounds(env, tuple(left_bounds), left)
    right_id = child_region_for_bounds(env, tuple(right_bounds), right)
    recursive_cut_or_place(
        env,
        rooms_by_index,
        left,
        left_id,
        records,
        depth + 1,
    )
    recursive_cut_or_place(
        env,
        rooms_by_index,
        right,
        right_id,
        records,
        depth + 1,
    )


def decompose_mask(mask: np.ndarray, floor: int) -> list[tuple[int, int, int, int, int, int]]:
    remaining = mask.astype(bool).copy()
    boxes = []
    z0 = (floor - 1) * FLOOR_CELLS
    z1 = z0 + FLOOR_CELLS
    while remaining.any():
        x0, y0 = np.argwhere(remaining)[0]
        x1 = int(x0) + 1
        while x1 < remaining.shape[0] and remaining[x1, y0]:
            x1 += 1
        y1 = int(y0) + 1
        while y1 < remaining.shape[1] and remaining[x0:x1, y1].all():
            y1 += 1
        remaining[x0:x1, y0:y1] = False
        boxes.append((int(x0), int(y0), z0, int(x1), int(y1), z1))
    return boxes


def build_room_actions(
    env: StepwiseDecisionEnvironment,
    rooms: list[dict],
    cells: tuple[int, int, int],
    phase7_array_path: Path,
    records: list[dict],
) -> None:
    cross_floor = [room for room in rooms if len(room["floors"]) > 1]
    single_floor = [room for room in rooms if len(room["floors"]) == 1]

    for room in sorted(cross_floor, key=room_sort_key):
        apply_recorded(
            env,
            StepAction(
                kind=ActionKind.PLACE,
                region_id="root",
                node_ids=(room["index"],),
                bounds=room["bounds"],
                reason="place cross-floor function before floor split",
            ),
            "place_cross_floor",
            records,
        )

    floor1 = tuple(room["index"] for room in single_floor if room["floors"] == (1,))
    floor2 = tuple(room["index"] for room in single_floor if room["floors"] == (2,))
    if floor1 and floor2:
        apply_recorded(
            env,
            StepAction(
                kind=ActionKind.CUT,
                region_id="root",
                axis=2,
                cut=FLOOR_CELLS,
                left_node_ids=floor1,
                right_node_ids=floor2,
                reason="split remaining unresolved rooms into floor regions",
            ),
            "floor_split",
            records,
        )

    floor_regions = floor_region_ids(env) if floor1 and floor2 else {}
    first_single = sorted(single_floor, key=room_sort_key)[0] if single_floor else None
    if first_single is not None:
        region_id = (
            floor_regions[first_single["floors"][0]]
            if floor_regions
            else region_id_for_bounds(env, first_single["bounds"])
        )
        apply_probe_then_rollback(
            env,
            StepAction(
                kind=ActionKind.PLACE,
                region_id=region_id,
                node_ids=(first_single["index"],),
                bounds=first_single["bounds"],
                reason="synthetic accepted probe before rollback supervision",
            ),
            "probe_then_rollback",
            records,
        )
        apply_negative_attempt(
            env,
            StepAction(
                kind=ActionKind.PLACE,
                region_id=region_id,
                node_ids=(first_single["index"],),
                bounds=(0, 0, 0, cells[0] + 1, 1, 1),
                reason="synthetic rejected out-of-site attempt for retry supervision",
            ),
            "negative_attempt",
            records,
        )

    build_empty_actions(env, phase7_array_path, records)

    rooms_by_index = {room["index"]: room for room in rooms}
    if floor_regions:
        recursive_cut_or_place(
            env,
            rooms_by_index,
            floor1,
            floor_regions[1],
            records,
        )
        recursive_cut_or_place(
            env,
            rooms_by_index,
            floor2,
            floor_regions[2],
            records,
        )
    else:
        root_or_region = "root" if "root" in env.state.regions else next(iter(env.state.regions))
        recursive_cut_or_place(
            env,
            rooms_by_index,
            tuple(room["index"] for room in single_floor),
            root_or_region,
            records,
        )


def build_empty_actions(
    env: StepwiseDecisionEnvironment,
    phase7_array_path: Path,
    records: list[dict],
) -> int:
    with np.load(phase7_array_path) as arrays:
        empty = arrays["empty_mask"].copy()
    count = 0
    for floor in (1, 2):
        for bounds in decompose_mask(empty[floor - 1], floor):
            region_id = region_id_for_bounds(env, bounds)
            apply_recorded(
                env,
                StepAction(
                    kind=ActionKind.RESERVE_EMPTY,
                    region_id=region_id,
                    bounds=bounds,
                    reason="reserve explicit interior empty cells",
                ),
                "reserve_empty",
                records,
            )
            count += 1
    return count


def validate_replay(env: StepwiseDecisionEnvironment, rooms: list[dict]) -> dict:
    missing = [
        room["index"]
        for room in rooms
        if room["index"] not in env.state.assignments
    ]
    mismatched = []
    for room in rooms:
        boxes = env.state.assignments.get(room["index"], [])
        if boxes != [room["bounds"]]:
            mismatched.append(room["index"])
    return {
        "missing_assignment_count": len(missing),
        "mismatched_assignment_count": len(mismatched),
        "empty_box_count": len(env.state.empty_regions),
        "accepted_action_count": len(env.state.history),
        "attempt_count": len(env.attempt_log),
        "rejected_attempt_count": sum(
            1 for attempt in env.attempt_log if not attempt.accepted
        ),
    }


def build_sample(
    processed_path: Path,
    phase2_dir: Path = PHASE2_DIR,
    phase7_dir: Path = PHASE7_DIR,
) -> dict:
    rooms = load_rooms(processed_path)
    cells = site_cells(processed_path)
    phase7 = read_json(phase7_dir / f"{processed_path.stem}.json")
    env = StepwiseDecisionEnvironment(
        site_bounds=(0, 0, 0, *cells),
        node_ids=tuple(range(len(rooms))),
    )
    records: list[dict] = []
    phase7_array_path = phase7_dir / f"{processed_path.stem}.npz"
    build_room_actions(
        env,
        rooms,
        cells,
        phase7_array_path,
        records,
    )
    empty_box_count = sum(
        1 for record in records if record["kind"] == ActionKind.RESERVE_EMPTY.value
    )
    replay = validate_replay(env, rooms)
    action_counts = Counter(record["kind"] for record in records)
    return {
        "schema": "graphspace_v5_stepwise_spatial_supervision_v1",
        "house_id": processed_path.stem,
        "site_cells": list(cells),
        "graph": phase7["graph"],
        "rooms": rooms,
        "actions": records,
        "stats": {
            "room_count": len(rooms),
            "action_count": len(records),
            "accepted_action_count": replay["accepted_action_count"],
            "attempt_count": replay["attempt_count"],
            "rejected_attempt_count": replay["rejected_attempt_count"],
            "cut_action_count": action_counts[ActionKind.CUT.value],
            "place_action_count": action_counts[ActionKind.PLACE.value],
            "empty_action_count": action_counts[ActionKind.RESERVE_EMPTY.value],
            "rollback_action_count": action_counts[ActionKind.ROLLBACK.value],
            "explicit_empty_box_count": empty_box_count,
            "missing_assignment_count": replay["missing_assignment_count"],
            "mismatched_assignment_count": replay["mismatched_assignment_count"],
        },
    }


def main() -> None:
    args = parse_args()
    split = read_json(args.split_path)
    house_ids = split["train"] + split["validation"] + split["test"]
    if args.max_samples is not None:
        house_ids = house_ids[: args.max_samples]
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for house_id in house_ids:
        sample = build_sample(
            args.input_dir / f"{house_id}.json",
            args.phase2_dir,
            args.phase7_dir,
        )
        (samples_dir / f"{house_id}.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        reports.append(sample["stats"])
    summary = {
        "schema": "graphspace_v5_stepwise_spatial_summary_v1",
        "sample_count": len(reports),
        "total_action_count": sum(item["action_count"] for item in reports),
        "total_rejected_attempt_count": sum(
            item["rejected_attempt_count"] for item in reports
        ),
        "total_cut_action_count": sum(
            item["cut_action_count"] for item in reports
        ),
        "total_place_action_count": sum(
            item["place_action_count"] for item in reports
        ),
        "total_empty_action_count": sum(
            item["empty_action_count"] for item in reports
        ),
        "total_rollback_action_count": sum(
            item["rollback_action_count"] for item in reports
        ),
        "samples_with_rejected_attempt": sum(
            item["rejected_attempt_count"] > 0 for item in reports
        ),
        "samples_with_rollback": sum(
            item["rollback_action_count"] > 0 for item in reports
        ),
        "samples_with_empty_action": sum(
            item["empty_action_count"] > 0 for item in reports
        ),
        "replay_complete_count": sum(
            item["missing_assignment_count"] == 0
            and item["mismatched_assignment_count"] == 0
            for item in reports
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
