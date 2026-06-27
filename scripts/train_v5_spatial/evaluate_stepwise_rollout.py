#!/usr/bin/env python3
"""Evaluate Phase9 stepwise policy by replaying complete action rollouts."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for import_dir in (
    ROOT,
    SCRIPT_DIR,
    ROOT / "scripts" / "data_phase4",
):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.data_phase4.evaluate_candidates import evaluate_candidate  # noqa: E402
from staged_dataset import graph_arrays, split_ids  # noqa: E402
from stepwise_dataset import (  # noqa: E402
    ACTION_TO_ID,
    DEFAULT_DATA_DIR,
    DEFAULT_SPLIT_PATH,
    GRID_X,
    GRID_Y,
    GRID_Z,
    ID_TO_ACTION,
    protocol_action_mask,
    read_json,
    record_to_action,
    state_volume,
)
from stepwise_decision import (  # noqa: E402
    ActionKind,
    DecisionRegion,
    StepAction,
    StepwiseDecisionEnvironment,
    intersects,
    volume,
)
from stepwise_model import StepwiseActionPolicy  # noqa: E402

CELL_MM = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("oracle", "model"), default="oracle")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-houses", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def site_mm(site_cells: list[int]) -> tuple[float, float]:
    return float(site_cells[0]) * CELL_MM, float(site_cells[1]) * CELL_MM


def bounds_to_mm(bounds: tuple[int, int, int, int, int, int]) -> tuple[list[float], list[float]]:
    x0, y0, z0, x1, y1, z1 = bounds
    return (
        [x0 * CELL_MM, y0 * CELL_MM, z0 * CELL_MM],
        [x1 * CELL_MM, y1 * CELL_MM, z1 * CELL_MM],
    )


def requested_counts(payload: dict) -> dict[str, int]:
    return dict(Counter(room["type"] for room in payload["rooms"]))


def env_to_rooms(payload: dict, env: StepwiseDecisionEnvironment) -> list[dict]:
    rooms = []
    for node_id, boxes in sorted(env.state.assignments.items()):
        if not boxes:
            continue
        room = payload["rooms"][int(node_id)]
        for part_index, bounds in enumerate(boxes):
            box_min, box_max = bounds_to_mm(tuple(bounds))
            floors = sorted(
                {
                    1 if z < 10 else 2
                    for z in range(int(bounds[2]), int(bounds[5]))
                }
            )
            rooms.append(
                {
                    "id": f"{room['type']}_{node_id}_{part_index}",
                    "type": room["type"],
                    "box_min": box_min,
                    "box_max": box_max,
                    "floors": floors,
                    "floor": floors[0] if floors else 1,
                }
            )
    return rooms


def replay_oracle(payload: dict) -> tuple[StepwiseDecisionEnvironment, dict]:
    env = StepwiseDecisionEnvironment(
        site_bounds=(0, 0, 0, *payload["site_cells"]),
        node_ids=tuple(range(len(payload["rooms"]))),
    )
    invalid_records = []
    for index, record in enumerate(payload["actions"]):
        if not record["accepted"]:
            continue
        result = env.apply(record_to_action(record))
        if not result.accepted:
            invalid_records.append({"index": index, "issues": result.issues})
    return env, {
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records[:10],
    }


def load_model(checkpoint_path: Path, device: torch.device) -> StepwiseActionPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = StepwiseActionPolicy(
        base_channels=int(config.get("base_channels", 16)),
        hidden=int(config.get("hidden", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def active_region(env: StepwiseDecisionEnvironment) -> DecisionRegion | None:
    candidates = [
        region
        for region in env.state.regions.values()
        if region.status == "open" and region.node_ids
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda region: (len(region.node_ids), volume(region.bounds)))


def graph_tensors(payload: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes, edge_index, edge_type = graph_arrays(payload["graph"])
    node_tensor = torch.from_numpy(nodes)[None].to(device)
    mask = torch.ones(1, nodes.shape[0], device=device)
    adjacency = torch.zeros(1, 2, nodes.shape[0], nodes.shape[0], device=device)
    for offset in range(edge_index.shape[1]):
        source = int(edge_index[0, offset])
        target = int(edge_index[1, offset])
        relation = int(edge_type[offset])
        adjacency[0, relation, source, target] = 1.0
    return node_tensor, mask, adjacency


def legalize_bounds(
    values: list[int],
    container: tuple[int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    output = []
    for axis in range(3):
        low = max(container[axis], min(values[axis], values[axis + 3]))
        high = min(container[axis + 3], max(values[axis], values[axis + 3]))
        if high <= low:
            midpoint = int(round((values[axis] + values[axis + 3]) / 2))
            low = min(max(midpoint, container[axis]), container[axis + 3] - 1)
            high = low + 1
        output.append(int(low))
    for axis in range(3):
        output.append(
            int(min(max(output[axis] + 1, max(values[axis], values[axis + 3])), container[axis + 3]))
        )
    return tuple(output)


def decode_bounds(
    raw: torch.Tensor,
    container: tuple[int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    scale = torch.tensor([GRID_X, GRID_Y, GRID_Z, GRID_X, GRID_Y, GRID_Z], device=raw.device)
    values = torch.round(raw * scale).to(torch.int64).detach().cpu().tolist()
    return legalize_bounds([int(value) for value in values], container)


def taken_bounds(env: StepwiseDecisionEnvironment) -> list[tuple[int, int, int, int, int, int]]:
    assigned = [
        bounds
        for boxes in env.state.assignments.values()
        for bounds in boxes
    ]
    return assigned + list(env.state.empty_regions)


def overlaps_taken(
    env: StepwiseDecisionEnvironment,
    bounds: tuple[int, int, int, int, int, int],
) -> bool:
    return any(intersects(bounds, existing) for existing in taken_bounds(env))


def largest_free_xy_bounds(
    env: StepwiseDecisionEnvironment,
    region: DecisionRegion,
) -> tuple[int, int, int, int, int, int] | None:
    x0, y0, z0, x1, y1, z1 = region.bounds
    width = max(x1 - x0, 0)
    height = max(y1 - y0, 0)
    if width == 0 or height == 0:
        return None
    occupied = [[False for _ in range(height)] for _ in range(width)]
    for bounds in taken_bounds(env):
        if not intersects(region.bounds, bounds):
            continue
        ix0 = max(x0, bounds[0]) - x0
        iy0 = max(y0, bounds[1]) - y0
        ix1 = min(x1, bounds[3]) - x0
        iy1 = min(y1, bounds[4]) - y0
        for ix in range(ix0, ix1):
            for iy in range(iy0, iy1):
                occupied[ix][iy] = True

    heights = [0] * height
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for ix in range(width):
        for iy in range(height):
            heights[iy] = 0 if occupied[ix][iy] else heights[iy] + 1
        stack: list[int] = []
        for iy in range(height + 1):
            current = heights[iy] if iy < height else 0
            while stack and heights[stack[-1]] > current:
                top = stack.pop()
                rect_width = heights[top]
                y_start = stack[-1] + 1 if stack else 0
                rect_height = iy - y_start
                area = rect_width * rect_height
                if area > best_area:
                    best_area = area
                    best = (
                        ix - rect_width + 1,
                        y_start,
                        ix + 1,
                        iy,
                    )
            stack.append(iy)
    if best is None or best_area <= 0:
        return None
    bx0, by0, bx1, by1 = best
    return (x0 + bx0, y0 + by0, z0, x0 + bx1, y0 + by1, z1)


def repair_overlapping_bounds(
    env: StepwiseDecisionEnvironment,
    region: DecisionRegion,
    bounds: tuple[int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    if not overlaps_taken(env, bounds):
        return bounds
    repaired = largest_free_xy_bounds(env, region)
    return repaired or bounds


def decode_node_set(
    logits: torch.Tensor,
    active_nodes: tuple[int, ...],
    threshold: float = 0.5,
    require_one: bool = False,
    allow_all: bool = True,
) -> tuple[int, ...]:
    probs = logits.sigmoid().detach().cpu()
    selected = tuple(
        node_id
        for node_id in active_nodes
        if float(probs[node_id]) >= threshold
    )
    if require_one and not selected and active_nodes:
        selected = (max(active_nodes, key=lambda node_id: float(probs[node_id])),)
    if not allow_all and len(selected) == len(active_nodes) and len(active_nodes) > 1:
        lowest = min(active_nodes, key=lambda node_id: float(probs[node_id]))
        selected = tuple(node_id for node_id in selected if node_id != lowest)
    return selected


def predicted_action(
    output: dict,
    region: DecisionRegion,
    env: StepwiseDecisionEnvironment,
) -> StepAction | None:
    mask = torch.from_numpy(protocol_action_mask(env, region.id)).to(
        output["action_logits"].device
    )
    logits = output["action_logits"][0].masked_fill(mask <= 0, -1.0e4)
    action_id = int(logits.argmax(dim=-1))
    action_name = ID_TO_ACTION[action_id]
    if action_name == "reject":
        return None
    if action_name == "rollback":
        return StepAction(kind=ActionKind.ROLLBACK)
    if action_name == "cut":
        axis = int(output["axis_logits"].argmax(dim=-1)[0])
        scale = [GRID_X, GRID_Y, GRID_Z][axis]
        raw_cut = int(round(float(output["cut"][0]) * scale))
        cut = min(
            max(raw_cut, region.bounds[axis] + 1),
            region.bounds[axis + 3] - 1,
        )
        left = decode_node_set(
            output["node_logits"][0],
            region.node_ids,
            require_one=True,
            allow_all=False,
        )
        right = tuple(node for node in region.node_ids if node not in set(left))
        return StepAction(
            kind=ActionKind.CUT,
            region_id=region.id,
            axis=axis,
            cut=cut,
            left_node_ids=left,
            right_node_ids=right,
            reason="model rollout prediction",
        )
    if action_name == "place":
        node_ids = decode_node_set(
            output["node_logits"][0],
            region.node_ids,
            require_one=True,
        )
        bounds = repair_overlapping_bounds(
            env,
            region,
            decode_bounds(output["box"][0], region.bounds),
        )
        return StepAction(
            kind=ActionKind.PLACE,
            region_id=region.id,
            node_ids=node_ids,
            bounds=bounds,
            reason="model rollout prediction",
        )
    if action_name == "reserve_empty":
        bounds = repair_overlapping_bounds(
            env,
            region,
            decode_bounds(output["box"][0], region.bounds),
        )
        return StepAction(
            kind=ActionKind.RESERVE_EMPTY,
            region_id=region.id,
            bounds=bounds,
            reason="model rollout prediction",
        )
    raise ValueError(f"unsupported predicted action: {action_name}")


def rollout_model(
    payload: dict,
    model: StepwiseActionPolicy,
    device: torch.device,
    max_steps: int,
) -> tuple[StepwiseDecisionEnvironment, dict]:
    env = StepwiseDecisionEnvironment(
        site_bounds=(0, 0, 0, *payload["site_cells"]),
        node_ids=tuple(range(len(payload["rooms"]))),
    )
    nodes, node_mask, adjacency = graph_tensors(payload, device)
    counters = Counter()
    consecutive_no_progress = 0
    issues = []
    for step in range(max_steps):
        region = active_region(env)
        if region is None:
            counters["completed"] += 1
            break
        current = {"region_id": region.id}
        volume_tensor = torch.from_numpy(
            state_volume(env, payload["site_cells"], current)
        )[None].to(device)
        with torch.no_grad():
            output = model(volume_tensor, nodes, node_mask, adjacency)
        action = predicted_action(output, region, env)
        if action is None:
            counters["predicted_reject"] += 1
            consecutive_no_progress += 1
        else:
            result = env.apply(action)
            counters[f"predicted_{action.kind.value}"] += 1
            if result.accepted:
                consecutive_no_progress = 0
                counters["accepted"] += 1
            else:
                consecutive_no_progress += 1
                counters["invalid"] += 1
                issues.append(
                    {
                        "step": step,
                        "action": action.kind.value,
                        "issues": result.issues,
                    }
                )
        if consecutive_no_progress >= 8:
            counters["stopped_no_progress"] += 1
            break
    else:
        counters["stopped_max_steps"] += 1
    return env, {"rollout_counters": dict(counters), "issues": issues[:20]}


def assignment_report(payload: dict, env: StepwiseDecisionEnvironment) -> dict:
    missing = [
        index
        for index in range(len(payload["rooms"]))
        if index not in env.state.assignments
    ]
    assigned = len(payload["rooms"]) - len(missing)
    return {
        "expected_room_count": len(payload["rooms"]),
        "assigned_room_count": assigned,
        "missing_assignment_count": len(missing),
        "missing_node_ids": missing[:20],
        "empty_box_count": len(env.state.empty_regions),
        "accepted_action_count": len(env.state.history),
        "attempt_count": len(env.attempt_log),
        "rejected_attempt_count": sum(
            1 for attempt in env.attempt_log if not attempt.accepted
        ),
        "complete": len(missing) == 0,
    }


def evaluate_payload(
    house_id: str,
    payload: dict,
    env: StepwiseDecisionEnvironment,
    rollout_extra: dict,
) -> dict:
    rooms = env_to_rooms(payload, env)
    report, _ = evaluate_candidate(
        house_id,
        rooms,
        requested_counts(payload),
        site_mm(payload["site_cells"]),
    )
    return {
        "house_id": house_id,
        "assignment": assignment_report(payload, env),
        "p0_pass": report["p0"]["pass"],
        "p1_hard_geometry_pass": report["p1_spatial_organization"]["hard_geometry_pass"],
        "p1_spatial_organization_pass": report["p1_spatial_organization"][
            "spatial_organization_pass"
        ],
        "p2_pass": bool(report["p0"]["pass"] and report["p2"]["quality_gate_pass"]),
        "rollout": rollout_extra,
        "candidate_report": report,
        "rooms": rooms,
    }


def selected_house_ids(split: str, max_houses: int) -> list[str]:
    return split_ids(DEFAULT_SPLIT_PATH, split)[:max_houses]


def summarize(results: list[dict], mode: str) -> dict:
    total = len(results)
    return {
        "mode": mode,
        "house_count": total,
        "complete_count": sum(result["assignment"]["complete"] for result in results),
        "p0_pass_count": sum(result["p0_pass"] for result in results),
        "p1_hard_geometry_pass_count": sum(
            result["p1_hard_geometry_pass"] for result in results
        ),
        "p1_spatial_organization_pass_count": sum(
            result["p1_spatial_organization_pass"] for result in results
        ),
        "p2_pass_count": sum(result["p2_pass"] for result in results),
        "assigned_room_rate": sum(
            result["assignment"]["assigned_room_count"]
            for result in results
        )
        / max(
            sum(result["assignment"]["expected_room_count"] for result in results),
            1,
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = None
    if args.mode == "model":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required in model mode")
        model = load_model(args.checkpoint, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for house_id in selected_house_ids(args.split, args.max_houses):
        payload = read_json(DEFAULT_DATA_DIR / f"{house_id}.json")
        if args.mode == "oracle":
            env, extra = replay_oracle(payload)
        else:
            assert model is not None
            env, extra = rollout_model(payload, model, device, args.max_steps)
        result = evaluate_payload(house_id, payload, env, extra)
        results.append(result)
        write_json(args.output_dir / f"{house_id}.json", result)

    summary = summarize(results, args.mode)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
