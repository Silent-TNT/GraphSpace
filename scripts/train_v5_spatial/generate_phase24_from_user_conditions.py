#!/usr/bin/env python3
"""Bridge user site/program conditions into the Phase24 multi-part decoder.

This is an experimental adapter, not the final V6 generator. Phase24 was
trained on known Phase10 functional groups, so this script creates a temporary
Phase10-like group set from the site/program prior, assigns coarse non-overlap
boxes, then lets the Phase24 decoder and P0-safe topology repair produce the
final blocks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
for import_dir in (
    ROOT,
    ROOT / "scripts" / "train_v5_spatial",
    ROOT / "scripts" / "spatial_modal_infer",
    ROOT / "scripts" / "data_phase4",
):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from scripts.spatial_modal_infer.config import DEFAULT_ROOM_SIZE, ROOM_TYPES  # noqa: E402
from scripts.train_v5_spatial.generate_from_user_conditions import (  # noqa: E402
    TEST_CASES,
    request_graph,
    validate_request,
)
from scripts.train_v5_spatial.program_prior import ProgramPrior  # noqa: E402
from scripts.train_v5_spatial.v6_multipart_decoder import (  # noqa: E402
    FLOOR_Z,
    MAX_PARTS,
    VOXEL_MM,
    GroupSample,
    MultiPartDecoder,
    decode_parts,
    feature_vector,
    group_neighbor_map,
    layout_report,
    repair_overlaps,
    topology_placement_search,
    write_json,
)
from scripts.train_v5_spatial.v6_coarse_layout_head import (  # noqa: E402
    load_coarse_layout_head,
    predict_coarse_layout_ratios,
)
from scripts.train_v5_spatial.v6_graph_coarse_layout_model import (  # noqa: E402
    load_graph_coarse_layout_model,
    predict_graph_layout_ratios,
)
from scripts.train_v5_spatial.v6_graph_topology_generator import (  # noqa: E402
    GraphTopologySample,
    load_graph_topology_generator,
    predict_graph,
)
from scripts.train_v5_spatial.v6_candidate_scorer import (  # noqa: E402
    candidate_feature_vector,
    load_candidate_scorer,
)
from scripts.train_v5_spatial.v6_whole_hetero_graph_generator import (  # noqa: E402
    generate_whole_graph,
)
from scripts.train_v5_spatial.v6_topology_learner import PairSample, TopologyNode  # noqa: E402


TYPE_PRIORITY = {
    "stairs": 0,
    "entryway": 1,
    "living_room": 2,
    "dining_room": 3,
    "kitchen": 4,
    "corridor": 5,
    "bedroom": 6,
    "bathroom": 7,
    "utility": 8,
    "multi_purpose": 9,
    "balcony": 10,
}

PASSAGE_TYPES = {
    "entryway",
    "living_room",
    "dining_room",
    "kitchen",
    "corridor",
    "multi_purpose",
}
VERTICAL_TYPES = {"stairs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(TEST_CASES))
    parser.add_argument("--site-x", type=float)
    parser.add_argument("--site-y", type=float)
    parser.add_argument("--rooms-json")
    parser.add_argument("--rooms-file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--program-prior",
        type=Path,
        default=ROOT / "data" / "phase8_program_prior" / "program_prior.json",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=ROOT
        / "outputs"
        / "v6_multipart_graph_topology_linked_size_full_phase24"
        / "decoder.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parts", type=int, default=MAX_PARTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-topology-move-mm", type=float, default=1800.0)
    parser.add_argument("--max-size-adjustment-mm", type=float, default=600.0)
    parser.add_argument(
        "--coarse-layout-checkpoint",
        type=Path,
        help="Optional v6_coarse_layout_head.py checkpoint for learned group bbox priors.",
    )
    parser.add_argument(
        "--coarse-layout-strategy",
        choices=["rule", "graph", "learned_graph"],
        default="rule",
        help=(
            "rule uses preferred-position packing; graph scores candidate boxes by target-topology contacts; "
            "learned_graph uses a trained whole-graph coarse layout model plus P0 safety placement."
        ),
    )
    parser.add_argument(
        "--graph-coarse-layout-checkpoint",
        type=Path,
        help="Optional v6_graph_coarse_layout_model.py checkpoint used by --coarse-layout-strategy learned_graph.",
    )
    parser.add_argument(
        "--candidate-scorer-checkpoint",
        type=Path,
        help="Optional v6_candidate_scorer.py checkpoint blended into graph-aware legal candidate ranking.",
    )
    parser.add_argument(
        "--candidate-scorer-weight",
        type=float,
        default=150.0,
        help="Blend strength for --candidate-scorer-checkpoint. Higher values let learned ranking affect candidates more.",
    )
    parser.add_argument(
        "--access-aware-placement",
        action="store_true",
        help=(
            "During graph-aware coarse placement, penalize legal candidate boxes "
            "that create blocked direct contacts such as stairs-bedroom or "
            "served-space to served-space contact."
        ),
    )
    parser.add_argument(
        "--blocked-contact-penalty",
        type=float,
        default=1800.0,
        help="Score penalty applied per blocked direct contact when --access-aware-placement is enabled.",
    )
    parser.add_argument(
        "--untargeted-access-contact-penalty",
        type=float,
        default=120.0,
        help=(
            "Small score penalty for access-capable contacts that are not target "
            "topology neighbors when --access-aware-placement is enabled."
        ),
    )
    parser.add_argument(
        "--topology-generator-checkpoint",
        type=Path,
        help=(
            "Optional v6_graph_topology_generator.py checkpoint. When provided, "
            "target guidance edges are predicted by the learned graph generator. "
            "With --whole-graph-count-checkpoint, learned whole-graph nodes and "
            "attributes are kept and only guidance_relation edges are replaced."
        ),
    )
    parser.add_argument(
        "--whole-graph-count-checkpoint",
        type=Path,
        help=(
            "Optional v6_whole_hetero_graph_generator.py count checkpoint. "
            "When used with --whole-graph-program-checkpoint, functional node "
            "counts, node attributes and target topology come from learned models."
        ),
    )
    parser.add_argument(
        "--whole-graph-program-checkpoint",
        type=Path,
        help="ProgramGraphModel checkpoint used by --whole-graph-count-checkpoint.",
    )
    parser.add_argument(
        "--whole-graph-edge-threshold",
        type=float,
        help="Optional target-adjacent threshold for learned whole-graph generation.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def snap(value: float) -> float:
    return round(value / VOXEL_MM) * VOXEL_MM


def floor_text_to_floors(room_type: str, value: Any) -> list[int]:
    if room_type == "stairs":
        return [1, 2]
    text = str(value)
    if text == "1&2":
        return [1, 2]
    if text == "2":
        return [2]
    return [1]


def normalized_positions(topology: dict[str, Any]) -> dict[str, tuple[float, float]]:
    raw = {
        str(node["id"]): (
            float(node.get("position", [0.0, 0.0])[0]),
            float(node.get("position", [0.0, 0.0])[1]),
        )
        for node in topology.get("nodes", [])
    }
    if not raw:
        return {}
    xs = [value[0] for value in raw.values()]
    ys = [value[1] for value in raw.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {
        node_id: (
            0.1 + 0.8 * (x - min_x) / max(max_x - min_x, 1e-9),
            0.1 + 0.8 * (y - min_y) / max(max_y - min_y, 1e-9),
        )
        for node_id, (x, y) in raw.items()
    }


def desired_size(room_type: str, area_ratio: float, site_x: float, site_y: float) -> tuple[float, float]:
    default_w, default_d, _ = DEFAULT_ROOM_SIZE.get(room_type, (3000.0, 3000.0, 3000.0))
    default_area = default_w * default_d
    requested_area = max(default_area * 0.75, min(area_ratio * site_x * site_y, default_area * 2.4))
    aspect = max(default_w / max(default_d, VOXEL_MM), 0.35)
    width = math.sqrt(requested_area * aspect)
    depth = requested_area / max(width, VOXEL_MM)
    width = min(max(width, default_w * 0.75), site_x)
    depth = min(max(depth, default_d * 0.75), site_y)
    return max(VOXEL_MM, snap(width)), max(VOXEL_MM, snap(depth))


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def overlaps(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> bool:
    return min(box[2], other[2]) > max(box[0], other[0]) and min(box[3], other[3]) > max(box[1], other[1])


def touches(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> bool:
    horizontal_touch = (box[2] == other[0] or other[2] == box[0]) and min(box[3], other[3]) > max(box[1], other[1])
    vertical_touch = (box[3] == other[1] or other[3] == box[1]) and min(box[2], other[2]) > max(box[0], other[0])
    return horizontal_touch or vertical_touch


def grid_gap(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> int:
    dx = max(other[0] - box[2], box[0] - other[2], 0)
    dy = max(other[1] - box[3], box[1] - other[3], 0)
    return dx + dy


def topology_neighbors(topology: dict[str, Any]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = {}
    for edge in topology.get("edges", []):
        source = str(edge["source"])
        target = str(edge["target"])
        neighbors.setdefault(source, set()).add(target)
        neighbors.setdefault(target, set()).add(source)
    return neighbors


def access_label_between_types(left_type: str, right_type: str) -> str:
    if left_type in VERTICAL_TYPES or right_type in VERTICAL_TYPES:
        other_type = right_type if left_type in VERTICAL_TYPES else left_type
        return "access" if other_type in PASSAGE_TYPES else "blocked"
    return "access" if left_type in PASSAGE_TYPES or right_type in PASSAGE_TYPES else "blocked"


def share_floor(left: list[int], right: list[int]) -> bool:
    return bool(set(left) & set(right))


def place_box(
    node_id: str,
    width: float,
    depth: float,
    floors: list[int],
    preferred: tuple[float, float],
    site_x: float,
    site_y: float,
    occupied: dict[int, list[tuple[int, int, int, int]]],
) -> tuple[float, float, float, float]:
    sx = int(round(site_x / VOXEL_MM))
    sy = int(round(site_y / VOXEL_MM))
    wc = max(1, min(sx, int(round(width / VOXEL_MM))))
    dc = max(1, min(sy, int(round(depth / VOXEL_MM))))
    preferred_x = int(round(preferred[0] * max(sx - wc, 0)))
    preferred_y = int(round(preferred[1] * max(sy - dc, 0)))
    candidates: list[tuple[float, int, int]] = []
    for x in range(0, max(sx - wc, 0) + 1):
        for y in range(0, max(sy - dc, 0) + 1):
            distance = abs(x - preferred_x) + abs(y - preferred_y)
            candidates.append((distance, x, y))
    for _distance, x, y in sorted(candidates):
        candidate = (x, y, x + wc, y + dc)
        if all(not any(overlaps(candidate, other) for other in occupied[floor]) for floor in floors):
            for floor in floors:
                occupied[floor].append(candidate)
            return tuple(float(value * VOXEL_MM) for value in candidate)

    # If the coarse layout is too crowded, shrink in 300 mm steps rather than
    # failing the whole bridge. The downstream evaluator will still catch bad geometry.
    if wc > 1 or dc > 1:
        return place_box(node_id, (wc - 1) * VOXEL_MM, max(VOXEL_MM, (dc - 1) * VOXEL_MM), floors, preferred, site_x, site_y, occupied)
    raise ValueError(f"no coarse placement found for {node_id}")


def place_box_graph_aware(
    node_id: str,
    width: float,
    depth: float,
    floors: list[int],
    preferred: tuple[float, float],
    site_x: float,
    site_y: float,
    occupied: dict[int, list[tuple[int, int, int, int]]],
    placed: dict[str, dict[str, Any]],
    neighbors_by_node: dict[str, set[str]],
    room_type: str | None = None,
    candidate_scorer: tuple[torch.nn.Module, torch.device] | None = None,
    candidate_scorer_weight: float = 150.0,
    access_aware_placement: bool = False,
    blocked_contact_penalty: float = 1800.0,
    untargeted_access_contact_penalty: float = 120.0,
) -> tuple[float, float, float, float]:
    sx = int(round(site_x / VOXEL_MM))
    sy = int(round(site_y / VOXEL_MM))
    wc = max(1, min(sx, int(round(width / VOXEL_MM))))
    dc = max(1, min(sy, int(round(depth / VOXEL_MM))))
    preferred_x = int(round(preferred[0] * max(sx - wc, 0)))
    preferred_y = int(round(preferred[1] * max(sy - dc, 0)))
    target_neighbors = neighbors_by_node.get(node_id, set())
    floor_occupied_cells = {
        floor: sum(max(other[2] - other[0], 0) * max(other[3] - other[1], 0) for other in occupied[floor])
        for floor in occupied
    }
    candidates: list[tuple[float, int, int]] = []
    for x in range(0, max(sx - wc, 0) + 1):
        for y in range(0, max(sy - dc, 0) + 1):
            candidate = (x, y, x + wc, y + dc)
            if any(any(overlaps(candidate, other) for other in occupied[floor]) for floor in floors):
                continue
            score = abs(x - preferred_x) + abs(y - preferred_y)
            scored_neighbor = False
            for neighbor_id in target_neighbors:
                neighbor = placed.get(neighbor_id)
                if not neighbor or not share_floor(floors, neighbor["floors"]):
                    continue
                scored_neighbor = True
                neighbor_box = neighbor["box"]
                if touches(candidate, neighbor_box):
                    score -= 5000.0
                else:
                    score += 35.0 * grid_gap(candidate, neighbor_box) + 250.0
            if access_aware_placement and room_type:
                for other_id, other in placed.items():
                    if other_id in target_neighbors:
                        continue
                    if not share_floor(floors, other["floors"]):
                        continue
                    if not touches(candidate, other["box"]):
                        continue
                    relation = access_label_between_types(room_type, str(other.get("room_type", "")))
                    if relation == "blocked":
                        score += float(blocked_contact_penalty)
                    else:
                        score += float(untargeted_access_contact_penalty)
            if not scored_neighbor:
                center_x = x + wc * 0.5
                center_y = y + dc * 0.5
                score += 0.15 * (abs(center_x - sx * 0.5) + abs(center_y - sy * 0.5))
            if candidate_scorer and room_type:
                model, scorer_device = candidate_scorer
                features = candidate_feature_vector(
                    room_type=room_type,
                    floors=floors,
                    candidate=candidate,
                    preferred_xy=(preferred_x, preferred_y),
                    site_cells=(sx, sy),
                    placed=placed,
                    neighbors_by_node=neighbors_by_node,
                    node_id=node_id,
                    floor_occupied_cells=floor_occupied_cells,
                )
                with torch.no_grad():
                    learned_score = float(model(features.unsqueeze(0).to(scorer_device)).detach().cpu().item())
                score -= float(candidate_scorer_weight) * learned_score
            candidates.append((score, x, y))
    if candidates:
        _score, x, y = min(candidates)
        candidate = (x, y, x + wc, y + dc)
        for floor in floors:
            occupied[floor].append(candidate)
        placed[node_id] = {"box": candidate, "floors": list(floors), "room_type": room_type}
        return tuple(float(value * VOXEL_MM) for value in candidate)

    if wc > 1 or dc > 1:
        return place_box_graph_aware(
            node_id,
            (wc - 1) * VOXEL_MM,
            max(VOXEL_MM, (dc - 1) * VOXEL_MM),
            floors,
            preferred,
            site_x,
            site_y,
            occupied,
            placed,
            neighbors_by_node,
            room_type=room_type,
            candidate_scorer=candidate_scorer,
            candidate_scorer_weight=candidate_scorer_weight,
            access_aware_placement=access_aware_placement,
            blocked_contact_penalty=blocked_contact_penalty,
            untargeted_access_contact_penalty=untargeted_access_contact_penalty,
        )
    raise ValueError(f"no graph-aware coarse placement found for {node_id}")


def split_dummy_parts(
    group_id: str,
    room_type: str,
    box: tuple[float, float, float, float],
    floors: list[int],
    part_count: int,
) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = box
    z0 = min(FLOOR_Z[floor][0] for floor in floors)
    z1 = max(FLOOR_Z[floor][1] for floor in floors)
    count = max(1, min(part_count, MAX_PARTS))
    parts = []
    split_x = (x1 - x0) >= (y1 - y0)
    for index in range(count):
        if split_x:
            px0 = snap(x0 + (x1 - x0) * index / count)
            px1 = snap(x0 + (x1 - x0) * (index + 1) / count)
            py0, py1 = y0, y1
        else:
            px0, px1 = x0, x1
            py0 = snap(y0 + (y1 - y0) * index / count)
            py1 = snap(y0 + (y1 - y0) * (index + 1) / count)
        if px1 <= px0:
            px1 = px0 + VOXEL_MM
        if py1 <= py0:
            py1 = py0 + VOXEL_MM
        parts.append(
            {
                "id": f"{group_id}_seed_part_{index}",
                "functional_id": group_id,
                "type": room_type,
                "box_min": [float(px0), float(py0), float(z0)],
                "box_max": [float(px1), float(py1), float(z1)],
                "floors": list(floors),
                "floor": min(floors),
            }
        )
    return parts


def distribute_part_counts(counts: dict[str, int], part_counts: dict[str, int]) -> dict[str, list[int]]:
    output = {}
    for room_type, group_count in counts.items():
        total = max(group_count, int(part_counts.get(room_type, group_count)))
        base = total // max(group_count, 1)
        remainder = total % max(group_count, 1)
        output[room_type] = [base + (1 if index < remainder else 0) for index in range(group_count)]
    return output


def user_conditions(args: argparse.Namespace) -> tuple[float, float, dict[str, int], bool]:
    if args.case:
        case = TEST_CASES[args.case]
        return float(case["site"][0]), float(case["site"][1]), dict(case["rooms"]), False
    if args.site_x is None or args.site_y is None:
        raise ValueError("provide --case, or provide --site-x and --site-y")
    if args.rooms_file:
        counts = read_json(args.rooms_file)
    elif args.rooms_json:
        counts = json.loads(args.rooms_json)
    else:
        counts = {}
    return (
        float(args.site_x),
        float(args.site_y),
        {str(key): int(value) for key, value in counts.items() if int(value) >= 0},
        not bool(counts),
    )


def build_user_group_samples(
    site_x: float,
    site_y: float,
    counts: dict[str, int],
    part_counts: dict[str, int],
    topology: dict[str, Any],
    coarse_priors: dict[str, tuple[float, float, float, float]] | None = None,
    coarse_layout_strategy: str = "rule",
    candidate_scorer: tuple[torch.nn.Module, torch.device] | None = None,
    candidate_scorer_weight: float = 150.0,
    access_aware_placement: bool = False,
    blocked_contact_penalty: float = 1800.0,
    untargeted_access_contact_penalty: float = 120.0,
) -> tuple[list[GroupSample], list[dict[str, Any]], dict[str, int]]:
    positions = normalized_positions(topology)
    conditions = topology.get("evidence", {}).get("node_conditions", {})
    per_type_parts = distribute_part_counts(counts, part_counts)
    used_type_index = {room_type: 0 for room_type in ROOM_TYPES}
    occupied = {1: [], 2: []}
    placed: dict[str, dict[str, Any]] = {}
    neighbors_by_node = topology_neighbors(topology)
    samples: list[GroupSample] = []
    seed_rooms: list[dict[str, Any]] = []
    base_nodes = sorted(
        topology.get("nodes", []),
        key=lambda node: (
            min(floor_text_to_floors(str(node["type"]), node.get("floor", 1))),
            TYPE_PRIORITY.get(str(node["type"]), 99),
            str(node["id"]),
        ),
    )
    if coarse_layout_strategy in {"graph", "learned_graph"}:
        nodes = []
        remaining = {str(node["id"]): node for node in base_nodes}
        ordered_ids: set[str] = set()
        while remaining:
            next_id, next_node = min(
                remaining.items(),
                key=lambda item: (
                    -len(neighbors_by_node.get(item[0], set()) & ordered_ids),
                    min(floor_text_to_floors(str(item[1]["type"]), item[1].get("floor", 1))),
                    TYPE_PRIORITY.get(str(item[1]["type"]), 99),
                    -len(neighbors_by_node.get(item[0], set())),
                    item[0],
                ),
            )
            nodes.append(next_node)
            ordered_ids.add(next_id)
            del remaining[next_id]
    else:
        nodes = base_nodes
    for node in nodes:
        node_id = str(node["id"])
        room_type = str(node["type"])
        floors = floor_text_to_floors(room_type, node.get("floor", 1))
        area_ratio = float(conditions.get(node_id, {}).get("area_ratio", 0.04))
        if coarse_priors and node_id in coarse_priors:
            center_x, center_y, width_ratio, depth_ratio = coarse_priors[node_id]
            width = max(VOXEL_MM, snap(width_ratio * site_x))
            depth = max(VOXEL_MM, snap(depth_ratio * site_y))
            preferred = (
                clamp01((center_x * site_x - width * 0.5) / max(site_x - width, VOXEL_MM)),
                clamp01((center_y * site_y - depth * 0.5) / max(site_y - depth, VOXEL_MM)),
            )
        else:
            width, depth = desired_size(room_type, area_ratio, site_x, site_y)
            preferred = positions.get(node_id, (0.5, 0.5))
        if coarse_layout_strategy in {"graph", "learned_graph"}:
            x0, y0, x1, y1 = place_box_graph_aware(
                node_id,
                width,
                depth,
                floors,
                preferred,
                site_x,
                site_y,
                occupied,
                placed,
                neighbors_by_node,
                room_type=room_type,
                candidate_scorer=candidate_scorer,
                candidate_scorer_weight=candidate_scorer_weight,
                access_aware_placement=access_aware_placement,
                blocked_contact_penalty=blocked_contact_penalty,
                untargeted_access_contact_penalty=untargeted_access_contact_penalty,
            )
        else:
            x0, y0, x1, y1 = place_box(node_id, width, depth, floors, preferred, site_x, site_y, occupied)
            placed[node_id] = {
                "box": (
                    int(round(x0 / VOXEL_MM)),
                    int(round(y0 / VOXEL_MM)),
                    int(round(x1 / VOXEL_MM)),
                    int(round(y1 / VOXEL_MM)),
                ),
                "floors": list(floors),
                "room_type": room_type,
            }
        z0 = min(FLOOR_Z[floor][0] for floor in floors)
        z1 = max(FLOOR_Z[floor][1] for floor in floors)
        type_index = used_type_index.get(room_type, 0)
        used_type_index[room_type] = type_index + 1
        part_count = per_type_parts.get(room_type, [1] * max(counts.get(room_type, 1), 1))[type_index]
        dummy_parts = split_dummy_parts(node_id, room_type, (x0, y0, x1, y1), floors, part_count)
        seed_rooms.extend(dummy_parts)
        samples.append(
            GroupSample(
                house_id="user_phase24_bridge",
                group_id=node_id,
                room_type=room_type,
                site=(site_x, site_y),
                floors=floors,
                box_min=[float(x0), float(y0), float(z0)],
                box_max=[float(x1), float(y1), float(z1)],
                target_parts=dummy_parts,
            )
        )
    return samples, seed_rooms, used_type_index


def user_size_prior(
    room_type: str,
    area_ratio: float,
    site_x: float,
    site_y: float,
    part_count: int,
) -> tuple[float, float, float, float]:
    width, depth = desired_size(room_type, area_ratio, site_x, site_y)
    return (
        clamp01(area_ratio),
        clamp01(width / max(site_x, 1.0)),
        clamp01(depth / max(site_y, 1.0)),
        clamp01(part_count / max(MAX_PARTS, 1)),
    )


def user_position_prior(node_id: str, positions: dict[str, tuple[float, float]]) -> tuple[float, float, float, float]:
    center_x, center_y = positions.get(node_id, (0.5, 0.5))
    return (
        clamp01(center_x),
        clamp01(center_y),
        clamp01(round(clamp01(center_x) * 2.0) / 2.0),
        clamp01(round(clamp01(center_y) * 2.0) / 2.0),
    )


def predict_user_coarse_priors(
    checkpoint_path: Path,
    device: torch.device,
    site_x: float,
    site_y: float,
    counts: dict[str, int],
    part_counts: dict[str, int],
    topology: dict[str, Any],
) -> dict[str, tuple[float, float, float, float]]:
    model = load_coarse_layout_head(checkpoint_path, device)
    positions = normalized_positions(topology)
    conditions = topology.get("evidence", {}).get("node_conditions", {})
    per_type_parts = distribute_part_counts(counts, part_counts)
    nodes = sorted(
        topology.get("nodes", []),
        key=lambda node: (
            min(floor_text_to_floors(str(node["type"]), node.get("floor", 1))),
            TYPE_PRIORITY.get(str(node["type"]), 99),
            str(node["id"]),
        ),
    )
    type_counts: dict[str, int] = {}
    for node in nodes:
        room_type = str(node["type"])
        type_counts[room_type] = type_counts.get(room_type, 0) + 1
    seen_by_type = {room_type: 0 for room_type in ROOM_TYPES}
    priors: dict[str, tuple[float, float, float, float]] = {}
    for node in nodes:
        node_id = str(node["id"])
        room_type = str(node["type"])
        floors = floor_text_to_floors(room_type, node.get("floor", 1))
        type_index = seen_by_type.get(room_type, 0)
        seen_by_type[room_type] = type_index + 1
        part_count = per_type_parts.get(room_type, [1] * max(counts.get(room_type, 1), 1))[type_index]
        area_ratio = float(conditions.get(node_id, {}).get("area_ratio", 0.04))
        priors[node_id] = predict_coarse_layout_ratios(
            model,
            device,
            room_type,
            floors,
            (site_x, site_y),
            type_index,
            max(type_counts.get(room_type, 1), 1),
            len(nodes),
            user_size_prior(room_type, area_ratio, site_x, site_y, part_count),
            user_position_prior(node_id, positions),
        )
    return priors


def predict_user_graph_coarse_priors(
    checkpoint_path: Path,
    device: torch.device,
    site_x: float,
    site_y: float,
    counts: dict[str, int],
    part_counts: dict[str, int],
    topology: dict[str, Any],
) -> dict[str, tuple[float, float, float, float]]:
    model = load_graph_coarse_layout_model(checkpoint_path, device)
    positions = normalized_positions(topology)
    conditions = topology.get("evidence", {}).get("node_conditions", {})
    per_type_parts = distribute_part_counts(counts, part_counts)
    nodes = sorted(
        topology.get("nodes", []),
        key=lambda node: (
            str(node["type"]),
            str(node["id"]),
        ),
    )
    type_counts: dict[str, int] = {}
    for node in nodes:
        room_type = str(node["type"])
        type_counts[room_type] = type_counts.get(room_type, 0) + 1
    seen_by_type = {room_type: 0 for room_type in ROOM_TYPES}
    node_fields = []
    for node in nodes:
        node_id = str(node["id"])
        room_type = str(node["type"])
        floors = floor_text_to_floors(room_type, node.get("floor", 1))
        type_index = seen_by_type.get(room_type, 0)
        seen_by_type[room_type] = type_index + 1
        part_count = per_type_parts.get(room_type, [1] * max(counts.get(room_type, 1), 1))[type_index]
        area_ratio = float(conditions.get(node_id, {}).get("area_ratio", 0.04))
        node_fields.append(
            {
                "node_id": node_id,
                "room_type": room_type,
                "floors": floors,
                "site": (site_x, site_y),
                "type_index": type_index,
                "type_count": max(type_counts.get(room_type, 1), 1),
                "group_count": len(nodes),
                "size_prior": user_size_prior(room_type, area_ratio, site_x, site_y, part_count),
                "position_prior": user_position_prior(node_id, positions),
            }
        )
    edges = [(str(edge["source"]), str(edge["target"])) for edge in topology.get("edges", [])]
    return predict_graph_layout_ratios(model, device, node_fields, edges)


def learned_topology_sample_from_user_topology(
    topology: dict[str, Any],
    site_x: float,
    site_y: float,
    counts: dict[str, int],
    part_counts: dict[str, int],
) -> GraphTopologySample:
    positions = normalized_positions(topology)
    conditions = topology.get("evidence", {}).get("node_conditions", {})
    per_type_parts = distribute_part_counts(counts, part_counts)
    sorted_nodes = sorted(topology.get("nodes", []), key=lambda node: str(node["id"]))
    type_counts: dict[str, int] = {}
    for node in sorted_nodes:
        room_type = str(node["type"])
        type_counts[room_type] = type_counts.get(room_type, 0) + 1
    seen_by_type = {room_type: 0 for room_type in ROOM_TYPES}
    topology_nodes: dict[str, TopologyNode] = {}
    for node in sorted_nodes:
        node_id = str(node["id"])
        room_type = str(node["type"])
        floors = tuple(floor_text_to_floors(room_type, node.get("floor", 1)))
        type_index = seen_by_type.get(room_type, 0)
        seen_by_type[room_type] = type_index + 1
        area_ratio = float(conditions.get(node_id, {}).get("area_ratio", 0.04))
        part_count = per_type_parts.get(room_type, [1] * max(counts.get(room_type, 1), 1))[type_index]
        size_prior = user_size_prior(room_type, area_ratio, site_x, site_y, part_count)
        position_prior = user_position_prior(node_id, positions)
        width, depth = desired_size(room_type, area_ratio, site_x, site_y)
        topology_nodes[node_id] = TopologyNode(
            house_id="user_phase24_bridge",
            node_id=node_id,
            room_type=room_type,
            site=(site_x, site_y),
            floors=floors,
            box_min=(0.0, 0.0, min(FLOOR_Z[floor][0] for floor in floors)),
            box_max=(width, depth, max(FLOOR_Z[floor][1] for floor in floors)),
            size_prior=size_prior,
            position_prior=position_prior,
        )
    target_pairs = {
        tuple(sorted((str(edge["source"]), str(edge["target"]))))
        for edge in topology.get("edges", [])
    }
    node_ids = sorted(topology_nodes)
    pairs = []
    for left_index, left_id in enumerate(node_ids):
        for right_id in node_ids[left_index + 1 :]:
            label = 1.0 if tuple(sorted((left_id, right_id))) in target_pairs else 0.0
            pairs.append(
                PairSample(
                    house_id="user_phase24_bridge",
                    source=left_id,
                    target=right_id,
                    left=topology_nodes[left_id],
                    right=topology_nodes[right_id],
                    label=label,
                )
            )
    return GraphTopologySample(
        house_id="user_phase24_bridge",
        target_topology=topology,
        nodes=[topology_nodes[node_id] for node_id in node_ids],
        pairs=pairs,
    )


def predict_user_learned_topology(
    checkpoint_path: Path,
    device: torch.device,
    topology: dict[str, Any],
    site_x: float,
    site_y: float,
    counts: dict[str, int],
    part_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model, feature_mode, enforce_connectivity = load_graph_topology_generator(checkpoint_path, device)
    if feature_mode == "full":
        raise ValueError(
            "--topology-generator-checkpoint uses feature_mode=full, which requires target boxes "
            "and is not valid for raw user-condition generation"
        )
    sample = learned_topology_sample_from_user_topology(topology, site_x, site_y, counts, part_counts)
    learned, metrics = predict_graph(
        model,
        sample,
        feature_mode,
        device,
        enforce_connectivity=enforce_connectivity,
    )
    learned["seed"] = topology.get("seed")
    learned["evidence"] = {
        **topology.get("evidence", {}),
        "source": "learned_graph_topology_generator",
        "fallback_program_prior_source": topology.get("evidence", {}).get("source"),
        "topology_generator_checkpoint": str(checkpoint_path),
        "topology_generator_feature_mode": feature_mode,
        "topology_generator_metrics_against_prior_graph": metrics,
    }
    return learned, metrics


def topology_for_evaluation(topology: dict[str, Any]) -> dict[str, Any]:
    edges = [
        {
            "source": str(edge["source"]),
            "target": str(edge["target"]),
            "relation": str(edge.get("relation", "horizontal")),
        }
        for edge in topology.get("edges", [])
    ]
    required = sorted({tuple(sorted((edge["source"], edge["target"]))) for edge in edges})
    return {
        "schema": "graphspace_phase24_user_bridge_topology_v1",
        "seed": topology.get("seed"),
        "nodes": [
            {
                "id": str(node["id"]),
                "type": str(node["type"]),
                "floor": node.get("floor"),
                "floors": floor_text_to_floors(str(node["type"]), node.get("floor", 1)),
                "position": node.get("position", [0.0, 0.0]),
            }
            for node in topology.get("nodes", [])
        ],
        "edges": edges,
        "required_edges": [list(edge) for edge in required],
        "source": str(topology.get("source", "program_prior_user_bridge")),
        "evidence": topology.get("evidence", {}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    site_x, site_y, explicit_counts, infer_missing = user_conditions(args)
    use_whole_graph = bool(args.whole_graph_count_checkpoint or args.whole_graph_program_checkpoint)
    if use_whole_graph and not (args.whole_graph_count_checkpoint and args.whole_graph_program_checkpoint):
        raise ValueError(
            "--whole-graph-count-checkpoint and --whole-graph-program-checkpoint must be provided together"
        )
    program_prior = None if use_whole_graph else ProgramPrior(args.program_prior)
    topology_generator_metrics = None
    whole_graph_metadata = None
    if use_whole_graph:
        whole_graph_explicit_counts = (
            explicit_counts
            if infer_missing
            else {room_type: int(explicit_counts.get(room_type, 0)) for room_type in ROOM_TYPES}
        )
        raw_topology, counts, whole_graph_metadata = generate_whole_graph(
            args.whole_graph_count_checkpoint,
            args.whole_graph_program_checkpoint,
            site_x,
            site_y,
            whole_graph_explicit_counts,
            seed=args.seed,
            edge_threshold=args.whole_graph_edge_threshold,
            device=torch.device(args.device),
        )
        count_evidence = {
            "source": "learned_whole_heterogeneous_graph",
            "explicit_counts": explicit_counts,
            "raw_count_prediction": whole_graph_metadata["raw_count_prediction"],
        }
        part_counts = dict(counts)
        part_count_evidence = {
            "source": "learned_whole_heterogeneous_graph_default_one_part_per_group",
            "note": "Temporary Phase24 compatibility: each learned functional group is decoded as one part unless a later learned part-count head is provided.",
        }
        if args.topology_generator_checkpoint:
            learned_guidance_topology, topology_generator_metrics = predict_user_learned_topology(
                args.topology_generator_checkpoint,
                torch.device(args.device),
                raw_topology,
                site_x,
                site_y,
                counts,
                part_counts,
            )
            contains_edges = [
                edge
                for edge in raw_topology.get("heterogeneous_edges", [])
                if str(edge.get("edge_type")) == "contains"
            ]
            geometric_observed = list(raw_topology.get("geometric_contact_observed", []))
            guidance_edges = [
                {**edge, "edge_type": "guidance_relation"}
                for edge in learned_guidance_topology.get("edges", [])
            ]
            learned_guidance_topology.update(
                {
                    "schema": "graphspace_learned_whole_heterogeneous_guidance_topology_v1",
                    "source": "learned_whole_heterogeneous_graph_with_learned_guidance_relation",
                    "site": raw_topology.get("site"),
                    "heterogeneous_nodes": raw_topology.get("heterogeneous_nodes", []),
                    "geometric_contact_observed": geometric_observed,
                    "heterogeneous_edges": [
                        *contains_edges,
                        *guidance_edges,
                        *geometric_observed,
                    ],
                }
            )
            learned_guidance_topology["evidence"] = {
                **raw_topology.get("evidence", {}),
                **learned_guidance_topology.get("evidence", {}),
                "source": "learned_whole_heterogeneous_graph_with_learned_guidance_relation",
                "whole_graph_node_source": raw_topology.get("source"),
                "guidance_relation_source": "learned_graph_topology_generator",
                "guidance_relation_checkpoint": str(args.topology_generator_checkpoint),
                "geometric_contact_observed_count": len(geometric_observed),
            }
            raw_topology = learned_guidance_topology
    else:
        assert program_prior is not None
        neighbors = program_prior.neighbors(site_x, site_y)
        counts, count_evidence = program_prior.infer_counts(
            neighbors,
            args.seed,
            explicit_counts=explicit_counts,
            infer_missing=infer_missing,
        )
        part_counts, part_count_evidence = program_prior.infer_part_counts(counts, neighbors, args.seed)
        _model_graph, raw_topology = request_graph(counts, site_x, site_y, args.seed, program_prior)
        if args.topology_generator_checkpoint:
            raw_topology, topology_generator_metrics = predict_user_learned_topology(
                args.topology_generator_checkpoint,
                torch.device(args.device),
                raw_topology,
                site_x,
                site_y,
                counts,
                part_counts,
            )
    validate_request(site_x, site_y, counts)
    topology = topology_for_evaluation(raw_topology)
    topology["count_evidence"] = count_evidence
    topology["part_count_evidence"] = part_count_evidence
    if whole_graph_metadata:
        topology["whole_graph_metadata"] = whole_graph_metadata

    device = torch.device(args.device)
    if args.coarse_layout_strategy == "learned_graph":
        if not args.graph_coarse_layout_checkpoint:
            raise ValueError("--coarse-layout-strategy learned_graph requires --graph-coarse-layout-checkpoint")
        coarse_priors = predict_user_graph_coarse_priors(
            args.graph_coarse_layout_checkpoint,
            device,
            site_x,
            site_y,
            counts,
            part_counts,
            topology,
        )
    else:
        coarse_priors = (
            predict_user_coarse_priors(
            args.coarse_layout_checkpoint,
            device,
            site_x,
            site_y,
            counts,
            part_counts,
            topology,
            )
            if args.coarse_layout_checkpoint
            else None
        )
    candidate_scorer = (
        (load_candidate_scorer(args.candidate_scorer_checkpoint, device), device)
        if args.candidate_scorer_checkpoint
        else None
    )
    samples, seed_rooms, _used = build_user_group_samples(
        site_x,
        site_y,
        counts,
        part_counts,
        topology,
        coarse_priors,
        coarse_layout_strategy=args.coarse_layout_strategy,
        candidate_scorer=candidate_scorer,
        candidate_scorer_weight=float(args.candidate_scorer_weight),
        access_aware_placement=bool(args.access_aware_placement),
        blocked_contact_penalty=float(args.blocked_contact_penalty),
        untargeted_access_contact_penalty=float(args.untargeted_access_contact_penalty),
    )

    checkpoint = torch.load(args.decoder_checkpoint, map_location=device)
    feature_dim = int(checkpoint["config"]["feature_dim"])
    max_parts = int(checkpoint["config"]["max_parts"])
    if max_parts != int(args.max_parts):
        raise ValueError(f"checkpoint max_parts={max_parts}, got --max-parts={args.max_parts}")
    actual_feature_dim = int(feature_vector(samples[0], max_parts).numel())
    if actual_feature_dim != feature_dim:
        raise ValueError(f"checkpoint feature_dim={feature_dim}, generated feature_dim={actual_feature_dim}")
    model = MultiPartDecoder(feature_dim, max_parts).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    decoded_rooms: list[dict[str, Any]] = []
    occupied: list[dict[str, Any]] = []
    neighbors_by_group = group_neighbor_map(topology)
    with torch.no_grad():
        for sample in samples:
            pred = model(feature_vector(sample, max_parts).unsqueeze(0).to(device))[0]
            decoded_rooms.extend(
                decode_parts(
                    sample,
                    pred,
                    max_parts,
                    occupied=occupied,
                    target_neighbors=neighbors_by_group.get(sample.group_id, set()),
                )
            )

    report = layout_report("user_phase24_bridge", decoded_rooms, counts, (site_x, site_y), topology)
    decoded_rooms, report, overlap_repair = repair_overlaps(
        "user_phase24_bridge",
        decoded_rooms,
        {sample.group_id: sample for sample in samples},
        counts,
        (site_x, site_y),
        topology,
    )
    decoded_rooms, report, placement_search = topology_placement_search(
        "user_phase24_bridge",
        decoded_rooms,
        {sample.group_id: sample for sample in samples},
        counts,
        (site_x, site_y),
        topology,
        max_move_mm=float(args.max_topology_move_mm),
        expand_search_to_site=True,
        enable_linked_part_placement=True,
        enable_controlled_size_adjustment=True,
        max_size_adjustment_mm=float(args.max_size_adjustment_mm),
    )

    candidate = {
        "house_id": "user_phase24_bridge",
        "metadata": {
            "building_size": {"x": site_x, "y": site_y, "z": 6000.0},
            "stats": counts,
            "bridge_source": "phase24_user_conditions_adapter_v1",
            "coarse_layout_source": (
                "learned_graph_coarse_layout_model"
                if args.coarse_layout_strategy == "learned_graph"
                else "learned_coarse_layout_head"
                if coarse_priors
                else ("graph_aware_packer" if args.coarse_layout_strategy == "graph" else "rule_packer")
            ),
            "candidate_scorer": str(args.candidate_scorer_checkpoint) if args.candidate_scorer_checkpoint else None,
            "candidate_scorer_weight": float(args.candidate_scorer_weight),
            "access_aware_placement": bool(args.access_aware_placement),
            "blocked_contact_penalty": float(args.blocked_contact_penalty),
            "untargeted_access_contact_penalty": float(args.untargeted_access_contact_penalty),
            "topology_source": topology.get("source"),
            "topology_generator": (
                str(args.topology_generator_checkpoint) if args.topology_generator_checkpoint else None
            ),
            "whole_graph_count_checkpoint": (
                str(args.whole_graph_count_checkpoint) if args.whole_graph_count_checkpoint else None
            ),
            "whole_graph_program_checkpoint": (
                str(args.whole_graph_program_checkpoint) if args.whole_graph_program_checkpoint else None
            ),
        },
        "rooms": decoded_rooms,
    }
    request = {
        "site_x": site_x,
        "site_y": site_y,
        "seed": args.seed,
        "program_source": "training_data_knn" if infer_missing else "user_complete_or_partial",
        "requested_counts": explicit_counts,
        "functional_group_counts": counts,
        "part_counts": part_counts,
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "coarse_layout_checkpoint": str(args.coarse_layout_checkpoint) if args.coarse_layout_checkpoint else None,
        "graph_coarse_layout_checkpoint": (
            str(args.graph_coarse_layout_checkpoint) if args.graph_coarse_layout_checkpoint else None
        ),
        "candidate_scorer_checkpoint": (
            str(args.candidate_scorer_checkpoint) if args.candidate_scorer_checkpoint else None
        ),
        "candidate_scorer_weight": float(args.candidate_scorer_weight),
        "access_aware_placement": bool(args.access_aware_placement),
        "blocked_contact_penalty": float(args.blocked_contact_penalty),
        "untargeted_access_contact_penalty": float(args.untargeted_access_contact_penalty),
        "topology_generator_checkpoint": (
            str(args.topology_generator_checkpoint) if args.topology_generator_checkpoint else None
        ),
        "whole_graph_count_checkpoint": (
            str(args.whole_graph_count_checkpoint) if args.whole_graph_count_checkpoint else None
        ),
        "whole_graph_program_checkpoint": (
            str(args.whole_graph_program_checkpoint) if args.whole_graph_program_checkpoint else None
        ),
        "whole_graph_edge_threshold": args.whole_graph_edge_threshold,
        "coarse_layout_strategy": args.coarse_layout_strategy,
    }
    summary = {
        "schema": "graphspace_phase24_user_bridge_summary_v1",
        "warning": (
            "Experimental bridge only. It synthesizes coarse functional-group boxes "
            "before using the Phase24 decoder; it is not a trained end-to-end user generator."
        ),
        "request": request,
        "group_count": len(samples),
        "seed_part_count": len(seed_rooms),
        "coarse_layout_source": (
            "learned_graph_coarse_layout_model"
            if args.coarse_layout_strategy == "learned_graph"
            else "learned_coarse_layout_head"
            if coarse_priors
            else ("graph_aware_packer" if args.coarse_layout_strategy == "graph" else "rule_packer")
        ),
        "candidate_scorer_checkpoint": (
            str(args.candidate_scorer_checkpoint) if args.candidate_scorer_checkpoint else None
        ),
        "candidate_scorer_weight": float(args.candidate_scorer_weight),
        "access_aware_placement": bool(args.access_aware_placement),
        "blocked_contact_penalty": float(args.blocked_contact_penalty),
        "untargeted_access_contact_penalty": float(args.untargeted_access_contact_penalty),
        "topology_source": topology.get("source"),
        "topology_generator_checkpoint": (
            str(args.topology_generator_checkpoint) if args.topology_generator_checkpoint else None
        ),
        "topology_generator_metrics_against_prior_graph": topology_generator_metrics,
        "whole_graph_metadata": whole_graph_metadata,
        "rectangular_part_count": len(decoded_rooms),
        "p0_pass": report["p0"]["pass"],
        "p1_hard_geometry_pass": report["p1_spatial_organization"]["hard_geometry_pass"],
        "p1_spatial_organization_pass": report["p1_spatial_organization"]["spatial_organization_pass"],
        "topology": {
            "target_edge_count": report["p1_spatial_organization"]["target_topology"]["target_edge_count"],
            "realized_edge_count": report["p1_spatial_organization"]["target_topology"]["realized_edge_count"],
            "realization_rate": report["p1_spatial_organization"]["target_topology"]["realization_rate"],
        },
        "overlap_repair": overlap_repair,
        "topology_placement_search": placement_search,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "request.json", request)
    write_json(args.output_dir / "seed_layout.json", {"metadata": candidate["metadata"], "rooms": seed_rooms})
    write_json(args.output_dir / "topology.json", topology)
    write_json(args.output_dir / "generated_layout.json", candidate)
    write_json(args.output_dir / "evaluation.json", report)
    write_json(args.output_dir / "overlap_repair.json", overlap_repair)
    write_json(args.output_dir / "topology_placement_search.json", placement_search)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
