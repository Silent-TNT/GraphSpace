#!/usr/bin/env python3
"""Build a site-conditioned room-program prior from the training split only."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = ROOT / "data" / "phase1" / "split_v1.json"
DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_GRAPH_DIR = ROOT / "data" / "phase7_staged_spatial" / "samples"
DEFAULT_FUNCTIONAL_GROUP_DIR = ROOT / "data" / "phase10_functional_parts" / "samples"
DEFAULT_OUTPUT = ROOT / "data" / "phase8_program_prior" / "program_prior.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument(
        "--functional-group-dir",
        type=Path,
        default=DEFAULT_FUNCTIONAL_GROUP_DIR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def floor_signature(node: dict) -> str:
    if bool(node["floor_1"]) and bool(node["floor_2"]):
        return "1&2"
    return "1" if bool(node["floor_1"]) else "2"


def group_floor_signature(floors: list[int]) -> str:
    floor_set = {int(value) for value in floors}
    if floor_set == {1, 2}:
        return "1&2"
    return "2" if floor_set == {2} else "1"


def combine_lighting(values: list[str]) -> str:
    if "direct" in values:
        return "direct"
    if "indirect" in values:
        return "indirect"
    return "none"


def part_index(part_id: str) -> int | None:
    if part_id.startswith("room_"):
        try:
            return int(part_id.split("_", 1)[1])
        except ValueError:
            return None
    return None


def build_group_graph(graph: dict, grouped: dict | None) -> tuple[list[dict], list[list[int]], dict]:
    """Compress part-level graph nodes into inferred functional-group nodes."""
    if not grouped:
        nodes = [
            {
                "type": str(node["type"]),
                "floor": floor_signature(node),
                "area_ratio": float(node["target_area_ratio"]),
                "lighting_access": str(node.get("lighting_access", "none")),
                "lighting_priority": int(node.get("lighting_priority", 0)),
                "part_count": 1,
                "inference": "legacy_part_node",
            }
            for node in graph["nodes"]
        ]
        unique_edges = {}
        for left, right, relation in graph["edges"]:
            pair = tuple(sorted((int(left), int(right))))
            if pair[0] != pair[1]:
                unique_edges[pair] = int(relation)
        return (
            nodes,
            [
                [left, right, relation]
                for (left, right), relation in sorted(unique_edges.items())
            ],
            {"mode": "legacy_part_nodes", "compressed_edge_count": len(unique_edges)},
        )

    graph_nodes = list(graph["nodes"])
    part_to_graph_index = {
        str(room["part_id"]): index
        for index, room in enumerate(grouped.get("rooms", []))
        if index < len(graph_nodes)
    }
    groups = list(grouped.get("functional_groups", []))
    group_index = {
        str(group["functional_id"]): index for index, group in enumerate(groups)
    }
    part_to_group = {
        str(part_id): str(group["functional_id"])
        for group in groups
        for part_id in group.get("part_ids", [])
    }
    nodes = []
    for group in groups:
        part_ids = [str(value) for value in group.get("part_ids", [])]
        indices = [
            part_to_graph_index[part_id]
            for part_id in part_ids
            if part_id in part_to_graph_index
        ]
        part_nodes = [graph_nodes[index] for index in indices]
        area_ratio = sum(float(node.get("target_area_ratio", 0.0)) for node in part_nodes)
        lighting_values = [str(node.get("lighting_access", "none")) for node in part_nodes]
        nodes.append(
            {
                "type": str(group["type"]),
                "floor": group_floor_signature(list(group.get("floors", []))),
                "area_ratio": area_ratio,
                "lighting_access": combine_lighting(lighting_values),
                "lighting_priority": max(
                    [int(node.get("lighting_priority", 0)) for node in part_nodes] or [0]
                ),
                "part_count": int(group.get("part_count", len(part_ids))),
                "inference": str(group.get("inference", "unknown")),
            }
        )

    unique_edges = {}
    skipped_same_group = 0
    for left, right, relation in graph["edges"]:
        left_part = f"room_{int(left)}"
        right_part = f"room_{int(right)}"
        left_group = part_to_group.get(left_part)
        right_group = part_to_group.get(right_part)
        if left_group is None or right_group is None:
            continue
        left_index = group_index[left_group]
        right_index = group_index[right_group]
        if left_index == right_index:
            skipped_same_group += 1
            continue
        pair = tuple(sorted((left_index, right_index)))
        unique_edges[pair] = int(relation)
    return (
        nodes,
        [
            [left, right, relation]
            for (left, right), relation in sorted(unique_edges.items())
        ],
        {
            "mode": "functional_group_nodes",
            "part_node_count": len(graph_nodes),
            "functional_group_node_count": len(nodes),
            "compressed_edge_count": len(unique_edges),
            "skipped_same_group_part_edges": skipped_same_group,
        },
    )


def count_group_parts(grouped: dict | None, fallback_counts: dict[str, int]) -> tuple[dict[str, int], dict[str, int], dict[str, list[int]]]:
    if not grouped:
        counts = dict(fallback_counts)
        return counts, counts, {room_type: [1] * count for room_type, count in counts.items()}
    group_counts = {
        str(key): int(value)
        for key, value in grouped.get("stats", {}).get("functional_group_counts", {}).items()
    }
    part_counts = {
        str(key): int(value)
        for key, value in grouped.get("stats", {}).get("raw_part_counts", {}).items()
    }
    part_counts_per_group: dict[str, list[int]] = defaultdict(list)
    for group in grouped.get("functional_groups", []):
        part_counts_per_group[str(group["type"])].append(int(group.get("part_count", 1)))
    return group_counts, part_counts, dict(part_counts_per_group)


def aggregate_part_stats(houses: list[dict]) -> dict:
    totals = Counter()
    groups = Counter()
    multipart_groups = Counter()
    group_part_histogram: dict[str, Counter] = defaultdict(Counter)
    for house in houses:
        totals.update(house.get("part_counts", {}))
        groups.update(house.get("functional_group_counts", house.get("room_counts", {})))
        for room_type, values in house.get("part_counts_per_group", {}).items():
            for value in values:
                group_part_histogram[room_type][str(value)] += 1
                if int(value) > 1:
                    multipart_groups[room_type] += 1
    return {
        "part_counts": dict(sorted(totals.items())),
        "functional_group_counts": dict(sorted(groups.items())),
        "multipart_group_counts": dict(sorted(multipart_groups.items())),
        "part_count_per_group_histogram": {
            key: dict(sorted(value.items(), key=lambda item: int(item[0])))
            for key, value in sorted(group_part_histogram.items())
        },
    }


def main() -> None:
    args = parse_args()
    train_ids = [str(value) for value in read_json(args.split_path)["train"]]
    houses = []
    for house_id in train_ids:
        processed = read_json(args.processed_dir / f"{house_id}.json")
        graph = read_json(args.graph_dir / f"{house_id}.json")["graph"]
        grouped_path = args.functional_group_dir / f"{house_id}.json"
        grouped = read_json(grouped_path) if grouped_path.exists() else None
        raw_counts = {
            str(key): int(value)
            for key, value in processed["metadata"]["stats"].items()
        }
        group_counts, part_counts, part_counts_per_group = count_group_parts(
            grouped,
            raw_counts,
        )
        nodes, edges, group_graph_stats = build_group_graph(graph, grouped)
        building = processed["metadata"]["building_size"]
        houses.append(
            {
                "house_id": house_id,
                "site_x": float(building["x"]),
                "site_y": float(building["y"]),
                "room_counts": group_counts,
                "functional_group_counts": group_counts,
                "part_counts": part_counts,
                "raw_part_counts": raw_counts,
                "part_counts_per_group": part_counts_per_group,
                "nodes": nodes,
                "edges": edges,
                "group_graph_stats": group_graph_stats,
            }
        )
    payload = {
        "schema": "graphspace_v6_group_program_prior_v1",
        "source_split": "train",
        "split_path": str(args.split_path.relative_to(ROOT)),
        "functional_group_dir": str(args.functional_group_dir.relative_to(ROOT)),
        "house_count": len(houses),
        "relation_types": {"horizontal_contact": 0, "vertical_contact": 1},
        "topology_semantics": (
            "Functional-group face contact learned from inferred group layouts; "
            "it is not door or guaranteed circulation connectivity. Group labels "
            "come from Phase10 inferred functional_id when available."
        ),
        "count_semantics": {
            "room_counts": "functional group counts, kept for ProgramPrior compatibility",
            "functional_group_counts": "number of functional nodes by type",
            "part_counts": "number of rectangular parts by type",
        },
        "part_statistics": aggregate_part_stats(houses),
        "houses": houses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "house_count": len(houses)}))


if __name__ == "__main__":
    main()
