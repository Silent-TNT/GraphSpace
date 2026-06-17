#!/usr/bin/env python3
"""Build a site-conditioned room-program prior from the training split only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = ROOT / "data" / "phase1" / "split_v1.json"
DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_GRAPH_DIR = ROOT / "data" / "phase7_staged_spatial" / "samples"
DEFAULT_OUTPUT = ROOT / "data" / "phase8_program_prior" / "program_prior.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def floor_signature(node: dict) -> str:
    if bool(node["floor_1"]) and bool(node["floor_2"]):
        return "1&2"
    return "1" if bool(node["floor_1"]) else "2"


def main() -> None:
    args = parse_args()
    train_ids = [str(value) for value in read_json(args.split_path)["train"]]
    houses = []
    for house_id in train_ids:
        processed = read_json(args.processed_dir / f"{house_id}.json")
        graph = read_json(args.graph_dir / f"{house_id}.json")["graph"]
        nodes = [
            {
                "type": str(node["type"]),
                "floor": floor_signature(node),
                "area_ratio": float(node["target_area_ratio"]),
                "lighting_access": str(node.get("lighting_access", "none")),
                "lighting_priority": int(node.get("lighting_priority", 0)),
            }
            for node in graph["nodes"]
        ]
        unique_edges = {}
        for left, right, relation in graph["edges"]:
            pair = tuple(sorted((int(left), int(right))))
            if pair[0] != pair[1]:
                unique_edges[pair] = int(relation)
        building = processed["metadata"]["building_size"]
        houses.append(
            {
                "house_id": house_id,
                "site_x": float(building["x"]),
                "site_y": float(building["y"]),
                "room_counts": {
                    str(key): int(value)
                    for key, value in processed["metadata"]["stats"].items()
                },
                "nodes": nodes,
                "edges": [
                    [left, right, relation]
                    for (left, right), relation in sorted(unique_edges.items())
                ],
            }
        )
    payload = {
        "schema": "graphspace_v5_program_prior_v1",
        "source_split": "train",
        "split_path": str(args.split_path.relative_to(ROOT)),
        "house_count": len(houses),
        "relation_types": {"horizontal_contact": 0, "vertical_contact": 1},
        "topology_semantics": (
            "Geometric face contact learned from training layouts; "
            "it is not door or guaranteed circulation connectivity."
        ),
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
