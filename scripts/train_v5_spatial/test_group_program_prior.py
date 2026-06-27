from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_program_prior import build_group_graph  # noqa: E402
from program_prior import ProgramPrior  # noqa: E402


class GroupProgramPriorTests(unittest.TestCase):
    def test_build_group_graph_compresses_part_edges(self) -> None:
        graph = {
            "nodes": [
                {
                    "type": "corridor",
                    "floor_1": 1,
                    "floor_2": 0,
                    "target_area_ratio": 0.02,
                    "lighting_access": "none",
                    "lighting_priority": 0,
                },
                {
                    "type": "corridor",
                    "floor_1": 1,
                    "floor_2": 0,
                    "target_area_ratio": 0.03,
                    "lighting_access": "none",
                    "lighting_priority": 0,
                },
                {
                    "type": "living_room",
                    "floor_1": 1,
                    "floor_2": 0,
                    "target_area_ratio": 0.12,
                    "lighting_access": "direct",
                    "lighting_priority": 10,
                },
            ],
            "edges": [[0, 1, 0], [1, 0, 0], [1, 2, 0], [2, 1, 0]],
        }
        grouped = {
            "rooms": [
                {"part_id": "room_0", "functional_id": "corridor_0"},
                {"part_id": "room_1", "functional_id": "corridor_0"},
                {"part_id": "room_2", "functional_id": "living_room_0"},
            ],
            "functional_groups": [
                {
                    "functional_id": "corridor_0",
                    "type": "corridor",
                    "part_ids": ["room_0", "room_1"],
                    "part_count": 2,
                    "floors": [1],
                    "inference": "same_type_adjacent_component",
                },
                {
                    "functional_id": "living_room_0",
                    "type": "living_room",
                    "part_ids": ["room_2"],
                    "part_count": 1,
                    "floors": [1],
                    "inference": "groupable_singleton",
                },
            ],
        }

        nodes, edges, stats = build_group_graph(graph, grouped)

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0]["part_count"], 2)
        self.assertEqual(nodes[0]["area_ratio"], 0.05)
        self.assertEqual(edges, [[0, 1, 0]])
        self.assertEqual(stats["mode"], "functional_group_nodes")
        self.assertEqual(stats["skipped_same_group_part_edges"], 2)

    def test_infer_part_counts_uses_group_to_part_ratio(self) -> None:
        payload = {
            "schema": "graphspace_v6_group_program_prior_v1",
            "source_split": "train",
            "houses": [
                {
                    "house_id": "donor",
                    "site_x": 12000,
                    "site_y": 12000,
                    "room_counts": {"corridor": 2, "living_room": 1},
                    "functional_group_counts": {"corridor": 2, "living_room": 1},
                    "part_counts": {"corridor": 5, "living_room": 1},
                    "nodes": [],
                    "edges": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program_prior.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            prior = ProgramPrior(path)
            neighbors = prior.neighbors(12000, 12000, count=1)
            part_counts, evidence = prior.infer_part_counts(
                {"corridor": 2, "living_room": 1},
                neighbors,
                seed=7,
            )

        self.assertEqual(part_counts["corridor"], 5)
        self.assertEqual(part_counts["living_room"], 1)
        self.assertEqual(evidence["source"], "phase10_group_to_part_prior")


if __name__ == "__main__":
    unittest.main()
