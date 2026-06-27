from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_from_user_conditions import (  # noqa: E402
    edge_quality,
    expand_functional_parts,
    is_trunk_edge,
    placement_order,
    repair_priority_edges,
    repair_missing_rooms,
    trunk_edge_scope,
)


def node(token: str, room_type: str, area_ratio: float = 0.02) -> dict:
    return {
        "instance_token": token,
        "type": room_type,
        "floor_1": 1,
        "floor_2": 0,
        "target_area_ratio": area_ratio,
    }


def floor_node(
    token: str,
    room_type: str,
    floor_1: int,
    floor_2: int,
    area_ratio: float = 0.02,
) -> dict:
    item = node(token, room_type, area_ratio)
    item["floor_1"] = floor_1
    item["floor_2"] = floor_2
    return item


class UserGenerationRepairTest(unittest.TestCase):
    def test_trunk_rooms_are_placed_before_bedroom_branches(self) -> None:
        nodes = [
            node("bedroom_0", "bedroom"),
            node("dining_room_0", "dining_room"),
            node("living_room_0", "living_room"),
            node("kitchen_0", "kitchen"),
        ]
        order = placement_order(nodes)
        ordered_types = [nodes[index]["type"] for index in order]
        self.assertLess(ordered_types.index("living_room"), ordered_types.index("bedroom"))
        self.assertLess(ordered_types.index("dining_room"), ordered_types.index("bedroom"))
        self.assertLess(ordered_types.index("kitchen"), ordered_types.index("bedroom"))

    def test_final_repair_places_missing_core_room_when_space_exists(self) -> None:
        model_graph = {
            "nodes": [
                node("living_room_0", "living_room"),
                node("dining_room_0", "dining_room"),
            ],
            "edges": [[0, 1, 0], [1, 0, 0]],
        }
        site_cells = [20, 10]
        building = np.ones((2, 20, 10), dtype=bool)
        occupied = np.zeros_like(building, dtype=np.uint8)
        occupied[0, 0:8, 0:10] = 1
        predictions = {0: (0, 0, 8, 10)}
        predicted_floors = {0: [1]}
        failures = ["dining_room_0"]
        remaining, repairs = repair_missing_rooms(
            model_graph,
            site_cells,
            building,
            occupied,
            predictions,
            predicted_floors,
            {0: [(1, 0, True)], 1: [(0, 0, True)]},
            {1: np.asarray([0.75, 0.5, 0.25, 0.8], dtype=np.float32)},
            failures,
        )
        self.assertEqual(remaining, [])
        self.assertIn(1, predictions)
        self.assertEqual(repairs[0]["node"], "dining_room_0")

    def test_final_repair_places_branch_after_core_priority(self) -> None:
        model_graph = {
            "nodes": [
                node("dining_room_0", "dining_room"),
                node("bedroom_0", "bedroom"),
            ],
            "edges": [[0, 1, 0], [1, 0, 0]],
        }
        site_cells = [20, 10]
        building = np.ones((2, 20, 10), dtype=bool)
        occupied = np.zeros_like(building, dtype=np.uint8)
        failures = ["dining_room_0", "bedroom_0"]
        remaining, repairs = repair_missing_rooms(
            model_graph,
            site_cells,
            building,
            occupied,
            {},
            {},
            {0: [(1, 0, True)], 1: [(0, 0, True)]},
            {
                0: np.asarray([0.25, 0.5, 0.2, 0.8], dtype=np.float32),
                1: np.asarray([0.75, 0.5, 0.2, 0.8], dtype=np.float32),
            },
            failures,
        )
        self.assertEqual(remaining, [])
        self.assertEqual([item["node"] for item in repairs], ["dining_room_0", "bedroom_0"])
        self.assertEqual(repairs[0]["details"]["repair_priority"], "core_trunk")
        self.assertEqual(repairs[1]["details"]["repair_priority"], "branch")

    def test_priority_edge_repair_moves_trunk_room_to_contact(self) -> None:
        model_graph = {
            "nodes": [
                node("living_room_0", "living_room"),
                node("dining_room_0", "dining_room"),
            ],
            "edges": [[0, 1, 0], [1, 0, 0]],
            "required_edges": [],
        }
        site_cells = [20, 10]
        building = np.ones((2, 20, 10), dtype=bool)
        occupied = np.zeros_like(building, dtype=np.uint8)
        predictions = {0: (0, 0, 5, 5), 1: (15, 0, 20, 5)}
        predicted_floors = {0: [1], 1: [1]}
        occupied[0, 0:5, 0:5] = 1
        occupied[0, 15:20, 0:5] = 1
        repairs = repair_priority_edges(
            model_graph,
            site_cells,
            building,
            occupied,
            predictions,
            predicted_floors,
            {0: [(1, 0, False)], 1: [(0, 0, False)]},
            {1: np.asarray([0.8, 0.25, 0.25, 0.5], dtype=np.float32)},
            placement_order(model_graph["nodes"]),
            set(),
        )
        self.assertTrue(any(item["trunk"] and item["repaired"] for item in repairs))
        self.assertGreater(edge_quality(0, 1, 0, predictions, predicted_floors), 0.0)

    def test_second_floor_private_spine_edges_are_trunk(self) -> None:
        nodes = [
            floor_node("corridor_0", "corridor", 0, 1),
            floor_node("bedroom_0", "bedroom", 0, 1),
            floor_node("bathroom_0", "bathroom", 0, 1),
        ]
        self.assertTrue(is_trunk_edge(nodes, 0, 1))
        self.assertTrue(is_trunk_edge(nodes, 0, 2))
        self.assertEqual(trunk_edge_scope(nodes, 0, 1), "floor_2")

    def test_first_floor_bedroom_corridor_edge_is_not_private_trunk(self) -> None:
        nodes = [
            floor_node("corridor_0", "corridor", 1, 0),
            floor_node("bedroom_0", "bedroom", 1, 0),
        ]
        self.assertFalse(is_trunk_edge(nodes, 0, 1))
        self.assertIsNone(trunk_edge_scope(nodes, 0, 1))

    def test_rule_expander_splits_corridor_into_functional_parts(self) -> None:
        rooms = [
            {
                "id": "corridor_0",
                "type": "corridor",
                "floor": 1,
                "floors": [1],
                "box_min": [0.0, 0.0, 0.0],
                "box_max": [3600.0, 1200.0, 3000.0],
            }
        ]
        expanded, report = expand_functional_parts(
            rooms,
            {"corridor": 1},
            {"corridor": 3},
        )
        self.assertEqual(len(expanded), 3)
        self.assertEqual(report["expanded_group_count"], 1)
        self.assertEqual({part["functional_id"] for part in expanded}, {"corridor_0"})
        self.assertEqual([part["id"] for part in expanded], [
            "corridor_0_part_0",
            "corridor_0_part_1",
            "corridor_0_part_2",
        ])
        self.assertEqual(expanded[0]["box_max"][0], expanded[1]["box_min"][0])
        self.assertEqual(expanded[1]["box_max"][0], expanded[2]["box_min"][0])

    def test_rule_expander_does_not_split_non_expandable_room_type(self) -> None:
        rooms = [
            {
                "id": "bedroom_0",
                "type": "bedroom",
                "floor": 2,
                "floors": [2],
                "box_min": [0.0, 0.0, 3000.0],
                "box_max": [3600.0, 3600.0, 6000.0],
            }
        ]
        expanded, report = expand_functional_parts(
            rooms,
            {"bedroom": 1},
            {"bedroom": 3},
        )
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["id"], "bedroom_0")
        self.assertEqual(expanded[0]["functional_id"], "bedroom_0")
        self.assertEqual(report["expanded_group_count"], 0)


if __name__ == "__main__":
    unittest.main()
