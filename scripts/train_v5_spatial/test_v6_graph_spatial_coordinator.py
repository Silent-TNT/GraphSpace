from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.train_v5_spatial.v6_graph_spatial_coordinator import coordinated_repair_house
from scripts.train_v5_spatial.v6_multipart_decoder import write_json


class V6GraphSpatialCoordinatorTest(unittest.TestCase):
    def test_component_move_repairs_missing_edge_while_preserving_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            house_dir = root / "input" / "house_test"
            house_dir.mkdir(parents=True)
            layout = {
                "house_id": "house_test",
                "metadata": {"building_size": {"x": 1800.0, "y": 2400.0, "z": 6000.0}},
                "rooms": [
                    {
                        "id": "corridor_0_part_0",
                        "functional_id": "corridor_0",
                        "type": "corridor",
                        "floor": 1,
                        "floors": [1],
                        "box_min": [0.0, 0.0, 0.0],
                        "box_max": [600.0, 1200.0, 3000.0],
                    },
                    {
                        "id": "entryway_0_part_0",
                        "functional_id": "entryway_0",
                        "type": "entryway",
                        "floor": 1,
                        "floors": [1],
                        "box_min": [0.0, 1200.0, 0.0],
                        "box_max": [600.0, 2400.0, 3000.0],
                    },
                    {
                        "id": "living_room_0_part_0",
                        "functional_id": "living_room_0",
                        "type": "living_room",
                        "floor": 1,
                        "floors": [1],
                        "box_min": [1200.0, 0.0, 0.0],
                        "box_max": [1800.0, 1200.0, 3000.0],
                    },
                ],
            }
            topology = {
                "schema": "test",
                "nodes": [
                    {"id": "corridor_0", "type": "corridor", "floor": 1, "floors": [1]},
                    {"id": "entryway_0", "type": "entryway", "floor": 1, "floors": [1]},
                    {"id": "living_room_0", "type": "living_room", "floor": 1, "floors": [1]},
                ],
                "edges": [
                    {"source": "corridor_0", "target": "entryway_0", "relation": "horizontal"},
                    {"source": "corridor_0", "target": "living_room_0", "relation": "horizontal"},
                ],
                "required_edges": [
                    ["corridor_0", "entryway_0"],
                    ["corridor_0", "living_room_0"],
                ],
            }
            evaluation = {
                "requested_counts": {
                    "corridor": 1,
                    "entryway": 1,
                    "living_room": 1,
                    "dining_room": 1,
                }
            }
            # Add dining_room to satisfy the project-level mandatory type check
            # without affecting the synthetic component under test.
            layout["rooms"].append(
                {
                    "id": "dining_room_0_part_0",
                    "functional_id": "dining_room_0",
                    "type": "dining_room",
                    "floor": 2,
                    "floors": [2],
                    "box_min": [0.0, 0.0, 3000.0],
                    "box_max": [600.0, 1200.0, 6000.0],
                }
            )
            topology["nodes"].append(
                {"id": "dining_room_0", "type": "dining_room", "floor": 2, "floors": [2]}
            )
            write_json(house_dir / "generated_layout.json", layout)
            write_json(house_dir / "topology.json", topology)
            write_json(house_dir / "evaluation.json", evaluation)

            report = coordinated_repair_house(
                house_dir,
                root / "output" / "house_test",
                max_iterations=2,
                max_component_size=2,
                candidate_limit=16,
            )

            self.assertTrue(report["p0_pass"])
            self.assertEqual(report["initial_topology"]["realized_edge_count"], 1)
            self.assertEqual(report["final_topology"]["realized_edge_count"], 2)
            self.assertEqual(report["accepted_move_count"], 1)


if __name__ == "__main__":
    unittest.main()
