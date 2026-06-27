from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.data_phase4.evaluate_candidates import evaluate_candidate
from scripts.train_v5_spatial.v6_inflation_topology_repair import repair_house
from scripts.train_v5_spatial.v6_multipart_decoder import write_json


class V6InflationTopologyRepairTest(unittest.TestCase):
    def test_bridges_empty_cell_between_missing_topology_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            house_dir = root / "input" / "house_test"
            house_dir.mkdir(parents=True)
            site = (900.0, 300.0)
            topology = {
                "nodes": [
                    {"id": "dining_room_0", "type": "dining_room"},
                    {"id": "living_room_0", "type": "living_room"},
                ],
                "edges": [
                    {
                        "source": "dining_room_0",
                        "target": "living_room_0",
                        "type": "horizontal",
                        "required": True,
                    }
                ],
            }
            rooms = [
                {
                    "id": "dining_room_0_cell_1_0",
                    "functional_id": "dining_room_0",
                    "type": "dining_room",
                    "floor": 1,
                    "floors": [1],
                    "box_min": [0.0, 0.0, 0.0],
                    "box_max": [300.0, 300.0, 3000.0],
                },
                {
                    "id": "living_room_0_cell_1_0",
                    "functional_id": "living_room_0",
                    "type": "living_room",
                    "floor": 1,
                    "floors": [1],
                    "box_min": [600.0, 0.0, 0.0],
                    "box_max": [900.0, 300.0, 3000.0],
                },
            ]
            counts = {"dining_room": 1, "living_room": 1}
            initial_eval, _ = evaluate_candidate("house_test", rooms, counts, site, topology=topology)
            self.assertFalse(
                initial_eval["p1_spatial_organization"]["target_topology"]["edges"][0]["realized_in_dual"]
            )
            layout = {
                "house_id": "house_test",
                "metadata": {"building_size": {"x": site[0], "y": site[1], "z": 6000.0}},
                "rooms": rooms,
            }
            write_json(house_dir / "generated_layout.json", layout)
            write_json(house_dir / "topology.json", topology)
            write_json(house_dir / "evaluation.json", initial_eval)

            summary = repair_house(house_dir, root / "out" / "house_test", max_iterations=4, max_endpoint_pairs=8)

            self.assertTrue(summary["p0_pass"])
            self.assertEqual(summary["accepted_move_count"], 1)
            self.assertEqual(summary["initial_topology"]["realized_edge_count"], 0)
            self.assertEqual(summary["final_topology"]["realized_edge_count"], 1)


if __name__ == "__main__":
    unittest.main()
