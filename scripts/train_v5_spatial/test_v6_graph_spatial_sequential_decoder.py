from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.train_v5_spatial.v6_graph_spatial_sequential_decoder import decode_house
from scripts.train_v5_spatial.v6_multipart_decoder import write_json


class V6GraphSpatialSequentialDecoderTest(unittest.TestCase):
    def test_decodes_simple_chain_from_empty_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "house_test.json"
            payload = {
                "house_id": "house_test",
                "metadata": {"building_size": {"x": 1800.0, "y": 900.0, "z": 6000.0}},
                "functional_groups": [
                    {"functional_id": "entryway_0", "type": "entryway", "floors": [1]},
                    {"functional_id": "corridor_0", "type": "corridor", "floors": [1]},
                    {"functional_id": "dining_room_0", "type": "dining_room", "floors": [1]},
                ],
                "rooms": [
                    {
                        "id": "entryway_0_part_0",
                        "functional_id": "entryway_0",
                        "type": "entryway",
                        "floor": 1,
                        "floors": [1],
                        "box_min": [0.0, 0.0, 0.0],
                        "box_max": [600.0, 900.0, 3000.0],
                    },
                    {
                        "id": "corridor_0_part_0",
                        "functional_id": "corridor_0",
                        "type": "corridor",
                        "floor": 1,
                        "floors": [1],
                        "box_min": [600.0, 0.0, 0.0],
                        "box_max": [1200.0, 900.0, 3000.0],
                    },
                    {
                        "id": "dining_room_0_part_0",
                        "functional_id": "dining_room_0",
                        "type": "dining_room",
                        "floor": 1,
                        "floors": [1],
                        "box_min": [1200.0, 0.0, 0.0],
                        "box_max": [1800.0, 900.0, 3000.0],
                    },
                ],
            }
            write_json(source, payload)

            report = decode_house(source, root / "out" / "house_test", candidate_limit=32)

            self.assertTrue(report["p0_pass"])
            self.assertTrue(report["p1_spatial_organization_pass"])
            self.assertEqual(report["topology"]["realized_edge_count"], 2)
            self.assertEqual(report["topology"]["target_edge_count"], 2)


if __name__ == "__main__":
    unittest.main()
