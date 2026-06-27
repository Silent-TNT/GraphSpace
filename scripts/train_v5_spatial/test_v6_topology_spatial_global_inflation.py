from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.train_v5_spatial.v6_multipart_decoder import write_json
from scripts.train_v5_spatial.v6_topology_spatial_global_inflation import decode_house


class V6TopologySpatialGlobalInflationTest(unittest.TestCase):
    def test_global_inflation_prioritizes_shared_topology_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "house_test.json"
            payload = {
                "house_id": "house_test",
                "metadata": {"building_size": {"x": 1800.0, "y": 900.0, "z": 6000.0}},
                "functional_groups": [
                    {"functional_id": "stairs_0", "type": "stairs", "floors": [1, 2]},
                    {"functional_id": "living_room_0", "type": "living_room", "floors": [1]},
                    {"functional_id": "dining_room_0", "type": "dining_room", "floors": [1]},
                ],
                "rooms": [
                    {
                        "id": "stairs_0_part_0",
                        "functional_id": "stairs_0",
                        "type": "stairs",
                        "floor": 1,
                        "floors": [1, 2],
                        "box_min": [0.0, 0.0, 0.0],
                        "box_max": [300.0, 300.0, 6000.0],
                    },
                    {
                        "id": "living_room_0_part_0",
                        "functional_id": "living_room_0",
                        "type": "living_room",
                        "floor": 1,
                        "floors": [1],
                        "box_min": [300.0, 0.0, 0.0],
                        "box_max": [900.0, 900.0, 3000.0],
                    },
                    {
                        "id": "dining_room_0_part_0",
                        "functional_id": "dining_room_0",
                        "type": "dining_room",
                        "floor": 1,
                        "floors": [1],
                        "box_min": [900.0, 0.0, 0.0],
                        "box_max": [1800.0, 900.0, 3000.0],
                    },
                ],
            }
            write_json(source, payload)

            report = decode_house(
                source,
                root / "out" / "house_test",
                fill_ratio=0.55,
                max_iterations=200,
                paired_edge_passes=2,
            )

            self.assertTrue(report["p0_pass"])
            self.assertTrue(report["p1_spatial_organization_pass"])
            self.assertGreater(report["topology"]["realized_edge_count"], 0)


if __name__ == "__main__":
    unittest.main()
