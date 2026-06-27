from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_topology_dual import build_report  # noqa: E402


class TopologyDualReportTest(unittest.TestCase):
    def test_report_compares_target_edges_and_voxel_assignment(self) -> None:
        topology = {
            "nodes": [
                {"id": "living_room_0", "type": "living_room", "floor": "1"},
                {"id": "dining_room_0", "type": "dining_room", "floor": "1"},
                {"id": "stairs_0", "type": "stairs", "floor": "1-2"},
            ],
            "edges": [
                {
                    "source": "living_room_0",
                    "target": "dining_room_0",
                    "relation": "horizontal",
                },
                {
                    "source": "stairs_0",
                    "target": "living_room_0",
                    "relation": "vertical",
                },
            ],
            "required_edges": [
                ["living_room_0", "dining_room_0"],
                ["stairs_0", "living_room_0"],
            ],
        }
        layout = {
            "metadata": {"building_size": {"x": 6000.0, "y": 3000.0, "z": 6000.0}},
            "rooms": [
                {
                    "id": "living_room_0",
                    "type": "living_room",
                    "floors": [1],
                    "box_min": [0.0, 0.0, 0.0],
                    "box_max": [3000.0, 3000.0, 3000.0],
                },
                {
                    "id": "dining_room_0",
                    "type": "dining_room",
                    "floors": [1],
                    "box_min": [3000.0, 0.0, 0.0],
                    "box_max": [6000.0, 3000.0, 3000.0],
                },
                {
                    "id": "stairs_0",
                    "type": "stairs",
                    "floors": [1, 2],
                    "box_min": [0.0, 0.0, 0.0],
                    "box_max": [1500.0, 1500.0, 6000.0],
                },
            ],
        }
        report = build_report(topology, layout)
        self.assertEqual(report["heterogeneous_topology"]["room_node_count"], 3)
        self.assertEqual(report["target_vs_realized"]["required_edge_count"], 2)
        self.assertEqual(report["target_vs_realized"]["required_realized_edge_count"], 2)
        self.assertGreater(report["realized_planar_dual"]["horizontal_edge_count"], 0)
        self.assertGreater(report["realized_planar_dual"]["vertical_overlap_edge_count"], 0)
        self.assertGreater(report["voxel_assignment"]["assigned_voxel_count"], 0)

    def test_multipart_functional_group_realizes_target_edge(self) -> None:
        topology = {
            "nodes": [
                {"id": "living_room_0", "type": "living_room", "floor": "1"},
                {"id": "corridor_0", "type": "corridor", "floor": "1"},
                {"id": "bedroom_0", "type": "bedroom", "floor": "1"},
            ],
            "edges": [
                {
                    "source": "living_room_0",
                    "target": "corridor_0",
                    "relation": "horizontal",
                },
                {
                    "source": "corridor_0",
                    "target": "bedroom_0",
                    "relation": "horizontal",
                },
            ],
            "required_edges": [
                ["living_room_0", "corridor_0"],
                ["corridor_0", "bedroom_0"],
            ],
        }
        layout = {
            "metadata": {"building_size": {"x": 9000.0, "y": 3000.0, "z": 6000.0}},
            "rooms": [
                {
                    "id": "living_room_0",
                    "type": "living_room",
                    "floors": [1],
                    "box_min": [0.0, 0.0, 0.0],
                    "box_max": [3000.0, 3000.0, 3000.0],
                },
                {
                    "id": "corridor_0_part_0",
                    "functional_id": "corridor_0",
                    "type": "corridor",
                    "floors": [1],
                    "box_min": [3000.0, 0.0, 0.0],
                    "box_max": [6000.0, 1500.0, 3000.0],
                },
                {
                    "id": "corridor_0_part_1",
                    "functional_id": "corridor_0",
                    "type": "corridor",
                    "floors": [1],
                    "box_min": [6000.0, 0.0, 0.0],
                    "box_max": [7500.0, 3000.0, 3000.0],
                },
                {
                    "id": "bedroom_0",
                    "type": "bedroom",
                    "floors": [1],
                    "box_min": [7500.0, 0.0, 0.0],
                    "box_max": [9000.0, 3000.0, 3000.0],
                },
            ],
        }
        report = build_report(topology, layout)
        self.assertEqual(report["multipart_groups"]["part_count"], 4)
        self.assertEqual(report["multipart_groups"]["functional_group_count"], 3)
        self.assertEqual(
            report["multipart_groups"]["groups_with_multiple_parts"]["corridor_0"],
            2,
        )
        self.assertEqual(report["target_vs_realized"]["required_realized_edge_count"], 2)


if __name__ == "__main__":
    unittest.main()
