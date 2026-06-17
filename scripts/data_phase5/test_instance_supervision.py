import unittest

import numpy as np

from build_instance_supervision import (
    build_instance_supervision,
    decode_floor_instances,
    evaluate_supervision,
)


class InstanceSupervisionTest(unittest.TestCase):
    def sample(self):
        class_grid = np.full((2, 8, 8), 255, dtype=np.uint8)
        instance_grid = np.full((2, 8, 8), 65535, dtype=np.uint16)
        class_grid[:, 1:7, 1:7] = 0
        instance_grid[:, 1:7, 1:7] = 0
        class_grid[0, 1:4, 1:4] = 5
        instance_grid[0, 1:4, 1:4] = 1
        class_grid[0, 4:7, 1:4] = 5
        instance_grid[0, 4:7, 1:4] = 2
        class_grid[:, 1:4, 4:7] = 8
        instance_grid[:, 1:4, 4:7] = 3
        building_mask = (
            (class_grid != 0) & (class_grid != 255)
        ).astype(np.uint8)
        arrays = {
            "class_grid": class_grid,
            "instance_grid": instance_grid,
            "building_mask": building_mask,
            "cross_floor_mask": (instance_grid == 3).astype(np.uint8),
        }
        metadata = {
            "house_id": "synthetic",
            "placement": {"canvas_x0": 1, "canvas_y0": 1},
            "class_map": {"bedroom": 5, "stairs": 8},
            "instance_table": [
                {
                    "instance_index": 1, "id": "bedroom_1", "type": "bedroom",
                    "floors": [1], "box_min": [0, 0, 0], "box_max": [900, 900, 3000],
                },
                {
                    "instance_index": 2, "id": "bedroom_2", "type": "bedroom",
                    "floors": [1], "box_min": [900, 0, 0], "box_max": [1800, 900, 3000],
                },
                {
                    "instance_index": 3, "id": "stairs", "type": "stairs",
                    "floors": [1, 2], "box_min": [0, 900, 0], "box_max": [900, 1800, 6000],
                },
            ],
        }
        return arrays, metadata

    def test_same_class_rooms_get_distinct_centers(self):
        arrays, metadata = self.sample()
        supervision, _ = build_instance_supervision(arrays, metadata)
        decoded = decode_floor_instances(
            arrays["class_grid"],
            arrays["building_mask"],
            supervision["center_offset"],
        )
        bedroom_ids = np.unique(decoded[0][arrays["class_grid"][0] == 5])
        self.assertEqual(len(bedroom_ids), 2)

    def test_cross_floor_instance_is_merged(self):
        arrays, metadata = self.sample()
        supervision, supervision_metadata = build_instance_supervision(
            arrays, metadata
        )
        report = evaluate_supervision(arrays, metadata, supervision)
        self.assertEqual(supervision_metadata["cross_floor_instance_count"], 1)
        self.assertTrue(report["floor_partition_exact"])
        self.assertTrue(report["building_instances_exact"])

    def test_boundary_marks_room_interface(self):
        arrays, metadata = self.sample()
        supervision, _ = build_instance_supervision(arrays, metadata)
        self.assertEqual(int(supervision["boundary_mask"][0, 3, 2]), 1)
        self.assertEqual(int(supervision["boundary_mask"][0, 4, 2]), 1)


if __name__ == "__main__":
    unittest.main()
