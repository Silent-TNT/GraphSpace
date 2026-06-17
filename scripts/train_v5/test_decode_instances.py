"""Synthetic tests for noisy V5 instance decoding."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from decode_instances import (  # noqa: E402
    decode_floor_instances,
    matched_instance_metrics,
    merge_building_instances,
)


class DecodeInstancesTest(unittest.TestCase):
    def test_noisy_offsets_separate_adjacent_same_class_rooms(self) -> None:
        class_grid = np.zeros((2, 12, 12), dtype=np.uint8)
        expected = np.zeros_like(class_grid, dtype=np.uint16)
        class_grid[0, 1:5, 1:5] = 5
        class_grid[0, 5:9, 1:5] = 5
        expected[0, 1:5, 1:5] = 1
        expected[0, 5:9, 1:5] = 2
        offsets = np.zeros((2, 2, 12, 12), dtype=np.float32)
        rng = np.random.default_rng(7)
        for instance_id, center in ((1, (2.5, 2.5)), (2, (6.5, 2.5))):
            for x, y in np.argwhere(expected[0] == instance_id):
                offsets[0, :, x, y] = (
                    np.asarray(center) - np.asarray((x, y))
                    + rng.normal(0.0, 0.25, size=2)
                )
        heatmap = np.zeros((2, 12, 12), dtype=np.float32)
        heatmap[0, 2:4, 2:4] = 0.9
        heatmap[0, 6:8, 2:4] = 0.9
        boundary = np.zeros_like(heatmap)
        boundary[0, 4:6, 1:5] = 0.9
        counts = np.zeros((2, 11), dtype=np.float32)
        counts[0, 4] = 2
        decoded = decode_floor_instances(
            class_grid, heatmap, offsets, boundary, counts
        )
        metrics = matched_instance_metrics(decoded, expected, class_grid)
        self.assertEqual(metrics["count_error"], 0.0)
        self.assertGreater(metrics["mean_matched_iou"], 0.9)
        self.assertEqual(metrics["instance_recall_iou50"], 1.0)

    def test_cross_floor_probability_merges_stair(self) -> None:
        class_grid = np.zeros((2, 8, 8), dtype=np.uint8)
        instances = np.zeros((2, 8, 8), dtype=np.uint16)
        class_grid[:, 2:5, 2:5] = 8
        instances[0, 2:5, 2:5] = 1
        instances[1, 2:5, 2:5] = 2
        cross_floor = np.zeros((2, 8, 8), dtype=np.float32)
        cross_floor[:, 2:5, 2:5] = 0.95
        records = merge_building_instances(instances, class_grid, cross_floor)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["floors"], [1, 2])


if __name__ == "__main__":
    unittest.main()
