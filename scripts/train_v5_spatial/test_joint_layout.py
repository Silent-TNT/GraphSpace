from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from joint_dataset import JointLayoutDataset, collate_joint  # noqa: E402
from joint_model import JointLayoutPolicy  # noqa: E402
from train_joint_layout import compute_loss, soft_coverage_loss  # noqa: E402


class JointLayoutTest(unittest.TestCase):
    def test_whole_house_forward_and_joint_loss(self):
        dataset = JointLayoutDataset("train", max_houses=2)
        batch = collate_joint([dataset[0], dataset[1]])
        model = JointLayoutPolicy(base_channels=8)
        output = model(
            batch["volume"],
            batch["nodes"],
            batch["node_mask"],
            batch["adjacency"],
        )
        self.assertEqual(output["boxes"].shape[:2], batch["nodes"].shape[:2])
        args = Namespace(
            box_weight=1.0,
            overlap_weight=0.5,
            contact_weight=0.5,
            coverage_weight=0.5,
            shape_weight=1.0,
            bounds_weight=0.5,
        )
        loss, parts = compute_loss(output, batch, args)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(parts),
            {"box", "overlap", "contact", "coverage", "shape", "bounds"},
        )
        loss.backward()

    def test_stairs_cannot_satisfy_ordinary_room_coverage(self):
        dataset = JointLayoutDataset("train", max_houses=1)
        batch = collate_joint([dataset[0]])
        boxes = batch["target_boxes"].clone()
        stair_mask = batch["nodes"][:, :, 7] > 0
        ordinary_mask = batch["node_mask"].bool() & ~stair_mask
        boxes[:, :, 2][ordinary_mask] = 0.001
        boxes[:, :, 3][ordinary_mask] = 0.001
        baseline = soft_coverage_loss(
            boxes,
            batch["node_mask"],
            batch["floors"],
            batch["volume"],
            batch["nodes"],
        )
        boxes[:, :, 2][stair_mask] = 0.95
        boxes[:, :, 3][stair_mask] = 0.95
        enlarged_stairs = soft_coverage_loss(
            boxes,
            batch["node_mask"],
            batch["floors"],
            batch["volume"],
            batch["nodes"],
        )
        self.assertAlmostEqual(
            float(baseline),
            float(enlarged_stairs),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
