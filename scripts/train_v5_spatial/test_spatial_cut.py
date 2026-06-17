from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_cut_supervision import build_sample  # noqa: E402
from dataset import SpatialCutDataset, collate_cut_actions  # noqa: E402
from model import SpatialModalCutPolicy  # noqa: E402


ROOT = SCRIPT_DIR.parents[1]


class SpatialCutTest(unittest.TestCase):
    def test_supervision_contains_3d_cut_actions(self):
        sample = build_sample(ROOT / "data" / "processed" / "house_1232.json")
        axes = {action["axis"] for action in sample["actions"]}
        self.assertIn(2, axes)
        self.assertIn(3, axes)
        self.assertGreater(len(sample["graph"]["edges"]), 0)

    def test_model_forward(self):
        dataset = SpatialCutDataset("train", max_houses=1)
        batch = collate_cut_actions([dataset[0], dataset[1]])
        model = SpatialModalCutPolicy()
        output = model(
            batch["volume"],
            batch["nodes"],
            batch["active"],
            batch["adjacency"],
        )
        self.assertEqual(tuple(output["axis_logits"].shape), (2, 4))
        self.assertEqual(tuple(output["cut_ratio"].shape), (2,))
        self.assertEqual(tuple(output["left_fraction"].shape), (2,))
        self.assertEqual(output["side_logits"].shape[:2], batch["nodes"].shape[:2])
        loss = torch.nn.functional.cross_entropy(
            output["axis_logits"], batch["axis"]
        )
        loss.backward()


if __name__ == "__main__":
    unittest.main()
