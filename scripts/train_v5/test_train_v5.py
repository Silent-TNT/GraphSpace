"""Tests for V5 lazy loading, model heads, and losses."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import make_dataset  # noqa: E402
from losses import compute_losses  # noqa: E402
from model import V5MinimalNet  # noqa: E402


class V5TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = make_dataset("train", max_samples=2)

    def test_lazy_sample_shapes(self) -> None:
        sample = self.dataset[0]
        self.assertEqual(tuple(sample["condition"].shape), (24,))
        self.assertEqual(tuple(sample["site_mask"].shape), (1, 88, 88))
        self.assertEqual(tuple(sample["class_grid"].shape), (2, 88, 88))
        self.assertEqual(tuple(sample["center_offset"].shape), (2, 2, 88, 88))

    def test_forward_and_backward(self) -> None:
        batch = next(iter(DataLoader(self.dataset, batch_size=2)))
        model = V5MinimalNet(base_channels=8)
        output = model(batch["condition"], batch["site_mask"])
        self.assertEqual(tuple(output["class_logits"].shape), (2, 2, 12, 88, 88))
        self.assertEqual(tuple(output["center_logits"].shape), (2, 2, 88, 88))
        self.assertEqual(tuple(output["center_offset"].shape), (2, 2, 2, 88, 88))
        self.assertEqual(tuple(output["boundary_logits"].shape), (2, 2, 88, 88))
        self.assertEqual(tuple(output["cross_floor_logits"].shape), (2, 2, 88, 88))
        self.assertEqual(tuple(output["count_prediction"].shape), (2, 24))
        losses = compute_losses(output, batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()
