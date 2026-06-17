"""Tests for the learned user-condition program graph."""
from __future__ import annotations

import unittest

import torch

from program_graph_dataset import ProgramGraphDataset, collate_program_graph
from program_graph_model import ProgramGraphModel
from train_program_graph import compute_loss


class ProgramGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.item = ProgramGraphDataset("train", max_houses=1)[0]

    def test_input_contains_no_target_geometry(self) -> None:
        self.assertEqual(self.item["node_input"].shape[1], 19)
        self.assertEqual(
            self.item["node_input"].shape[0],
            self.item["floor_target"].shape[0],
        )

    def test_forward_and_loss(self) -> None:
        batch = collate_program_graph([self.item])
        model = ProgramGraphModel(hidden=32, layers=1)
        output = model(batch["node_input"], batch["node_mask"])
        loss, parts = compute_loss(output, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(parts),
            {"loss", "floor", "area", "lighting", "exterior", "relation"},
        )
        count = self.item["node_input"].shape[0]
        self.assertEqual(tuple(output["relation_logits"].shape), (1, count, count, 3))


if __name__ == "__main__":
    unittest.main()
