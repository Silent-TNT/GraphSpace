from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stepwise_dataset import (  # noqa: E402
    ACTION_TO_ID,
    STEPWISE_VOLUME_CHANNELS,
    StepwiseActionDataset,
    collate_stepwise,
)
from stepwise_model import StepwiseActionPolicy  # noqa: E402
from train_stepwise import compute_loss, quiz_metrics  # noqa: E402


class StepwiseTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = StepwiseActionDataset("train", max_houses=1)

    def test_dataset_replays_state_before_action(self) -> None:
        item = self.dataset[0]
        self.assertEqual(
            tuple(item["volume"].shape),
            (len(STEPWISE_VOLUME_CHANNELS), 88, 88, 20),
        )
        self.assertIn(
            int(item["action_target"]),
            set(ACTION_TO_ID.values()),
        )
        self.assertEqual(tuple(item["action_mask"].shape), (len(ACTION_TO_ID),))
        self.assertEqual(float(item["action_mask"][ACTION_TO_ID["reserve_empty"]]), 0.0)
        self.assertEqual(float(item["progress_target"]), 1.0)
        self.assertEqual(item["nodes"].shape[0], item["node_target"].shape[0])

    def test_dataset_encodes_last_rejected_attempt(self) -> None:
        rejected_index = next(
            index
            for index, target in enumerate(self.dataset.action_targets)
            if target == ACTION_TO_ID["reject"]
        )
        next_item = self.dataset[rejected_index + 1]
        rejected_bounds_channel = STEPWISE_VOLUME_CHANNELS.index("last_rejected_bounds")
        bounds_invalid_channel = STEPWISE_VOLUME_CHANNELS.index(
            "last_rejected_bounds_invalid"
        )
        self.assertGreater(float(next_item["volume"][rejected_bounds_channel].sum()), 0.0)
        self.assertGreater(float(next_item["volume"][bounds_invalid_channel].sum()), 0.0)

    def test_model_forward_loss_and_quiz_metrics(self) -> None:
        batch = collate_stepwise([self.dataset[0], self.dataset[1]])
        model = StepwiseActionPolicy(base_channels=8, hidden=64)
        output = model(
            batch["volume"],
            batch["nodes"],
            batch["node_mask"],
            batch["adjacency"],
        )
        self.assertEqual(tuple(output["action_logits"].shape), (2, len(ACTION_TO_ID)))
        self.assertEqual(tuple(output["progress_logit"].shape), (2,))
        self.assertEqual(tuple(output["box"].shape), (2, 6))
        self.assertEqual(tuple(output["node_logits"].shape), tuple(batch["node_mask"].shape))
        loss, parts = compute_loss(output, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(parts),
            {"loss", "action", "accept", "progress", "axis", "cut", "box", "node"},
        )
        quizzes = quiz_metrics(output, batch)
        self.assertIn("quiz_action_correct", quizzes)
        self.assertIn("quiz_accept_correct", quizzes)
        self.assertIn("quiz_progress_correct", quizzes)


if __name__ == "__main__":
    unittest.main()
