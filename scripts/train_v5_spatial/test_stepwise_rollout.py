from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_stepwise_rollout import (  # noqa: E402
    assignment_report,
    evaluate_payload,
    replay_oracle,
)
from stepwise_dataset import (  # noqa: E402
    ACTION_TO_ID,
    DEFAULT_DATA_DIR,
    StepwiseActionDataset,
    read_json,
)


class StepwiseRolloutTest(unittest.TestCase):
    def test_oracle_rollout_completes_assignments(self) -> None:
        payload = read_json(DEFAULT_DATA_DIR / "house_10.json")
        env, extra = replay_oracle(payload)
        report = assignment_report(payload, env)
        self.assertEqual(extra["invalid_record_count"], 0)
        self.assertTrue(report["complete"])
        self.assertEqual(report["missing_assignment_count"], 0)

    def test_oracle_rollout_can_use_unified_evaluator(self) -> None:
        payload = read_json(DEFAULT_DATA_DIR / "house_10.json")
        env, extra = replay_oracle(payload)
        result = evaluate_payload("house_10", payload, env, extra)
        self.assertTrue(result["p0_pass"])
        self.assertTrue(result["p1_hard_geometry_pass"])

    def test_cut_node_target_is_one_side_partition(self) -> None:
        dataset = StepwiseActionDataset("train", max_houses=1)
        cut_index = next(
            index
            for index, target in enumerate(dataset.action_targets)
            if target == ACTION_TO_ID["cut"]
        )
        house_id, action_index = dataset.items[cut_index]
        payload = read_json(DEFAULT_DATA_DIR / f"{house_id}.json")
        record = payload["actions"][action_index]
        item = dataset[cut_index]
        marked = {
            index
            for index, value in enumerate(item["node_target"].tolist())
            if value > 0.5
        }
        self.assertEqual(marked, set(record["left_node_ids"]))
        self.assertNotEqual(marked, set(record["left_node_ids"]) | set(record["right_node_ids"]))


if __name__ == "__main__":
    unittest.main()
