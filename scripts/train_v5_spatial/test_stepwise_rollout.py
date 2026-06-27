from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_stepwise_rollout import (  # noqa: E402
    assignment_report,
    decode_node_set,
    evaluate_payload,
    legalize_bounds,
    repair_overlapping_bounds,
    replay_oracle,
)
from stepwise_decision import ActionKind, StepAction, StepwiseDecisionEnvironment  # noqa: E402
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

    def test_action_decoder_legalizes_empty_node_and_invalid_box(self) -> None:
        bounds = legalize_bounds([8, 8, 8, 2, 2, 2], (0, 0, 0, 10, 10, 10))
        self.assertLess(bounds[0], bounds[3])
        self.assertLess(bounds[1], bounds[4])
        self.assertLess(bounds[2], bounds[5])
        logits = torch.tensor([-10.0, -9.0, -8.0])
        self.assertEqual(
            decode_node_set(logits, (0, 1, 2), require_one=True),
            (2,),
        )

    def test_repair_overlapping_bounds_finds_free_region(self) -> None:
        env = StepwiseDecisionEnvironment(
            site_bounds=(0, 0, 0, 10, 10, 10),
            node_ids=(0, 1),
        )
        result = env.apply(
            StepAction(
                kind=ActionKind.PLACE,
                region_id="root",
                node_ids=(0,),
                bounds=(0, 0, 0, 5, 10, 10),
            )
        )
        self.assertTrue(result.accepted)
        repaired = repair_overlapping_bounds(
            env,
            env.state.regions["root"],
            (0, 0, 0, 5, 10, 10),
        )
        self.assertEqual(repaired, (5, 0, 0, 10, 10, 10))


if __name__ == "__main__":
    unittest.main()
