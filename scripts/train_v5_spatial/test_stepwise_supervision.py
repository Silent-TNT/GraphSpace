from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_stepwise_supervision import build_sample  # noqa: E402


ROOT = SCRIPT_DIR.parents[1]


class StepwiseSupervisionTest(unittest.TestCase):
    def test_sample_contains_mixed_actions_and_replays(self) -> None:
        sample = build_sample(ROOT / "data" / "processed" / "house_10.json")
        stats = sample["stats"]
        kinds = {action["kind"] for action in sample["actions"]}
        self.assertEqual(sample["schema"], "graphspace_v5_stepwise_spatial_supervision_v1")
        self.assertIn("cut", kinds)
        self.assertIn("place", kinds)
        self.assertIn("reserve_empty", kinds)
        self.assertIn("rollback", kinds)
        self.assertGreater(stats["cut_action_count"], 10)
        self.assertGreater(stats["rejected_attempt_count"], 0)
        self.assertGreater(stats["rollback_action_count"], 0)
        self.assertEqual(stats["missing_assignment_count"], 0)
        self.assertEqual(stats["mismatched_assignment_count"], 0)

    def test_rejected_attempt_is_recorded_without_acceptance(self) -> None:
        sample = build_sample(ROOT / "data" / "processed" / "house_10.json")
        rejected = [
            action for action in sample["actions"] if not action["accepted"]
        ]
        self.assertGreaterEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["kind"], "place")
        self.assertTrue(rejected[0]["issues"])


if __name__ == "__main__":
    unittest.main()
