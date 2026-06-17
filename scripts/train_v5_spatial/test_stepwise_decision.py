from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stepwise_decision import (  # noqa: E402
    ActionKind,
    StepAction,
    StepwiseDecisionEnvironment,
)


class StepwiseDecisionEnvironmentTest(unittest.TestCase):
    def test_invalid_cut_is_rejected_without_mutating_state(self) -> None:
        env = StepwiseDecisionEnvironment(node_ids=(1, 2))
        before = list(env.state.regions)
        result = env.apply(
            StepAction(
                kind=ActionKind.CUT,
                region_id="root",
                axis=0,
                cut=99,
                left_node_ids=(1,),
                right_node_ids=(2,),
            )
        )
        self.assertFalse(result.accepted)
        self.assertEqual(before, list(env.state.regions))
        self.assertEqual(len(env.state.history), 0)
        self.assertEqual(len(env.attempt_log), 1)

    def test_valid_cut_creates_two_child_regions(self) -> None:
        env = StepwiseDecisionEnvironment(node_ids=(1, 2, 3, 4))
        result = env.apply(
            StepAction(
                kind=ActionKind.CUT,
                region_id="root",
                axis=2,
                cut=10,
                left_node_ids=(1, 2),
                right_node_ids=(3, 4),
            )
        )
        self.assertTrue(result.accepted)
        self.assertEqual(len(env.state.regions), 2)
        self.assertEqual(
            sorted(region.bounds for region in env.state.regions.values()),
            [(0, 0, 0, 88, 88, 10), (0, 0, 10, 88, 88, 20)],
        )

    def test_place_assigns_function_without_cutting_whole_region(self) -> None:
        env = StepwiseDecisionEnvironment(node_ids=(7, 8))
        result = env.apply(
            StepAction(
                kind=ActionKind.PLACE,
                region_id="root",
                node_ids=(7,),
                bounds=(0, 0, 0, 20, 20, 10),
            )
        )
        self.assertTrue(result.accepted)
        self.assertEqual(env.state.assignments[7], [(0, 0, 0, 20, 20, 10)])
        self.assertEqual(env.state.regions["root"].node_ids, (8,))
        self.assertEqual(len(env.state.regions), 1)

    def test_reserve_empty_records_void_space(self) -> None:
        env = StepwiseDecisionEnvironment(node_ids=(1,))
        result = env.apply(
            StepAction(
                kind=ActionKind.RESERVE_EMPTY,
                region_id="root",
                bounds=(70, 70, 0, 88, 88, 10),
            )
        )
        self.assertTrue(result.accepted)
        self.assertEqual(env.state.empty_regions, [(70, 70, 0, 88, 88, 10)])

    def test_overlapping_place_is_rejected(self) -> None:
        env = StepwiseDecisionEnvironment(node_ids=(1, 2))
        first = env.apply(
            StepAction(
                kind=ActionKind.PLACE,
                region_id="root",
                node_ids=(1,),
                bounds=(0, 0, 0, 20, 20, 10),
            )
        )
        second = env.apply(
            StepAction(
                kind=ActionKind.PLACE,
                region_id="root",
                node_ids=(2,),
                bounds=(10, 10, 0, 30, 30, 10),
            )
        )
        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertNotIn(2, env.state.assignments)

    def test_rollback_restores_previous_accepted_state(self) -> None:
        env = StepwiseDecisionEnvironment(node_ids=(1, 2))
        env.apply(
            StepAction(
                kind=ActionKind.PLACE,
                region_id="root",
                node_ids=(1,),
                bounds=(0, 0, 0, 20, 20, 10),
            )
        )
        result = env.apply(StepAction(kind=ActionKind.ROLLBACK))
        self.assertTrue(result.accepted)
        self.assertEqual(env.state.assignments, {})
        self.assertEqual(env.state.regions["root"].node_ids, (1, 2))
        self.assertEqual(env.state.history[-1].kind, ActionKind.ROLLBACK)

    def test_invalid_rollback_target_is_rejected(self) -> None:
        env = StepwiseDecisionEnvironment(node_ids=(1,))
        result = env.apply(
            StepAction(kind=ActionKind.ROLLBACK, target_action_index=99)
        )
        self.assertFalse(result.accepted)
        self.assertEqual(env.state.history, [])

    def test_merge_combines_adjacent_child_regions(self) -> None:
        env = StepwiseDecisionEnvironment(node_ids=(1, 2))
        env.apply(
            StepAction(
                kind=ActionKind.CUT,
                region_id="root",
                axis=0,
                cut=40,
                left_node_ids=(1,),
                right_node_ids=(2,),
            )
        )
        source_ids = tuple(env.state.regions)
        result = env.apply(
            StepAction(kind=ActionKind.MERGE, source_region_ids=source_ids)
        )
        self.assertTrue(result.accepted)
        self.assertEqual(len(env.state.regions), 1)
        region = next(iter(env.state.regions.values()))
        self.assertEqual(region.bounds, (0, 0, 0, 88, 88, 20))
        self.assertEqual(sorted(region.node_ids), [1, 2])

    def test_non_cut_actions_can_describe_l_shaped_occupied_area(self) -> None:
        env = StepwiseDecisionEnvironment(node_ids=(1, 2))
        first = env.apply(
            StepAction(
                kind=ActionKind.PLACE,
                region_id="root",
                node_ids=(1,),
                bounds=(0, 0, 0, 60, 30, 10),
            )
        )
        second = env.apply(
            StepAction(
                kind=ActionKind.PLACE,
                region_id="root",
                node_ids=(2,),
                bounds=(0, 30, 0, 30, 60, 10),
            )
        )
        empty = env.apply(
            StepAction(
                kind=ActionKind.RESERVE_EMPTY,
                region_id="root",
                bounds=(30, 30, 0, 60, 60, 10),
            )
        )
        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertTrue(empty.accepted)
        occupied = env.state.assignments[1] + env.state.assignments[2]
        self.assertEqual(
            occupied,
            [(0, 0, 0, 60, 30, 10), (0, 30, 0, 30, 60, 10)],
        )
        self.assertEqual(env.state.empty_regions, [(30, 30, 0, 60, 60, 10)])


if __name__ == "__main__":
    unittest.main()
