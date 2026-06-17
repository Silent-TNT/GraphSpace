from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_instance_rollout import (  # noqa: E402
    box_face_contact,
    candidate_valid,
    place_nearest_box,
    place_topology_box,
)
from instance_dataset import (  # noqa: E402
    InstancePlacementDataset,
    collate_instances,
)
from instance_model import InstancePlacementPolicy  # noqa: E402


class InstancePlacementTest(unittest.TestCase):
    def test_dataset_has_one_action_per_room(self):
        dataset = InstancePlacementDataset("train", max_houses=1)
        self.assertEqual(len(dataset), 26)
        self.assertEqual(dataset[0]["volume"].shape[0], 9)

    def test_model_forward_and_backward(self):
        dataset = InstancePlacementDataset("train", max_houses=1)
        batch = collate_instances([dataset[0], dataset[1]])
        model = InstancePlacementPolicy(base_channels=8)
        output = model(
            batch["volume"],
            batch["nodes"],
            batch["node_mask"],
            batch["adjacency"],
            batch["room_index"],
            batch["step_ratio"],
        )
        self.assertEqual(tuple(output["box"].shape), (2, 4))
        output["box"].mean().backward()

    def test_parameterized_placement_respects_free_space(self):
        import numpy as np

        building = np.ones((2, 12, 12), dtype=bool)
        occupied = np.zeros_like(building, dtype=np.uint8)
        occupied[0, :6, :6] = 1
        prediction = np.asarray([0.25, 0.25, 0.4, 0.4], dtype=np.float32)
        box = place_nearest_box(
            prediction,
            [12, 12],
            [1],
            building,
            occupied,
        )
        self.assertIsNotNone(box)
        self.assertTrue(candidate_valid(box, [1], building, occupied))

    def test_topology_projection_prefers_face_contact(self):
        import numpy as np

        building = np.ones((2, 12, 12), dtype=bool)
        occupied = np.zeros_like(building, dtype=np.uint8)
        occupied[0, 1:5, 1:5] = 1
        prediction = np.asarray([0.8, 0.8, 0.3, 0.3], dtype=np.float32)
        box, details = place_topology_box(
            prediction,
            [12, 12],
            [1],
            building,
            occupied,
            {0: (1, 1, 5, 5)},
            {0: [1]},
            [(0, 0, True)],
        )
        self.assertIsNotNone(box)
        self.assertGreater(box_face_contact(box, (1, 1, 5, 5)), 0)
        self.assertEqual(details["realized_topology_neighbors"], 1)



if __name__ == "__main__":
    unittest.main()
