from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_staged_supervision import build_sample  # noqa: E402
from staged_dataset import (  # noqa: E402
    StagedSpatialDataset,
    collate_staged,
    staged_volume,
)
from staged_model import StagedSpatialPolicy  # noqa: E402


ROOT = SCRIPT_DIR.parents[1]


class StagedSupervisionTest(unittest.TestCase):
    def setUp(self):
        self.sample, self.arrays = build_sample(
            ROOT / "data" / "processed" / "house_10.json"
        )

    def test_stage_order_and_protected_stairs(self):
        actions = self.sample["actions"]
        self.assertEqual(actions[0]["stage"], "stair_core")
        self.assertEqual(actions[1]["stage"], "floor_split")
        self.assertEqual(actions[1]["cut_cell"], 10)
        self.assertTrue(actions[0]["protected_indices"])
        self.assertEqual(
            actions[0]["protected_indices"],
            actions[1]["protected_indices"],
        )

    def test_site_is_partitioned_into_building_and_empty(self):
        site = self.arrays["site_mask"]
        for floor in range(2):
            partition = (
                self.arrays["building_mask"][floor]
                + self.arrays["empty_mask"][floor]
            )
            np.testing.assert_array_equal(partition, site)
        self.assertGreater(int(self.arrays["empty_mask"].sum()), 0)

    def test_reachability_is_functional_block_contact(self):
        report = self.sample["actions"][-1]["oracle_report"]
        self.assertIn("unreachable_indices", report)
        self.assertIn("no doors", report["semantics"])

    def test_staged_volume_has_protected_and_empty_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            sample_path = Path(directory) / "house_10.json"
            sample_path.write_text(
                __import__("json").dumps(self.sample),
                encoding="utf-8",
            )
            np.savez_compressed(sample_path.with_suffix(".npz"), **self.arrays)
            volume = staged_volume(sample_path, 4)
        self.assertEqual(volume.shape, (8, 44, 44, 20))
        self.assertGreater(float(volume[2].sum()), 0)
        self.assertGreater(float(volume[3].sum()), 0)
        self.assertGreater(float(volume[4].sum()), 0)

    def test_stage_input_does_not_contain_its_target(self):
        stage_zero = staged_volume(
            ROOT / "data" / "phase7_staged_spatial" / "samples" / "house_10.json",
            0,
        )
        self.assertEqual(float(stage_zero[2].sum()), 0.0)
        stage_two = staged_volume(
            ROOT / "data" / "phase7_staged_spatial" / "samples" / "house_10.json",
            2,
        )
        self.assertEqual(float(stage_two[1].sum()), 0.0)
        self.assertEqual(float(stage_two[3].sum()), 0.0)

    def test_staged_model_forward_and_backward(self):
        dataset = StagedSpatialDataset("train", max_houses=1)
        batch = collate_staged([dataset[0], dataset[2]])
        model = StagedSpatialPolicy(base_channels=8)
        output = model(
            batch["volume"],
            batch["nodes"],
            batch["node_mask"],
            batch["adjacency"],
            batch["stage_id"],
        )
        self.assertEqual(
            tuple(output["mask_logits"].shape),
            (2, 2, 44, 44, 20),
        )
        self.assertEqual(tuple(output["cut_ratio"].shape), (2,))
        loss = output["mask_logits"].mean()
        loss.backward()


if __name__ == "__main__":
    unittest.main()
