from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from scripts.train_v5_spatial.v6_topology_learner import (
    TopologyEdgeClassifier,
    TopologyPairDataset,
    load_house_pair_samples,
    node_feature,
    pair_feature,
    read_size_priors,
)
from scripts.train_v5_spatial.v6_multipart_decoder import write_json


def sample_phase10_house(path: Path) -> None:
    write_json(
        path,
        {
            "house_id": "house_test",
            "metadata": {"building_size": {"x": 3600.0, "y": 1200.0, "z": 6000.0}},
            "functional_groups": [
                {"functional_id": "entryway_0", "type": "entryway", "floors": [1]},
                {"functional_id": "corridor_0", "type": "corridor", "floors": [1]},
                {"functional_id": "living_room_0", "type": "living_room", "floors": [1]},
            ],
            "rooms": [
                {
                    "id": "entryway_0_part_0",
                    "functional_id": "entryway_0",
                    "type": "entryway",
                    "box_min": [0.0, 0.0, 0.0],
                    "box_max": [900.0, 1200.0, 3000.0],
                },
                {
                    "id": "corridor_0_part_0",
                    "functional_id": "corridor_0",
                    "type": "corridor",
                    "box_min": [900.0, 0.0, 0.0],
                    "box_max": [1500.0, 1200.0, 3000.0],
                },
                {
                    "id": "corridor_0_part_1",
                    "functional_id": "corridor_0",
                    "type": "corridor",
                    "box_min": [1500.0, 0.0, 0.0],
                    "box_max": [2100.0, 1200.0, 3000.0],
                },
                {
                    "id": "living_room_0_part_0",
                    "functional_id": "living_room_0",
                    "type": "living_room",
                    "box_min": [2100.0, 0.0, 0.0],
                    "box_max": [3600.0, 1200.0, 3000.0],
                },
            ],
        },
    )


class V6TopologyLearnerTest(unittest.TestCase):
    def test_pair_samples_use_functional_group_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            topology, pairs = load_house_pair_samples(path)
        self.assertEqual(len(topology["edges"]), 2)
        labels = {(pair.source, pair.target): pair.label for pair in pairs}
        self.assertEqual(labels[("corridor_0", "entryway_0")], 1.0)
        self.assertEqual(labels[("corridor_0", "living_room_0")], 1.0)
        self.assertEqual(labels[("entryway_0", "living_room_0")], 0.0)

    def test_program_only_features_remove_target_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            _topology, pairs = load_house_pair_samples(path)
        sample = pairs[0]
        full = pair_feature(sample, "full")
        program_only = pair_feature(sample, "program_only")
        self.assertLess(program_only.numel(), full.numel())
        self.assertEqual(program_only.numel(), node_feature(sample.left, "program_only").__len__() * 2 + 1)
        shifted_left = type(sample.left)(
            house_id=sample.left.house_id,
            node_id=sample.left.node_id,
            room_type=sample.left.room_type,
            site=sample.left.site,
            floors=sample.left.floors,
            box_min=(999.0, 999.0, 0.0),
            box_max=(1299.0, 1299.0, 3000.0),
        )
        shifted_sample = type(sample)(
            house_id=sample.house_id,
            source=sample.source,
            target=sample.target,
            left=shifted_left,
            right=sample.right,
            label=sample.label,
        )
        self.assertTrue(torch.equal(program_only, pair_feature(shifted_sample, "program_only")))
        self.assertFalse(torch.equal(full, pair_feature(shifted_sample, "full")))

    def test_program_size_features_use_size_priors_without_target_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "house_test.json"
            sample_phase10_house(path)
            write_json(
                root / "size_predictions" / "house_test" / "predicted_sizes.json",
                {
                    "house_id": "house_test",
                    "groups": [
                        {
                            "functional_id": "entryway_0",
                            "predicted": {
                                "area_ratio": 0.10,
                                "width_ratio": 0.25,
                                "depth_ratio": 1.0,
                                "part_count": 1,
                            },
                        },
                        {
                            "functional_id": "corridor_0",
                            "predicted": {
                                "area_ratio": 0.20,
                                "width_ratio": 0.33,
                                "depth_ratio": 1.0,
                                "part_count": 2,
                            },
                        },
                    ],
                },
            )
            _topology, pairs = load_house_pair_samples(path, read_size_priors(root))
        sample = pairs[0]
        program_size = pair_feature(sample, "program_size")
        program_only = pair_feature(sample, "program_only")
        self.assertGreater(program_size.numel(), program_only.numel())
        shifted = replace(
            sample,
            left=replace(
                sample.left,
                box_min=(999.0, 999.0, 0.0),
                box_max=(1299.0, 1299.0, 3000.0),
            ),
        )
        self.assertTrue(torch.equal(program_size, pair_feature(shifted, "program_size")))
        changed_size = replace(
            sample,
            left=replace(sample.left, size_prior=(0.9, 0.9, 0.9, 0.9)),
        )
        self.assertFalse(torch.equal(program_size, pair_feature(changed_size, "program_size")))

    def test_model_can_overfit_tiny_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            _topology, pairs = load_house_pair_samples(path)
        dataset = TopologyPairDataset(pairs)
        feature_dim = int(dataset[0]["features"].numel())
        model = TopologyEdgeClassifier(feature_dim, hidden=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        features = torch.stack([pair_feature(pair) for pair in pairs])
        labels = torch.tensor([[pair.label] for pair in pairs], dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss()
        initial = float(criterion(model(features), labels).detach())
        for _ in range(200):
            loss = criterion(model(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final = float(criterion(model(features), labels).detach())
        self.assertLess(final, initial * 0.1)


if __name__ == "__main__":
    unittest.main()
