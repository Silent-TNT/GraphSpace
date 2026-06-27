from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_v5_spatial.test_v6_size_area_head import sample_phase10_house
from scripts.train_v5_spatial.v6_position_head import (
    PositionDataset,
    PositionHead,
    load_position_samples,
    position_feature,
    position_loss_weight,
    position_target,
)


class V6PositionHeadTest(unittest.TestCase):
    def test_position_targets_use_group_center_and_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            samples = load_position_samples(Path(tmp), None)
        by_id = {sample.group_id: sample for sample in samples}
        living_center, living_zone = position_target(by_id["living_room_0"])
        corridor_center, corridor_zone = position_target(by_id["corridor_0"])
        self.assertAlmostEqual(float(living_center[0]), 0.25)
        self.assertAlmostEqual(float(living_center[1]), 0.5)
        self.assertEqual(int(living_zone), 3)
        self.assertAlmostEqual(float(corridor_center[0]), 2.0 / 3.0)
        self.assertEqual(int(corridor_zone), 5)

    def test_position_feature_does_not_include_target_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            samples = load_position_samples(Path(tmp), None)
        sample = samples[0]
        moved = type(sample)(
            house_id=sample.house_id,
            group_id=sample.group_id,
            room_type=sample.room_type,
            site=sample.site,
            floors=sample.floors,
            type_index=sample.type_index,
            type_count=sample.type_count,
            group_count=sample.group_count,
            box_min=(900.0, 900.0, 0.0),
            box_max=(2700.0, 2400.0, 3000.0),
        )
        self.assertTrue(torch.equal(position_feature(sample), position_feature(moved)))

    def test_position_loss_weights_prioritize_primary_rooms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            samples = load_position_samples(Path(tmp), None)
        by_type = {sample.room_type: sample for sample in samples}
        self.assertGreater(position_loss_weight(by_type["living_room"]), position_loss_weight(by_type["corridor"]))
        dataset = PositionDataset(samples, primary_weight=3.0, circulation_weight=0.5)
        by_id = {sample.group_id: index for index, sample in enumerate(samples)}
        living_weight = float(dataset[by_id["living_room_0"]]["weight"])
        corridor_weight = float(dataset[by_id["corridor_0"]]["weight"])
        self.assertEqual(living_weight, 3.0)
        self.assertEqual(corridor_weight, 0.5)

    def test_model_can_overfit_tiny_position_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            samples = load_position_samples(Path(tmp), None)
        dataset = PositionDataset(samples)
        features = torch.stack([dataset[index]["features"] for index in range(len(dataset))])
        centers = torch.stack([dataset[index]["center"] for index in range(len(dataset))])
        zones = torch.stack([dataset[index]["zone"] for index in range(len(dataset))])
        model = PositionHead(int(features.shape[1]), hidden=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        center_criterion = torch.nn.SmoothL1Loss()
        zone_criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            pred_center, pred_zone = model(features)
            initial = float(center_criterion(pred_center, centers) + 0.2 * zone_criterion(pred_zone, zones))
        for _ in range(300):
            pred_center, pred_zone = model(features)
            loss = center_criterion(pred_center, centers) + 0.2 * zone_criterion(pred_zone, zones)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred_center, pred_zone = model(features)
            final = float(center_criterion(pred_center, centers) + 0.2 * zone_criterion(pred_zone, zones))
        self.assertLess(final, initial * 0.1)


if __name__ == "__main__":
    unittest.main()
