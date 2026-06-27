from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_v5_spatial.v6_multipart_decoder import write_json
from scripts.train_v5_spatial.v6_size_area_head import (
    SizeAreaHead,
    SizeDataset,
    load_size_samples,
    size_feature,
    size_target,
)


def sample_phase10_house(path: Path) -> None:
    write_json(
        path,
        {
            "house_id": "house_test",
            "metadata": {"building_size": {"x": 3600.0, "y": 2400.0, "z": 6000.0}},
            "functional_groups": [
                {"functional_id": "living_room_0", "type": "living_room", "floors": [1]},
                {"functional_id": "corridor_0", "type": "corridor", "floors": [1]},
            ],
            "rooms": [
                {
                    "id": "living_room_0_part_0",
                    "functional_id": "living_room_0",
                    "type": "living_room",
                    "box_min": [0.0, 0.0, 0.0],
                    "box_max": [1800.0, 2400.0, 3000.0],
                    "area": 4320000.0,
                },
                {
                    "id": "corridor_0_part_0",
                    "functional_id": "corridor_0",
                    "type": "corridor",
                    "box_min": [1800.0, 0.0, 0.0],
                    "box_max": [2400.0, 2400.0, 3000.0],
                    "area": 1440000.0,
                },
                {
                    "id": "corridor_0_part_1",
                    "functional_id": "corridor_0",
                    "type": "corridor",
                    "box_min": [2400.0, 0.0, 0.0],
                    "box_max": [3000.0, 2400.0, 3000.0],
                    "area": 1440000.0,
                },
            ],
        },
    )


class V6SizeAreaHeadTest(unittest.TestCase):
    def test_size_targets_use_group_area_and_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            samples = load_size_samples(Path(tmp), None)
        by_id = {sample.group_id: sample for sample in samples}
        living_target = size_target(by_id["living_room_0"])
        corridor_target = size_target(by_id["corridor_0"])
        self.assertAlmostEqual(float(living_target[0]), 0.5)
        self.assertAlmostEqual(float(living_target[1]), 0.5)
        self.assertAlmostEqual(float(corridor_target[0]), 1.0 / 3.0)
        self.assertAlmostEqual(float(corridor_target[3]), 2.0 / 8.0)

    def test_size_feature_does_not_include_target_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            samples = load_size_samples(Path(tmp), None)
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
            part_count=sample.part_count,
            box_min=(900.0, 900.0, 0.0),
            box_max=(2700.0, 2400.0, 3000.0),
            area_mm2=sample.area_mm2,
        )
        self.assertTrue(torch.equal(size_feature(sample), size_feature(moved)))

    def test_model_can_overfit_tiny_size_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "house_test.json"
            sample_phase10_house(path)
            samples = load_size_samples(Path(tmp), None)
        dataset = SizeDataset(samples)
        features = torch.stack([dataset[index]["features"] for index in range(len(dataset))])
        targets = torch.stack([dataset[index]["target"] for index in range(len(dataset))])
        model = SizeAreaHead(int(features.shape[1]), hidden=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        criterion = torch.nn.SmoothL1Loss()
        initial = float(criterion(model(features), targets).detach())
        for _ in range(300):
            loss = criterion(model(features), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final = float(criterion(model(features), targets).detach())
        self.assertLess(final, initial * 0.1)


if __name__ == "__main__":
    unittest.main()
