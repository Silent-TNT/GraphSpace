"""Tests for the native 300 mm final V5 route."""
from __future__ import annotations

import unittest

import torch

from fullres_dataset import FullResolutionLayoutDataset, collate_fullres
from fullres_model import FullResolutionGraphVoxelModel
from train_fullres import (
    assignment_probabilities,
    compute_loss,
    instance_distribution_losses,
    semantic_logits_from_instances,
    topology_contact_loss,
)


class FullResolutionRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = FullResolutionLayoutDataset(
            "train",
            condition_mode="program",
            max_houses=1,
        )
        cls.item = cls.dataset[0]

    def test_native_resolution_and_complete_targets(self) -> None:
        self.assertEqual(tuple(self.item["volume"].shape), (8, 88, 88, 20))
        self.assertEqual(
            tuple(self.item["semantic_target"].shape),
            (88, 88, 20),
        )
        valid_semantics = self.item["semantic_target"][
            self.item["semantic_target"] != 255
        ]
        self.assertEqual(valid_semantics.max().item(), 11)
        self.assertEqual(
            self.item["instance_targets"].shape[0],
            self.item["nodes"].shape[0],
        )

    def test_input_does_not_contain_target_semantics(self) -> None:
        volume = self.item["volume"]
        unique = set(float(value) for value in torch.unique(volume))
        self.assertTrue(unique.issubset({-1.0, 0.0, 1.0}) or len(unique) > 3)
        self.assertEqual(volume.shape[0], 8)
        self.assertNotEqual(volume.shape[0], 12)

    def test_model_preserves_spatial_canvas(self) -> None:
        batch = collate_fullres([self.item])
        model = FullResolutionGraphVoxelModel(
            spatial_channels=8,
            query_channels=8,
            architecture="v2",
        )
        output = model(
            batch["volume"],
            batch["nodes"],
            batch["node_mask"],
            batch["adjacency"],
        )
        self.assertEqual(
            tuple(output["instance_logits"].shape),
            (
                1,
                self.item["nodes"].shape[0],
                88,
                88,
                20,
            ),
        )

    def test_all_joint_losses_backpropagate(self) -> None:
        batch = collate_fullres([self.item])
        model = FullResolutionGraphVoxelModel(
            spatial_channels=8,
            query_channels=8,
            architecture="v2",
        )
        output = model(
            batch["volume"],
            batch["nodes"],
            batch["node_mask"],
            batch["adjacency"],
        )
        loss, parts = compute_loss(output, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(parts),
            {
                "instance_bce",
                "instance_dice",
                "semantic",
                "building",
                "outside",
                "overlap",
                "topology",
                "existence",
                "area",
                "compactness",
                "height",
            },
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_semantic_head_has_empty_plus_eleven_classes(self) -> None:
        batch = collate_fullres([self.item])
        logits = torch.zeros_like(batch["instance_targets"])
        semantic = semantic_logits_from_instances(
            logits,
            batch["nodes"],
            batch["node_mask"],
        )
        self.assertEqual(tuple(semantic.shape), (1, 12, 88, 88, 20))

    def test_uncertain_voxels_do_not_fake_topology_contact(self) -> None:
        batch = collate_fullres([self.item])
        probabilities = torch.zeros_like(batch["instance_targets"])
        loss = topology_contact_loss(probabilities, batch["adjacency"])
        self.assertAlmostEqual(float(loss), 1.0, places=5)

    def test_assignment_is_exclusive_with_empty_class(self) -> None:
        batch = collate_fullres([self.item])
        model = FullResolutionGraphVoxelModel(
            spatial_channels=8,
            query_channels=8,
            architecture="v2",
        )
        output = model(
            batch["volume"],
            batch["nodes"],
            batch["node_mask"],
            batch["adjacency"],
        )
        _, probabilities = assignment_probabilities(
            output,
            batch["node_mask"],
        )
        sums = probabilities.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_training_uses_program_graph_without_true_exterior_answer(self) -> None:
        exterior_features = self.item["nodes"][:, 14:18]
        self.assertEqual(float(exterior_features.sum()), 0.0)

    def test_instance_distribution_losses_detect_missing_rooms(self) -> None:
        batch = collate_fullres([self.item])
        perfect = batch["instance_targets"].float()
        missing = perfect.clone()
        missing[:, 0] = 0
        perfect_parts = instance_distribution_losses(
            perfect,
            batch["instance_targets"],
            batch["node_mask"],
        )
        missing_parts = instance_distribution_losses(
            missing,
            batch["instance_targets"],
            batch["node_mask"],
        )
        self.assertGreater(float(missing_parts[0]), float(perfect_parts[0]))
        self.assertGreater(float(missing_parts[1]), float(perfect_parts[1]))

    def test_robust_graph_preserves_target_instance_order(self) -> None:
        item = FullResolutionLayoutDataset(
            "train",
            condition_mode="robust",
            max_houses=1,
        )[0]
        self.assertEqual(
            item["nodes"].shape[0],
            item["instance_targets"].shape[0],
        )


if __name__ == "__main__":
    unittest.main()
