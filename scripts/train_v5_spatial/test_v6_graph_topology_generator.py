from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_v5_spatial.test_v6_topology_learner import sample_phase10_house
from scripts.train_v5_spatial.v6_graph_topology_generator import (
    GraphTopologyGenerator,
    decode_graph_edges,
    load_graph_samples,
    predict_graph,
    tensors_for_graph,
)
from scripts.train_v5_spatial.v6_multipart_decoder import write_json
from scripts.train_v5_spatial.v6_topology_learner import (
    PairSample,
    TopologyNode,
    read_position_priors,
    read_size_priors,
)


class V6GraphTopologyGeneratorTest(unittest.TestCase):
    def test_budget_decode_keeps_sparse_graph_instead_of_threshold_density(self) -> None:
        nodes = [
            TopologyNode("house", "a", "entryway", (3000.0, 3000.0), (1,), (0, 0, 0), (300, 300, 3000)),
            TopologyNode("house", "b", "corridor", (3000.0, 3000.0), (1,), (0, 0, 0), (300, 300, 3000)),
            TopologyNode("house", "c", "living_room", (3000.0, 3000.0), (1,), (0, 0, 0), (300, 300, 3000)),
        ]
        pairs = [
            PairSample("house", "a", "b", nodes[0], nodes[1], 1.0),
            PairSample("house", "a", "c", nodes[0], nodes[2], 0.0),
            PairSample("house", "b", "c", nodes[1], nodes[2], 1.0),
        ]
        selected, info = decode_graph_edges(pairs, [0.95, 0.90, 0.85], 2.0 / 3.0, ["a", "b", "c"])
        self.assertEqual(len(selected), 2)
        self.assertEqual(info["selected_edge_count"], 2)
        self.assertTrue(info["connected_after_decode"])

    def test_graph_samples_read_program_size_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase10 = root / "phase10"
            phase10.mkdir()
            sample_phase10_house(phase10 / "house_test.json")
            write_json(
                root / "size_predictions" / "house_test" / "predicted_sizes.json",
                {
                    "house_id": "house_test",
                    "groups": [
                        {
                            "functional_id": "entryway_0",
                            "predicted": {
                                "area_ratio": 0.1,
                                "width_ratio": 0.25,
                                "depth_ratio": 1.0,
                                "part_count": 1,
                            },
                        }
                    ],
                },
            )
            samples = load_graph_samples(phase10, None, read_size_priors(root))
        self.assertEqual(len(samples), 1)
        entry = next(node for node in samples[0].nodes if node.node_id == "entryway_0")
        self.assertEqual(entry.size_prior, (0.1, 0.25, 1.0, 0.125))

    def test_graph_samples_read_program_position_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase10 = root / "phase10"
            phase10.mkdir()
            sample_phase10_house(phase10 / "house_test.json")
            write_json(
                root / "position_predictions" / "house_test" / "predicted_positions.json",
                {
                    "house_id": "house_test",
                    "groups": [
                        {
                            "functional_id": "entryway_0",
                            "predicted": {
                                "center_x_ratio": 0.25,
                                "center_y_ratio": 0.75,
                                "zone_x": 1,
                                "zone_y": 2,
                            },
                        }
                    ],
                },
            )
            samples = load_graph_samples(phase10, None, {}, read_position_priors(root))
        entry = next(node for node in samples[0].nodes if node.node_id == "entryway_0")
        self.assertEqual(entry.position_prior, (0.25, 0.75, 0.5, 1.0))

    def test_model_prediction_exports_complete_topology_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase10 = Path(tmp)
            sample_phase10_house(phase10 / "house_test.json")
            sample = load_graph_samples(phase10, None, {})[0]
        device = torch.device("cpu")
        node_features, _pair_indices, relations, _labels, _stats, _target = tensors_for_graph(
            sample, "program_only", device
        )
        model = GraphTopologyGenerator(node_features.shape[1], relations.shape[1], hidden=16)
        topology, metrics = predict_graph(model, sample, "program_only", device)
        self.assertEqual(topology["schema"], "graphspace_v6_graph_topology_generator_v1")
        self.assertIn("graph_decode", topology)
        self.assertEqual(len(topology["nodes"]), 3)
        self.assertGreaterEqual(metrics["predicted_edge_count"], 2)


if __name__ == "__main__":
    unittest.main()
