from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_v5_spatial.v6_multipart_decoder import (
    FunctionalPartDataset,
    GroupSample,
    MultiPartDecoder,
    build_target_topology,
    boxes_overlap,
    decode_parts,
    feature_vector,
    masked_mse,
    read_conditioning_topologies,
    repair_overlaps,
    target_tensor,
    topology_placement_search,
    write_json,
)


def sample_group() -> GroupSample:
    return GroupSample(
        house_id="house_test",
        group_id="corridor_0",
        room_type="corridor",
        site=(12000.0, 9000.0),
        floors=[1],
        box_min=[0.0, 0.0, 0.0],
        box_max=[3600.0, 1200.0, 3000.0],
        target_parts=[
            {
                "id": "room_0",
                "functional_id": "corridor_0",
                "type": "corridor",
                "box_min": [0.0, 0.0, 0.0],
                "box_max": [1200.0, 1200.0, 3000.0],
            },
            {
                "id": "room_1",
                "functional_id": "corridor_0",
                "type": "corridor",
                "box_min": [1200.0, 0.0, 0.0],
                "box_max": [2400.0, 1200.0, 3000.0],
            },
            {
                "id": "room_2",
                "functional_id": "corridor_0",
                "type": "corridor",
                "box_min": [2400.0, 0.0, 0.0],
                "box_max": [3600.0, 1200.0, 3000.0],
            },
        ],
    )


class V6MultiPartDecoderTest(unittest.TestCase):
    def test_target_tensor_encodes_relative_part_boxes(self) -> None:
        target, mask = target_tensor(sample_group(), max_parts=4)
        self.assertEqual(mask.tolist(), [1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(float(target[0, 0]), 0.0)
        self.assertAlmostEqual(float(target[0, 3]), 1.0 / 3.0, places=4)
        self.assertAlmostEqual(float(target[2, 0]), 2.0 / 3.0, places=4)
        self.assertAlmostEqual(float(target[2, 3]), 1.0, places=4)

    def test_decode_parts_preserves_functional_group_protocol(self) -> None:
        sample = sample_group()
        target, _mask = target_tensor(sample, max_parts=4)
        decoded = decode_parts(sample, target, max_parts=4)
        self.assertEqual(len(decoded), 3)
        self.assertEqual({part["functional_id"] for part in decoded}, {"corridor_0"})
        self.assertEqual(decoded[0]["id"], "corridor_0_part_0")
        self.assertEqual(decoded[1]["box_min"][0], decoded[0]["box_max"][0])
        self.assertEqual(decoded[2]["box_max"][0], 3600.0)

    def test_decode_parts_uses_floor_z_bounds_not_predicted_z(self) -> None:
        sample = GroupSample(
            house_id="house_test",
            group_id="dining_room_0",
            room_type="dining_room",
            site=(12000.0, 9000.0),
            floors=[2],
            box_min=[0.0, 0.0, 3000.0],
            box_max=[2400.0, 2400.0, 6000.0],
            target_parts=[
                {
                    "id": "room_0",
                    "functional_id": "bedroom_0",
                    "type": "bedroom",
                    "box_min": [0.0, 0.0, 3000.0],
                    "box_max": [2400.0, 2400.0, 6000.0],
                }
            ],
        )
        pred = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 0.1]])
        decoded = decode_parts(sample, pred, max_parts=1)
        self.assertEqual(decoded[0]["box_min"][2], 3000.0)
        self.assertEqual(decoded[0]["box_max"][2], 6000.0)

    def test_decode_parts_searches_non_overlapping_candidate(self) -> None:
        sample = sample_group()
        pred = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.34, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.34, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.34, 1.0, 1.0],
            ]
        )
        decoded = decode_parts(sample, pred, max_parts=3)
        for left_index, left in enumerate(decoded):
            for right in decoded[left_index + 1 :]:
                self.assertFalse(boxes_overlap(left, right))
        self.assertEqual(decoded[0]["box_min"][0], 0.0)
        self.assertEqual(decoded[1]["box_min"][0], 1200.0)
        self.assertEqual(decoded[2]["box_min"][0], 2400.0)

    def test_decode_parts_can_shrink_single_part_to_avoid_existing_occupancy(self) -> None:
        sample = GroupSample(
            house_id="house_test",
            group_id="entryway_0",
            room_type="entryway",
            site=(6000.0, 3000.0),
            floors=[1],
            box_min=[0.0, 0.0, 0.0],
            box_max=[1800.0, 900.0, 3000.0],
            target_parts=[
                {
                    "id": "room_0",
                    "functional_id": "entryway_0",
                    "type": "entryway",
                    "box_min": [0.0, 0.0, 0.0],
                    "box_max": [1800.0, 900.0, 3000.0],
                }
            ],
        )
        occupied = [
            {
                "id": "occupied",
                "type": "corridor",
                "box_min": [0.0, 0.0, 0.0],
                "box_max": [1500.0, 900.0, 3000.0],
            }
        ]
        pred = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
        decoded = decode_parts(sample, pred, max_parts=1, occupied=occupied)
        self.assertFalse(boxes_overlap(decoded[0], occupied[0]))
        self.assertEqual(decoded[0]["box_min"][0], 1500.0)
        self.assertEqual(decoded[0]["box_max"][0], 1800.0)

    def test_build_target_topology_uses_group_level_edges(self) -> None:
        source = {
            "house_id": "house_test",
            "metadata": {"building_size": {"x": 3000.0, "y": 1200.0, "z": 6000.0}},
            "functional_groups": [
                {"functional_id": "corridor_0", "type": "corridor", "floors": [1]},
                {"functional_id": "living_room_0", "type": "living_room", "floors": [1]},
            ],
            "rooms": [
                {
                    "id": "corridor_0_part_0",
                    "functional_id": "corridor_0",
                    "type": "corridor",
                    "box_min": [0.0, 0.0, 0.0],
                    "box_max": [600.0, 1200.0, 3000.0],
                },
                {
                    "id": "corridor_0_part_1",
                    "functional_id": "corridor_0",
                    "type": "corridor",
                    "box_min": [600.0, 0.0, 0.0],
                    "box_max": [1200.0, 1200.0, 3000.0],
                },
                {
                    "id": "living_room_0_part_0",
                    "functional_id": "living_room_0",
                    "type": "living_room",
                    "box_min": [1200.0, 0.0, 0.0],
                    "box_max": [3000.0, 1200.0, 3000.0],
                },
            ],
        }
        topology = build_target_topology(source)
        edges = {
            (edge["source"], edge["target"], edge["relation"])
            for edge in topology["edges"]
        }
        self.assertEqual(edges, {("corridor_0", "living_room_0", "horizontal")})
        self.assertEqual(topology["required_edges"], [["corridor_0", "living_room_0"]])

    def test_decode_parts_prefers_topology_preserving_candidate(self) -> None:
        sample = GroupSample(
            house_id="house_test",
            group_id="entryway_0",
            room_type="entryway",
            site=(1800.0, 900.0),
            floors=[1],
            box_min=[0.0, 0.0, 0.0],
            box_max=[900.0, 900.0, 3000.0],
            target_parts=[
                {
                    "id": "entryway_0_part_0",
                    "functional_id": "entryway_0",
                    "type": "entryway",
                    "box_min": [600.0, 0.0, 0.0],
                    "box_max": [900.0, 900.0, 3000.0],
                }
            ],
        )
        occupied = [
            {
                "id": "blocker",
                "functional_id": "other_0",
                "type": "utility",
                "box_min": [300.0, 0.0, 0.0],
                "box_max": [600.0, 900.0, 3000.0],
            },
            {
                "id": "corridor_0_part_0",
                "functional_id": "corridor_0",
                "type": "corridor",
                "box_min": [900.0, 0.0, 0.0],
                "box_max": [1200.0, 900.0, 3000.0],
            },
        ]
        pred = torch.tensor([[1.0 / 3.0, 0.0, 0.0, 2.0 / 3.0, 1.0, 1.0]])
        decoded = decode_parts(
            sample,
            pred,
            max_parts=1,
            occupied=occupied,
            target_neighbors={"corridor_0"},
        )
        self.assertEqual(decoded[0]["box_min"][0], 600.0)
        self.assertEqual(decoded[0]["box_max"][0], 900.0)

    def test_reads_predicted_topology_conditioning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topology_path = root / "topologies" / "house_test" / "predicted_topology.json"
            write_json(
                topology_path,
                {
                    "schema": "test",
                    "nodes": [],
                    "edges": [
                        {
                            "source": "entryway_0",
                            "target": "corridor_0",
                            "relation": "horizontal",
                        }
                    ],
                    "source": "learned_pairwise_topology_classifier_smoke",
                },
            )
            topologies = read_conditioning_topologies(root)
        self.assertEqual(set(topologies), {"house_test"})
        self.assertEqual(len(topologies["house_test"]["edges"]), 1)

    def test_topology_placement_search_moves_part_to_realize_missing_edge(self) -> None:
        sample = GroupSample(
            house_id="house_test",
            group_id="dining_room_0",
            room_type="dining_room",
            site=(1800.0, 900.0),
            floors=[1],
            box_min=[0.0, 0.0, 0.0],
            box_max=[900.0, 900.0, 3000.0],
            target_parts=[
                {
                    "id": "dining_room_0_part_0",
                    "functional_id": "dining_room_0",
                    "type": "dining_room",
                    "floor": 1,
                    "floors": [1],
                    "box_min": [600.0, 0.0, 0.0],
                    "box_max": [900.0, 900.0, 3000.0],
                }
            ],
        )
        rooms = [
            {
                "id": "dining_room_0_part_0",
                "functional_id": "dining_room_0",
                "type": "dining_room",
                "floor": 1,
                "floors": [1],
                "box_min": [0.0, 0.0, 0.0],
                "box_max": [300.0, 900.0, 3000.0],
            },
            {
                "id": "living_room_0_part_0",
                "functional_id": "living_room_0",
                "type": "living_room",
                "floor": 1,
                "floors": [1],
                "box_min": [900.0, 0.0, 0.0],
                "box_max": [1200.0, 900.0, 3000.0],
            },
        ]
        topology = {
            "nodes": [
                {"id": "dining_room_0", "type": "dining_room", "floor": 1, "floors": [1]},
                {"id": "living_room_0", "type": "living_room", "floor": 1, "floors": [1]},
            ],
            "edges": [
                {"source": "dining_room_0", "target": "living_room_0", "relation": "horizontal"}
            ],
            "required_edges": [["dining_room_0", "living_room_0"]],
        }
        repaired, report, summary = topology_placement_search(
            "house_test",
            rooms,
            {"dining_room_0": sample},
            {"dining_room": 1, "living_room": 1},
            (1800.0, 900.0),
            topology,
        )
        self.assertTrue(report["p0"]["pass"])
        self.assertEqual(summary["accepted_move_count"], 1)
        self.assertEqual(summary["final_topology"]["required_realized_edge_count"], 1)
        moved = next(room for room in repaired if room["functional_id"] == "dining_room_0")
        self.assertEqual(moved["box_min"][0], 600.0)
        self.assertEqual(moved["box_max"][0], 900.0)

    def test_overlap_repair_moves_colliding_part_before_topology_search(self) -> None:
        sample = GroupSample(
            house_id="house_test",
            group_id="dining_room_0",
            room_type="dining_room",
            site=(1800.0, 900.0),
            floors=[1],
            box_min=[0.0, 0.0, 0.0],
            box_max=[1800.0, 900.0, 3000.0],
            target_parts=[
                {
                    "id": "dining_room_0_part_0",
                    "functional_id": "dining_room_0",
                    "type": "dining_room",
                    "floor": 1,
                    "floors": [1],
                    "box_min": [900.0, 0.0, 0.0],
                    "box_max": [1500.0, 900.0, 3000.0],
                }
            ],
        )
        rooms = [
            {
                "id": "dining_room_0_part_0",
                "functional_id": "dining_room_0",
                "type": "dining_room",
                "floor": 1,
                "floors": [1],
                "box_min": [0.0, 0.0, 0.0],
                "box_max": [600.0, 900.0, 3000.0],
            },
            {
                "id": "living_room_0_part_0",
                "functional_id": "living_room_0",
                "type": "living_room",
                "floor": 1,
                "floors": [1],
                "box_min": [300.0, 0.0, 0.0],
                "box_max": [900.0, 900.0, 3000.0],
            },
        ]
        repaired, report, summary = repair_overlaps(
            "house_test",
            rooms,
            {"dining_room_0": sample},
            {"dining_room": 1, "living_room": 1},
            (1800.0, 900.0),
            {"nodes": [], "edges": [], "required_edges": []},
        )
        self.assertFalse(summary["initial_p0_pass"])
        self.assertTrue(report["p0"]["pass"])
        self.assertEqual(summary["initial_overlap_count"], 1)
        self.assertEqual(summary["final_overlap_count"], 0)
        self.assertEqual(summary["accepted_move_count"], 1)
        dining = next(room for room in repaired if room["functional_id"] == "dining_room_0")
        self.assertGreaterEqual(dining["box_min"][0], 900.0)

    def test_topology_placement_search_respects_max_move_distance(self) -> None:
        sample = GroupSample(
            house_id="house_test",
            group_id="dining_room_0",
            room_type="dining_room",
            site=(3600.0, 900.0),
            floors=[1],
            box_min=[0.0, 0.0, 0.0],
            box_max=[3600.0, 900.0, 3000.0],
            target_parts=[
                {
                    "id": "dining_room_0_part_0",
                    "functional_id": "dining_room_0",
                    "type": "dining_room",
                    "floor": 1,
                    "floors": [1],
                    "box_min": [2700.0, 0.0, 0.0],
                    "box_max": [3000.0, 900.0, 3000.0],
                }
            ],
        )
        rooms = [
            {
                "id": "dining_room_0_part_0",
                "functional_id": "dining_room_0",
                "type": "dining_room",
                "floor": 1,
                "floors": [1],
                "box_min": [0.0, 0.0, 0.0],
                "box_max": [300.0, 900.0, 3000.0],
            },
            {
                "id": "living_room_0_part_0",
                "functional_id": "living_room_0",
                "type": "living_room",
                "floor": 1,
                "floors": [1],
                "box_min": [3000.0, 0.0, 0.0],
                "box_max": [3300.0, 900.0, 3000.0],
            },
        ]
        topology = {
            "nodes": [
                {"id": "dining_room_0", "type": "dining_room", "floor": 1, "floors": [1]},
                {"id": "living_room_0", "type": "living_room", "floor": 1, "floors": [1]},
            ],
            "edges": [
                {"source": "dining_room_0", "target": "living_room_0", "relation": "horizontal"}
            ],
            "required_edges": [["dining_room_0", "living_room_0"]],
        }
        repaired, _report, summary = topology_placement_search(
            "house_test",
            rooms,
            {"dining_room_0": sample},
            {"dining_room": 1, "living_room": 1},
            (3600.0, 900.0),
            topology,
            max_move_mm=600.0,
        )
        self.assertEqual(summary["accepted_move_count"], 0)
        self.assertEqual(summary["final_topology"]["required_realized_edge_count"], 0)
        dining = next(room for room in repaired if room["functional_id"] == "dining_room_0")
        self.assertEqual(dining["box_min"][0], 0.0)

    def test_topology_placement_search_can_expand_beyond_group_bbox_locally(self) -> None:
        sample = GroupSample(
            house_id="house_test",
            group_id="dining_room_0",
            room_type="dining_room",
            site=(1800.0, 900.0),
            floors=[1],
            box_min=[0.0, 0.0, 0.0],
            box_max=[300.0, 900.0, 3000.0],
            target_parts=[
                {
                    "id": "dining_room_0_part_0",
                    "functional_id": "dining_room_0",
                    "type": "dining_room",
                    "floor": 1,
                    "floors": [1],
                    "box_min": [0.0, 0.0, 0.0],
                    "box_max": [300.0, 900.0, 3000.0],
                }
            ],
        )
        rooms = [
            {
                "id": "dining_room_0_part_0",
                "functional_id": "dining_room_0",
                "type": "dining_room",
                "floor": 1,
                "floors": [1],
                "box_min": [0.0, 0.0, 0.0],
                "box_max": [300.0, 900.0, 3000.0],
            },
            {
                "id": "living_room_0_part_0",
                "functional_id": "living_room_0",
                "type": "living_room",
                "floor": 1,
                "floors": [1],
                "box_min": [900.0, 0.0, 0.0],
                "box_max": [1200.0, 900.0, 3000.0],
            },
        ]
        topology = {
            "nodes": [
                {"id": "dining_room_0", "type": "dining_room", "floor": 1, "floors": [1]},
                {"id": "living_room_0", "type": "living_room", "floor": 1, "floors": [1]},
            ],
            "edges": [
                {"source": "dining_room_0", "target": "living_room_0", "relation": "horizontal"}
            ],
            "required_edges": [["dining_room_0", "living_room_0"]],
        }
        constrained, _report, constrained_summary = topology_placement_search(
            "house_test",
            rooms,
            {"dining_room_0": sample},
            {"dining_room": 1, "living_room": 1},
            (1800.0, 900.0),
            topology,
            max_move_mm=900.0,
        )
        expanded, expanded_report, expanded_summary = topology_placement_search(
            "house_test",
            rooms,
            {"dining_room_0": sample},
            {"dining_room": 1, "living_room": 1},
            (1800.0, 900.0),
            topology,
            max_move_mm=900.0,
            expand_search_to_site=True,
        )
        self.assertEqual(constrained_summary["accepted_move_count"], 0)
        constrained_dining = next(room for room in constrained if room["functional_id"] == "dining_room_0")
        self.assertEqual(constrained_dining["box_min"][0], 0.0)
        self.assertTrue(expanded_report["p0"]["pass"])
        self.assertEqual(expanded_summary["accepted_move_count"], 1)
        self.assertTrue(expanded_summary["expand_search_to_site"])
        expanded_dining = next(room for room in expanded if room["functional_id"] == "dining_room_0")
        self.assertEqual(expanded_dining["box_min"][0], 600.0)
        self.assertEqual(expanded_summary["final_topology"]["required_realized_edge_count"], 1)

    def test_topology_placement_search_can_move_linked_parts_together(self) -> None:
        sample = GroupSample(
            house_id="house_test",
            group_id="bedroom_0",
            room_type="bedroom",
            site=(1800.0, 900.0),
            floors=[2],
            box_min=[0.0, 0.0, 3000.0],
            box_max=[1200.0, 900.0, 6000.0],
            target_parts=[
                {
                    "id": "dining_room_0_part_0",
                    "functional_id": "dining_room_0",
                    "type": "dining_room",
                    "floor": 2,
                    "floors": [2],
                    "box_min": [600.0, 0.0, 3000.0],
                    "box_max": [900.0, 900.0, 6000.0],
                },
                {
                    "id": "dining_room_0_part_1",
                    "functional_id": "dining_room_0",
                    "type": "dining_room",
                    "floor": 2,
                    "floors": [2],
                    "box_min": [900.0, 0.0, 3000.0],
                    "box_max": [1200.0, 900.0, 6000.0],
                },
            ],
        )
        rooms = [
            {
                "id": "dining_room_0_part_0",
                "functional_id": "dining_room_0",
                "type": "dining_room",
                "floor": 2,
                "floors": [2],
                "box_min": [0.0, 0.0, 3000.0],
                "box_max": [300.0, 900.0, 6000.0],
            },
            {
                "id": "dining_room_0_part_1",
                "functional_id": "dining_room_0",
                "type": "dining_room",
                "floor": 2,
                "floors": [2],
                "box_min": [300.0, 0.0, 3000.0],
                "box_max": [600.0, 900.0, 6000.0],
            },
            {
                "id": "living_room_0_part_0",
                "functional_id": "living_room_0",
                "type": "living_room",
                "floor": 2,
                "floors": [2],
                "box_min": [1200.0, 0.0, 3000.0],
                "box_max": [1500.0, 900.0, 6000.0],
            },
        ]
        topology = {
            "nodes": [
                {"id": "dining_room_0", "type": "dining_room", "floor": 2, "floors": [2]},
                {"id": "living_room_0", "type": "living_room", "floor": 2, "floors": [2]},
            ],
            "edges": [{"source": "dining_room_0", "target": "living_room_0", "relation": "horizontal"}],
            "required_edges": [["dining_room_0", "living_room_0"]],
        }
        repaired, report, summary = topology_placement_search(
            "house_test",
            rooms,
            {"dining_room_0": sample},
            {"dining_room": 1, "living_room": 1},
            (1800.0, 900.0),
            topology,
            enable_linked_part_placement=True,
        )
        self.assertTrue(report["p0"]["pass"])
        self.assertEqual(summary["accepted_move_count"], 1)
        self.assertEqual(summary["moves"][0]["move_type"], "linked_part_group_move")
        moved_parts = [room for room in repaired if room["functional_id"] == "dining_room_0"]
        self.assertEqual([room["box_min"][0] for room in moved_parts], [600.0, 900.0])
        self.assertEqual(summary["final_topology"]["required_realized_edge_count"], 1)

    def test_topology_placement_search_can_use_controlled_size_adjustment(self) -> None:
        sample = GroupSample(
            house_id="house_test",
            group_id="dining_room_0",
            room_type="dining_room",
            site=(1500.0, 900.0),
            floors=[1],
            box_min=[0.0, 0.0, 0.0],
            box_max=[1200.0, 900.0, 3000.0],
            target_parts=[
                {
                    "id": "dining_room_0_part_0",
                    "functional_id": "dining_room_0",
                    "type": "dining_room",
                    "floor": 1,
                    "floors": [1],
                    "box_min": [900.0, 0.0, 0.0],
                    "box_max": [1200.0, 900.0, 3000.0],
                }
            ],
        )
        rooms = [
            {
                "id": "dining_room_0_part_0",
                "functional_id": "dining_room_0",
                "type": "dining_room",
                "floor": 1,
                "floors": [1],
                "box_min": [0.0, 0.0, 0.0],
                "box_max": [600.0, 900.0, 3000.0],
            },
            {
                "id": "utility_0_part_0",
                "functional_id": "utility_0",
                "type": "utility",
                "floor": 1,
                "floors": [1],
                "box_min": [600.0, 0.0, 0.0],
                "box_max": [900.0, 900.0, 3000.0],
            },
            {
                "id": "living_room_0_part_0",
                "functional_id": "living_room_0",
                "type": "living_room",
                "floor": 1,
                "floors": [1],
                "box_min": [1200.0, 0.0, 0.0],
                "box_max": [1500.0, 900.0, 3000.0],
            },
        ]
        topology = {
            "nodes": [
                {"id": "dining_room_0", "type": "dining_room", "floor": 1, "floors": [1]},
                {"id": "living_room_0", "type": "living_room", "floor": 1, "floors": [1]},
            ],
            "edges": [
                {"source": "dining_room_0", "target": "living_room_0", "relation": "horizontal"}
            ],
            "required_edges": [["dining_room_0", "living_room_0"]],
        }
        unchanged, _unchanged_report, unchanged_summary = topology_placement_search(
            "house_test",
            rooms,
            {"dining_room_0": sample},
            {"dining_room": 1, "living_room": 1, "utility": 1},
            (1500.0, 900.0),
            topology,
            max_move_mm=900.0,
        )
        adjusted, adjusted_report, adjusted_summary = topology_placement_search(
            "house_test",
            rooms,
            {"dining_room_0": sample},
            {"dining_room": 1, "living_room": 1, "utility": 1},
            (1500.0, 900.0),
            topology,
            max_move_mm=900.0,
            enable_controlled_size_adjustment=True,
            max_size_adjustment_mm=300.0,
        )
        self.assertEqual(unchanged_summary["accepted_move_count"], 0)
        unchanged_dining = next(room for room in unchanged if room["functional_id"] == "dining_room_0")
        self.assertEqual(unchanged_dining["box_min"][0], 0.0)
        self.assertTrue(adjusted_report["p0"]["pass"])
        self.assertEqual(adjusted_summary["accepted_move_count"], 1)
        self.assertEqual(adjusted_summary["moves"][0]["move_type"], "controlled_size_topology_adjustment")
        adjusted_dining = next(room for room in adjusted if room["functional_id"] == "dining_room_0")
        self.assertEqual(adjusted_dining["box_min"][0], 900.0)
        self.assertEqual(adjusted_dining["box_max"][0], 1200.0)
        self.assertEqual(adjusted_summary["final_topology"]["required_realized_edge_count"], 1)

    def test_model_can_overfit_one_group_interface(self) -> None:
        sample = sample_group()
        dataset = FunctionalPartDataset([sample], max_parts=4)
        feature_dim = int(dataset[0]["features"].numel())
        model = MultiPartDecoder(feature_dim, max_parts=4, hidden=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        features = feature_vector(sample, max_parts=4).unsqueeze(0)
        target, mask = target_tensor(sample, max_parts=4)
        target = target.unsqueeze(0)
        mask = mask.unsqueeze(0)
        initial = float(masked_mse(model(features), target, mask).detach())
        for _ in range(200):
            loss = masked_mse(model(features), target, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final = float(masked_mse(model(features), target, mask).detach())
        self.assertLess(final, initial * 0.05)


if __name__ == "__main__":
    unittest.main()
