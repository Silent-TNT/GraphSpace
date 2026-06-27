import torch

from scripts.train_v5_spatial.generate_phase24_from_user_conditions import (
    build_user_group_samples,
    learned_topology_sample_from_user_topology,
)
from scripts.train_v5_spatial.v6_graph_topology_generator import (
    GraphTopologyGenerator,
    predict_graph,
    tensors_for_graph,
)


def _touches(left, right) -> bool:
    lx0, ly0, _lz0 = left["box_min"]
    lx1, ly1, _lz1 = left["box_max"]
    rx0, ry0, _rz0 = right["box_min"]
    rx1, ry1, _rz1 = right["box_max"]
    horizontal = (lx1 == rx0 or rx1 == lx0) and min(ly1, ry1) > max(ly0, ry0)
    vertical = (ly1 == ry0 or ry1 == ly0) and min(lx1, rx1) > max(lx0, rx0)
    return horizontal or vertical


def test_graph_coarse_layout_places_target_neighbors_in_contact() -> None:
    topology = {
        "nodes": [
            {"id": "living_room_0", "type": "living_room", "floor": 1, "position": [0.1, 0.1]},
            {"id": "dining_room_0", "type": "dining_room", "floor": 1, "position": [0.9, 0.9]},
        ],
        "edges": [{"source": "living_room_0", "target": "dining_room_0", "relation": "horizontal"}],
        "evidence": {
            "node_conditions": {
                "living_room_0": {"area_ratio": 0.08},
                "dining_room_0": {"area_ratio": 0.06},
            }
        },
    }

    _samples, seed_rooms, _used = build_user_group_samples(
        12000.0,
        9000.0,
        {"living_room": 1, "dining_room": 1},
        {"living_room": 1, "dining_room": 1},
        topology,
        coarse_layout_strategy="graph",
    )

    by_group = {room["functional_id"]: room for room in seed_rooms}
    assert _touches(by_group["living_room_0"], by_group["dining_room_0"])


def test_user_topology_can_be_decoded_by_learned_graph_generator_shape() -> None:
    topology = {
        "seed": 42,
        "nodes": [
            {"id": "entryway_0", "type": "entryway", "floor": 1, "position": [0.1, 0.5]},
            {"id": "living_room_0", "type": "living_room", "floor": 1, "position": [0.5, 0.5]},
            {"id": "stairs_0", "type": "stairs", "floor": "1&2", "position": [0.8, 0.5]},
        ],
        "edges": [
            {"source": "entryway_0", "target": "living_room_0", "relation": "horizontal"},
            {"source": "living_room_0", "target": "stairs_0", "relation": "horizontal"},
        ],
        "evidence": {
            "node_conditions": {
                "entryway_0": {"area_ratio": 0.03},
                "living_room_0": {"area_ratio": 0.10},
                "stairs_0": {"area_ratio": 0.04},
            }
        },
    }
    sample = learned_topology_sample_from_user_topology(
        topology,
        12000.0,
        9000.0,
        {"entryway": 1, "living_room": 1, "stairs": 1},
        {"entryway": 1, "living_room": 1, "stairs": 1},
    )
    device = torch.device("cpu")
    node_features, _pair_indices, relations, _labels, _stats, _target = tensors_for_graph(
        sample,
        "program_size_position",
        device,
    )
    model = GraphTopologyGenerator(node_features.shape[1], relations.shape[1], hidden=16)
    predicted, metrics = predict_graph(model, sample, "program_size_position", device)
    assert predicted["schema"] == "graphspace_v6_graph_topology_generator_v1"
    assert len(predicted["nodes"]) == 3
    assert metrics["predicted_edge_count"] >= 2
