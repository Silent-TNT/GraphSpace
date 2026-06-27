import torch

from scripts.train_v5_spatial.v6_graph_coarse_layout_model import (
    GraphCoarseLayoutModel,
    boundary_loss,
    contact_loss,
    edge_gap_loss,
    floor_occupancy_loss,
    graph_feature_from_fields,
    predict_graph_layout_ratios,
    repulsion_loss,
)


def test_graph_coarse_layout_model_predicts_one_bbox_per_node() -> None:
    model = GraphCoarseLayoutModel(feature_dim=24, hidden=32, steps=2)
    features = torch.rand(3, 24)
    adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    pred = model(features, adjacency)

    assert pred.shape == (3, 4)
    assert torch.all(pred >= 0.0)
    assert torch.all(pred <= 1.0)
    assert edge_gap_loss(pred, adjacency) >= 0.0
    assert contact_loss(pred, adjacency) >= 0.0
    assert repulsion_loss(pred, [(1,), (1,), (2,)]) >= 0.0
    assert floor_occupancy_loss(pred, pred, [(1,), (1,), (2,)]) == 0.0
    assert boundary_loss(pred) >= 0.0


def test_graph_coarse_layout_exported_inference_interface() -> None:
    node_fields = [
        {
            "node_id": "living_room_0",
            "room_type": "living_room",
            "floors": [1],
            "site": (12000.0, 9000.0),
            "type_index": 0,
            "type_count": 1,
            "group_count": 2,
            "size_prior": (0.08, 0.25, 0.25, 0.125),
            "position_prior": (0.4, 0.5, 0.5, 0.5),
        },
        {
            "node_id": "dining_room_0",
            "room_type": "dining_room",
            "floors": [1],
            "site": (12000.0, 9000.0),
            "type_index": 0,
            "type_count": 1,
            "group_count": 2,
            "size_prior": (0.06, 0.2, 0.2, 0.125),
            "position_prior": (0.6, 0.5, 0.5, 0.5),
        },
    ]
    feature_dim = int(
        graph_feature_from_fields(
            "living_room",
            (0,),
            site=(18_000, 15_000),
            type_index=0,
            type_count=1,
            group_count=len(node_fields),
            size_prior=(0.2, 0.3, 0.4, 0.12),
            position_prior=(0.5, 0.5, 0.5, 0.5),
        ).numel()
    )
    model = GraphCoarseLayoutModel(feature_dim=feature_dim, hidden=32, steps=2)

    ratios = predict_graph_layout_ratios(
        model,
        torch.device("cpu"),
        node_fields,
        [("living_room_0", "dining_room_0")],
    )

    assert set(ratios) == {"living_room_0", "dining_room_0"}
    assert all(len(value) == 4 for value in ratios.values())
