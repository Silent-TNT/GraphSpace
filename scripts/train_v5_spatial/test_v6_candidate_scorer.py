#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from scripts.train_v5_spatial.v6_candidate_scorer import (
    CandidateScorer,
    build_examples,
    candidate_feature_vector,
    load_candidate_scorer,
)


def test_candidate_feature_vector_shape_and_range() -> None:
    features = candidate_feature_vector(
        room_type="living_room",
        floors=[1],
        candidate=(2, 3, 12, 11),
        preferred_xy=(1, 2),
        site_cells=(60, 50),
        placed={
            "stairs_0": {
                "box": (12, 3, 18, 11),
                "floors": [1, 2],
            }
        },
        neighbors_by_node={"living_room_0": {"stairs_0"}},
        node_id="living_room_0",
        floor_occupied_cells={1: 100, 2: 80},
    )
    assert features.ndim == 1
    assert features.numel() == len(torch.zeros(11)) + 15
    assert torch.isfinite(features).all()
    assert float(features.min()) >= 0.0


def test_candidate_scorer_forward_and_checkpoint_roundtrip() -> None:
    model = CandidateScorer(feature_dim=26, hidden=32)
    batch = torch.rand(4, 26)
    pred = model(batch)
    assert pred.shape == (4,)
    assert float(pred.min()) >= 0.0
    assert float(pred.max()) <= 1.0

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "candidate_scorer.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": {"feature_dim": 26, "hidden": 32},
            },
            path,
        )
        loaded = load_candidate_scorer(path, torch.device("cpu"))
        loaded_pred = loaded(batch)
        assert torch.allclose(pred, loaded_pred)


def test_build_examples_smoke() -> None:
    examples = build_examples(
        Path("data/phase10_functional_parts/samples"),
        max_houses=1,
        candidates_per_group=8,
        seed=123,
    )
    assert examples
    assert examples[0].features.numel() == 26
    assert all(0.0 <= example.target <= 1.0 for example in examples[:20])
