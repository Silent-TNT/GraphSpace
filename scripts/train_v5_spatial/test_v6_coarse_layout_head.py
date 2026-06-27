import torch

from scripts.train_v5_spatial.generate_phase24_from_user_conditions import user_position_prior, user_size_prior
from scripts.train_v5_spatial.v6_coarse_layout_head import (
    CoarseLayoutHead,
    CoarseLayoutSample,
    coarse_layout_feature,
    coarse_layout_target,
    predict_coarse_layout_ratios,
)


def test_coarse_layout_feature_and_target_are_program_to_bbox_priors() -> None:
    sample = CoarseLayoutSample(
        house_id="house_test",
        group_id="g_living_0",
        room_type="living_room",
        site=(12000.0, 9000.0),
        floors=(1,),
        type_index=0,
        type_count=1,
        group_count=5,
        box_min=(1500.0, 900.0, 0.0),
        box_max=(5100.0, 3900.0, 3000.0),
        size_prior=(0.1, 0.3, 0.3333333333, 0.25),
        position_prior=(0.25, 0.3, 0.0, 0.5),
    )

    feature = coarse_layout_feature(sample)
    target = coarse_layout_target(sample)

    assert feature.shape == (len(feature),)
    assert torch.allclose(target, torch.tensor([0.275, 0.26666668, 0.3, 0.33333334]))
    assert torch.allclose(feature[-8:][:4], torch.tensor([0.1, 0.3, 0.33333334, 0.25]))


def test_predict_coarse_layout_ratios_uses_exported_inference_interface() -> None:
    feature = coarse_layout_feature(
        CoarseLayoutSample(
            house_id="house_test",
            group_id="g_bedroom_0",
            room_type="bedroom",
            site=(15000.0, 12000.0),
            floors=(2,),
            type_index=0,
            type_count=2,
            group_count=8,
            box_min=(6000.0, 3000.0, 3000.0),
            box_max=(9000.0, 6000.0, 6000.0),
            size_prior=(0.05, 0.2, 0.25, 0.125),
            position_prior=(0.5, 0.4, 0.5, 0.5),
        )
    )
    model = CoarseLayoutHead(int(feature.numel()))
    ratios = predict_coarse_layout_ratios(
        model,
        torch.device("cpu"),
        "bedroom",
        (2,),
        (15000.0, 12000.0),
        0,
        2,
        8,
        (0.05, 0.2, 0.25, 0.125),
        (0.5, 0.4, 0.5, 0.5),
    )

    assert len(ratios) == 4
    assert all(0.0 <= value <= 1.0 for value in ratios)


def test_user_prior_helpers_stay_normalized() -> None:
    size_prior = user_size_prior("living_room", 0.2, 12000.0, 9000.0, 2)
    position_prior = user_position_prior("g_living_0", {"g_living_0": (1.2, -0.1)})

    assert all(0.0 <= value <= 1.0 for value in size_prior)
    assert position_prior == (1.0, 0.0, 1.0, 0.0)
