"""Losses and lightweight metrics for V5 minimal training."""
from __future__ import annotations

import torch
from torch.nn import functional as F


LOSS_WEIGHTS = {
    "class": 1.0,
    "center": 2.0,
    "offset": 0.2,
    "boundary": 1.0,
    "cross_floor": 1.0,
    "count": 0.1,
}


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def focal_heatmap_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    positive = target.eq(1.0).float()
    negative = target.lt(1.0).float()
    negative_weight = (1.0 - target).pow(4)
    positive_loss = -torch.log(probability.clamp_min(1e-6))
    positive_loss = positive_loss * (1.0 - probability).pow(2) * positive
    negative_loss = -torch.log((1.0 - probability).clamp_min(1e-6))
    negative_loss = negative_loss * probability.pow(2) * negative_weight * negative
    return masked_mean(positive_loss + negative_loss, mask)


def compute_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
) -> dict[str, torch.Tensor]:
    site = batch["site_mask"].float()
    site_two = site.expand(-1, 2, -1, -1)
    class_target = batch["class_grid"].long()
    class_loss = F.cross_entropy(
        output["class_logits"].flatten(0, 1),
        class_target.flatten(0, 1),
        ignore_index=255,
    )
    center_loss = focal_heatmap_loss(
        output["center_logits"],
        batch["center_heatmap"].float(),
        site_two,
    )
    valid_offset = batch["center_valid_mask"].float().unsqueeze(2)
    offset_error = F.smooth_l1_loss(
        output["center_offset"],
        batch["center_offset"].float(),
        reduction="none",
    )
    offset_loss = masked_mean(offset_error, valid_offset)
    boundary_target = batch["boundary_mask"].float()
    boundary_error = F.binary_cross_entropy_with_logits(
        output["boundary_logits"], boundary_target, reduction="none"
    )
    boundary_loss = masked_mean(boundary_error, site_two)
    cross_floor_target = batch["cross_floor_mask"].float()
    cross_floor_error = -(
        8.0
        * cross_floor_target
        * F.logsigmoid(output["cross_floor_logits"])
        + (1.0 - cross_floor_target)
        * F.logsigmoid(-output["cross_floor_logits"])
    )
    cross_floor_loss = masked_mean(cross_floor_error, site_two)
    count_target = torch.cat(
        (
            batch["floor_instance_counts"].float(),
            batch["class_instance_counts"].float().flatten(1),
        ),
        dim=1,
    )
    count_loss = F.smooth_l1_loss(
        output["count_prediction"], count_target
    )
    total = (
        LOSS_WEIGHTS["class"] * class_loss
        + LOSS_WEIGHTS["center"] * center_loss
        + LOSS_WEIGHTS["offset"] * offset_loss
        + LOSS_WEIGHTS["boundary"] * boundary_loss
        + LOSS_WEIGHTS["cross_floor"] * cross_floor_loss
        + LOSS_WEIGHTS["count"] * count_loss
    )
    return {
        "total": total,
        "class": class_loss,
        "center": center_loss,
        "offset": offset_loss,
        "boundary": boundary_loss,
        "cross_floor": cross_floor_loss,
        "count": count_loss,
    }


@torch.no_grad()
def compute_metrics(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
) -> dict[str, float]:
    class_target = batch["class_grid"].long()
    valid = class_target.ne(255)
    predicted = output["class_logits"].argmax(dim=2)
    class_accuracy = (
        predicted.eq(class_target).logical_and(valid).sum()
        / valid.sum().clamp_min(1)
    )
    count_target = torch.cat(
        (
            batch["floor_instance_counts"].float(),
            batch["class_instance_counts"].float().flatten(1),
        ),
        dim=1,
    )
    count_mae = (output["count_prediction"] - count_target).abs().mean()
    return {
        "class_accuracy": float(class_accuracy.cpu()),
        "count_mae": float(count_mae.cpu()),
    }
