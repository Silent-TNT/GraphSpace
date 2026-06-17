"""A small 2D two-floor decoder for testing the V5 output targets."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class V5MinimalNet(nn.Module):
    """Decode site and room-count conditions into V5 dense supervision heads."""

    def __init__(
        self,
        condition_dim: int = 24,
        base_channels: int = 32,
        class_count: int = 12,
    ) -> None:
        super().__init__()
        hidden = base_channels * 4
        self.base_channels = base_channels
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
        )
        self.spatial_projection = nn.Linear(hidden, hidden * 11 * 11)
        self.decode_22 = ConvBlock(hidden, base_channels * 3)
        self.decode_44 = ConvBlock(base_channels * 3, base_channels * 2)
        self.decode_88 = ConvBlock(base_channels * 2, base_channels)
        self.fuse_site = ConvBlock(base_channels + 3, base_channels)
        self.class_head = nn.Conv2d(base_channels, 2 * class_count, 1)
        self.center_head = nn.Conv2d(base_channels, 2, 1)
        self.offset_head = nn.Conv2d(base_channels, 4, 1)
        self.boundary_head = nn.Conv2d(base_channels, 2, 1)
        self.cross_floor_head = nn.Conv2d(base_channels, 2, 1)
        self.count_head = nn.Linear(hidden, 24)
        coords = torch.linspace(-1.0, 1.0, 88)
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer(
            "coordinates", torch.stack((grid_x, grid_y), dim=0)[None]
        )

    def forward(
        self, condition: torch.Tensor, site_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch_size = condition.shape[0]
        condition_features = self.condition_encoder(condition)
        value = self.spatial_projection(condition_features).view(
            batch_size, self.base_channels * 4, 11, 11
        )
        value = self.decode_22(
            F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        )
        value = self.decode_44(
            F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        )
        value = self.decode_88(
            F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        )
        coords = self.coordinates.expand(batch_size, -1, -1, -1)
        features = self.fuse_site(torch.cat((value, site_mask, coords), dim=1))
        return {
            "class_logits": self.class_head(features).view(
                batch_size, 2, 12, 88, 88
            ),
            "center_logits": self.center_head(features),
            "center_offset": self.offset_head(features).view(
                batch_size, 2, 2, 88, 88
            ),
            "boundary_logits": self.boundary_head(features),
            "cross_floor_logits": self.cross_floor_head(features),
            "count_prediction": self.count_head(condition_features),
        }
