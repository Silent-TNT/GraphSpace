"""Spatially aligned graph-voxel model for native 300 mm instance masks."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from fullres_dataset import INPUT_CHANNELS
from instance_model import InstanceGraphEncoder


class ResidualBlock3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.block(value))


class FullResolutionGraphVoxelModel(nn.Module):
    """Predict every room directly on the 88x88x20 voxel canvas."""

    def __init__(
        self,
        spatial_channels: int = 24,
        query_channels: int = 32,
        architecture: str = "v2",
    ) -> None:
        super().__init__()
        if architecture not in {"v1", "v2"}:
            raise ValueError(f"unknown architecture: {architecture}")
        self.architecture = architecture
        self.graph = InstanceGraphEncoder(hidden=64)
        self.instance_embedding = nn.Embedding(64, 16)
        self.stem = nn.Sequential(
            nn.Conv3d(INPUT_CHANNELS, spatial_channels, 3, padding=1),
            nn.GroupNorm(8, spatial_channels),
            nn.SiLU(),
            ResidualBlock3d(spatial_channels),
        )
        if architecture == "v1":
            self.down = nn.Sequential(
                nn.Conv3d(
                    spatial_channels,
                    spatial_channels * 2,
                    3,
                    stride=(2, 2, 1),
                    padding=1,
                ),
                nn.GroupNorm(8, spatial_channels * 2),
                nn.SiLU(),
                ResidualBlock3d(spatial_channels * 2),
            )
            self.up = nn.Sequential(
                nn.Conv3d(
                    spatial_channels * 2,
                    spatial_channels,
                    3,
                    padding=1,
                ),
                nn.GroupNorm(8, spatial_channels),
                nn.SiLU(),
            )
        else:
            self.down1 = self._down_block(
                spatial_channels,
                spatial_channels * 2,
                (2, 2, 2),
            )
            self.down2 = self._down_block(
                spatial_channels * 2,
                spatial_channels * 4,
                (2, 2, 2),
            )
            self.down3 = self._down_block(
                spatial_channels * 4,
                spatial_channels * 4,
                (2, 2, 1),
            )
            self.bottleneck = ResidualBlock3d(spatial_channels * 4)
            self.decode2 = self._fusion_block(
                spatial_channels * 8,
                spatial_channels * 4,
            )
            self.decode1 = self._fusion_block(
                spatial_channels * 6,
                spatial_channels * 2,
            )
            self.decode0 = self._fusion_block(
                spatial_channels * 3,
                spatial_channels,
            )
            self.graph_film = nn.Linear(64, spatial_channels * 2)
        self.spatial_projection = nn.Conv3d(
            spatial_channels,
            query_channels,
            1,
        )
        self.empty_head = nn.Conv3d(spatial_channels, 1, 1)
        self.query_projection = nn.Sequential(
            nn.Linear(64 + 64 + 16, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, query_channels + 1),
        )
        self.scale = math.sqrt(query_channels)

    @staticmethod
    def _down_block(
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int, int],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(
                input_channels,
                output_channels,
                3,
                stride=stride,
                padding=1,
            ),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            ResidualBlock3d(output_channels),
        )

    @staticmethod
    def _fusion_block(
        input_channels: int,
        output_channels: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            ResidualBlock3d(output_channels),
        )

    def forward(
        self,
        volume: torch.Tensor,
        nodes: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_features, graph_features = self.graph(
            nodes,
            node_mask,
            adjacency,
        )
        base = self.stem(volume)
        if self.architecture == "v1":
            coarse = self.down(base)
            coarse = F.interpolate(
                coarse,
                size=base.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            fused_spatial = base + self.up(coarse)
        else:
            level1 = self.down1(base)
            level2 = self.down2(level1)
            level3 = self.bottleneck(self.down3(level2))
            decoded2 = self.decode2(
                torch.cat(
                    (
                        F.interpolate(
                            level3,
                            size=level2.shape[-3:],
                            mode="trilinear",
                            align_corners=False,
                        ),
                        level2,
                    ),
                    dim=1,
                )
            )
            decoded1 = self.decode1(
                torch.cat(
                    (
                        F.interpolate(
                            decoded2,
                            size=level1.shape[-3:],
                            mode="trilinear",
                            align_corners=False,
                        ),
                        level1,
                    ),
                    dim=1,
                )
            )
            fused_spatial = self.decode0(
                torch.cat(
                    (
                        F.interpolate(
                            decoded1,
                            size=base.shape[-3:],
                            mode="trilinear",
                            align_corners=False,
                        ),
                        base,
                    ),
                    dim=1,
                )
            )
            scale, shift = self.graph_film(graph_features).chunk(2, dim=-1)
            fused_spatial = fused_spatial * (
                1.0 + 0.25 * torch.tanh(scale)[:, :, None, None, None]
            ) + shift[:, :, None, None, None]
        spatial = self.spatial_projection(fused_spatial)
        empty_logits = self.empty_head(fused_spatial)[:, 0]

        indices = torch.arange(nodes.shape[1], device=nodes.device)
        indices = indices.clamp_max(self.instance_embedding.num_embeddings - 1)
        instance = self.instance_embedding(indices)[None].expand(
            nodes.shape[0],
            -1,
            -1,
        )
        graph = graph_features[:, None].expand(-1, nodes.shape[1], -1)
        query_and_bias = self.query_projection(
            torch.cat((node_features, graph, instance), dim=-1)
        )
        query = query_and_bias[:, :, :-1]
        bias = query_and_bias[:, :, -1]
        logits = torch.einsum("bnc,bcxyz->bnxyz", query, spatial) / self.scale
        logits = logits + bias[:, :, None, None, None]
        invalid = node_mask[:, :, None, None, None] == 0
        logits = logits.masked_fill(invalid, -20.0)
        return {
            "instance_logits": logits,
            "empty_logits": empty_logits,
        }
