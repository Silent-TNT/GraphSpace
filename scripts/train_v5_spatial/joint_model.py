"""Joint whole-house room box predictor."""
from __future__ import annotations

import torch
from torch import nn

from instance_dataset import INSTANCE_VOLUME_CHANNELS
from instance_model import InstanceGraphEncoder


class JointLayoutPolicy(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.graph = InstanceGraphEncoder(hidden=64)
        self.instance_embedding = nn.Embedding(64, 16)
        self.voxel = nn.Sequential(
            nn.Conv3d(
                len(INSTANCE_VOLUME_CHANNELS),
                base_channels,
                3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
            nn.Conv3d(
                base_channels,
                base_channels * 2,
                3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(4, base_channels * 2),
            nn.SiLU(),
            nn.Conv3d(
                base_channels * 2,
                base_channels * 4,
                3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        feature_dim = 64 + 16 + 64 + base_channels * 4
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        volume: torch.Tensor,
        nodes: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_features, graph_features = self.graph(nodes, node_mask, adjacency)
        voxel_features = self.voxel(volume)
        instance_indices = torch.arange(
            nodes.shape[1],
            device=nodes.device,
        ).clamp_max(self.instance_embedding.num_embeddings - 1)
        instance_features = self.instance_embedding(instance_indices)[None].expand(
            nodes.shape[0],
            -1,
            -1,
        )
        global_features = torch.cat((graph_features, voxel_features), dim=-1)
        expanded = global_features[:, None, :].expand(
            -1,
            nodes.shape[1],
            -1,
        )
        boxes = self.box_head(
            torch.cat((node_features, instance_features, expanded), dim=-1)
        )
        return {"boxes": boxes * node_mask[:, :, None]}
