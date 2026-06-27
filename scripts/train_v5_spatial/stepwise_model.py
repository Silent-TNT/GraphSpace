"""Graph-voxel policy for Phase9 stepwise spatial actions."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from instance_model import InstanceGraphEncoder
from stepwise_dataset import ACTION_TO_ID, STEPWISE_VOLUME_CHANNELS


class StepwiseVoxelEncoder(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                len(STEPWISE_VOLUME_CHANNELS),
                base_channels,
                3,
                stride=(2, 2, 1),
                padding=1,
            ),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
            nn.Conv3d(
                base_channels,
                base_channels * 2,
                3,
                stride=(2, 2, 2),
                padding=1,
            ),
            nn.GroupNorm(4, base_channels * 2),
            nn.SiLU(),
            nn.Conv3d(
                base_channels * 2,
                base_channels * 4,
                3,
                stride=(2, 2, 2),
                padding=1,
            ),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.output_dim = base_channels * 4

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.net(volume)


class StepwiseActionPolicy(nn.Module):
    def __init__(self, base_channels: int = 16, hidden: int = 128) -> None:
        super().__init__()
        self.graph = InstanceGraphEncoder(hidden=64)
        self.voxel = StepwiseVoxelEncoder(base_channels)
        feature_dim = 64 + self.voxel.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(hidden, len(ACTION_TO_ID))
        self.accept_head = nn.Linear(hidden, 1)
        self.progress_head = nn.Linear(hidden, 1)
        self.axis_head = nn.Linear(hidden, 3)
        self.cut_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.box_head = nn.Sequential(nn.Linear(hidden, 6), nn.Sigmoid())
        self.node_head = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
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
        fused = self.fusion(torch.cat((graph_features, voxel_features), dim=-1))
        node_context = fused[:, None].expand(-1, nodes.shape[1], -1)
        node_logits = self.node_head(
            torch.cat((node_features, node_context), dim=-1)
        ).squeeze(-1)
        node_logits = node_logits.masked_fill(node_mask == 0, -20.0)
        return {
            "action_logits": self.action_head(fused),
            "accept_logit": self.accept_head(fused).squeeze(-1),
            "progress_logit": self.progress_head(fused).squeeze(-1),
            "axis_logits": self.axis_head(fused),
            "cut": self.cut_head(fused).squeeze(-1),
            "box": self.box_head(fused),
            "node_logits": node_logits,
        }
