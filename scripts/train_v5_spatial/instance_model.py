"""Node-conditioned 3D room-instance box predictor."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from instance_dataset import INSTANCE_VOLUME_CHANNELS
from staged_dataset import NODE_DIM


class InstanceGraphEncoder(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.input = nn.Linear(NODE_DIM, hidden)
        self.self_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(3)]
        )
        self.rel_layers = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
                for _ in range(3)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])

    def forward(
        self,
        nodes: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = F.silu(self.input(nodes))
        for layer_index in range(3):
            update = self.self_layers[layer_index](value)
            for relation_index in range(2):
                relation = adjacency[:, relation_index]
                degree = relation.sum(dim=-1, keepdim=True).clamp_min(1.0)
                message = torch.bmm(relation, value) / degree
                update = update + self.rel_layers[layer_index][relation_index](
                    message
                )
            value = F.silu(self.norms[layer_index](update))
        weights = node_mask.unsqueeze(-1)
        pooled = (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return value, pooled


class InstancePlacementPolicy(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.graph = InstanceGraphEncoder(hidden=64)
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
        feature_dim = 64 + 64 + base_channels * 4 + 1
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
        room_index: torch.Tensor,
        step_ratio: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_features, graph_features = self.graph(
            nodes,
            node_mask,
            adjacency,
        )
        batch_indices = torch.arange(nodes.shape[0], device=nodes.device)
        query = node_features[batch_indices, room_index]
        voxel = self.voxel(volume)
        features = torch.cat(
            (query, graph_features, voxel, step_ratio[:, None]),
            dim=-1,
        )
        return {"box": self.box_head(features)}
