"""Topology and 3D voxel fusion network for block-cut actions."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RelationalGraphEncoder(nn.Module):
    def __init__(self, node_dim: int = 15, hidden: int = 64) -> None:
        super().__init__()
        self.input = nn.Linear(node_dim, hidden)
        self.self_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.rel_layers = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
                for _ in range(2)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])

    def forward(
        self,
        nodes: torch.Tensor,
        active: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = F.silu(self.input(nodes))
        for layer_index in range(2):
            update = self.self_layers[layer_index](value)
            for relation_index in range(2):
                degree = adjacency[:, relation_index].sum(dim=-1, keepdim=True).clamp_min(1)
                message = torch.bmm(adjacency[:, relation_index], value) / degree
                update = update + self.rel_layers[layer_index][relation_index](message)
            value = F.silu(self.norms[layer_index](update))
        weights = active.unsqueeze(-1)
        pooled = (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return value, pooled


class VoxelStateEncoder(nn.Module):
    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.SiLU(),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.net(volume)


class SpatialModalCutPolicy(nn.Module):
    def __init__(self, hidden: int = 96) -> None:
        super().__init__()
        self.graph = RelationalGraphEncoder(hidden=64)
        self.voxel = VoxelStateEncoder(hidden=hidden)
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.axis_head = nn.Linear(hidden, 4)
        self.cut_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.partition_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.side_head = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        volume: torch.Tensor,
        nodes: torch.Tensor,
        active: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_features, graph_features = self.graph(nodes, active, adjacency)
        voxel_features = self.voxel(volume)
        features = self.fusion(torch.cat((graph_features, voxel_features), dim=-1))
        return {
            "axis_logits": self.axis_head(features),
            "cut_ratio": self.cut_head(features).squeeze(-1),
            "left_fraction": self.partition_head(features).squeeze(-1),
            "side_logits": self.side_head(
                torch.cat(
                    (
                        node_features,
                        features.unsqueeze(1).expand(
                            -1, node_features.shape[1], -1
                        ),
                    ),
                    dim=-1,
                )
            ),
        }
