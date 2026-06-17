"""Shared graph and 3D voxel policy for the seven V5 generation stages."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from staged_dataset import NODE_DIM, STAGE_COUNT, VOLUME_CHANNELS


class StagedGraphEncoder(nn.Module):
    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.input = nn.Linear(NODE_DIM, hidden)
        self.self_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(2)]
        )
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
        node_mask: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        value = F.silu(self.input(nodes))
        for layer_index in range(2):
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
        return (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class StagedSpatialPolicy(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.graph = StagedGraphEncoder(hidden=48)
        self.stage_embedding = nn.Embedding(STAGE_COUNT, 24)
        self.encoder = nn.Sequential(
            nn.Conv3d(len(VOLUME_CHANNELS), base_channels, 3, padding=1),
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
        )
        condition_dim = 48 + 24
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )
        self.decoder_block_1 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.SiLU(),
        )
        self.decoder_block_2 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
        )
        self.mask_head = nn.Conv3d(base_channels, 2, 1)
        self.reachability_head = nn.Sequential(
            nn.Linear(base_channels * 4 + condition_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.cut_head = nn.Sequential(
            nn.Linear(base_channels * 4 + condition_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        volume: torch.Tensor,
        nodes: torch.Tensor,
        node_mask: torch.Tensor,
        adjacency: torch.Tensor,
        stage_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        graph = self.graph(nodes, node_mask, adjacency)
        stage = self.stage_embedding(stage_id)
        condition = torch.cat((graph, stage), dim=-1)
        encoded = self.encoder(volume)
        conditioned = encoded + self.condition(condition)[:, :, None, None, None]
        pooled = F.adaptive_avg_pool3d(conditioned, 1).flatten(1)
        decoded = F.interpolate(
            conditioned,
            size=(22, 22, 10),
            mode="trilinear",
            align_corners=False,
        )
        decoded = self.decoder_block_1(decoded)
        decoded = F.interpolate(
            decoded,
            size=(44, 44, 20),
            mode="trilinear",
            align_corners=False,
        )
        decoded = self.decoder_block_2(decoded)
        action_features = torch.cat((pooled, condition), dim=-1)
        return {
            "mask_logits": self.mask_head(decoded),
            "reachability_logit": self.reachability_head(
                action_features
            ).squeeze(-1),
            "cut_ratio": self.cut_head(action_features).squeeze(-1),
        }
