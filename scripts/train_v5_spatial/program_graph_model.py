"""Learned user-condition to room-program and topology model."""
from __future__ import annotations

import torch
from torch import nn

from program_graph_dataset import NODE_INPUT_DIM


class ProgramGraphModel(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 4) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(NODE_INPUT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.floor_head = nn.Linear(hidden, 3)
        self.area_head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.lighting_head = nn.Linear(hidden, 3)
        self.exterior_head = nn.Linear(hidden, 4)
        self.relation_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(
        self,
        node_input: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(
            self.input(node_input),
            src_key_padding_mask=node_mask == 0,
        )
        left = encoded[:, :, None, :].expand(
            -1,
            -1,
            encoded.shape[1],
            -1,
        )
        right = encoded[:, None, :, :].expand(
            -1,
            encoded.shape[1],
            -1,
            -1,
        )
        pair = torch.cat(
            (left, right, torch.abs(left - right), left * right),
            dim=-1,
        )
        relation = self.relation_head(pair)
        relation = 0.5 * (relation + relation.transpose(1, 2))
        return {
            "floor_logits": self.floor_head(encoded),
            "area": self.area_head(encoded)[:, :, 0],
            "lighting_logits": self.lighting_head(encoded),
            "exterior_logits": self.exterior_head(encoded),
            "relation_logits": relation,
        }
