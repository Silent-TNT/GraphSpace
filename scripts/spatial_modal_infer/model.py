from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv, global_mean_pool

try:
    from .config import (
        COND_COUNT_SCALE,
        COND_DIM,
        COND_EMBED_DIM,
        DECODER_INIT_X,
        DECODER_INIT_Y,
        DECODER_INIT_Z,
        HIDDEN_DIM,
        LATENT_DIM,
        NODE_IN_DIM,
        NUM_CHANNELS,
        NUM_ROOM_TYPES,
        RES_X,
        RES_Y,
        RES_Z,
    )
except ImportError:
    from config import (
        COND_COUNT_SCALE,
        COND_DIM,
        COND_EMBED_DIM,
        DECODER_INIT_X,
        DECODER_INIT_Y,
        DECODER_INIT_Z,
        HIDDEN_DIM,
        LATENT_DIM,
        NODE_IN_DIM,
        NUM_CHANNELS,
        NUM_ROOM_TYPES,
        RES_X,
        RES_Y,
        RES_Z,
    )


NODE_TYPES = ["living", "service", "circulation"]


def build_hetero_convs(in_dim, out_dim, vertical_gat=False):
    h_conv = {}
    for s in NODE_TYPES:
        for d in NODE_TYPES:
            h_conv[(s, "horizontal", d)] = SAGEConv(in_dim, out_dim)
    vertical_pairs = [
        ("circulation", "living"),
        ("living", "circulation"),
        ("circulation", "service"),
        ("service", "circulation"),
        ("circulation", "circulation"),
    ]
    for s, d in vertical_pairs:
        if vertical_gat:
            h_conv[(s, "vertical", d)] = GATv2Conv(
                in_dim, out_dim, heads=2, concat=False, add_self_loops=False
            )
        else:
            h_conv[(s, "vertical", d)] = SAGEConv(in_dim, out_dim)
    return HeteroConv(h_conv, aggr="sum")


def _valid_groups(groups: int, channels: int) -> int:
    g = min(groups, channels)
    while channels % g != 0:
        g -= 1
    return max(1, g)


class CondEncoder(nn.Module):
    """Raw V4 condition vector to the FiLM conditioning embedding."""

    def __init__(self, cond_dim=COND_DIM, embed_dim=COND_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(self, cond):
        return self.net(cond)


class FiLM3d(nn.Module):
    """GroupNorm plus condition modulation: norm(x) * (1 + gamma) + beta."""

    def __init__(self, num_channels, embed_dim=COND_EMBED_DIM, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(_valid_groups(groups, num_channels), num_channels, affine=False)
        self.to_film = nn.Linear(embed_dim, num_channels * 2)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def forward(self, x, cond_embed):
        x = self.norm(x)
        gamma, beta = self.to_film(cond_embed).chunk(2, dim=1)
        gamma = gamma.view(x.size(0), -1, 1, 1, 1)
        beta = beta.view(x.size(0), -1, 1, 1, 1)
        return x * (1.0 + gamma) + beta


class HeteroGraphVAEEncoder(nn.Module):
    def __init__(self, node_in_dim=NODE_IN_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = build_hetero_convs(node_in_dim, hidden_dim, vertical_gat=True)
        self.conv2 = build_hetero_convs(hidden_dim, hidden_dim, vertical_gat=False)
        self.node_types = NODE_TYPES
        self.to_mu = nn.Linear(hidden_dim + len(self.node_types), latent_dim)
        self.to_logvar = nn.Linear(hidden_dim + len(self.node_types), latent_dim)

    def _device(self, x_dict):
        for x in x_dict.values():
            return x.device
        return torch.device("cpu")

    def _pool(self, x_dict, batch_dict):
        bs = 1
        for b in batch_dict.values():
            if b.numel() > 0:
                bs = max(bs, int(b.max()) + 1)
        dev = self._device(x_dict)
        feats = []
        counts = []
        for ntype in self.node_types:
            if ntype in batch_dict and ntype in x_dict and x_dict[ntype].size(0) > 0:
                feats.append(global_mean_pool(x_dict[ntype], batch_dict[ntype], size=bs))
                counts.append(torch.bincount(batch_dict[ntype], minlength=bs).float().unsqueeze(1))
            else:
                feats.append(torch.zeros(bs, HIDDEN_DIM, device=dev))
                counts.append(torch.zeros(bs, 1, device=dev))
        h = torch.stack(feats, dim=0).mean(dim=0)
        count_feat = torch.cat(counts, dim=1) / COND_COUNT_SCALE
        return torch.cat([h, count_feat], dim=1)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        conv1 = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(conv1[k]) for k in x_dict}
        conv2 = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(conv2[k]) for k in x_dict}
        h = self._pool(x_dict, batch_dict)
        return self.to_mu(h), self.to_logvar(h)


class ConditionalVoxelDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, cond_dim=COND_DIM, channels=NUM_CHANNELS):
        super().__init__()
        self.init_volume_size = (256, DECODER_INIT_X, DECODER_INIT_Y, DECODER_INIT_Z)
        self.cond_mlp = CondEncoder(cond_dim, COND_EMBED_DIM)
        self.fc = nn.Linear(latent_dim + COND_EMBED_DIM, int(np.prod(self.init_volume_size)))
        self.deconv1 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)
        self.film1 = FiLM3d(128)
        self.deconv2 = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)
        self.film2 = FiLM3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1)
        self.film3 = FiLM3d(32)
        self.deconv4 = nn.ConvTranspose3d(32, channels, 4, stride=2, padding=1)

    def forward(self, z, cond):
        cond_embed = self.cond_mlp(cond)
        x = torch.cat([z, cond_embed], dim=-1)
        d = self.fc(x).view(z.size(0), *self.init_volume_size)
        h = torch.relu(self.film1(self.deconv1(d), cond_embed))
        h = torch.relu(self.film2(self.deconv2(h), cond_embed))
        h = torch.relu(self.film3(self.deconv3(h), cond_embed))
        x = self.deconv4(h)
        return x[:, :, :RES_X, :RES_Y, :RES_Z]


class SpatialModalCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HeteroGraphVAEEncoder()
        self.decoder = ConditionalVoxelDecoder()
        self.count_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, NUM_ROOM_TYPES),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def predict_counts(self, z):
        return self.count_head(z)

    def forward(self, batch):
        try:
            from .graph_utils import graph_batch_dict, graph_condition
        except ImportError:
            from graph_utils import graph_batch_dict, graph_condition

        mu, logvar = self.encoder(batch.x_dict, batch.edge_index_dict, graph_batch_dict(batch))
        z = self.reparameterize(mu, logvar)
        cond = graph_condition(batch)
        logits = self.decoder(z, cond)
        return logits, mu, logvar
