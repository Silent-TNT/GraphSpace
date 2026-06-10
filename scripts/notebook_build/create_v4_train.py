# -*- coding: utf-8 -*-
"""从 V3 notebook 生成 V4 训练 notebook（88×88×24）。

V4 主题：解决"条件被淹没"（诊断见 notebooks/train/V4 实现方案（诊断结论与设计）.md）。
核心改动：
  1) 条件向量 占比→绝对数量（软缩放 COND_COUNT_SCALE）
  2) CondEncoder 升维(17→64) + FiLM 注入解码器每层 + GroupNorm 替代 BatchNorm
  3) 编码器计数感知 pooling（拼接每超类节点计数）
  4) 计数辅助损失（count_head 从 mu 预测房间数）

运行: python scripts/notebook_build/create_v4_train.py
"""
from __future__ import annotations

import json
import os
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "notebooks", "train", "260609-V3训练（88x88x24）.ipynb")
DST = os.path.join(ROOT, "notebooks", "train", "260609-V4训练（88x88x24）.ipynb")

RES_X, RES_Y, RES_Z = 88, 88, 24
COND_COUNT_SCALE = 4.0
COND_EMBED_DIM = 64
AUX_COUNT_WEIGHT = 0.1


def src_lines(code: str) -> list[str]:
    return [line + "\n" for line in textwrap.dedent(code).strip("\n").splitlines()]


def cell_text(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, code: str) -> None:
    cell["source"] = src_lines(code)
    cell["execution_count"] = None
    cell["outputs"] = []


# ===================== 替换块 =====================

MARKDOWN_0 = f"""\
# V4 训练专用 · {RES_X}×{RES_Y}×{RES_Z} · 条件 CVAE（修复"条件被淹没"）

基于 V3 训练管线。**诊断结论**（见 `V4 实现方案（诊断结论与设计）.md`）：V3 非后验坍缩，
真正病灶是**解码器无视条件**（改 cond 体素变化仅 0.034%，cond/z 影响比 0.025）。

## 相对 V3 的核心变更

| 项目 | V3 | **V4** |
|---|---|---|
| 条件语义 | 房间**占比** | **绝对数量**（软缩放 /{COND_COUNT_SCALE:.0f}） |
| 条件注入 | 入口 concat 一次 | **CondEncoder 升维(17→{COND_EMBED_DIM}) + FiLM 每层注入** |
| 解码器归一化 | BatchNorm3d | **GroupNorm(affine=False) + FiLM(γ,β)** |
| 编码器 pooling | mean-pool | **mean-pool + 每超类计数** |
| 辅助损失 | 无 | **计数头**（mu→房间数，smooth_l1×{AUX_COUNT_WEIGHT}） |
| 画布 | 88×88×24 | 88×88×24（不变） |

> 解码器签名仍为 `decoder(z, cond_raw)`（内部做 embed），便于推理包对接。
> KL 保持 V3 设置（坍缩非问题，隔离变量）。

## 步骤
| 步骤 | 作用 |
|---|---|
| Step 0–4 | 依赖、配置、预处理(QC)、V4 建模、训练 |
| Step 1b | 合成拓扑探针 |
| **Step D** | V4-a 诊断（训练后复测：cond/z 影响比应大幅上升） |
| Step 5 | 训练结果摘要 |
"""


# ---- Step 1（cell 含 config + 条件函数）的定点替换 ----
STEP1_REPLACEMENTS = [
    # 条件维度后追加 V4 常量
    (
        "NODE_IN_DIM = 8\nCOND_DIM = 3 + len(ROOM_TYPES) + 3",
        "NODE_IN_DIM = 8\n"
        "COND_DIM = 3 + len(ROOM_TYPES) + 3\n"
        f"NUM_ROOM_TYPES = len(ROOM_TYPES)\n"
        f"COND_COUNT_SCALE = {COND_COUNT_SCALE}  # V4: 房间数软缩放（4间→1.0，保留 3vs5 差异）\n"
        f"COND_EMBED_DIM = {COND_EMBED_DIM}     # V4: 条件升维（FiLM 注入）\n"
        f"AUX_COUNT_WEIGHT = {AUX_COUNT_WEIGHT}  # V4: 计数辅助损失权重",
    ),
    # 条件：房间占比 → 绝对数量
    (
        "    for rt in ROOM_TYPES:\n"
        "        cond.append(float(stats.get(rt, 0)) / total)",
        "    for rt in ROOM_TYPES:\n"
        "        cond.append(float(stats.get(rt, 0)) / COND_COUNT_SCALE)  # V4: 绝对数量",
    ),
    # 采光：占比 → 绝对数量
    (
        "    cond.extend([direct / total, indirect / total, none / total])",
        "    cond.extend([direct / COND_COUNT_SCALE, indirect / COND_COUNT_SCALE, none / COND_COUNT_SCALE])  # V4",
    ),
    # 缓存目录 / 版本 / 权重命名 v3 → v4
    ("OUT_DIR = os.path.join(DATA_DIR, 'processed_tensors_v3')",
     "OUT_DIR = os.path.join(DATA_DIR, 'processed_tensors_v4')"),
    ("# OUT_DIR = '/content/processed_tensors_v3'",
     "# OUT_DIR = '/content/processed_tensors_v4'"),
    ("TENSOR_CACHE_VERSION = 'v3_tensors_88x88x24'",
     "TENSOR_CACHE_VERSION = 'v4_tensors_88x88x24'"),
    ("TRAIN_CACHE_VERSION = 'v3_train_88x88x24'",
     "TRAIN_CACHE_VERSION = 'v4_train_88x88x24'"),
    ("WEIGHT_FILENAME = 'spatial_modal_cvae_v3_88x88x24.pth'",
     "WEIGHT_FILENAME = 'spatial_modal_cvae_v4_88x88x24.pth'"),
    ("WEIGHT_PROBE_FILENAME = 'spatial_modal_cvae_v3_88x88x24_probe_best.pth'",
     "WEIGHT_PROBE_FILENAME = 'spatial_modal_cvae_v4_88x88x24_probe_best.pth'"),
    ("CHECKPOINT_FILENAME = 'spatial_modal_cvae_v3_88x88x24_checkpoint.pt'",
     "CHECKPOINT_FILENAME = 'spatial_modal_cvae_v4_88x88x24_checkpoint.pt'"),
    ("SPLIT_FILENAME = 'train_val_split_v3.json'",
     "SPLIT_FILENAME = 'train_val_split_v4.json'"),
]


# ---- Step 2 ----
STEP2_REPLACEMENTS = [
    ("# Step 2: 批量预处理 JSON -> .pt（V3 画布 88×88×24 + QC）",
     "# Step 2: 批量预处理 JSON -> .pt（V4 画布 88×88×24 + 绝对数量条件 + QC）"),
    ("'preprocess_qc_v3_tensors_88x88x24.json'",
     "'preprocess_qc_v4_tensors_88x88x24.json'"),
]


# ---- Step 3：整段替换为 V4 模型 ----
STEP3_MODEL = f"""\
# Step 3: V4 CVAE 模型（绝对数量条件 + CondMLP + FiLM + GroupNorm + 计数辅助头）
import math
LATENT_DIM = 256
HIDDEN_DIM = 128


def build_hetero_convs(in_dim, out_dim, vertical_gat=False):
    h_conv = {{}}
    node_types = ['living', 'service', 'circulation']
    for s in node_types:
        for d in node_types:
            h_conv[(s, 'horizontal', d)] = SAGEConv(in_dim, out_dim)
    vertical_pairs = [
        ('circulation', 'living'), ('living', 'circulation'),
        ('circulation', 'service'), ('service', 'circulation'),
        ('circulation', 'circulation'),
    ]
    for s, d in vertical_pairs:
        if vertical_gat:
            h_conv[(s, 'vertical', d)] = GATv2Conv(
                in_dim, out_dim, heads=2, concat=False, add_self_loops=False
            )
        else:
            h_conv[(s, 'vertical', d)] = SAGEConv(in_dim, out_dim)
    return HeteroConv(h_conv, aggr='sum')


def _valid_groups(groups, channels):
    g = min(groups, channels)
    while channels % g != 0:
        g -= 1
    return max(1, g)


class CondEncoder(nn.Module):
    \"\"\"原始条件(绝对数量, COND_DIM) → cond_embed(COND_EMBED_DIM)。\"\"\"
    def __init__(self, cond_dim=COND_DIM, embed_dim=COND_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(),
        )

    def forward(self, cond):
        return self.net(cond)


class FiLM3d(nn.Module):
    \"\"\"GroupNorm(affine=False) + 条件调制 (1+γ)·x + β；γ 初始≈0 → 初始恒等。\"\"\"
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
    \"\"\"计数感知 pooling：mean-pool 拼接每超类节点计数，缓解 mean-pool 稀释。\"\"\"
    def __init__(self, node_in_dim=NODE_IN_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = build_hetero_convs(node_in_dim, hidden_dim, vertical_gat=True)
        self.conv2 = build_hetero_convs(hidden_dim, hidden_dim, vertical_gat=False)
        self.node_types = ['living', 'service', 'circulation']
        self.to_mu = nn.Linear(hidden_dim + len(self.node_types), latent_dim)
        self.to_logvar = nn.Linear(hidden_dim + len(self.node_types), latent_dim)

    def _device(self, x_dict):
        for x in x_dict.values():
            return x.device
        return torch.device('cpu')

    def _pool(self, x_dict, batch_dict):
        bs = 1
        for b in batch_dict.values():
            if b.numel() > 0:
                bs = max(bs, int(b.max()) + 1)
        dev = self._device(x_dict)
        feats, counts = [], []
        for nt in self.node_types:
            if nt in batch_dict and nt in x_dict and x_dict[nt].size(0) > 0:
                feats.append(global_mean_pool(x_dict[nt], batch_dict[nt], size=bs))
                c = torch.bincount(batch_dict[nt], minlength=bs).float().unsqueeze(1)
            else:
                feats.append(torch.zeros(bs, HIDDEN_DIM, device=dev))
                c = torch.zeros(bs, 1, device=dev)
            counts.append(c)
        h = torch.stack(feats, dim=0).mean(dim=0)            # (bs, hidden)
        cnt = torch.cat(counts, dim=1) / COND_COUNT_SCALE     # (bs, 3)
        return torch.cat([h, cnt], dim=1)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        x_dict = {{k: torch.relu(self.conv1(x_dict, edge_index_dict)[k]) for k in x_dict}}
        x_dict = {{k: torch.relu(self.conv2(x_dict, edge_index_dict)[k]) for k in x_dict}}
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
        ce = self.cond_mlp(cond)
        d = self.fc(torch.cat([z, ce], dim=-1)).view(z.size(0), *self.init_volume_size)
        h = torch.relu(self.film1(self.deconv1(d), ce))
        h = torch.relu(self.film2(self.deconv2(h), ce))
        h = torch.relu(self.film3(self.deconv3(h), ce))
        x = self.deconv4(h)
        return x[:, :, :RES_X, :RES_Y, :RES_Z]


class SpatialModalCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HeteroGraphVAEEncoder()
        self.decoder = ConditionalVoxelDecoder()
        self.count_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.SiLU(), nn.Linear(128, NUM_ROOM_TYPES)
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
        mu, logvar = self.encoder(
            batch.x_dict, batch.edge_index_dict, graph_batch_dict(batch)
        )
        z = self.reparameterize(mu, logvar)
        cond = graph_condition(batch)
        logits = self.decoder(z, cond)
        return logits, mu, logvar


model = SpatialModalCVAE().to(device)
print(f'V4 参数量: {{sum(p.numel() for p in model.parameters()):,}}')
"""


# ---- Step 4：损失加 aux + 版本号/标题 ----
STEP4_REPLACEMENTS = [
    (
        "def compute_batch_loss(batch, kl_weight):\n"
        "    bs = graph_batch_size(batch)\n"
        "    target = batch.y.view(bs, NUM_CHANNELS, RES_X, RES_Y, RES_Z)\n"
        "    logits, mu, logvar = forward_model(model, batch)\n"
        "    bce = F.binary_cross_entropy_with_logits(logits, target)\n"
        "    dice = dice_loss_with_logits(logits, target)\n"
        "    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())\n"
        "    loss = bce + DICE_WEIGHT * dice + kl_weight * kl\n"
        "    return loss, bs",
        "def compute_batch_loss(batch, kl_weight):\n"
        "    bs = graph_batch_size(batch)\n"
        "    target = batch.y.view(bs, NUM_CHANNELS, RES_X, RES_Y, RES_Z)\n"
        "    logits, mu, logvar = forward_model(model, batch)\n"
        "    bce = F.binary_cross_entropy_with_logits(logits, target)\n"
        "    dice = dice_loss_with_logits(logits, target)\n"
        "    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())\n"
        "    # V4: 计数辅助损失（mu → 房间数），监督潜变量编码房间程序\n"
        "    cond = graph_condition(batch)\n"
        "    target_counts = cond[:, 3:3 + NUM_ROOM_TYPES] * COND_COUNT_SCALE\n"
        "    aux = F.smooth_l1_loss(model.predict_counts(mu), target_counts)\n"
        "    loss = bce + DICE_WEIGHT * dice + kl_weight * kl + AUX_COUNT_WEIGHT * aux\n"
        "    return loss, bs",
    ),
    ("print(f'开始训练 ({device}) | V3 {RES_X}×{RES_Y}×{RES_Z} | AMP=",
     "print(f'开始训练 ({device}) | V4 {RES_X}×{RES_Y}×{RES_Z} | AMP="),
    ("        'model_version': 'v3',\n"
     "        'grid': f'{RES_X}x{RES_Y}x{RES_Z}',\n"
     "        'init_volume': [256, DECODER_INIT_X, DECODER_INIT_Y, DECODER_INIT_Z],",
     "        'model_version': 'v4',\n"
     "        'grid': f'{RES_X}x{RES_Y}x{RES_Z}',\n"
     "        'init_volume': [256, DECODER_INIT_X, DECODER_INIT_Y, DECODER_INIT_Z],\n"
     "        'cond_semantics': 'absolute_counts',\n"
     "        'cond_count_scale': COND_COUNT_SCALE,\n"
     "        'cond_embed_dim': COND_EMBED_DIM,\n"
     "        'decoder_norm': 'groupnorm+film',\n"
     "        'aux_count_weight': AUX_COUNT_WEIGHT,"),
    ("    'model_version': 'v3',\n"
     "    'grid': f'{RES_X}x{RES_Y}x{RES_Z}',\n"
     "    'cache_version': CACHE_VERSION,",
     "    'model_version': 'v4',\n"
     "    'grid': f'{RES_X}x{RES_Y}x{RES_Z}',\n"
     "    'cache_version': CACHE_VERSION,"),
    ("axes[0].set_title(f'V3 Loss ({RES_X}×{RES_Y}×{RES_Z})')",
     "axes[0].set_title(f'V4 Loss ({RES_X}×{RES_Y}×{RES_Z})')"),
]


# ---- Step 5 markdown：示例命令 v3 → v4 ----
STEP5_REPLACEMENTS = [
    ("spatial_modal_cvae_v3_88x88x24.pth", "spatial_modal_cvae_v4_88x88x24.pth"),
]


def apply_all(text: str, replacements: list[tuple[str, str]], cell_tag: str) -> str:
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"[{cell_tag}] 替换失败，未找到锚点:\n{old[:80]}...")
        text = text.replace(old, new)
    return text


def main():
    with open(SRC, encoding="utf-8") as f:
        nb = json.load(f)

    patched = {"step1": False, "step2": False, "step3": False, "step4": False, "step5": False}

    for cell in nb["cells"]:
        text = cell_text(cell)
        if cell["cell_type"] == "markdown" and text.startswith("# V3 训练专用"):
            set_source(cell, MARKDOWN_0)
            continue
        if cell["cell_type"] == "markdown" and "## Step 5: 训练结果摘要" in text:
            set_source(cell, apply_all(text, STEP5_REPLACEMENTS, "step5"))
            patched["step5"] = True
            continue
        if cell["cell_type"] != "code":
            continue
        if "def build_condition_vector" in text and "WEIGHT_FILENAME" in text:
            set_source(cell, apply_all(text, STEP1_REPLACEMENTS, "step1"))
            patched["step1"] = True
        elif "# Step 2: 批量预处理" in text:
            set_source(cell, apply_all(text, STEP2_REPLACEMENTS, "step2"))
            patched["step2"] = True
        elif "class SpatialModalCVAE" in text:
            set_source(cell, STEP3_MODEL)
            patched["step3"] = True
        elif "def compute_batch_loss" in text:
            set_source(cell, apply_all(text, STEP4_REPLACEMENTS, "step4"))
            patched["step4"] = True

    missing = [k for k, v in patched.items() if not v]
    if missing:
        raise RuntimeError(f"以下步骤未被定位/替换: {missing}")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("Wrote:", DST)
    print("Cells:", len(nb["cells"]))
    print(f"V4: 绝对数量条件(/{COND_COUNT_SCALE:.0f}) + CondMLP({COND_EMBED_DIM}) + FiLM/GroupNorm "
          f"+ 计数pooling + 计数辅助头(w={AUX_COUNT_WEIGHT})")


if __name__ == "__main__":
    main()
