# -*- coding: utf-8 -*-
"""从 MVP notebook 生成 470 套训练优化 v1 副本。"""
from __future__ import annotations

import json
import os
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "notebooks", "train", "260523-74条件生成与量化评估（MVP）.ipynb")
DST = os.path.join(ROOT, "notebooks", "train", "260607-470条件生成与量化评估（训练优化v1）.ipynb")


def src_lines(code: str) -> list[str]:
    return [line + "\n" for line in textwrap.dedent(code).strip("\n").splitlines()]


def set_cell_source(nb: dict, idx: int, code: str) -> None:
    nb["cells"][idx]["source"] = src_lines(code)
    nb["cells"][idx]["execution_count"] = None
    nb["cells"][idx]["outputs"] = []


def insert_cell(nb: dict, idx: int, cell_type: str, code: str) -> None:
    nb["cells"].insert(
        idx,
        {
            "cell_type": cell_type,
            "metadata": {},
            "source": src_lines(code),
            "execution_count": None,
            "outputs": [],
        },
    )


MARKDOWN_0 = """\
# 470 套数据 · 条件 CVAE · 生成与量化评估（训练优化 v1）

基于 `260523-74条件生成与量化评估（MVP）.ipynb` 副本，面向 `data/processed/**/house_*.json` 全量数据。

| 优化项 | 说明 |
|---|---|
| 递归数据扫描 | `data/processed` 三批子目录 |
| 持久化划分 | `train_val_split_v470.json`，训练与 Step 7 共用 |
| 拓扑加噪 | 训练时边 dropout + 节点坐标微扰 |
| KL Annealing | β 从 0 线性升至 `KL_WEIGHT_MAX` |
| 梯度积累 + AMP | 有效 batch 更大、显存更省 |
| 合成拓扑探针 | 每 `PROBE_EVERY` epoch 监控非空体素率 |
| Best-of-K 评估 | Step 7 条件生成多次采样取最优 |

| 步骤 | 作用 |
|---|---|
| Step 0–4 | 安装依赖、配置、预处理、建模、训练 |
| **Step 1b** | 合成拓扑工具（训练探针 + Step 6 共用） |
| **Step R** | 加载已训练权重 |
| Step 5–7 | 重建评估、条件生成、量化对比 |
| Step 8 | 交互式生成 |

> **未纳入本版**：FiLM、Attention Pooling、LR Warmup（留作 v2 架构实验）
"""

STEP1_APPEND = """

# ---------- 470 训练优化 v1 配置 ----------
JSON_ROOT = os.path.join(DATA_DIR, 'processed')
OUT_DIR = os.path.join(DATA_DIR, 'processed_tensors_v470')
# Drive 空间不足时可改 Colab 本地（~6GB，会话结束丢失，需重跑 Step 2）：
# OUT_DIR = '/content/processed_tensors_v470'
CACHE_VERSION = 'v470_train_opt_v1'
WEIGHT_FILENAME = 'spatial_modal_cvae_470_v1.pth'
SPLIT_FILENAME = 'train_val_split_v470.json'

# 训练超参（Step 4 读取）
BATCH_SIZE = 4
ACCUM_STEPS = 4          # 有效 batch ≈ BATCH_SIZE * ACCUM_STEPS
USE_AMP = True
EPOCHS = 120
PATIENCE = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
DICE_WEIGHT = 0.3
KL_WEIGHT_MAX = 1e-4
KL_ANNEAL_EPOCHS = 25
EDGE_DROP_PROB = 0.10
NODE_JITTER = 0.02
AUGMENT_PROB = 0.5       # 每个训练样本施加拓扑加噪的概率
PROBE_EVERY = 10         # 合成拓扑随堂测试间隔（epoch）
GEN_EVAL_K = 8           # Step 7 条件生成 best-of-K

VAL_RATIO = 0.2
SPLIT_SEED = 42


def discover_json_files(root=None):
    root = root or JSON_ROOT
    if not os.path.isdir(root):
        return []
    return sorted(
        f for f in glob.glob(os.path.join(root, '**', 'house_*.json'), recursive=True)
        if not f.endswith('_topology.json')
    )


def split_path():
    return os.path.join(DATA_DIR, SPLIT_FILENAME)


def weight_path():
    return os.path.join(DATA_DIR, WEIGHT_FILENAME)


def load_train_val_split(pt_names, seed=SPLIT_SEED, val_ratio=VAL_RATIO):
    path = split_path()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        train_names = saved.get('train', [])
        val_names = saved.get('val', [])
        known = set(pt_names)
        train_names = [n for n in train_names if n in known]
        val_names = [n for n in val_names if n in known]
        if train_names or val_names:
            return train_names, val_names, saved
    rng = random.Random(seed)
    names = list(pt_names)
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = names[-n_val:]
    train_names = names[:-n_val]
    meta = {
        'seed': seed,
        'val_ratio': val_ratio,
        'total': len(names),
        'train_count': len(train_names),
        'val_count': len(val_names),
        'cache_version': CACHE_VERSION,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'train': train_names, 'val': val_names, 'meta': meta}, f, indent=2, ensure_ascii=False)
    return train_names, val_names, meta


def resolve_source_json(sample, pt_basename=None):
    if isinstance(sample, dict) and sample.get('source_json') and os.path.exists(sample['source_json']):
        return sample['source_json']
    if pt_basename:
        cand = os.path.join(DATA_DIR, pt_basename.replace('.pt', '.json'))
        if os.path.exists(cand):
            return cand
    return None


def augment_batch_topology(batch, edge_drop_prob=EDGE_DROP_PROB, node_jitter=NODE_JITTER):
    """直接在 HeteroDataBatch 上加噪，避免 to_data_list 拆分 batch 元数据出错。"""
    for edge_key in batch.edge_types:
        store = batch[edge_key]
        ei = store.edge_index
        if ei.numel() == 0:
            continue
        keep = torch.rand(ei.size(1), device=ei.device) > edge_drop_prob
        if not keep.any():
            keep = torch.zeros(ei.size(1), dtype=torch.bool, device=ei.device)
            keep[0] = True
        store.edge_index = ei[:, keep]
    for ntype in batch.node_types:
        x = batch[ntype].x
        if x is None or x.numel() == 0:
            continue
        noise = torch.zeros_like(x)
        if x.size(1) >= 5:
            noise[:, 2:5] = torch.randn(x.size(0), 3, device=x.device) * node_jitter
        batch[ntype].x = (x + noise).clamp(0.0, 5.0)
    return batch


# 覆盖 TOTAL_JSON_COUNT（递归统计）
TOTAL_JSON_COUNT = len(discover_json_files())
print(f'JSON 根目录: {JSON_ROOT}  发现 JSON: {TOTAL_JSON_COUNT}')
print(f'缓存目录: {OUT_DIR}  权重: {WEIGHT_FILENAME}')
"""

STEP1B = """\
# Step 1b: 合成拓扑工具（训练探针 & Step 6 条件生成共用）
import math
import networkx as nx

DEFAULT_ROOM_SIZE = {
    'entryway': (2400, 2400, 3000),
    'living_room': (6000, 4500, 3000),
    'dining_room': (3600, 3300, 3000),
    'kitchen': (3300, 3000, 3000),
    'bedroom': (3600, 3600, 3000),
    'bathroom': (2400, 2400, 3000),
    'corridor': (1800, 2400, 3000),
    'stairs': (3000, 3000, 6000),
    'utility': (2400, 2400, 3000),
    'balcony': (3000, 1800, 3000),
    'multi_purpose': (3600, 3300, 3000),
}

SYNTHETIC_PROBE_SPECS = [
    {'name': 'compact_2f', 'site_x': 15000, 'site_y': 12000, 'seed': 11,
     'room_counts': {'entryway': 1, 'living_room': 1, 'dining_room': 1, 'kitchen': 1,
                     'bedroom': 2, 'bathroom': 1, 'corridor': 1, 'stairs': 1}},
    {'name': 'standard_3br', 'site_x': 18000, 'site_y': 15000, 'seed': 22,
     'room_counts': {'entryway': 1, 'living_room': 1, 'dining_room': 1, 'kitchen': 1,
                     'bedroom': 3, 'bathroom': 2, 'corridor': 2, 'stairs': 1, 'balcony': 1}},
    {'name': 'large_4br', 'site_x': 21000, 'site_y': 18000, 'seed': 33,
     'room_counts': {'entryway': 1, 'living_room': 1, 'dining_room': 1, 'kitchen': 1,
                     'bedroom': 4, 'bathroom': 2, 'corridor': 2, 'stairs': 1, 'balcony': 1, 'utility': 1}},
]


def snap_modulus(v):
    return round(float(v) / VOXEL_SIZE) * VOXEL_SIZE


def build_user_request(site_x, site_y, room_counts, site_z=6000):
    counts = {k: int(v) for k, v in room_counts.items() if int(v) > 0}
    return {
        'site_x': float(site_x), 'site_y': float(site_y), 'site_z': float(site_z),
        'room_counts': counts,
    }


def build_program_topology(room_counts, seed=42):
    G = nx.Graph()
    nodes = []
    bath_i = corr_i = 0
    for r_type, count in room_counts.items():
        for i in range(count):
            nid = f"{r_type}_{i}"
            if r_type in ['entryway', 'living_room', 'dining_room', 'kitchen']:
                floor = 1
            elif r_type in ['bedroom', 'balcony']:
                floor = 2
            elif r_type == 'bathroom':
                floor = 1 if bath_i % 2 == 0 else 2
                bath_i += 1
            elif r_type == 'corridor':
                floor = 1 if corr_i % 2 == 0 else 2
                corr_i += 1
            elif r_type == 'stairs':
                floor = '1&2'
            else:
                floor = 1
            nodes.append((nid, r_type, floor))
            G.add_node(nid, type=r_type, floor=floor)

    f1_corr = [n for n, t, f in nodes if t == 'corridor' and f == 1]
    f2_corr = [n for n, t, f in nodes if t == 'corridor' and f == 2]
    f1_hub = f1_corr[0] if f1_corr else next((n for n, t, f in nodes if t == 'living_room'), nodes[0][0])
    f2_hub = f2_corr[0] if f2_corr else next((n for n, t, f in nodes if t == 'bedroom'), nodes[-1][0])

    edge_types = {}
    for nid, r_type, floor in nodes:
        if r_type == 'stairs':
            continue
        hub = f1_hub if floor == 1 else f2_hub
        if nid != hub:
            G.add_edge(nid, hub)
            edge_types[(nid, hub)] = 'horizontal'
            edge_types[(hub, nid)] = 'horizontal'

    stairs = [n for n, t, f in nodes if t == 'stairs']
    if stairs:
        st = stairs[0]
        if f1_hub != st:
            G.add_edge(st, f1_hub)
            edge_types[(st, f1_hub)] = 'vertical'
        if f2_hub != st and f2_hub != f1_hub:
            G.add_edge(st, f2_hub)
            edge_types[(st, f2_hub)] = 'vertical'

    pos = nx.spring_layout(G, seed=seed, k=1.2)
    return G, pos, nodes, edge_types


def layout_rooms_from_program(user_req, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    G, pos, nodes, edge_types = build_program_topology(user_req['room_counts'], seed=seed)
    sx, sy = user_req['site_x'], user_req['site_y']
    rooms = []
    for nid, r_type, floor in nodes:
        w, d, h = DEFAULT_ROOM_SIZE.get(r_type, (3600, 3600, 3000))
        px, py = pos[nid]
        cx = (px + 1) / 2 * (sx * 0.7) + sx * 0.15
        cy = (py + 1) / 2 * (sy * 0.7) + sy * 0.15
        cx, cy = snap_modulus(cx), snap_modulus(cy)
        w, d = snap_modulus(w), snap_modulus(d)
        z0 = 0 if floor == 1 or floor == '1&2' else 3000
        z1 = z0 + h
        rooms.append({
            'id': nid, 'type': r_type, 'floor': 1 if floor == '1&2' else floor,
            'box_min': [max(0, cx - w / 2), max(0, cy - d / 2), z0],
            'box_max': [min(sx, cx + w / 2), min(sy, cy + d / 2), z1],
            'lighting_access': 'direct' if r_type in ['living_room', 'bedroom', 'balcony'] else 'indirect',
            'lighting_priority': 8 if r_type in ['living_room', 'bedroom'] else 4,
            'effective_lighting': [],
        })
    return rooms, G, pos, edge_types


def request_to_house_json(user_req, rooms):
    stats = {t: 0 for t in ROOM_TYPES}
    for r in rooms:
        stats[r['type']] = stats.get(r['type'], 0) + 1
    return {
        'metadata': {
            'stats': stats,
            'total_rooms': len(rooms),
            'building_size': {'x': user_req['site_x'], 'y': user_req['site_y'], 'z': user_req['site_z']},
            'constraints': {'modulus': int(VOXEL_SIZE)},
        },
        'rooms': rooms,
    }


@torch.no_grad()
def synthetic_graph_forward(user_req, model, seed=42):
    rooms, _, _, _ = layout_rooms_from_program(user_req, seed=seed)
    data = request_to_house_json(user_req, rooms)
    sample = json_to_sample(data)
    if sample is None:
        return 0, 0.0
    hg = sample['graph']
    batch = prepare_graph_batch(hg, condition=sample['condition']).to(device)
    model.eval()
    logits, _, _ = forward_model(model, batch, sample['condition'])
    pred = torch.argmax(logits[0], dim=0).cpu().numpy()
    n_occ = int((pred > 0).sum())
    occ_ratio = float(n_occ / max(pred.size, 1))
    return n_occ, occ_ratio


@torch.no_grad()
def run_synthetic_probe_eval(model, specs=None, verbose=True):
    specs = specs or SYNTHETIC_PROBE_SPECS
    results = []
    for spec in specs:
        req = build_user_request(spec['site_x'], spec['site_y'], spec['room_counts'])
        n_occ, occ_ratio = synthetic_graph_forward(req, model, seed=spec.get('seed', 42))
        row = {'name': spec['name'], 'n_occ': n_occ, 'occ_ratio': occ_ratio}
        results.append(row)
        if verbose:
            print(f"  [{spec['name']}] 非空体素={n_occ} ({occ_ratio*100:.4f}%)")
    mean_occ = float(np.mean([r['n_occ'] for r in results]))
    hit = sum(1 for r in results if r['n_occ'] > 0)
    return {'per_probe': results, 'mean_n_occ': mean_occ, 'nonempty_hits': hit, 'total': len(results)}


print('Step 1b 就绪: 合成拓扑探针 +', len(SYNTHETIC_PROBE_SPECS), '个固定探针')
"""

STEP2 = """\
# Step 2: 批量预处理 JSON -> .pt（递归扫描 processed/ 子目录）
os.makedirs(OUT_DIR, exist_ok=True)
meta_path = os.path.join(OUT_DIR, '_cache_meta.json')

json_files = discover_json_files()
if not json_files:
    raise FileNotFoundError(f'未找到 JSON: {JSON_ROOT}（请确认 data/processed 已上传）')

need_rebuild = True
if os.path.exists(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    need_rebuild = meta.get('version') != CACHE_VERSION or meta.get('count') != len(json_files)

if need_rebuild:
    ok, fail = 0, 0
    for fp in json_files:
        fname = os.path.basename(fp)
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sample = json_to_sample(data)
            if sample is None:
                continue
            sample['source_json'] = fp
            sample['pt_name'] = fname.replace('.json', '.pt')
            torch.save(sample, os.path.join(OUT_DIR, sample['pt_name']))
            ok += 1
        except Exception as exc:
            fail += 1
            print(f'[FAIL] {fname}: {exc}')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'version': CACHE_VERSION,
            'count': len(json_files),
            'ok': ok,
            'fail': fail,
            'json_root': JSON_ROOT,
        }, f, indent=2, ensure_ascii=False)
    print(f'预处理完成: 成功 {ok}, 失败 {fail}, 输出 {OUT_DIR}')
else:
    print(f'命中缓存 {CACHE_VERSION}，跳过预处理（共 {len(json_files)} 个 JSON）')
"""

STEP4 = """\
# Step 4: 训练（AMP + 梯度积累 + KL Annealing + 拓扑加噪 + 合成探针）

set_seed(SPLIT_SEED)

pt_files = sorted([
    p for p in glob.glob(os.path.join(OUT_DIR, '*.pt'))
    if not os.path.basename(p).startswith('_')
])
if not pt_files:
    raise RuntimeError('请先运行 Step 2 预处理')

items = []
for fp in pt_files:
    try:
        sample = torch.load(fp, weights_only=False)
        pt_name = sample.get('pt_name') or os.path.basename(fp)
        items.append((pt_name, sample, fp))
    except Exception as exc:
        print(f'读取失败 {fp}: {exc}')

pt_names = [x[0] for x in items]
train_names, val_names, split_meta = load_train_val_split(pt_names)
train_items = [x for x in items if x[0] in set(train_names)]
val_items = [x for x in items if x[0] in set(val_names)]
print(f'样本总数 {len(items)} | 训练 {len(train_items)} | 验证 {len(val_items)}')
print(f'划分文件: {split_path()}')

train_set, val_set = [], []
for pt_name, sample, _fp in train_items:
    hg = sample['graph']
    hg.y = sample['voxel']
    hg.condition = sample['condition'].unsqueeze(0) if sample['condition'].dim() == 1 else sample['condition']
    hg.pt_name = pt_name
    train_set.append(hg)
for pt_name, sample, _fp in val_items:
    hg = sample['graph']
    hg.y = sample['voxel']
    hg.condition = sample['condition'].unsqueeze(0) if sample['condition'].dim() == 1 else sample['condition']
    hg.pt_name = pt_name
    val_set.append(hg)

train_loader = GraphDataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = GraphDataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False) if val_set else None

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
_use_amp = USE_AMP and device.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=_use_amp)

best_val = float('inf')
bad_epochs = 0
history = {'train': [], 'val': [], 'kl_beta': [], 'synth_probe': []}
import time as _time
_train_start = _time.time()
save_path = weight_path()


def kl_beta_for_epoch(epoch):
    if KL_ANNEAL_EPOCHS <= 0:
        return KL_WEIGHT_MAX
    t = min(1.0, epoch / KL_ANNEAL_EPOCHS)
    return KL_WEIGHT_MAX * t


def compute_batch_loss(batch, kl_weight):
    bs = graph_batch_size(batch)
    target = batch.y.view(bs, NUM_CHANNELS, RES_X, RES_Y, RES_Z)
    logits, mu, logvar = forward_model(model, batch)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_with_logits(logits, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = bce + DICE_WEIGHT * dice + kl_weight * kl
    return loss, bs


def run_epoch(loader, train_mode=True, epoch=1):
    model.train(train_mode)
    total = 0.0
    n = 0
    kl_weight = kl_beta_for_epoch(epoch) if train_mode else KL_WEIGHT_MAX
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        if train_mode and AUGMENT_PROB > 0 and random.random() < AUGMENT_PROB:
            batch = augment_batch_topology(batch)

        with torch.amp.autocast('cuda', enabled=_use_amp):
            loss, bs = compute_batch_loss(batch, kl_weight)
            loss_scaled = loss / ACCUM_STEPS

        if train_mode:
            scaler.scale(loss_scaled).backward()
            if step % ACCUM_STEPS == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1), kl_weight


print(f'开始训练 ({device}) | AMP={USE_AMP and device.type == "cuda"} | accum={ACCUM_STEPS} | eff_batch≈{BATCH_SIZE * ACCUM_STEPS}')
for epoch in range(1, EPOCHS + 1):
    tr, beta = run_epoch(train_loader, True, epoch)
    history['train'].append(tr)
    history['kl_beta'].append(beta)
    va, _ = run_epoch(val_loader, False, epoch) if val_loader else (tr, beta)
    history['val'].append(va)
    scheduler.step(va)

    if va < best_val:
        best_val = va
        bad_epochs = 0
        torch.save(model.state_dict(), save_path)
    else:
        bad_epochs += 1

    if epoch == 1 or epoch % 10 == 0:
        print(f'Epoch {epoch:03d}/{EPOCHS} | train={tr:.4f} val={va:.4f} best={best_val:.4f} beta={beta:.2e}')
        if epoch % PROBE_EVERY == 0:
            print('  [合成拓扑探针]')
            probe = run_synthetic_probe_eval(model, verbose=True)
            history['synth_probe'].append({'epoch': epoch, **probe})

    if bad_epochs >= PATIENCE:
        print(f'早停于 epoch {epoch}，最佳 val={best_val:.4f}')
        break

set_experiment_meta(
    train_count=len(train_set),
    total_count=len(items),
    extra={
        'val_samples': len(val_set),
        'epochs_run': epoch,
        'best_val_loss': float(best_val),
        'train_seconds': round(_time.time() - _train_start, 1),
        'cache_version': CACHE_VERSION,
        'weight_file': WEIGHT_FILENAME,
        'split_file': SPLIT_FILENAME,
        'batch_size': BATCH_SIZE,
        'accum_steps': ACCUM_STEPS,
        'use_amp': USE_AMP,
        'kl_weight_max': KL_WEIGHT_MAX,
        'kl_anneal_epochs': KL_ANNEAL_EPOCHS,
        'edge_drop_prob': EDGE_DROP_PROB,
        'augment_prob': AUGMENT_PROB,
        'synth_probe': history['synth_probe'],
        'train_names': train_names,
        'val_names': val_names,
    },
)
meta_path = experiment_report_path('experiment_meta', ext='json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(EXPERIMENT_META, f, indent=2, ensure_ascii=False)
print(f'最佳权重已保存: {save_path}')
print(f'实验元数据: {meta_path}')
print(EXPERIMENT_META)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train'], label='train')
axes[0].plot(history['val'], label='val')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[1].plot(history['kl_beta'], label='kl_beta', color='tab:orange')
axes[1].set_title('KL beta schedule')
axes[1].set_xlabel('Epoch')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
plt.tight_layout()
plt.show()
"""

STEP_R = """\
# Step R: 快速恢复环境 + 加载权重（~10 秒）
# Colab 重启后请先跑 Step 1 → Step 1b → Step 3，再执行本 cell

import os, json, glob
from datetime import datetime
import torch

if 'device' not in globals():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'DATA_DIR' not in globals():
    USE_DRIVE = True
    if USE_DRIVE:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
        except ImportError:
            USE_DRIVE = False

    def resolve_data_dir():
        candidates = [
            '/content/drive/MyDrive/master_thesis/data',
            '/content/drive/MyDrive/260509_dataset',
            '/content/data',
            os.path.join(os.getcwd(), 'data'),
            os.path.abspath(os.path.join(os.getcwd(), '..', 'data')),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return candidates[0]

    DATA_DIR = resolve_data_dir()
    JSON_ROOT = os.path.join(DATA_DIR, 'processed')
    OUT_DIR = os.path.join(DATA_DIR, 'processed_tensors_v470')
    WEIGHT_FILENAME = 'spatial_modal_cvae_470_v1.pth'

    def discover_json_files(root=None):
        root = root or JSON_ROOT
        if not os.path.isdir(root):
            return []
        return sorted(
            f for f in glob.glob(os.path.join(root, '**', 'house_*.json'), recursive=True)
            if not f.endswith('_topology.json')
        )

if 'set_experiment_meta' not in globals():
    TOTAL_JSON_COUNT = len(discover_json_files()) if 'discover_json_files' in globals() else 0
    TRAIN_SAMPLE_COUNT = None
    TRAIN_TIMESTAMP = None
    EXPERIMENT_META = {}

    def set_experiment_meta(train_count, total_count=None, timestamp=None, extra=None):
        global TRAIN_SAMPLE_COUNT, TRAIN_TIMESTAMP, EXPERIMENT_META, TOTAL_JSON_COUNT
        TRAIN_SAMPLE_COUNT = int(train_count)
        if total_count is not None:
            TOTAL_JSON_COUNT = int(total_count)
        TRAIN_TIMESTAMP = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        EXPERIMENT_META = {
            'train_samples': TRAIN_SAMPLE_COUNT,
            'total_json': TOTAL_JSON_COUNT,
            'trained_at': TRAIN_TIMESTAMP,
        }
        if extra:
            EXPERIMENT_META.update(extra)
        return EXPERIMENT_META

_restore_deps = {
    'SpatialModalCVAE': 'Step 3（模型定义）',
    'prepare_graph_batch': 'Step 1（工具函数）',
    'forward_model': 'Step 1（工具函数）',
}
_missing = [f'{name} ← 请先运行 {step}' for name, step in _restore_deps.items() if name not in globals()]
if _missing:
    raise RuntimeError(
        'Colab 重启后内存已清空，不能只跑 Step R。\\n'
        '请依次执行：Step 0（如需）→ Step 1 → Step 1b → Step 3 → 再回来执行 Step R\\n'
        '当前缺少：\\n  - ' + '\\n  - '.join(_missing)
    )

WEIGHT_PATH = weight_path() if 'weight_path' in globals() else os.path.join(DATA_DIR, WEIGHT_FILENAME)

def quick_restore(force_rebuild_model=True):
    global model
    if force_rebuild_model or 'model' not in globals():
        model = SpatialModalCVAE().to(device)
    if os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device, weights_only=True))
        print(f'已加载权重: {WEIGHT_PATH}')
    else:
        print(f'警告: 未找到权重 {WEIGHT_PATH}，请先运行 Step 4 训练')
    model.eval()

    meta_files = sorted(glob.glob(os.path.join(DATA_DIR, 'eval_reports', 'experiment_meta_*.json')))
    if meta_files:
        with open(meta_files[-1], 'r', encoding='utf-8') as f:
            meta = json.load(f)
        set_experiment_meta(meta.get('train_samples', 0), meta.get('total_json'), meta.get('trained_at'), meta)
        print('实验元数据:', EXPERIMENT_META)
    if 'discover_json_files' in globals():
        TOTAL_JSON_COUNT = len(discover_json_files())
    print(f'设备: {device} | 数据: {DATA_DIR} | JSON 数: {TOTAL_JSON_COUNT}')
    return model

quick_restore()
"""

STEP5_WEIGHT = """weight_path = weight_path() if 'weight_path' in globals() else os.path.join(DATA_DIR, WEIGHT_FILENAME)"""

STEP7A_APPEND = """

@torch.no_grad()
def eval_conditional_generation_best_of_k(user_req, gt_voxel, model, k=None, base_seed=42):
    k = k or GEN_EVAL_K
    best = None
    best_score = (-1.0, -1.0, -1.0)
    trials = []
    for i in range(k):
        seed = base_seed + i * 17
        gen = eval_conditional_generation(user_req, gt_voxel, model, seed=seed)
        trials.append(gen)
        score = (gen['miou'], gen['program_acc'], float((gen['pred'] > 0).sum()))
        if score > best_score:
            best_score = score
            best = gen
    best = best or trials[0]
    best['eval_k'] = k
    best['trials'] = trials
    return best


print(f'Step 7a 就绪 | 条件生成 best-of-K={GEN_EVAL_K}')
"""

STEP7B = """\
# Step 7b: 验证集批量量化评估（重建 vs 条件生成 best-of-K）
set_seed(SPLIT_SEED)
wp = weight_path() if 'weight_path' in globals() else os.path.join(DATA_DIR, WEIGHT_FILENAME)
model.load_state_dict(torch.load(wp, map_location=device, weights_only=True))
model.eval()

pt_files = sorted([
    p for p in glob.glob(os.path.join(OUT_DIR, '*.pt'))
    if not os.path.basename(p).startswith('_')
])
if not pt_files:
    raise RuntimeError('请先运行 Step 2 预处理')

items = []
for fp in pt_files:
    s = torch.load(fp, weights_only=False)
    pt_name = s.get('pt_name') or os.path.basename(fp)
    items.append((pt_name, s))

all_names = [x[0] for x in items]
_, val_names, _ = load_train_val_split(all_names)
val_name_set = set(val_names)
val_items = [x for x in items if x[0] in val_name_set]
print(f'验证样本数: {len(val_items)}（来自持久化划分 {SPLIT_FILENAME}）')

rows = []
skipped = 0
for pt_name, sample in val_items:
    json_path = resolve_source_json(sample, pt_name)
    if not json_path:
        skipped += 1
        print('跳过，无 JSON:', pt_name)
        continue
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    user_req = json_to_user_request(raw)

    hg = sample['graph']
    hg.condition = sample['condition'].unsqueeze(0) if sample['condition'].dim() == 1 else sample['condition']
    rec_sample = {'graph': hg, 'voxel': sample['voxel'], 'condition': sample['condition']}

    rec = eval_reconstruction(rec_sample, model)
    gen = eval_conditional_generation_best_of_k(user_req, sample['voxel'], model, k=GEN_EVAL_K)

    rows.append({
        'sample': pt_name.replace('.pt', ''),
        'mode': 'reconstruction',
        'miou': rec['miou'], 'occupied_acc': rec['occupied_acc'], 'total_acc': rec['total_acc'],
        'program_mae': np.nan, 'program_acc': np.nan, 'modulus_rate': np.nan, 'site_fit': np.nan,
        'eval_k': np.nan,
    })
    rows.append({
        'sample': pt_name.replace('.pt', ''),
        'mode': 'conditional_generation_best_of_k',
        'miou': gen['miou'], 'occupied_acc': gen['occupied_acc'], 'total_acc': gen['total_acc'],
        'program_mae': gen['program_mae'], 'program_acc': gen['program_acc'],
        'modulus_rate': gen['modulus_rate'], 'site_fit': gen['site_fit'],
        'eval_k': gen.get('eval_k', GEN_EVAL_K),
    })

import pandas as pd
df_gen = pd.DataFrame(rows)
print(f'跳过（无 source_json）: {skipped}')

print('\\n=== 重建 vs 条件生成（best-of-K）对比 ===')
for mode in ['reconstruction', 'conditional_generation_best_of_k']:
    sub = df_gen[df_gen['mode'] == mode]
    if sub.empty:
        continue
    print(f"\\n[{mode}] n={len(sub)}")
    print(f"  mIoU:          {sub['miou'].mean()*100:5.2f}% ± {sub['miou'].std()*100:4.2f}%")
    print(f"  非空准确率:    {sub['occupied_acc'].mean()*100:5.2f}% ± {sub['occupied_acc'].std()*100:4.2f}%")
    if 'conditional' in mode:
        print(f"  功能数量MAE:   {sub['program_mae'].mean():5.2f} ± {sub['program_mae'].std():4.2f}")
        print(f"  功能数量匹配率:{sub['program_acc'].mean()*100:5.2f}% ± {sub['program_acc'].std()*100:4.2f}%")
        print(f"  模数合规率:    {sub['modulus_rate'].mean()*100:5.2f}%")
        print(f"  用地贴合率:    {sub['site_fit'].mean()*100:5.2f}%")
        print(f"  采样次数 K:    {sub['eval_k'].iloc[0]}")

csv_out = experiment_report_path('generation_metrics', subdir='eval_reports', ext='csv')
json_out = experiment_report_path('generation_summary', subdir='eval_reports', ext='json')
df_gen.to_csv(csv_out, index=False, encoding='utf-8-sig')

report = {}
for mode in df_gen['mode'].unique():
    sub = df_gen[df_gen['mode'] == mode]
    report[mode] = {
        c: float(sub[c].mean()) for c in [
            'miou', 'occupied_acc', 'total_acc', 'program_mae', 'program_acc', 'modulus_rate', 'site_fit'
        ] if c in sub.columns and sub[c].notna().any()
    }
report['eval_k'] = GEN_EVAL_K
with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\\n已导出: {csv_out}")
print(f"已导出: {json_out}")
try:
    display(df_gen.pivot_table(index='sample', columns='mode', values=['miou', 'occupied_acc', 'program_acc']))
except NameError:
    print(df_gen.head(10))
"""

STEP7_MD = """\
## Step 7: 量化评估（重建 vs 条件生成 best-of-K）

- **重建模式**：GT 异构图（上界）
- **生成模式**：仅用户约束 → 合成拓扑，**K 次采样取最优**（真实使用路径）
- 验证集与训练共用 `train_val_split_v470.json`

导出命名：`{报告名}_json{总数}_train{训练数}_{时间}.csv`
"""


def patch_step1_cell(source: str) -> str:
    source = source.replace(
        "OUT_DIR = os.path.join(DATA_DIR, 'processed_tensors_v2')\n",
        "",
    )
    source = source.replace(
        "CACHE_VERSION = 'mvp_v2_lighting_cvae'\n",
        "",
    )
    source = source.replace(
        "TOTAL_JSON_COUNT = len([\n"
        "    f for f in glob.glob(os.path.join(DATA_DIR, 'house_*.json'))\n"
        "    if not f.endswith('_topology.json')\n"
        "])",
        "TOTAL_JSON_COUNT = 0  # 在下方 v1 配置块中重新统计",
    )
    if "discover_json_files" not in source:
        source = source.rstrip() + STEP1_APPEND
    return source


def patch_step5_cell(source: str) -> str:
    return source.replace(
        "weight_path = os.path.join(DATA_DIR, 'spatial_modal_cvae_mvp.pth')",
        STEP5_WEIGHT,
    )


def patch_step7a_cell(source: str) -> str:
    source = source.replace("print('Step 7a 就绪')\n", "")
    if "eval_conditional_generation_best_of_k" not in source:
        source = source.rstrip() + STEP7A_APPEND
    return source


def main():
    with open(SRC, encoding="utf-8") as f:
        nb = json.load(f)

    set_cell_source(nb, 0, MARKDOWN_0)
    nb["cells"][2]["source"] = src_lines(patch_step1_cell("".join(nb["cells"][2]["source"])))
    insert_cell(nb, 3, "code", STEP1B)
    set_cell_source(nb, 4, STEP2)
    # Step 3 unchanged at index 5
    nb["cells"][6]["source"] = [line.replace(
        "Step 1 → Step 3",
        "Step 1 → Step 1b → Step 3",
    ) for line in nb["cells"][6]["source"]]
    # Re-load and redo with correct indices after insert
    with open(SRC, encoding="utf-8") as f:
        nb = json.load(f)

    set_cell_source(nb, 0, MARKDOWN_0)
    nb["cells"][2]["source"] = src_lines(patch_step1_cell("".join(nb["cells"][2]["source"])))
    insert_cell(nb, 3, "code", STEP1B)
    # 0 md, 1 step0, 2 step1, 3 step1b, 4 step2, 5 step3, 6 stepR md, 7 stepR, 8 step4...
    set_cell_source(nb, 4, STEP2)
    set_cell_source(nb, 7, STEP_R)
    set_cell_source(nb, 8, STEP4)
    set_cell_source(nb, 9, patch_step5_cell("".join(nb["cells"][9]["source"])))
    set_cell_source(nb, 13, STEP7_MD)
    set_cell_source(nb, 14, patch_step7a_cell("".join(nb["cells"][14]["source"])))
    set_cell_source(nb, 15, STEP7B)

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("Wrote:", DST)
    print("Cells:", len(nb["cells"]))


if __name__ == "__main__":
    main()
