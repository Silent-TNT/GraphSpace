# -*- coding: utf-8 -*-
"""从 v2 notebook 生成 V3 训练专用副本（88×88×24，仅训练）。

运行: python scripts/notebook_build/create_v3_train.py
"""
from __future__ import annotations

import json
import os
import re
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "notebooks", "train", "260607-470条件生成与量化评估（训练优化v2）.ipynb")
DST = os.path.join(ROOT, "notebooks", "train", "260609-V3训练（88x88x24）.ipynb")


class V3Config:
    RES_X, RES_Y, RES_Z = 88, 88, 24
    INIT_X = (RES_X + 15) // 16  # 6 → decoder 输出 96，裁切到 88
    INIT_Y = (RES_Y + 15) // 16
    INIT_Z = (RES_Z + 15) // 16  # 2 → decoder Z 输出 32，裁切到 24
    VOXEL_SIZE = 300.0

    BATCH_SIZE = 4
    ACCUM_STEPS = 4
    USE_AMP = True
    EPOCHS = 90
    PATIENCE = 18
    LOG_EVERY = 5
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DICE_WEIGHT = 0.3
    KL_WEIGHT_MAX = 5e-3
    KL_ANNEAL_EPOCHS = 40
    WARMUP_EPOCHS = 5
    WARMUP_LR_START = 1e-5

    EDGE_DROP_PROB = 0.20
    EDGE_ADD_PROB = 0.05
    NODE_JITTER = 0.02
    AUGMENT_PROB = 0.80
    PROBE_EVERY = 10

    VAL_RATIO = 0.2
    SPLIT_SEED = 42

    TENSOR_CACHE_VERSION = "v3_tensors_88x88x24"
    TRAIN_CACHE_VERSION = "v3_train_88x88x24"
    WEIGHT_FILENAME = "spatial_modal_cvae_v3_88x88x24.pth"
    WEIGHT_PROBE_FILENAME = "spatial_modal_cvae_v3_88x88x24_probe_best.pth"
    CHECKPOINT_FILENAME = "spatial_modal_cvae_v3_88x88x24_checkpoint.pt"
    SPLIT_FILENAME = "train_val_split_v3.json"


cfg = V3Config()


def src_lines(code: str) -> list[str]:
    return [line + "\n" for line in textwrap.dedent(code).strip("\n").splitlines()]


def set_cell_source(nb: dict, idx: int, code: str) -> None:
    nb["cells"][idx]["source"] = src_lines(code)
    nb["cells"][idx]["execution_count"] = None
    nb["cells"][idx]["outputs"] = []


MARKDOWN_0 = f"""\
# V3 训练专用 · {cfg.RES_X}×{cfg.RES_Y}×{cfg.RES_Z} · 条件 CVAE

基于 `260607-470条件生成与量化评估（训练优化v2）.ipynb` 训练管线，**仅负责训练与实验记录**。

## 相对 v1/v2 的核心变更

| 项目 | v1/v2 | V3 |
|---|---|---|
| 画布 | 64×128×32 | **{cfg.RES_X}×{cfg.RES_Y}×{cfg.RES_Z}** |
| 物理画布 (mm) | 19200×38400×9600 | {cfg.RES_X * cfg.VOXEL_SIZE:.0f}×{cfg.RES_Y * cfg.VOXEL_SIZE:.0f}×{cfg.RES_Z * cfg.VOXEL_SIZE:.0f} |
| init_volume | (256,4,8,2) | **(256,{cfg.INIT_X},{cfg.INIT_Y},{cfg.INIT_Z})** → 裁切到目标栅格 |
| 有效 batch | 8 (2×4) | **16 (4×4)** |
| 训练 epoch 上限 | 120 | **{cfg.EPOCHS}**（早停 patience={cfg.PATIENCE}） |
| 3D 生成 / 网页 | notebook 内 | **`scripts/spatial_modal_infer/`** |

> **画布 {cfg.RES_X}×{cfg.RES_Y}×{cfg.RES_Z}**：零裁切 + 比 96×96 更密，训练/生成更稳。decoder 输出 {cfg.INIT_X * 16}×{cfg.INIT_Y * 16}×{cfg.INIT_Z * 16} 后裁切。

## 步骤

| 步骤 | 作用 |
|---|---|
| Step 0–4 | 依赖、配置、预处理(QC)、建模、训练 |
| Step 1b | 合成拓扑探针（训练监控） |
| Step 5 | 训练结果摘要 + 实验记录索引 |

**生成与可视化**：下载权重后，在本地运行 `scripts/spatial_modal_infer/run_cli.py` 或 `app_gradio.py`。
"""

V3_CONFIG_BLOCK = f"""
# ========== V3 训练配置（{cfg.RES_X}×{cfg.RES_Y}×{cfg.RES_Z}）==========
JSON_ROOT = os.path.join(DATA_DIR, 'processed')
OUT_DIR = os.path.join(DATA_DIR, 'processed_tensors_v3')
# Drive 空间不足时可改 Colab 本地（会话结束丢失）：
# OUT_DIR = '/content/processed_tensors_v3'

TENSOR_CACHE_VERSION = '{cfg.TENSOR_CACHE_VERSION}'
TRAIN_CACHE_VERSION = '{cfg.TRAIN_CACHE_VERSION}'
CACHE_VERSION = TRAIN_CACHE_VERSION

WEIGHT_FILENAME = '{cfg.WEIGHT_FILENAME}'
WEIGHT_PROBE_FILENAME = '{cfg.WEIGHT_PROBE_FILENAME}'
CHECKPOINT_FILENAME = '{cfg.CHECKPOINT_FILENAME}'
SPLIT_FILENAME = '{cfg.SPLIT_FILENAME}'

RES_X, RES_Y, RES_Z = {cfg.RES_X}, {cfg.RES_Y}, {cfg.RES_Z}
VOXEL_SIZE = {cfg.VOXEL_SIZE}
DECODER_INIT_X, DECODER_INIT_Y, DECODER_INIT_Z = {cfg.INIT_X}, {cfg.INIT_Y}, {cfg.INIT_Z}

BATCH_SIZE = {cfg.BATCH_SIZE}
ACCUM_STEPS = {cfg.ACCUM_STEPS}
USE_AMP = {cfg.USE_AMP}
EPOCHS = {cfg.EPOCHS}
PATIENCE = {cfg.PATIENCE}
LOG_EVERY = {cfg.LOG_EVERY}
LR = {cfg.LR}
WEIGHT_DECAY = {cfg.WEIGHT_DECAY}
DICE_WEIGHT = {cfg.DICE_WEIGHT}
KL_WEIGHT_MAX = {cfg.KL_WEIGHT_MAX}
KL_ANNEAL_EPOCHS = {cfg.KL_ANNEAL_EPOCHS}
WARMUP_EPOCHS = {cfg.WARMUP_EPOCHS}
WARMUP_LR_START = {cfg.WARMUP_LR_START}

EDGE_DROP_PROB = {cfg.EDGE_DROP_PROB}
EDGE_ADD_PROB = {cfg.EDGE_ADD_PROB}
NODE_JITTER = {cfg.NODE_JITTER}
AUGMENT_PROB = {cfg.AUGMENT_PROB}
PROBE_EVERY = {cfg.PROBE_EVERY}

RESUME_IF_CHECKPOINT = True
FORCE_NEW_TRAINING = False
MANUAL_RESUME_EPOCH = 0

VAL_RATIO = {cfg.VAL_RATIO}
SPLIT_SEED = {cfg.SPLIT_SEED}

RUNS_LOG_PATH = os.path.join(DATA_DIR, 'eval_reports', 'training_runs.jsonl')


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


def probe_weight_path():
    return os.path.join(DATA_DIR, WEIGHT_PROBE_FILENAME)


def checkpoint_path():
    return os.path.join(DATA_DIR, CHECKPOINT_FILENAME)


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
    meta = {{
        'seed': seed,
        'val_ratio': val_ratio,
        'total': len(names),
        'train_count': len(train_names),
        'val_count': len(val_names),
        'cache_version': CACHE_VERSION,
        'grid': '{cfg.RES_X}x{cfg.RES_Y}x{cfg.RES_Z}',
    }}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({{'train': train_names, 'val': val_names, 'meta': meta}}, f, indent=2, ensure_ascii=False)
    return train_names, val_names, meta


def resolve_source_json(sample, pt_basename=None):
    if isinstance(sample, dict) and sample.get('source_json') and os.path.exists(sample['source_json']):
        return sample['source_json']
    if pt_basename:
        cand = os.path.join(DATA_DIR, pt_basename.replace('.pt', '.json'))
        if os.path.exists(cand):
            return cand
    return None


def augment_batch_topology(batch, edge_drop_prob=EDGE_DROP_PROB, node_jitter=NODE_JITTER, edge_add_prob=EDGE_ADD_PROB):
    \"\"\"拓扑加噪：边 dropout + 同图假边注入 + 节点坐标微扰。\"\"\"
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
        if edge_add_prob > 0:
            src_type, dst_type = edge_key[0], edge_key[2]
            if (hasattr(batch[src_type], 'x') and batch[src_type].x is not None and batch[src_type].x.numel() > 0
                and hasattr(batch[dst_type], 'x') and batch[dst_type].x is not None and batch[dst_type].x.numel() > 0):
                n_src = batch[src_type].x.size(0)
                n_dst = batch[dst_type].x.size(0)
                src_graph = batch[src_type].batch if hasattr(batch[src_type], 'batch') else torch.zeros(n_src, dtype=torch.long, device=ei.device)
                dst_graph = batch[dst_type].batch if hasattr(batch[dst_type], 'batch') else torch.zeros(n_dst, dtype=torch.long, device=ei.device)
                existing = set()
                for col in range(store.edge_index.size(1)):
                    existing.add((int(store.edge_index[0, col]), int(store.edge_index[1, col])))
                fake = []
                n_add = max(1, int(store.edge_index.size(1) * edge_add_prob))
                attempts = 0
                while len(fake) < n_add and attempts < n_add * 15:
                    si = torch.randint(0, n_src, (1,)).item()
                    di = torch.randint(0, n_dst, (1,)).item()
                    if src_graph[si] == dst_graph[di] and (si, di) not in existing:
                        fake.append([si, di])
                        existing.add((si, di))
                    attempts += 1
                if fake:
                    fake_tensor = torch.tensor(fake, dtype=torch.long, device=ei.device).t().contiguous()
                    store.edge_index = torch.cat([store.edge_index, fake_tensor], dim=1)
    for ntype in batch.node_types:
        x = batch[ntype].x
        if x is None or x.numel() == 0:
            continue
        noise = torch.zeros_like(x)
        if x.size(1) >= 5:
            noise[:, 2:5] = torch.randn(x.size(0), 3, device=x.device) * node_jitter
        batch[ntype].x = (x + noise).clamp(0.0, 5.0)
    return batch


def qc_sample_voxelization(data):
    \"\"\"统计体素化裁切与占用率，用于 Step 2 数据集 QC。\"\"\"
    rooms = data.get('rooms', [])
    if not rooms:
        return None
    all_coords = np.array([r['box_min'] for r in rooms] + [r['box_max'] for r in rooms])
    build_min = all_coords.min(axis=0)
    build_max = all_coords.max(axis=0)
    bsize = build_max - build_min
    phys_center_xy = (build_min[:2] + build_max[:2]) / 2.0
    offset_xy = np.array([RES_X * VOXEL_SIZE / 2, RES_Y * VOXEL_SIZE / 2]) - phys_center_xy
    z_min_phys = build_min[2]
    clipped_rooms = 0
    for r in rooms:
        ix_min = int((r['box_min'][0] + offset_xy[0]) / VOXEL_SIZE)
        ix_max = int((r['box_max'][0] + offset_xy[0]) / VOXEL_SIZE)
        iy_min = int((r['box_min'][1] + offset_xy[1]) / VOXEL_SIZE)
        iy_max = int((r['box_max'][1] + offset_xy[1]) / VOXEL_SIZE)
        iz_min = int((r['box_min'][2] - z_min_phys) / VOXEL_SIZE)
        iz_max = int((r['box_max'][2] - z_min_phys) / VOXEL_SIZE)
        raw_ix_min = ix_min
        raw_iy_min = iy_min
        raw_iz_min = iz_min
        raw_ix_max = ix_max
        raw_iy_max = iy_max
        raw_iz_max = iz_max
        ix_min = max(0, ix_min)
        iy_min = max(0, iy_min)
        iz_min = max(0, iz_min)
        ix_max = min(RES_X, ix_max)
        iy_max = min(RES_Y, iy_max)
        iz_max = min(RES_Z, iz_max)
        if (raw_ix_min < 0 or raw_iy_min < 0 or raw_iz_min < 0
                or raw_ix_max > RES_X or raw_iy_max > RES_Y or raw_iz_max > RES_Z
                or ix_max <= ix_min or iy_max <= iy_min or iz_max <= iz_min):
            clipped_rooms += 1
    sample = json_to_sample(data)
    if sample is None:
        return None
    occ = int((sample['voxel'].sum(dim=0) > 0).sum().item())
    total_vox = RES_X * RES_Y * RES_Z
    return {{
        'building_x': float(bsize[0]),
        'building_y': float(bsize[1]),
        'building_z': float(bsize[2]),
        'clipped_rooms': clipped_rooms,
        'total_rooms': len(rooms),
        'n_occ': occ,
        'occ_ratio': occ / total_vox,
    }}


def append_training_run_log(record):
    os.makedirs(os.path.dirname(RUNS_LOG_PATH), exist_ok=True)
    with open(RUNS_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + chr(10))


TOTAL_JSON_COUNT = len(discover_json_files())
print(f'JSON 根目录: {{JSON_ROOT}}  发现 JSON: {{TOTAL_JSON_COUNT}}')
print(f'缓存目录: {{OUT_DIR}}  权重: {{WEIGHT_FILENAME}}')
print(f'画布: {{RES_X}}×{{RES_Y}}×{{RES_Z}}  体素: {{VOXEL_SIZE}}mm  有效batch≈{{BATCH_SIZE * ACCUM_STEPS}}')
"""

STEP2 = f"""\
# Step 2: 批量预处理 JSON -> .pt（V3 画布 {cfg.RES_X}×{cfg.RES_Y}×{cfg.RES_Z} + QC）
os.makedirs(OUT_DIR, exist_ok=True)
meta_path = os.path.join(OUT_DIR, '_cache_meta.json')
qc_report_path = os.path.join(DATA_DIR, 'eval_reports', 'preprocess_qc_{cfg.TENSOR_CACHE_VERSION}.json')

json_files = discover_json_files()
if not json_files:
    raise FileNotFoundError(f'未找到 JSON: {{JSON_ROOT}}（请确认 data/processed 已上传）')

need_rebuild = True
if os.path.exists(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    need_rebuild = meta.get('version') != TENSOR_CACHE_VERSION or meta.get('count') != len(json_files)

if need_rebuild:
    ok, fail, skip = 0, 0, 0
    qc_rows = []
    for fp in json_files:
        fname = os.path.basename(fp)
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            qc = qc_sample_voxelization(data)
            if qc is None:
                skip += 1
                continue
            sample = json_to_sample(data)
            if sample is None:
                skip += 1
                continue
            sample['source_json'] = fp
            sample['pt_name'] = fname.replace('.json', '.pt')
            sample['qc'] = qc
            torch.save(sample, os.path.join(OUT_DIR, sample['pt_name']))
            qc_rows.append({{'file': fname, **qc}})
            ok += 1
        except Exception as exc:
            fail += 1
            print(f'[FAIL] {{fname}}: {{exc}}')
    clipped_any = sum(1 for r in qc_rows if r['clipped_rooms'] > 0)
    clipped_all = sum(1 for r in qc_rows if r['clipped_rooms'] == r['total_rooms'] and r['total_rooms'] > 0)
    qc_summary = {{
        'version': TENSOR_CACHE_VERSION,
        'grid': '{cfg.RES_X}x{cfg.RES_Y}x{cfg.RES_Z}',
        'count_ok': ok,
        'count_fail': fail,
        'count_skip': skip,
        'samples_with_clipped_rooms': clipped_any,
        'samples_fully_clipped': clipped_all,
        'clip_rate': clipped_any / max(ok, 1),
        'mean_occ_ratio': float(np.mean([r['occ_ratio'] for r in qc_rows])) if qc_rows else 0.0,
        'max_building_x': float(max((r['building_x'] for r in qc_rows), default=0)),
        'max_building_y': float(max((r['building_y'] for r in qc_rows), default=0)),
        'max_building_z': float(max((r['building_z'] for r in qc_rows), default=0)),
    }}
    os.makedirs(os.path.dirname(qc_report_path), exist_ok=True)
    with open(qc_report_path, 'w', encoding='utf-8') as f:
        json.dump({{'summary': qc_summary, 'per_sample': qc_rows}}, f, indent=2, ensure_ascii=False)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({{
            'version': TENSOR_CACHE_VERSION,
            'count': len(json_files),
            'ok': ok,
            'fail': fail,
            'skip': skip,
            'json_root': JSON_ROOT,
            'grid': '{cfg.RES_X}x{cfg.RES_Y}x{cfg.RES_Z}',
            'qc_report': qc_report_path,
        }}, f, indent=2, ensure_ascii=False)
    print(f'预处理完成: 成功 {{ok}}, 失败 {{fail}}, 跳过 {{skip}}')
    print(f'QC: 有裁切房间样本 {{clipped_any}}/{{ok}} ({{qc_summary["clip_rate"]*100:.2f}}%) | 报告 {{qc_report_path}}')
else:
    print(f'命中缓存 {{TENSOR_CACHE_VERSION}}，跳过预处理（共 {{len(json_files)}} 个 JSON）')
    if os.path.exists(qc_report_path):
        with open(qc_report_path, 'r', encoding='utf-8') as f:
            qc_saved = json.load(f)
        print('QC 摘要:', qc_saved.get('summary', {{}}))
"""

STEP3_DECODER_PATCH = {
    "old_init": "self.init_volume_size = (256, 4, 8, 2)",
    "new_init": f"self.init_volume_size = (256, {cfg.INIT_X}, {cfg.INIT_Y}, {cfg.INIT_Z})",
    "old_forward": "return self.decoder(d)",
    "new_forward": f"""x = self.decoder(d)
        return x[:, :, :RES_X, :RES_Y, :RES_Z]""",
}

STEP4_EXTRA = """
best_probe_score = -1.0
probe_save_path = probe_weight_path()

"""

STEP4_PROBE_BLOCK = (
    "        if epoch % PROBE_EVERY == 0:\n"
    "            print('  [合成拓扑探针]')\n"
    "            probe = run_synthetic_probe_eval(model, verbose=True)\n"
    "            history['synth_probe'].append({'epoch': epoch, **probe})\n"
    "            if float(probe.get('mean_n_occ', 0)) > best_probe_score:\n"
    "                best_probe_score = float(probe['mean_n_occ'])\n"
    "                torch.save(model.state_dict(), probe_save_path)\n"
    "                print(f'  [探针最佳] mean_n_occ={best_probe_score:.0f} → {probe_save_path}')\n"
)

STEP5_MD = """\
## Step 5: 训练结果摘要

本 notebook **不包含 3D 可视化**。请下载权重到本地，使用 `scripts/spatial_modal_infer/`：

```bash
cd scripts/spatial_modal_infer
python run_cli.py --weights /path/to/spatial_modal_cvae_v3_88x88x24.pth \\
  --site-x 18000 --site-y 15000 --seed 123 \\
  --living-room 1 --bedroom 3 --kitchen 1 --out-dir ./outputs
```

实验记录目录：`data/eval_reports/`（`experiment_meta_*.json`、`training_runs.jsonl`、训练曲线 PNG）。
"""

STEP5_CODE = """\
# Step 5: 训练结果摘要与实验索引
import glob
import json
import os

print('=' * 60)
print('V3 训练产物')
print('=' * 60)
for label, path_fn in [
    ('最佳 val 权重', weight_path),
    ('最佳探针权重', probe_weight_path),
    ('断点归档', lambda: checkpoint_path().replace('.pt', '_done.pt')),
]:
    p = path_fn() if callable(path_fn) else path_fn
    status = '✓' if os.path.exists(p) else '✗'
    print(f'  [{status}] {label}: {p}')

meta_glob = sorted(glob.glob(os.path.join(DATA_DIR, 'eval_reports', 'experiment_meta_*.json')))
if meta_glob:
    with open(meta_glob[-1], 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print('\\n最近一次实验 meta:')
    for k in ['model_version', 'grid', 'epochs_run', 'best_val_loss', 'best_probe_mean_n_occ',
              'train_seconds', 'weight_file', 'trained_at']:
        if k in meta:
            print(f'  {k}: {meta[k]}')
    curve = meta_glob[-1].replace('experiment_meta_', 'training_curve_').replace('.json', '.png')
    if os.path.exists(curve):
        print(f'  训练曲线: {curve}')

if os.path.exists(RUNS_LOG_PATH):
    print(f'\\n训练历史索引: {RUNS_LOG_PATH}')
    with open(RUNS_LOG_PATH, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f if ln.strip()]
    print(f'  共 {len(lines)} 次训练记录（最近 3 条）:')
    for ln in lines[-3:]:
        rec = json.loads(ln)
        print(f"    {rec.get('trained_at')} | ep={rec.get('epochs_run')} | best_val={rec.get('best_val_loss')} | grid={rec.get('grid')}")
else:
    print('\\n尚无 training_runs.jsonl，完成 Step 4 后会自动追加。')
"""


def patch_step1(source: str) -> str:
    source = source.replace("RES_X, RES_Y, RES_Z = 64, 128, 32", f"RES_X, RES_Y, RES_Z = {cfg.RES_X}, {cfg.RES_Y}, {cfg.RES_Z}")
    source = re.sub(
        r"# ---------- 470 训练优化 v2 配置 ----------.*?print\(f'缓存目录: \{OUT_DIR\}  权重: \{WEIGHT_FILENAME\}'\)\n",
        V3_CONFIG_BLOCK,
        source,
        flags=re.DOTALL,
    )
    if "RUNS_LOG_PATH" not in source:
        raise RuntimeError("V3 配置块注入失败")
    return source


def patch_step3(source: str) -> str:
    s = source.replace(STEP3_DECODER_PATCH["old_init"], STEP3_DECODER_PATCH["new_init"])
    s = s.replace(STEP3_DECODER_PATCH["old_forward"], STEP3_DECODER_PATCH["new_forward"])
    return s


def patch_step4(source: str) -> str:
    s = source.replace(
        "history = {'train': [], 'val': [], 'kl_beta': [], 'synth_probe': []}",
        "history = {'train': [], 'val': [], 'kl_beta': [], 'synth_probe': []}\n" + STEP4_EXTRA.strip(),
    )
    s = s.replace(
        "print(f'开始训练 ({device}) | AMP=",
        "print(f'开始训练 ({device}) | V3 {RES_X}×{RES_Y}×{RES_Z} | AMP=",
    )
    s = s.replace(
        "print(f'检查点显示已完成 {EPOCHS} epoch，跳过训练。可直接 Step R / Step 7。')",
        "print(f'检查点显示已完成 {EPOCHS} epoch，跳过训练。请运行 Step 5 查看摘要。')",
    )
    old_probe = (
        "        if epoch % PROBE_EVERY == 0:\n"
        "            print('  [合成拓扑探针]')\n"
        "            probe = run_synthetic_probe_eval(model, verbose=True)\n"
        "            history['synth_probe'].append({'epoch': epoch, **probe})"
    )
    s = s.replace(old_probe, STEP4_PROBE_BLOCK.rstrip())

    s = s.replace(
        "    extra={\n        'val_samples': len(val_entries),",
        "    extra={\n"
        "        'model_version': 'v3',\n"
        "        'grid': f'{RES_X}x{RES_Y}x{RES_Z}',\n"
        "        'init_volume': [256, DECODER_INIT_X, DECODER_INIT_Y, DECODER_INIT_Z],\n"
        "        'best_probe_mean_n_occ': float(best_probe_score),\n"
        "        'weight_probe_file': WEIGHT_PROBE_FILENAME,\n"
        "        'val_samples': len(val_entries),",
    )
    s = s.replace("axes[0].set_title('Loss')", "axes[0].set_title(f'V3 Loss ({RES_X}×{RES_Y}×{RES_Z})')")

    append_block = (
        "\ncurve_path = experiment_report_path('training_curve', ext='png')\n"
        "fig.savefig(curve_path, dpi=120, bbox_inches='tight')\n"
        "print(f'训练曲线已保存: {curve_path}')\n"
        "\n"
        "append_training_run_log({\n"
        "    'trained_at': EXPERIMENT_META.get('trained_at'),\n"
        "    'model_version': 'v3',\n"
        "    'grid': f'{RES_X}x{RES_Y}x{RES_Z}',\n"
        "    'cache_version': CACHE_VERSION,\n"
        "    'epochs_run': epoch,\n"
        "    'best_val_loss': float(best_val),\n"
        "    'best_probe_mean_n_occ': float(best_probe_score),\n"
        "    'train_seconds': EXPERIMENT_META.get('train_seconds'),\n"
        "    'weight_file': WEIGHT_FILENAME,\n"
        "    'weight_probe_file': WEIGHT_PROBE_FILENAME,\n"
        "    'meta_file': meta_path,\n"
        "    'curve_file': curve_path,\n"
        "})\n"
        "print(f'训练记录已追加: {RUNS_LOG_PATH}')\n"
    )
    s = s.replace("plt.show()", append_block + "\nplt.show()")
    return s


def patch_step1b(source: str) -> str:
    return source.replace(
        "训练探针 & Step 6 条件生成共用",
        "训练探针（监控条件生成能力）",
    )


def main():
    with open(SRC, encoding="utf-8") as f:
        nb = json.load(f)

    set_cell_source(nb, 0, MARKDOWN_0)
    nb["cells"][2]["source"] = src_lines(patch_step1("".join(nb["cells"][2]["source"])))
    nb["cells"][3]["source"] = src_lines(patch_step1b("".join(nb["cells"][3]["source"])))
    set_cell_source(nb, 4, STEP2)
    nb["cells"][5]["source"] = src_lines(patch_step3("".join(nb["cells"][5]["source"])))
    nb["cells"][8]["source"] = src_lines(patch_step4("".join(nb["cells"][8]["source"])))

    # 仅保留 Step 0–4 + Step 1b（跳过 Step R），+ Step 5 摘要
    keep = [nb["cells"][i] for i in (0, 1, 2, 3, 4, 5, 8)]
    nb["cells"] = keep + [
        {"cell_type": "markdown", "metadata": {}, "source": src_lines(STEP5_MD), "execution_count": None, "outputs": []},
        {"cell_type": "code", "metadata": {}, "source": src_lines(STEP5_CODE), "execution_count": None, "outputs": []},
    ]

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("Wrote:", DST)
    print("Cells:", len(nb["cells"]))
    print(f"V3: {cfg.RES_X}×{cfg.RES_Y}×{cfg.RES_Z} | EPOCHS={cfg.EPOCHS} PATIENCE={cfg.PATIENCE} | batch={cfg.BATCH_SIZE}×{cfg.ACCUM_STEPS}")


if __name__ == "__main__":
    main()
