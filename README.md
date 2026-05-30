# GraphSpace

基于图神经网络 (GNN) 与空间拓扑算法的住宅布局生成研究项目。核心数据流：Rhino 3D 模型导出 JSON → 离线数据清洗与校验 → GNN 模型训练 → 结果记录与评估。

## 项目简介

GraphSpace 探索将建筑空间拓扑转化为可计算图网络，并实现住宅功能布局自动生成的学术工程流水线，涵盖从三维几何模型提取拓扑信息到深度学习训练的完整流程。

> 本项目相关论文正在撰写/审稿中。为遵守学术规范，内部文档、原始数据集与训练权重暂不公开，待论文正式发表后再评估是否同步发布。

## 核心工作流

### 1. 数据提取 (Rhino Export)

- **工具**: Rhino 3D / Grasshopper / Python Scripts
- **脚本**: `scripts/rhino_export/`（当前主版本为 `260524rhino-json-v14.py`）
- **流程**: 遍历 `.3dm` 住宅模型，提取房间节点、采光面、相邻关系等空间特征，导出为标准 `.json`

### 2. 离线清洗与校验 (Offline Processing)

- **脚本**: `scripts/offline_check/validate_json_dataset.py`
- **流程**: 对 Rhino 导出的 JSON 做二次离线检测，剔除异常值，验证拓扑合法性，并试转为神经网络输入张量

### 3. 模型训练与评估 (Training & Evaluation)

- **Notebook**: `notebooks/train/`
- **流程**: 将校验后的数据集输入 GNN / CVAE 进行训练与量化评估（MVP 指标）
- **产出**: Loss 曲线、评估指标与验证结果（本地保存，不上传仓库）

## 仓库结构

```text
GraphSpace/
├── scripts/                  # 可复现的核心脚本（公开）
│   ├── rhino_export/         # Rhino → JSON 导出
│   └── offline_check/        # JSON 离线校验
├── notebooks/                # 实验 notebook（公开）
│   ├── train/                # 模型训练与评估
│   └── experiments/          # 实验结果（本地，已 gitignore）
├── data/                     # 数据集（本地私有，不上传）
├── models/                   # 训练权重 .pth/.pt（本地私有，不上传）
├── docs/                     # 论文草稿与建模规范（本地私有，不上传）
├── requirements.txt
├── README.md
└── .gitignore
```

## 未公开内容 (Not Included)

以下目录/文件仅保留在本地，**不会**上传至 GitHub：

| 路径 | 说明 |
|------|------|
| `docs/` | 论文草稿、Rhino 建模规范等内部文档 |
| `data/` | 原始与 processed 户型 JSON、预处理 `.pt` 缓存 |
| `models/` | 训练完成的模型权重（`.pth` / `.pt`） |
| `logs/`、`experimental_results/` | 训练日志与批量实验输出 |
| `notebooks/experiments/` | 验证集指标 CSV/JSON 等结果文件 |

各私有目录下的 `README.md` 说明了本地放置规范；仓库中仅保留这些说明文件。

## 环境要求

- Python 3.8+
- Rhino 7+（仅 `scripts/rhino_export/` 需要，在 Rhino 内置 Python 或 Grasshopper 中运行）
- 训练 notebook 建议使用 GPU 环境（Colab / 本地 CUDA）

## 快速开始

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd GraphSpace

# 2. 安装 Python 依赖（离线校验脚本 + notebook 共用）
pip install -r requirements.txt

# 3. 准备数据（需自行准备，放入本地 data/ 目录）
#    参见 data/README.md

# 4. 离线校验
python scripts/offline_check/validate_json_dataset.py --data-dir data/processed

# 5. 打开 notebooks/train/ 中对应 notebook 进行训练
#    训练完成后将权重保存至本地 models/（参见 models/README.md）
```

## 项目网站

静态站点位于 [`website/`](website/)，域名为 **spacemodal.com** / spacemodal.cn：

| 页面 | 说明 |
|------|------|
| [首页](website/index.html) | SpaceModal 空间模态 — 项目概览 |
| [作品集](website/portfolio/index.html) | 实习作品集（安全模式，无数据下载） |
| [数据可视化](website/viewer/index.html) | 公网 10 套 Demo（`demo/datasets/`） |
| [论文](website/paper/index.html) | 摘要、方法、引用 |
| [演示](website/demo/index.html) | 本地演示说明 |
| [产品](website/product/index.html) | 模态户型 ModalPlan |

部署与安全说明见 [`website/DEPLOY.md`](website/DEPLOY.md)。上线前请编辑 [`website/js/site-config.js`](website/js/site-config.js) 并运行 `cd website && npm run build:viewer`。

```bash
cd website
npm run build:viewer
python -m http.server 8080
```

## 许可证

待定（论文发表后补充）。
