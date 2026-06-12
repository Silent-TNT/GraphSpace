# GraphSpace

基于异构图神经网络与条件 CVAE 的独立住宅二层功能体块布局生成研究。

> 根据用户输入的场地尺寸与房间功能需求，生成符合 300mm 模数、拓扑合理的二层建筑功能布局。
>
> **论文**：《基于空间模态编码的AI辅助设计研究——以独立住宅的量体生成为例》
> 草稿见 [`docs/thesis_draft.md`](docs/thesis_draft.md)。数据集与训练权重在论文发表前暂不公开。

## 仓库结构

```text
GraphSpace/
├── scripts/
│   ├── rhino_export/          # Rhino → JSON 导出
│   ├── offline_check/         # JSON 离线校验
│   ├── spatial_modal_infer/   # V4 推理、评分、可视化（CLI + Gradio）
│   └── notebook_build/        # 训练 notebook 生成器（源文件）
├── notebooks/
│   ├── train/                 # 模型训练与评估
│   └── sandbox/               # 参数化体块生成实验
├── website/                   # 静态站点（spacemodal.com）
├── docs/                      # 论文草稿与建模规范（不上传）
├── data/                      # 数据集（不上传）
├── weights/                   # 训练权重（不上传）
└── requirements.txt
```

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

Python 3.8+，训练/推理建议 GPU。Rhino 7+ 仅导出时需要。

### 数据校验

```bash
python scripts/offline_check/validate_json_dataset.py --data-dir data/processed
```

### 条件生成（CLI）

```bash
python scripts/spatial_modal_infer/run_cli.py \
  --weights weights/spatial_modal_cvae_v4_88x88x24.pth \
  --site-x 18000 --site-y 15000 \
  --living-room 1 --dining-room 1 --kitchen 1 \
  --bedroom 3 --bathroom 2 --corridor 2 --stairs 1 \
  --entryway 1 --balcony 1 \
  --seed 42 --out-dir ./outputs
```

### 交互界面（Gradio）

```bash
python scripts/spatial_modal_infer/app_gradio.py \
  --weights weights/spatial_modal_cvae_v4_88x88x24.pth
```

### 参数化体块生成（无需权重）

```bash
# 纯规则递归切割
python notebooks/sandbox/demo_block_cut.py --length 18 --width 15 --seed 123

# V4 概率引导分区布局
python notebooks/sandbox/demo_zoned_layout_with_prior.py \
  --weights weights/spatial_modal_cvae_v4_88x88x24_probe_best.pth \
  --length 18 --width 15 --seed 703
```

## 项目网站

域名为 [spacemodal.com](https://spacemodal.com) / spacemodal.cn，源码在 [`website/`](website/)：

| 页面 | 路径 | 说明 |
|---|---|---|
| 首页 | [`website/index.html`](website/index.html) | 项目概览 |
| 数据可视化 | [`website/viewer/`](website/viewer/) | 10 套公开 Demo（Vite + Three.js） |
| 论文 | [`website/paper/`](website/paper/) | 摘要、方法、引用 |
| 作品集 | [`website/portfolio/`](website/portfolio/) | 实习作品集 |

部署说明见 [`website/DEPLOY.md`](website/DEPLOY.md)。

## 许可证

待定（论文发表后补充）。当前仅供学术审阅。
