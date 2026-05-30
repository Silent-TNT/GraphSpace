# 作品集安全模式 — 检查清单

本网站以**静态页 + 无数据下载**方式展示项目，适用于论文审稿期与实习作品集。

## 不会泄露的内容

以下仅存在于本地，已被 `.gitignore` 排除，**不应**出现在网站或公开 GitHub：

- `data/**/*.json` — 原始 / processed 户型拓扑
- `models/*.pth` — 训练权重
- `docs/` — 论文草稿与建模规范
- `notebooks/experiments/` — 逐样本评估结果
- Rhino 源文件 `.3dm`

## 网站可安全展示

- 方法框架与流水线说明
- **数据可视化**（`/viewer/`）— 10 套 Demo（`demo/datasets/`）
- 技术栈与个人贡献
- **聚合**指标（如整体 mIoU、program_acc），非单样本结果
- 脱敏示意图 / 录屏（无项目名、无完整 JSON）
- 脚本与 notebook **结构**（仓库公开时）

## 高风险行为（避免）

| 行为 | 风险 |
|------|------|
| 挂载私有数据的公开 Colab | 数据可被复制 |
| 公网 Gradio + 真实权重 | 模型与生成策略泄露 |
| 提供 JSON / 权重下载 | 直接数据流失 |
| 部署 `website/qc-viewer/public/dataset/` 全量 96 套 | **完整训练集泄露** |
| 使用 `npm run build:local` 部署公网 | 同上 |
| 展示未脱敏真实户型图 | 间接泄露数据集 |
| `gradio_app.py` 绑定 `0.0.0.0` 公网 | 本地权重暴露 |

当前 `gradio_app.py` 已限制为 `127.0.0.1` 本地访问。

## 面试演示建议

1. 优先展示本网站 `/portfolio/`
2. 需要动态演示时，**本地**打开 notebook 或 Gradio
3. 提前准备 1 段脱敏录屏作为 backup
4. 口头说明「数据与权重因审稿未公开」— 体现规范意识

## 部署前命令检查

```bash
# 确认 website 目录无敏感扩展名
find website -type f \( -name "*.json" ! -name "config.json" -o -name "*.pth" -o -name "*.pt" \)

# 确认 git 未跟踪 data/models
git status
git ls-files data/ models/
```

`website/demo/config.json` 仅含 URL 配置，不含训练数据，可提交。

## 论文发表后

若需公开 Demo，应：

1. 使用合成或脱敏样例
2. 不提供批量下载
3. 权重通过 Releases 发布并注明 license
4. 更新 `/demo/` 页说明与隐私政策
