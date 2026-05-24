# models/ — 本地模型权重目录

此目录**不会**上传至 GitHub，用于存放训练完成的权重文件（`.pth` / `.pt`）。

## 推荐结构

```text
models/
├── spatial_modal_cvae_mvp.pth    # 示例：主模型 checkpoint
└── checkpoints/                  # 可选：按 epoch 或实验编号归档
```

## 与 notebook 的路径约定

`notebooks/train/` 中的 notebook 可能默认将权重写入 `data/`。建议在本地训练完成后，将最终权重**移动或复制**到本目录，便于与数据集分离管理。

在 notebook 中可将保存路径改为：

```python
WEIGHT_PATH = "models/spatial_modal_cvae_mvp.pth"
```

## 是否开源权重？

**当前阶段：不建议公开。**

- 权重由私有数据集训练得到，可能间接反映数据分布与建模细节
- 论文尚未发表，公开权重可能影响审稿与首发声明
- 大文件（通常数十 MB 以上）也不适合直接放入 Git 仓库

**论文发表后**，可考虑通过以下方式发布：

- GitHub Releases（附 SHA256 校验）
- Hugging Face Hub / Zenodo 等学术托管平台
- 在 README 中注明版本、训练数据规模与评估指标

届时可从 `.gitignore` 中移除 `models/` 规则，或仅发布经脱敏/蒸馏后的公开权重。
