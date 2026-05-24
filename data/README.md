# data/ — 本地数据集目录

此目录**不会**上传至 GitHub，请在本地自行维护。

## 推荐结构

```text
data/
├── raw/              # 原始 Rhino 导出或未清洗 JSON（可选）
└── processed/        # 通过 QC 的 house_*.json，供校验与训练使用
```

## 使用方式

1. 使用 `scripts/rhino_export/260524rhino-json-v14.py` 在 Rhino 中导出 JSON，保存至 `processed/`
2. 运行离线校验：

   ```bash
   python scripts/offline_check/validate_json_dataset.py --data-dir data/processed
   ```

3. 训练 notebook（`notebooks/train/`）默认从本目录读取 JSON；预处理生成的 `.pt` 缓存也建议放在本地 `data/` 下（已被 `.gitignore` 排除）

## 说明

原始户型数据涉及学术研究与建模规范，在论文正式发表前不对外公开。
