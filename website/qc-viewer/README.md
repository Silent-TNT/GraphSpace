# 数据可视化工具（qc-viewer）

浏览拓扑 JSON：2D 平面图 + 3D 体块预览。

## 模式

| 命令 | 数据来源 | 用途 |
|------|----------|------|
| `npm run dev` | `data/processed/` 或 `dataset/`（全量） | 本地开发 |
| `npm run build:demo` | `website/demo/datasets/*.json` | 公网 `/viewer/` |

## 构建公网 Demo

```bash
cd website/qc-viewer
npm install
npm run build:demo
```

输出到 `website/viewer/`（10 套 JSON + 可视化 UI）。
