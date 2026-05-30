# graphspace.cn / spacemodal.com 信息架构

## 空间折叠五区

整站以「空间折叠」为隐喻：首页（`/`）为折叠态五维门户；各区展开为独立深度路径。

| 版块 | 路径 | 职责 |
|------|------|------|
| 折叠空间（首页） | `/` | 五区卡片门户、工作流摘要 |
| 个人数字门户 | `/portal/` | 作品集、贡献、指标、隐私说明 |
| 数据看板 | `/viewer/` | qc-viewer：10 套 Demo 2D/3D |
| 机器学习展示 | `/ml/` | 枢纽 → `/paper/` `/demo/` `/product/` |
| AI 对话检索 | `/ai/` | 站内检索 + 对话占位（二期 LLM） |
| 个人工具箱 | `/tools/` | Rhino/QC/Gradio 本地工具说明 |

## 站点地图

```
/
├── /portal/           个人数字门户（原作品集主内容）
├── /portfolio/        旧路径，保留兼容，指向 portal 说明
├── /viewer/           数据看板（Vite 构建产物）
├── /ml/               ML 展示枢纽
│   ├── /paper/        论文
│   ├── /demo/         演示说明
│   └── /product/      模态户型 ModalPlan
├── /ai/               AI 检索（search-index.json）
└── /tools/            个人工具箱
```

## 导航

顶栏由 `website/js/shell.js` 注入，五链：**折叠空间 | 门户 | 看板 | ML | AI | 工具箱**。

## URL 与 SEO

| 路径 | `<title>` | `<meta description>` |
|------|-----------|---------------------|
| `/` | SpaceModal — 空间折叠五区门户 | 五区门户：门户、看板、ML、AI、工具箱 |
| `/portal/` | 个人数字门户 — SpaceModal | 作品集、Rhino 管线、GNN/CVAE |
| `/viewer/` | 数据看板 — SpaceModal | 10 套 Demo 2D/3D 户型浏览 |
| `/ml/` | 机器学习展示 — SpaceModal | 论文、演示、产品枢纽 |
| `/ai/` | AI 对话检索 — SpaceModal | 全站与 Demo 户型检索 |
| `/tools/` | 个人工具箱 — SpaceModal | Rhino 导出、QC、Gradio 说明 |
| `/paper/` | 论文 — SpaceModal / GraphSpace | 论文摘要与方法 |
| `/demo/` | 演示说明 — SpaceModal | Demo 与 Gradio 本地说明 |
| `/product/` | 模态户型 ModalPlan — SpaceModal | 产品化方向介绍 |

## 构建与部署

```bash
cd website
npm run build:site    # 生成 search-index + 构建 viewer
# 或分别：
npm run build:search
npm run build:viewer
```

GitHub Pages 工作流：推送 `website/**` 时在 CI 中执行 `build:viewer`；本地发布前建议运行 `build:site` 以更新 `/ai/search-index.json` 与 `/viewer/`。

## 后期扩展

- `app.graphspace.cn` — SaaS（需备案若大陆服务器）
- `/ai/` 二期 — LLM API / Edge Function
- 可选 Vite 单页 shell 强化折叠动画

## 域名跳转

| 来源 | 目标 |
|------|------|
| `motaihuxing.cn` | `https://graphspace.cn/product/` |
| `modalplan.cn` | `https://graphspace.cn/product/` |
| `www.graphspace.cn` | `https://graphspace.cn/` |
