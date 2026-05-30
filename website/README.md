# SpaceModal 官网（统一目录）

所有网站相关内容都在 `website/` 下。

## 目录结构

```text
website/
├── index.html, portfolio/, paper/, demo/, product/   # 静态主页
├── demo/datasets/                                    # 公网 Demo JSON（10 套）
├── css/, js/, brand/
├── viewer/                                           # 数据可视化（构建产物 → /viewer/）
├── qc-viewer/                                        # 可视化工具源码（Vite + Three.js）
├── package.json
└── CNAME                                             # spacemodal.com
```

## 常用命令

```bash
# 构建公网可视化（10 套 Demo → website/viewer/）
cd website
npm run build:viewer

# 本地预览整站
python -m http.server 8080

# 本地全量数据（训练集，不上传公网）
cd website/qc-viewer
npm install
npm run dev
```

## 页面

| 路径 | 说明 |
|------|------|
| `/` | 项目首页 |
| `/portfolio/` | 实习作品集 |
| `/viewer/` | **数据可视化**（10 套 Demo） |
| `/demo/` | 演示说明 |
| `/paper/` | 论文摘要 |
| `/product/` | 模态户型 |

## 添加 / 更换 Demo 样例

将 `house_*.json` 放入 [`demo/datasets/`](demo/datasets/)，然后：

```bash
cd website && npm run build:viewer
```

## 绑定域名

见 [`DEPLOY.md`](DEPLOY.md) 中的「腾讯云 + GitHub Pages」步骤。

## 可视化工具源码

见 [`qc-viewer/README.md`](qc-viewer/README.md)。
