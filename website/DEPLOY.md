# SpaceModal 部署指南（spacemodal.com）

## 一、上线前准备

### 1. 编辑站点配置

[`js/site-config.js`](js/site-config.js)：

```javascript
window.SITE_CONFIG = {
  siteUrl: "https://spacemodal.com",
  contactEmail: "contact@spacemodal.com",
  githubUrl: "https://github.com/你的用户名/GraphSpace",
  authorName: "你的姓名",
  authorRole: "建筑学硕士 · 计算设计 / AI",
};
```

### 2. 确认 Demo 数据

10 套展示 JSON 放在 [`demo/datasets/`](demo/datasets/)。增删样例后重新构建：

```bash
cd website
npm run build:viewer
```

### 3. 本地预览

```bash
cd website
npm run build:viewer
python -m http.server 8080
```

- 首页：http://127.0.0.1:8080/
- 数据可视化：http://127.0.0.1:8080/viewer/

---

## 二、绑定域名（推荐：GitHub Pages + 腾讯云 DNS）

你已购买 **spacemodal.com** / **spacemodal.cn**，推荐用 GitHub Pages 托管（免费、免备案），DNS 在腾讯云解析。

### 步骤 1：推送代码到 GitHub

```bash
git add website/
git commit -m "deploy: SpaceModal website"
git push origin main
```

### 步骤 2：开启 GitHub Pages

1. 打开仓库 **Settings → Pages**
2. **Source** 选 **GitHub Actions**（仓库已含 [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)）
3. 推送 `website/**` 后 Actions 会自动：
   - 在 `website/qc-viewer` 执行 `npm run build:demo`
   - 部署整个 `website/` 目录

### 步骤 3：绑定自定义域

1. Settings → Pages → **Custom domain** 填入：`spacemodal.com`
2. 勾选 **Enforce HTTPS**
3. GitHub 会显示需要的 DNS 记录

### 步骤 4：腾讯云 DNS 解析

登录 [腾讯云 DNS 控制台](https://console.cloud.tencent.com/cns)，选择 `spacemodal.com`：

| 主机记录 | 记录类型 | 记录值 | 说明 |
|----------|----------|--------|------|
| `@` | A | `185.199.108.153` | GitHub Pages（需添加 4 条 A，见下） |
| `@` | A | `185.199.109.153` | |
| `@` | A | `185.199.110.153` | |
| `@` | A | `185.199.111.153` | |
| `www` | CNAME | `你的用户名.github.io` | 或 CNAME 到 `spacemodal.com` |

> GitHub 自定义域页面会给出最新 IP，以其为准。

**spacemodal.cn**（可选）：

- 方式 A：DNSPod **URL 转发** → `https://spacemodal.com`
- 方式 B：同样 CNAME 到 GitHub（与 .com 并存）

### 步骤 5：等待生效

- DNS：通常 10 分钟～24 小时
- GitHub Pages：首次部署约 1～3 分钟
- 验证：访问 https://spacemodal.com/viewer/ 应看到 10 套 Demo

### 步骤 6：邮箱（可选）

腾讯云 → 邮件推送 / 企业邮箱，将 `contact@spacemodal.com` 转发到你的常用邮箱。

---

## 三、备选：腾讯云 COS 静态托管

若希望全部在腾讯云：

1. COS 创建存储桶，开启**静态网站**（索引 `index.html`）
2. 本地执行 `npm run build:viewer` 后，上传整个 `website/` 目录
3. 存储桶绑定自定义域 `spacemodal.com`
4. **使用大陆 CDN 需 ICP 备案**；仅 COS 静态 + 境外加速可不备案

---

## 四、上线自检

- [ ] `website/viewer/dataset/` 仅有 `demo/datasets/` 中的 10 个 JSON
- [ ] 未上传 `data/processed/` 全量训练集
- [ ] `site-config.js` 已填写姓名、GitHub、邮箱
- [ ] https://spacemodal.com 与 https://spacemodal.com/viewer/ 均可访问
- [ ] 可视化页顶「← 返回 SpaceModal 首页」链接正常

## 五、页面结构

| URL | 用途 |
|-----|------|
| `/` | 项目首页 |
| `/portfolio/` | 实习作品集 |
| `/viewer/` | **数据可视化**（10 套 Demo） |
| `/demo/` | 演示说明 |
| `/paper/` | 论文摘要 |
| `/product/` | 模态户型 |

安全说明见 [`brand/PORTFOLIO_SAFETY.md`](brand/PORTFOLIO_SAFETY.md)。
