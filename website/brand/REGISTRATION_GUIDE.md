# 域名注册操作指南

> 域名注册需本人实名认证并支付，无法由代码仓库代完成。按本指南逐步操作即可。

## 推荐注册组合

| 域名 | 用途 | 参考年费 |
|------|------|----------|
| `graphspace.cn` | 母品牌主站 | 约 ¥29–39/首年 |
| `motaihuxing.cn` | 模态户型产品（中文口述） | 约 ¥29–39/首年 |
| `modalplan.cn` | 模态户型产品（英文物料） | 约 ¥29–39/首年 |

可选防御：`tukongjian.cn`（图空间谐音）

## 注册步骤（以阿里云为例）

### 1. 查询可用性

1. 访问 [阿里云万网域名注册](https://wanwang.aliyun.com/domain/)
2. 输入 `graphspace.cn`，点击查询
3. 对 `motaihuxing.cn`、`modalplan.cn` 重复查询
4. 可用则加入清单

### 2. 创建信息模板并实名

1. 控制台 → 域名 → 信息模板
2. 个人：身份证；企业：营业执照
3. 提交实名认证（通常 1–3 个工作日）

### 3. 购买域名

1. 选择注册年限（建议 ≥2 年，防遗忘续费）
2. 开启隐私保护（可选）
3. 支付完成

### 4. DNS 解析（网站上线后）

将域名解析到托管平台：

| 托管方式 | 记录类型 | 值 |
|----------|----------|-----|
| GitHub Pages | CNAME | `<user>.github.io` 或自定义 |
| Cloudflare Pages | CNAME | `<project>.pages.dev` |
| 国内服务器 | A | 服务器 IP |

示例（GitHub Pages）：

```
主机记录: @
记录类型: A
记录值: 185.199.108.153（及 GitHub Pages 其余 IP）

主机记录: www
记录类型: CNAME
记录值: <username>.github.io
```

### 5. 备案（若使用大陆服务器）

- 使用**境外** GitHub Pages / Cloudflare / Vercel：**无需 ICP 备案**
- 使用**腾讯云 / 阿里云大陆服务器**：必须备案，网站名称避免「中国」「国家」等敏感词
- 个人备案网站不宜涉及商业销售；企业备案可挂产品页

## 产品域跳转策略

初期只运维 `graphspace.cn`，产品域注册后设置 301：

```
motaihuxing.cn  →  https://graphspace.cn/product/
modalplan.cn    →  https://graphspace.cn/product/
```

等产品独立运营后再拆站。

## 商标同步申请（建议）

在 [阿里云商标服务](https://tm.aliyun.com/) 或 CNIPA 官网提交：

| 商标名 | 建议类别 |
|--------|----------|
| 模态户型 | 42（软件/design）、35（推广） |
| 图空间 GraphSpace | 42、9（如有客户端） |

## 注册后核对清单

- [ ] 三个 P0/P1 域名已注册且实名通过
- [ ] DNS 已指向网站托管
- [ ] `motaihuxing.cn` / `modalplan.cn` 301 到 `/product/`（可选）
- [ ] 商标申请已提交
- [ ] 微信公众号名称已预留（如需）
