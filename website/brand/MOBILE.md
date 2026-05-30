# 移动端开发约定（SpaceModal 网站）

后续改 UI、加页面、加交互时，请默认按 **手机端优先** 自检。

## 断点

| 断点 | 用途 |
|------|------|
| `≤768px` | 主手机样式（侧栏导航、单列布局） |
| `≤480px` | 小屏（更紧凑字号与间距） |
| `≤1024px` | 平板（首页两列卡片等） |

样式写在 [`css/style.css`](../css/style.css) 与页面专用 CSS（如 [`css/fold-home.css`](../css/fold-home.css)）。

## 必做清单

- [ ] `viewport` 含 `viewport-fit=cover`（刘海屏安全区）
- [ ] 可点击区域最小高度 **44px**（`--touch-min`）
- [ ] 表单/搜索框 `font-size: 16px`（避免 iOS 自动放大）
- [ ] 宽表格包在 `<div class="table-scroll">` 内
- [ ] 触控设备用 `@media (hover: none)` 弱化悬停-only 效果
- [ ] 新页面使用 `data-site-shell` + [`js/shell.js`](../js/shell.js)（含抽屉导航）

## 导航

- 桌面：顶栏横排链接
- 手机：右上角 **汉堡菜单** → 右侧滑出抽屉

## 数据看板 `/viewer/`

- 源码样式：[`qc-viewer/src/styles.css`](../qc-viewer/src/styles.css)
- 修改后执行：`cd website && npm run build:viewer`
- 手机布局：上列表 / 中画布 / 下信息面板；顶栏链接可横向滑动

## 首页折纸卡片

- 手机单列，「折入 →」常显，不依赖悬停
- 整页折叠动画在触控端略缩短（见 `fold.js`）
