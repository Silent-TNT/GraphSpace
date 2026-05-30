import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const websiteRoot = path.resolve(__dirname, "..");

const PAGE_ENTRIES = [
  {
    path: "/",
    title: "空间折叠五区门户",
    description: "SpaceModal 空间折叠门户：个人数字门户、数据看板、机器学习展示、AI 检索、个人工具箱。",
    tags: ["portal", "home", "GraphSpace"],
  },
  {
    path: "/portal/",
    title: "个人数字门户",
    description: "作品集、工作流、技术栈与量化指标。Rhino 管线与 SpatialModalCVAE。",
    tags: ["portal", "portfolio"],
  },
  {
    path: "/viewer/",
    title: "数据看板",
    description: "10 套 Demo 户型 2D 平面图与 3D 体块浏览，房间统计与建筑尺寸。",
    tags: ["dashboard", "viewer", "demo"],
  },
  {
    path: "/ml/",
    title: "机器学习展示",
    description: "论文、演示与模态户型产品枢纽。异构图 + CVAE + 体素解码。",
    tags: ["ml", "paper", "demo", "product"],
  },
  {
    path: "/paper/",
    title: "论文",
    description: "GraphSpace 住宅布局生成论文摘要、方法与评估指标。",
    tags: ["ml", "paper", "GNN", "CVAE"],
  },
  {
    path: "/demo/",
    title: "演示说明",
    description: "数据可视化 Demo 与本地 Gradio 条件生成说明。",
    tags: ["ml", "demo", "gradio"],
  },
  {
    path: "/product/",
    title: "模态户型 ModalPlan",
    description: "面向设计院的约束感知住宅体块生成产品方向。",
    tags: ["ml", "product", "ModalPlan"],
  },
  {
    path: "/ai/",
    title: "AI 对话检索",
    description: "全站内容与 Demo 户型 ID 检索。",
    tags: ["ai", "search"],
  },
  {
    path: "/tools/",
    title: "个人工具箱",
    description: "Rhino 导出、离线 QC、Gradio 与本地 qc-viewer。",
    tags: ["tools", "rhino", "scripts"],
  },
  {
    path: "/portfolio/",
    title: "作品集（旧路径）",
    description: "已迁移至 /portal/，内容相同。",
    tags: ["portal", "portfolio"],
  },
];

function loadManifestHouses() {
  const manifestPath = path.join(websiteRoot, "viewer", "manifest.json");
  if (!fs.existsSync(manifestPath)) {
    console.warn("viewer/manifest.json not found, skipping house entries");
    return [];
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  return (manifest.houses ?? []).map((h) => ({
    path: `/viewer/?house=${encodeURIComponent(h.id)}`,
    title: `Demo 户型 · ${h.id}`,
    description: `${h.totalRooms} 个房间 · ${(h.floors ?? []).length} 层 · 数据看板样例`,
    tags: ["dashboard", "house", h.id],
  }));
}

const items = [...PAGE_ENTRIES, ...loadManifestHouses()];
const outPath = path.join(websiteRoot, "ai", "search-index.json");
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify({ generatedAt: new Date().toISOString(), items }, null, 2));
console.log(`Wrote ${items.length} entries to ${outPath}`);
