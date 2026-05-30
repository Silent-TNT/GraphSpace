import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, "..");
const websiteRoot = path.resolve(webRoot, "..");
const demoSourceDir = path.join(websiteRoot, "demo", "datasets");
const publicDir = path.join(webRoot, "public");
const publicDatasetDir = path.join(publicDir, "dataset");
const manifestPath = path.join(publicDir, "manifest.json");

const LOCAL_SOURCE_DIRS = [
  path.resolve(websiteRoot, "../data/processed"),
  path.resolve(websiteRoot, "../dataset"),
];

function parseMode() {
  const arg = process.argv.find((a) => a.startsWith("--mode="));
  const mode = arg?.split("=")[1] ?? "local";
  if (mode !== "local" && mode !== "demo") {
    throw new Error(`Unknown mode "${mode}". Use --mode=local or --mode=demo`);
  }
  return mode;
}

function computeStats(rooms) {
  const stats = {};
  for (const room of rooms) {
    stats[room.type] = (stats[room.type] || 0) + 1;
  }
  return stats;
}

function computeBuildingSize(rooms) {
  let maxX = 0;
  let maxY = 0;
  let maxZ = 0;
  for (const room of rooms) {
    maxX = Math.max(maxX, room.box_max[0]);
    maxY = Math.max(maxY, room.box_max[1]);
    maxZ = Math.max(maxZ, room.box_max[2]);
  }
  return { x: maxX, y: maxY, z: maxZ };
}

function getFloors(rooms) {
  return [...new Set(rooms.map((r) => r.floor))].sort((a, b) => a - b);
}

function resolveLocalSourceDir() {
  for (const dir of LOCAL_SOURCE_DIRS) {
    if (!fs.existsSync(dir)) continue;
    const files = fs.readdirSync(dir).filter((f) => f.endsWith(".json"));
    if (files.length > 0) return dir;
  }
  return null;
}

function clearPublicDataset() {
  if (fs.existsSync(publicDatasetDir)) {
    fs.rmSync(publicDatasetDir, { recursive: true, force: true });
  }
  fs.mkdirSync(publicDatasetDir, { recursive: true });
}

function buildHouseEntry(sourcePath, fileName) {
  const raw = JSON.parse(fs.readFileSync(sourcePath, "utf-8"));
  const rooms = raw.rooms || [];
  const metadata = raw.metadata || {};
  const stats = metadata.stats || computeStats(rooms);
  const buildingSize = metadata.building_size || computeBuildingSize(rooms);
  const floors = getFloors(rooms);

  return {
    entry: {
      id: raw.house_id || fileName.replace(".json", ""),
      file: fileName,
      totalRooms: metadata.total_rooms ?? rooms.length,
      stats,
      buildingSize,
      floors,
    },
    raw,
  };
}

function syncLocalDataset() {
  const sourceDir = resolveLocalSourceDir();
  if (!sourceDir) {
    console.error(
      "Local dataset not found. Expected JSON files in one of:\n" +
        LOCAL_SOURCE_DIRS.map((d) => `  - ${d}`).join("\n")
    );
    process.exit(1);
  }

  clearPublicDataset();
  const files = fs.readdirSync(sourceDir).filter((f) => f.endsWith(".json"));
  for (const file of files) {
    fs.copyFileSync(path.join(sourceDir, file), path.join(publicDatasetDir, file));
  }
  return files.map((file) => {
    const { entry } = buildHouseEntry(path.join(sourceDir, file), file);
    return entry;
  });
}

function syncDemoDataset() {
  if (!fs.existsSync(demoSourceDir)) {
    console.error(
      `Demo datasets missing: ${demoSourceDir}\n` +
        "Place showcase JSON files in website/demo/datasets/"
    );
    process.exit(1);
  }

  clearPublicDataset();
  const files = fs
    .readdirSync(demoSourceDir)
    .filter((f) => f.endsWith(".json"))
    .sort();

  if (files.length === 0) {
    console.error(`No JSON files in ${demoSourceDir}`);
    process.exit(1);
  }

  for (const file of files) {
    fs.copyFileSync(
      path.join(demoSourceDir, file),
      path.join(publicDatasetDir, file)
    );
  }

  return files.map((file) => {
    const { entry } = buildHouseEntry(path.join(demoSourceDir, file), file);
    return entry;
  });
}

const mode = parseMode();
const houses = mode === "demo" ? syncDemoDataset() : syncLocalDataset();

fs.mkdirSync(publicDir, { recursive: true });
fs.writeFileSync(
  manifestPath,
  JSON.stringify(
    {
      mode,
      count: houses.length,
      houses,
      notice:
        mode === "demo"
          ? "Public demo datasets for visualization showcase. Full training set is local only."
          : "Local development manifest.",
    },
    null,
    2
  )
);

console.log(
  `[${mode}] synced ${houses.length} file(s) -> ${publicDatasetDir}\nmanifest -> ${manifestPath}`
);
