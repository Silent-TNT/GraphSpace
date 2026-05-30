import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, "..");
const candidates = [
  path.join(webRoot, "public/dataset/house_1380.json"),
  path.resolve(webRoot, "../../data/processed/house_1380.json"),
];

const outDir = path.join(webRoot, "demo-data");
const outFile = path.join(outDir, "demo_house.json");

const source = candidates.find((p) => fs.existsSync(p));
if (!source) {
  console.error("No source house JSON found to build demo sample.");
  process.exit(1);
}

const raw = JSON.parse(fs.readFileSync(source, "utf-8"));
raw.house_id = "demo_sample_01";
raw.metadata = {
  ...(raw.metadata || {}),
  note: "Public demo sample for portfolio. Not part of the training release.",
  demo: true,
};

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(outFile, JSON.stringify(raw, null, 2));
console.log(`Created ${outFile} from ${source}`);
