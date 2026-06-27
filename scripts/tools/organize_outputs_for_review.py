"""Build a report-friendly visual index for GraphSpace outputs.

The script is intentionally non-destructive: original experiment folders stay
where they are, while visual artifacts are copied into outputs/_report_visuals
and non-visual artifacts are cataloged by type.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


VISUAL_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".html"}
WEIGHT_EXTS = {".pt", ".pth", ".ckpt", ".onnx"}
ARRAY_EXTS = {".npz", ".npy"}
LOG_EXTS = {".log", ".jsonl", ".csv", ".md"}
SKIP_DIRS = {"_report_visuals", "_nonvisual_catalog"}


@dataclass
class VisualItem:
    source: Path
    copied: Path
    top_dir: str
    kind: str
    size: int
    modified: datetime
    p0: str = ""
    topology: str = ""
    edge_f1: str = ""


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_path(data: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def fmt_rate(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return ""


def summary_metrics(run_dir: Path) -> dict[str, str]:
    summary = safe_read_json(run_dir / "summary.json")
    evaluation = safe_read_json(run_dir / "evaluation.json")
    p0 = summary.get("p0_pass")
    if p0 is None:
        p0 = get_path(evaluation, ["p0", "pass"], None)

    topology = get_path(summary, ["topology", "realization_rate"], None)
    if topology is None:
        topology = get_path(evaluation, ["p1_spatial_organization", "target_topology", "realization_rate"], None)

    edge_f1 = get_path(summary, ["metrics", "edge_f1"], None)
    return {
        "p0": fmt_bool(p0),
        "topology": fmt_rate(topology),
        "edge_f1": fmt_rate(edge_f1),
    }


def visual_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".html"):
        return "interactive"
    if "layout_3d" in name or "3d" in name:
        return "3d"
    if "floor_1" in name:
        return "floor_1"
    if "floor_2" in name:
        return "floor_2"
    if "topology" in name:
        return "topology"
    if "contact_sheet" in name:
        return "contact_sheet"
    if "diagnostic" in name:
        return "diagnostic"
    return path.suffix.lower().lstrip(".")


def should_skip(path: Path, output_root: Path) -> bool:
    try:
        parts = path.relative_to(output_root).parts
    except ValueError:
        return False
    return any(part in SKIP_DIRS for part in parts)


def copy_visuals(output_root: Path, visual_root: Path) -> list[VisualItem]:
    items: list[VisualItem] = []
    for source in sorted(output_root.rglob("*")):
        if not source.is_file() or should_skip(source, output_root):
            continue
        if source.suffix.lower() not in VISUAL_EXTS:
            continue
        rel_source = source.relative_to(output_root)
        top_dir = rel_source.parts[0] if rel_source.parts else "."
        dest = visual_root / rel_source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        stat = source.stat()
        metrics = summary_metrics(output_root / top_dir)
        items.append(
            VisualItem(
                source=source,
                copied=dest,
                top_dir=top_dir,
                kind=visual_kind(source),
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
                p0=metrics["p0"],
                topology=metrics["topology"],
                edge_f1=metrics["edge_f1"],
            )
        )
    return sorted(items, key=lambda item: item.modified, reverse=True)


def write_visual_manifest(items: list[VisualItem], output_root: Path, report_root: Path) -> None:
    manifest = report_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["modified", "run", "kind", "p0", "topology", "edge_f1", "size_bytes", "source", "review_copy"])
        for item in items:
            writer.writerow(
                [
                    item.modified.isoformat(timespec="seconds"),
                    item.top_dir,
                    item.kind,
                    item.p0,
                    item.topology,
                    item.edge_f1,
                    item.size,
                    rel(item.source, output_root),
                    rel(item.copied, report_root),
                ]
            )


def write_index(items: list[VisualItem], report_root: Path) -> None:
    rows = []
    for item in items:
        copied_rel = rel(item.copied, report_root)
        source_rel = item.source.as_posix()
        title = f"{item.top_dir} / {item.source.name}"
        if item.source.suffix.lower() == ".html":
            preview = f'<a class="html-link" href="{html.escape(copied_rel)}">Open HTML</a>'
        else:
            preview = f'<a href="{html.escape(copied_rel)}"><img src="{html.escape(copied_rel)}" loading="lazy" alt="{html.escape(title)}"></a>'
        rows.append(
            f"""
<article class="card" data-run="{html.escape(item.top_dir)}" data-kind="{html.escape(item.kind)}">
  <div class="preview">{preview}</div>
  <div class="meta">
    <strong>{html.escape(title)}</strong>
    <span>{html.escape(item.modified.strftime('%Y-%m-%d %H:%M'))} | {html.escape(item.kind)}</span>
    <span>P0 {html.escape(item.p0 or '-')} | topology {html.escape(item.topology or '-')} | edge F1 {html.escape(item.edge_f1 or '-')}</span>
    <code>{html.escape(source_rel)}</code>
  </div>
</article>"""
        )

    run_options = "\n".join(
        f'<option value="{html.escape(run)}">{html.escape(run)}</option>'
        for run in sorted({item.top_dir for item in items})
    )
    kind_options = "\n".join(
        f'<option value="{html.escape(kind)}">{html.escape(kind)}</option>'
        for kind in sorted({item.kind for item in items})
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GraphSpace 汇报图索引</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f5; color: #202124; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 16px 20px; background: #ffffff; border-bottom: 1px solid #d9d9d6; }}
    h1 {{ margin: 0 0 10px; font-size: 22px; font-weight: 700; }}
    .controls {{ display: grid; grid-template-columns: minmax(220px, 1fr) 180px 180px; gap: 10px; }}
    input, select {{ height: 36px; border: 1px solid #c7c7c2; border-radius: 6px; padding: 0 10px; background: #fff; }}
    main {{ padding: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
    .preview {{ height: 210px; display: flex; align-items: center; justify-content: center; background: #ededeb; }}
    img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    .html-link {{ color: #075e54; font-weight: 700; text-decoration: none; }}
    .meta {{ display: grid; gap: 6px; padding: 10px; font-size: 12px; line-height: 1.35; }}
    .meta strong {{ font-size: 13px; }}
    code {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #5f6368; }}
    @media (max-width: 760px) {{ .controls {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>GraphSpace 汇报图索引</h1>
    <div class="controls">
      <input id="q" placeholder="搜索实验目录、文件名或原始路径">
      <select id="run"><option value="">全部实验</option>{run_options}</select>
      <select id="kind"><option value="">全部图类</option>{kind_options}</select>
    </div>
  </header>
  <main>
    <div class="grid" id="grid">
      {''.join(rows)}
    </div>
  </main>
  <script>
    const q = document.getElementById('q');
    const run = document.getElementById('run');
    const kind = document.getElementById('kind');
    const cards = [...document.querySelectorAll('.card')];
    function applyFilters() {{
      const text = q.value.toLowerCase();
      for (const card of cards) {{
        const okText = !text || card.innerText.toLowerCase().includes(text);
        const okRun = !run.value || card.dataset.run === run.value;
        const okKind = !kind.value || card.dataset.kind === kind.value;
        card.style.display = okText && okRun && okKind ? '' : 'none';
      }}
    }}
    q.addEventListener('input', applyFilters);
    run.addEventListener('change', applyFilters);
    kind.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""
    (report_root / "index.html").write_text(html_text, encoding="utf-8")


def write_readme(items: list[VisualItem], report_root: Path, output_root: Path) -> None:
    readme = f"""# GraphSpace 汇报图索引

这个目录是从 `{output_root}` 生成的非破坏性汇报索引。

- 打开 `index.html` 可以浏览所有可视化图和交互 HTML。
- `manifest.csv` 可以按实验目录、图类、P0、拓扑实现率或原始路径检索。
- 可视化副本在 `visuals/` 下，保留原始 `outputs/` 目录结构。
- 原始实验输出没有被移动或删除。

统计：

- 已复制可视化产物：{len(items)}
- 含可视化的实验目录：{len({item.top_dir for item in items})}
"""
    (report_root / "README.md").write_text(readme, encoding="utf-8")


def catalog_nonvisual(output_root: Path, catalog_root: Path) -> None:
    catalog_root.mkdir(parents=True, exist_ok=True)
    buckets = {
        "weights.csv": WEIGHT_EXTS,
        "arrays.csv": ARRAY_EXTS,
        "logs_and_tables.csv": LOG_EXTS,
        "json_reports.csv": {".json"},
    }
    for filename, exts in buckets.items():
        with (catalog_root / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["modified", "top_dir", "extension", "size_bytes", "path"])
            for path in sorted(output_root.rglob("*")):
                if not path.is_file() or should_skip(path, output_root):
                    continue
                if path.suffix.lower() not in exts:
                    continue
                stat = path.stat()
                rel_path = path.relative_to(output_root)
                top_dir = rel_path.parts[0] if rel_path.parts else "."
                writer.writerow(
                    [
                        datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                        top_dir,
                        path.suffix.lower(),
                        stat.st_size,
                        rel_path.as_posix(),
                    ]
                )
    (catalog_root / "README.md").write_text(
        "# GraphSpace 非可视化产物清单\n\n"
        "这个目录只索引非可视化产物，不移动原始文件。\n\n"
        "- `weights.csv`：checkpoint 和模型文件。\n"
        "- `json_reports.csv`：JSON 摘要、报告、布局和图导出。\n"
        "- `arrays.csv`：NumPy 数组产物。\n"
        "- `logs_and_tables.csv`：日志、JSONL、CSV 和 Markdown 输出。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs", help="GraphSpace outputs directory")
    args = parser.parse_args()

    output_root = Path(args.outputs).resolve()
    report_root = output_root / "_report_visuals"
    visual_root = report_root / "visuals"
    catalog_root = output_root / "_nonvisual_catalog"

    report_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)

    items = copy_visuals(output_root, visual_root)
    write_visual_manifest(items, output_root, report_root)
    write_index(items, report_root)
    write_readme(items, report_root, output_root)
    catalog_nonvisual(output_root, catalog_root)

    print(f"Copied visual artifacts: {len(items)}")
    print(f"Visual index: {report_root / 'index.html'}")
    print(f"Non-visual catalogs: {catalog_root}")


if __name__ == "__main__":
    main()
