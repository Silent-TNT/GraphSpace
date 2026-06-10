#!/usr/bin/env python3
"""同一用户条件下批量扫 seed，输出统计与可选 PNG。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from layout import build_user_request
from pipeline import generate_user_layout, load_model
from run_meta import build_run_meta, file_tag
from visualize import save_layout_figures

ROOM_FIELDS = [
    "entryway", "living_room", "dining_room", "kitchen", "bedroom", "bathroom",
    "corridor", "stairs", "utility", "balcony", "multi_purpose",
]


def parse_room_counts(args) -> dict:
    counts = {}
    for name in ROOM_FIELDS:
        v = getattr(args, name.replace("-", "_"), 0)
        if v and int(v) > 0:
            counts[name] = int(v)
    return counts


def main():
    parser = argparse.ArgumentParser(description="批量 seed 生成与统计")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out-dir", default="../../weights/outputs_batch")
    parser.add_argument("--site-x", type=float, default=18000)
    parser.add_argument("--site-y", type=float, default=15000)
    parser.add_argument("--seeds", default="0,42,123,456,789,1024", help="逗号分隔 seed 列表")
    parser.add_argument("--sample-k", type=int, default=8)
    parser.add_argument(
        "--display-style",
        choices=["footprint", "regions", "boxes"],
        default="footprint",
    )
    parser.add_argument("--save-images", action="store_true", help="为每个 seed 保存 PNG")
    parser.add_argument("--device", default=None)
    for name in ROOM_FIELDS:
        attr = name
        parser.add_argument(f"--{name.replace('_', '-')}", dest=attr, type=int, default=0)
    parser.set_defaults(
        living_room=1, dining_room=1, kitchen=1, bedroom=3, bathroom=2,
        corridor=2, stairs=1, entryway=1, balcony=1,
    )
    args = parser.parse_args()

    counts = parse_room_counts(args)
    if not counts:
        print("错误: 至少指定一个房间数量", file=sys.stderr)
        sys.exit(1)

    weights = Path(args.weights).resolve()
    if not weights.exists():
        print(f"错误: 权重不存在 {weights}", file=sys.stderr)
        sys.exit(1)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_model(weights, args.device)
    req = build_user_request(args.site_x, args.site_y, counts)
    run_meta = build_run_meta(
        weights,
        sample_k=args.sample_k,
        display_style=args.display_style,
    )

    rows = []
    best = None
    for seed in seeds:
        result = generate_user_layout(
            req, model, device,
            seed=seed, sample_k=args.sample_k, display_style=args.display_style,
        )
        row = {
            "seed": result["seed"],
            "n_occ": result["n_occ"],
            "quality_score": result.get("quality_score"),
            "quality_metrics": result.get("quality_metrics"),
            "decode_mode": result["decode_mode"],
            "display_source": result["display_source"],
        }
        rows.append(row)
        row_score = row["quality_score"] if row["quality_score"] is not None else -1e18
        best_score = best.get("quality_score") if best else None
        best_score = best_score if best_score is not None else -1e18
        if best is None or row_score > best_score or (row_score == best_score and result["n_occ"] > best["n_occ"]):
            best = {**row, "result": result}

        tag = "OK" if result["n_occ"] > 0 else "EMPTY"
        print(
            f"seed={seed:5d} | n_occ={result['n_occ']:5d} | "
            f"score={row_score:8.2f} | mode={result['decode_mode']:16s} | {tag}"
        )

        if args.save_images and result["n_occ"] > 0:
            save_layout_figures(
                result, req, out_dir / f"seed_{seed}",
                weights_path=weights, run_meta=run_meta,
            )

    n_occ_list = [r["n_occ"] for r in rows]
    score_list = [r["quality_score"] for r in rows if r.get("quality_score") is not None]
    summary = {
        **run_meta,
        "device": str(device),
        "site": [args.site_x, args.site_y],
        "room_counts": counts,
        "seeds": seeds,
        "rows": rows,
        "stats": {
            "mean_n_occ": float(np.mean(n_occ_list)),
            "max_n_occ": int(max(n_occ_list)) if n_occ_list else 0,
            "mean_quality_score": float(np.mean(score_list)) if score_list else None,
            "max_quality_score": float(max(score_list)) if score_list else None,
            "nonempty_hits": sum(1 for x in n_occ_list if x > 0),
            "total": len(rows),
        },
        "best_seed": best["seed"] if best else None,
    }

    if best and best["n_occ"] > 0 and not args.save_images:
        best_paths = save_layout_figures(
            best["result"], req, out_dir / "best",
            weights_path=weights, run_meta=run_meta,
        )
        summary["best_outputs"] = {k: str(v) for k, v in best_paths.items()}

    report_path = out_dir / f"batch_report_{file_tag(weights, run_meta['generated_stamp'])}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n设备: {device}")
    print(f"生成时间: {run_meta['generated_at']}")
    print(f"权重: {run_meta['weights_file']}")
    print(
        f"汇总: mean_n_occ={summary['stats']['mean_n_occ']:.1f} | "
        f"nonempty={summary['stats']['nonempty_hits']}/{summary['stats']['total']} | "
        f"best_seed={summary['best_seed']} (score={summary['stats']['max_quality_score']})"
    )
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
