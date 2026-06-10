#!/usr/bin/env python3
"""独立条件生成 CLI：只需权重文件，无需 notebook。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from layout import build_user_request
from pipeline import generate_user_layout, load_model
from run_meta import build_run_meta
from visualize import save_layout_figures


ROOM_FIELDS = [
    "entryway", "living_room", "dining_room", "kitchen", "bedroom", "bathroom",
    "corridor", "stairs", "utility", "balcony", "multi_purpose",
]


def parse_room_counts(args) -> dict:
    counts = {}
    for name in ROOM_FIELDS:
        v = getattr(args, name, 0)
        if v and int(v) > 0:
            counts[name] = int(v)
    return counts


def main():
    parser = argparse.ArgumentParser(description="SpatialModal CVAE 独立条件生成")
    parser.add_argument("--weights", required=True, help="训练得到的 .pth 权重路径")
    parser.add_argument("--out-dir", default="./outputs", help="输出目录（PNG + JSON）")
    parser.add_argument("--site-x", type=float, default=18000)
    parser.add_argument("--site-y", type=float, default=15000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sample-k", type=int, default=8, help="潜变量采样次数，按质量评分选择最优结果")
    parser.add_argument(
        "--display-style",
        choices=["footprint", "regions", "boxes"],
        default="regions",
        help="regions=简洁3D体块(默认); footprint=体素碎格; boxes=旧外接盒",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    for name in ROOM_FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=0)
    parser.set_defaults(
        living_room=1, dining_room=1, kitchen=1, bedroom=3, bathroom=2,
        corridor=2, stairs=1, entryway=1, balcony=1,
    )
    args = parser.parse_args()

    counts = parse_room_counts(args)
    if not counts:
        print("错误: 至少指定一个房间数量", file=sys.stderr)
        sys.exit(1)

    weights = Path(args.weights)
    if not weights.exists():
        print(f"错误: 权重不存在 {weights}", file=sys.stderr)
        sys.exit(1)

    model, device = load_model(weights, args.device)
    req = build_user_request(args.site_x, args.site_y, counts)
    result = generate_user_layout(
        req, model, device,
        seed=args.seed, sample_k=args.sample_k, display_style=args.display_style,
    )

    out_dir = Path(args.out_dir)
    run_meta = build_run_meta(
        weights,
        sample_k=args.sample_k,
        display_style=args.display_style,
        device=str(device),
    )
    paths = save_layout_figures(
        result, req, out_dir,
        weights_path=weights,
        run_meta=run_meta,
    )

    print(f"设备: {device}")
    print(f"生成时间: {run_meta['generated_at']}")
    print(f"权重: {run_meta['weights_file']}")
    print(f"seed={result['seed']} | 解码={result['decode_mode']} | 非空体素={result['n_occ']}")
    if result.get("quality_score") is not None:
        print(f"质量分={result['quality_score']:.3f} | 细节={result.get('quality_metrics')}")
    print(f"显示来源: {result['display_source']}")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    if "meta" in paths:
        print(f"元数据: {paths['meta']}")


if __name__ == "__main__":
    main()
