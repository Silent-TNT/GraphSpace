# -*- coding: utf-8 -*-
"""
扫描 house_*.json 的用地/建筑尺寸，推荐体素栅格 RES_X/Y/Z。

用法:
  python analyze_building_size_grid.py
  python analyze_building_size_grid.py --data-dir E:/Documents/GraphSpace/data/processed
  python analyze_building_size_grid.py --voxel-size 300 --margin 0.12

输出:
  - 控制台汇总（mm 与体素维度 P50/P95/P99）
  - 当前栅格下非空占比估算
  - 推荐 RES（向上取整到 8 的倍数，便于 3D 卷积）
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from statistics import mean, median

VOXEL_SIZE_DEFAULT = 300.0
# 与 notebook 260607 一致
CURRENT_RES = (64, 128, 32)


def resolve_data_dir(arg: str | None) -> str:
    if arg:
        return os.path.abspath(arg)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "data", "processed")


def discover_json_files(data_dir: str) -> list[str]:
    return sorted(
        f
        for f in glob.glob(os.path.join(data_dir, "**", "house_*.json"), recursive=True)
        if not f.endswith("_topology.json")
    )


def bbox_from_rooms(rooms: list) -> tuple[list[float], list[float]] | None:
    if not rooms:
        return None
    mins, maxs = [], []
    for r in rooms:
        mins.append(r["box_min"])
        maxs.append(r["box_max"])
    import numpy as np

    arr_min = np.array(mins)
    arr_max = np.array(maxs)
    return arr_min.min(axis=0).tolist(), arr_max.max(axis=0).tolist()


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def round_up_to(n: int, base: int) -> int:
    return int(math.ceil(n / base) * base)


def simulate_occupancy(
    rooms: list,
    res_x: int,
    res_y: int,
    res_z: int,
    voxel_size: float,
) -> tuple[int, float]:
    """复现 notebook json_to_sample 的栅格填充，返回 (非空体素数, 占比)。"""
    import numpy as np

    bmin, bmax = bbox_from_rooms(rooms)
    if bmin is None:
        return 0, 0.0
    build_min = np.array(bmin)
    build_max = np.array(bmax)
    phys_center_xy = (build_min[:2] + build_max[:2]) / 2.0
    offset_xy = np.array([res_x * voxel_size / 2, res_y * voxel_size / 2]) - phys_center_xy
    z_min_phys = build_min[2]

    grid = np.zeros((res_x, res_y, res_z), dtype=np.int8)
    for r in rooms:
        ix_min = int((r["box_min"][0] + offset_xy[0]) / voxel_size)
        ix_max = int((r["box_max"][0] + offset_xy[0]) / voxel_size)
        iy_min = int((r["box_min"][1] + offset_xy[1]) / voxel_size)
        iy_max = int((r["box_max"][1] + offset_xy[1]) / voxel_size)
        iz_min = int((r["box_min"][2] - z_min_phys) / voxel_size)
        iz_max = int((r["box_max"][2] - z_min_phys) / voxel_size)
        grid[
            max(0, ix_min) : min(res_x, ix_max),
            max(0, iy_min) : min(res_y, iy_max),
            max(0, iz_min) : min(res_z, iz_max),
        ] = 1

    n_occ = int((grid > 0).sum())
    total = res_x * res_y * res_z
    return n_occ, n_occ / max(total, 1)


def load_sample(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    rooms = data.get("rooms", [])
    meta = data.get("metadata", {})
    bsize = meta.get("building_size", {})
    bbox = bbox_from_rooms(rooms)
    if bbox is None:
        return None
    bmin, bmax = bbox
    extent = [bmax[i] - bmin[i] for i in range(3)]
    return {
        "path": path,
        "house_id": os.path.splitext(os.path.basename(path))[0],
        "meta_size": [
            float(bsize.get("x", 0) or 0),
            float(bsize.get("y", 0) or 0),
            float(bsize.get("z", 0) or 0),
        ],
        "bbox_extent": extent,
        "rooms": rooms,
    }


def summarize_mm(values: list[float], label: str, voxel_size: float) -> None:
    s = sorted(values)
    n = len(s)
    print(f"\n  [{label}] mm  (n={n})")
    print(f"    min / med / mean / P95 / P99 / max")
    print(
        f"    {s[0]:.0f}  {percentile(s, 50):.0f}  {mean(s):.0f}  "
        f"{percentile(s, 95):.0f}  {percentile(s, 99):.0f}  {s[-1]:.0f}"
    )
    vox = [math.ceil(v / voxel_size) for v in s]
    print(
        f"    体素 ceil(size/300): min={min(vox)} med={int(median(vox))} "
        f"P95={math.ceil(percentile(s, 95)/voxel_size)} max={max(vox)}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="分析数据集建筑尺寸并推荐体素栅格")
    parser.add_argument("--data-dir", default=None, help="processed 根目录")
    parser.add_argument("--voxel-size", type=float, default=VOXEL_SIZE_DEFAULT)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.12,
        help="推荐 RES 在 P95 体素需求上的余量比例（默认 12%%）",
    )
    parser.add_argument(
        "--align",
        type=int,
        default=8,
        help="RES 向上对齐倍数（ConvTranspose 友好，默认 8）",
    )
    args = parser.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    files = discover_json_files(data_dir)
    if not files:
        print(f"未找到 JSON: {data_dir}")
        return 1

    samples = []
    for fp in files:
        s = load_sample(fp)
        if s:
            samples.append(s)

    if not samples:
        print("无有效样本（rooms 为空）")
        return 1

    vx = args.voxel_size
    ex_x = [s["bbox_extent"][0] for s in samples]
    ex_y = [s["bbox_extent"][1] for s in samples]
    ex_z = [s["bbox_extent"][2] for s in samples]
    meta_x = [s["meta_size"][0] for s in samples if s["meta_size"][0] > 0]
    meta_y = [s["meta_size"][1] for s in samples if s["meta_size"][1] > 0]
    meta_z = [s["meta_size"][2] for s in samples if s["meta_size"][2] > 0]

    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"有效样本: {len(samples)} / 文件 {len(files)}")
    print(f"体素边长: {vx} mm")
    print(f"当前 notebook 栅格: RES_X,Y,Z = {CURRENT_RES[0]}, {CURRENT_RES[1]}, {CURRENT_RES[2]}")
    cur_phys = tuple(r * vx for r in CURRENT_RES)
    print(f"  物理覆盖: {cur_phys[0]/1000:.1f}m x {cur_phys[1]/1000:.1f}m x {cur_phys[2]/1000:.1f}m")
    print(f"  总体素/样本: {CURRENT_RES[0]*CURRENT_RES[1]*CURRENT_RES[2]:,}")

    print("\n--- 实际 rooms 包围盒跨度 (box_max - box_min) ---")
    summarize_mm(ex_x, "X 跨度", vx)
    summarize_mm(ex_y, "Y 跨度", vx)
    summarize_mm(ex_z, "Z 跨度", vx)

    if meta_x:
        print("\n--- metadata.building_size（若与上表差很多，以 rooms 为准）---")
        summarize_mm(meta_x, "meta X", vx)
        summarize_mm(meta_y, "meta Y", vx)
        summarize_mm(meta_z, "meta Z", vx)

    # 居中栅格最小 RES：ceil(extent / voxel)
    need_x = [math.ceil(e / vx) for e in ex_x]
    need_y = [math.ceil(e / vx) for e in ex_y]
    need_z = [math.ceil(e / vx) for e in ex_z]

    p95_nx = math.ceil(percentile(sorted(need_x), 95))
    p95_ny = math.ceil(percentile(sorted(need_y), 95))
    p95_nz = math.ceil(percentile(sorted(need_z), 95))
    max_nx, max_ny, max_nz = max(need_x), max(need_y), max(need_z)

    m = 1.0 + args.margin
    rec_x = round_up_to(math.ceil(p95_nx * m), args.align)
    rec_y = round_up_to(math.ceil(p95_ny * m), args.align)
    rec_z = round_up_to(math.ceil(p95_nz * m), args.align)
    # Z 常对齐 4 即可
    rec_z = round_up_to(rec_z, 4)

    cover_x = sum(1 for n in need_x if n <= CURRENT_RES[0]) / len(need_x) * 100
    cover_y = sum(1 for n in need_y if n <= CURRENT_RES[1]) / len(need_y) * 100
    cover_z = sum(1 for n in need_z if n <= CURRENT_RES[2]) / len(need_z) * 100
    clip_x = sum(1 for n in need_x if n > CURRENT_RES[0])
    clip_y = sum(1 for n in need_y if n > CURRENT_RES[1])
    clip_z = sum(1 for n in need_z if n > CURRENT_RES[2])

    print("\n--- 居中栅格最小体素需求 (ceil(extent/300)) ---")
    print(f"  P95 需求: X={p95_nx}  Y={p95_ny}  Z={p95_nz}")
    print(f"  最大需求: X={max_nx}  Y={max_ny}  Z={max_nz}")
    print(f"  当前 64x128x32 覆盖率: X {cover_x:.1f}%  Y {cover_y:.1f}%  Z {cover_z:.1f}%")
    if clip_x or clip_y or clip_z:
        print(f"  [WARN] 超出当前栅格（会被裁切）: X={clip_x}  Y={clip_y}  Z={clip_z} 套")

    # 非空占比
    occ_cur = []
    occ_rec = []
    for s in samples:
        n1, r1 = simulate_occupancy(s["rooms"], *CURRENT_RES, vx)
        n2, r2 = simulate_occupancy(s["rooms"], rec_x, rec_y, rec_z, vx)
        occ_cur.append(r1)
        occ_rec.append(r2)

    occ_cur_s = sorted(occ_cur)
    occ_rec_s = sorted(occ_rec)
    print("\n--- 非空体素占比（按 json_to_sample 规则模拟）---")
    print(
        f"  当前 {CURRENT_RES[0]}x{CURRENT_RES[1]}x{CURRENT_RES[2]}: "
        f"med={percentile(occ_cur_s,50)*100:.2f}%  mean={mean(occ_cur)*100:.2f}%  "
        f"P95={percentile(occ_cur_s,95)*100:.2f}%"
    )
    print(
        f"  推荐 {rec_x}x{rec_y}x{rec_z}: "
        f"med={percentile(occ_rec_s,50)*100:.2f}%  mean={mean(occ_rec)*100:.2f}%  "
        f"P95={percentile(occ_rec_s,95)*100:.2f}%"
    )

    cur_total = CURRENT_RES[0] * CURRENT_RES[1] * CURRENT_RES[2]
    rec_total = rec_x * rec_y * rec_z
    print("\n--- 推荐栅格（P95 + {:.0f}% 余量，对齐 {}）---".format(args.margin * 100, args.align))
    print(f"  RES_X, RES_Y, RES_Z = {rec_x}, {rec_y}, {rec_z}")
    print(
        f"  物理覆盖: {rec_x*vx/1000:.1f}m x {rec_y*vx/1000:.1f}m x {rec_z*vx/1000:.1f}m"
    )
    print(f"  总体素/样本: {rec_total:,}  （当前的 {rec_total/cur_total*100:.1f}%）")
    print(f"  预估 one-hot 体积: {rec_total * 12 * 4 / 1024 / 1024:.2f} MB / 样本")

    print("\n--- notebook 修改参考 ---")
    print(f"  RES_X, RES_Y, RES_Z = {rec_x}, {rec_y}, {rec_z}")
    print("  修改后需: bump TENSOR_CACHE_VERSION → 重跑 Step 2 → 重训")

    # 写出简要 JSON 报告
    report_dir = os.path.join(os.path.dirname(__file__))
    report_path = os.path.join(report_dir, "grid_size_report.json")
    report = {
        "data_dir": data_dir,
        "sample_count": len(samples),
        "voxel_size_mm": vx,
        "current_res": list(CURRENT_RES),
        "recommended_res": [rec_x, rec_y, rec_z],
        "p95_extent_mm": {
            "x": percentile(sorted(ex_x), 95),
            "y": percentile(sorted(ex_y), 95),
            "z": percentile(sorted(ex_z), 95),
        },
        "p95_voxel_need": {"x": p95_nx, "y": p95_ny, "z": p95_nz},
        "occupancy_current_mean": mean(occ_cur),
        "occupancy_recommended_mean": mean(occ_rec),
        "clipped_current": {"x": clip_x, "y": clip_y, "z": clip_z},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n已写入: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
