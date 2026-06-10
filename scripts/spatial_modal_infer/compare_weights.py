#!/usr/bin/env python3
"""对比两份 V3 权重在固定探针与用户条件下的生成表现。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from config import CHANNEL_MAP, NUM_CHANNELS, RES_X, RES_Y, RES_Z
from layout import build_user_request
from pipeline import generate_user_layout, load_model
from run_meta import build_run_meta, file_tag, now_iso, now_stamp

SYNTHETIC_PROBE_SPECS = [
    {
        "name": "compact_2f",
        "site_x": 15000,
        "site_y": 12000,
        "seed": 11,
        "room_counts": {
            "entryway": 1,
            "living_room": 1,
            "dining_room": 1,
            "kitchen": 1,
            "bedroom": 2,
            "bathroom": 1,
            "corridor": 1,
            "stairs": 1,
        },
    },
    {
        "name": "standard_3br",
        "site_x": 18000,
        "site_y": 15000,
        "seed": 22,
        "room_counts": {
            "entryway": 1,
            "living_room": 1,
            "dining_room": 1,
            "kitchen": 1,
            "bedroom": 3,
            "bathroom": 2,
            "corridor": 2,
            "stairs": 1,
            "balcony": 1,
        },
    },
    {
        "name": "large_4br",
        "site_x": 21000,
        "site_y": 18000,
        "seed": 33,
        "room_counts": {
            "entryway": 1,
            "living_room": 1,
            "dining_room": 1,
            "kitchen": 1,
            "bedroom": 4,
            "bathroom": 2,
            "corridor": 2,
            "stairs": 1,
            "balcony": 1,
            "utility": 1,
        },
    },
]

TYPE_NAMES = {v: k for k, v in CHANNEL_MAP.items() if v > 0}


def check_load(weights_path: Path, device: str | None) -> dict:
    model, dev = load_model(weights_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    state_keys = len(model.state_dict())
    return {
        "path": str(weights_path.resolve()),
        "device": str(dev),
        "params": n_params,
        "state_dict_keys": state_keys,
        "ok": True,
    }


def channel_stats(pred: np.ndarray) -> dict:
    total = int(pred.size)
    n_occ = int((pred > 0).sum())
    by_type = {}
    for cid in range(1, NUM_CHANNELS):
        n = int((pred == cid).sum())
        if n > 0:
            by_type[TYPE_NAMES[cid]] = n
    return {
        "n_occ": n_occ,
        "occ_ratio": round(n_occ / max(total, 1), 6),
        "by_type": by_type,
    }


def eval_one(weights_path: Path, user_req: dict, seed: int, sample_k: int, device: str | None) -> dict:
    model, dev = load_model(weights_path, device)
    result = generate_user_layout(
        user_req, model, dev, seed=seed, sample_k=sample_k, display_style="footprint"
    )
    stats = channel_stats(result["pred"])
    return {
        "weights": str(weights_path.resolve()),
        "seed": result["seed"],
        "decode_mode": result["decode_mode"],
        "display_source": result["display_source"],
        "quality_score": result.get("quality_score"),
        "quality_metrics": result.get("quality_metrics"),
        **stats,
    }


def run_probe_suite(weights_path: Path, sample_k: int, device: str | None) -> list[dict]:
    rows = []
    for spec in SYNTHETIC_PROBE_SPECS:
        req = build_user_request(spec["site_x"], spec["site_y"], spec["room_counts"])
        row = eval_one(weights_path, req, spec["seed"], sample_k, device)
        row["probe"] = spec["name"]
        rows.append(row)
    return rows


def summarize_probe_rows(rows: list[dict]) -> dict:
    n_occ = [r["n_occ"] for r in rows]
    scores = [r["quality_score"] for r in rows if r.get("quality_score") is not None]
    return {
        "mean_n_occ": float(np.mean(n_occ)) if n_occ else 0.0,
        "mean_quality_score": float(np.mean(scores)) if scores else None,
        "nonempty_hits": sum(1 for x in n_occ if x > 0),
        "total": len(rows),
    }


def print_load_report(label: str, info: dict):
    print(f"[{label}] OK")
    print(f"  路径: {info['path']}")
    print(f"  设备: {info['device']}")
    print(f"  参数量: {info['params']:,}")
    print(f"  state_dict keys: {info['state_dict_keys']}")


def print_probe_table(label: str, rows: list[dict], summary: dict):
    print(f"\n=== {label} 探针套件 ===")
    for r in rows:
        print(
            f"  [{r['probe']}] n_occ={r['n_occ']:5d} "
            f"({r['occ_ratio']*100:.3f}%) score={r.get('quality_score')} mode={r['decode_mode']} "
            f"source={r['display_source']}"
        )
    print(
        f"  汇总: mean_n_occ={summary['mean_n_occ']:.1f} "
        f"nonempty={summary['nonempty_hits']}/{summary['total']}"
    )


def main():
    parser = argparse.ArgumentParser(description="V3 权重加载检查与 probe/val 对比")
    parser.add_argument(
        "--probe",
        default="../../weights/spatial_modal_cvae_v3_88x88x24_probe_best.pth",
        help="探针最优权重",
    )
    parser.add_argument(
        "--val",
        default="../../weights/spatial_modal_cvae_v3_88x88x24.pth",
        help="val 最优权重（可选，不存在则跳过）",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-k", type=int, default=8)
    parser.add_argument("--out", default=None, help="JSON 报告输出路径")
    parser.add_argument("--check-only", action="store_true", help="仅检查权重能否 load")
    args = parser.parse_args()

    probe_path = Path(args.probe).resolve()
    val_path = Path(args.val).resolve()

    if not probe_path.exists():
        print(f"错误: probe 权重不存在 {probe_path}", file=sys.stderr)
        sys.exit(1)

    stamp = now_stamp()
    report = {
        "generated_at": now_iso(),
        "generated_stamp": stamp,
        "probe_weights": str(probe_path),
        "probe_weights_file": probe_path.name,
        "val_weights": str(val_path) if val_path.exists() else None,
        "val_weights_file": val_path.name if val_path.exists() else None,
        "grid": f"{RES_X}x{RES_Y}x{RES_Z}",
        "loads": {},
        "probe_suite": {},
        "comparison": {},
    }

    probe_load = check_load(probe_path, args.device)
    print_load_report("probe", probe_load)
    report["loads"]["probe"] = probe_load

    if val_path.exists():
        val_load = check_load(val_path, args.device)
        print_load_report("val", val_load)
        report["loads"]["val"] = val_load
    else:
        print(f"\n[val] 跳过（文件不存在）: {val_path}")

    if args.check_only:
        out_path = Path(args.out) if args.out else Path(
            f"../../weights/eval_reports/load_check_{file_tag(probe_path, stamp)}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n报告: {out_path.resolve()}")
        return

    probe_rows = run_probe_suite(probe_path, args.sample_k, args.device)
    probe_summary = summarize_probe_rows(probe_rows)
    print_probe_table("probe", probe_rows, probe_summary)
    report["probe_suite"]["probe"] = {"rows": probe_rows, "summary": probe_summary}

    if val_path.exists():
        val_rows = run_probe_suite(val_path, args.sample_k, args.device)
        val_summary = summarize_probe_rows(val_rows)
        print_probe_table("val", val_rows, val_summary)
        report["probe_suite"]["val"] = {"rows": val_rows, "summary": val_summary}

        print("\n=== probe vs val 差值 (probe - val) ===")
        for pr, vr in zip(probe_rows, val_rows):
            delta = pr["n_occ"] - vr["n_occ"]
            sign = "+" if delta >= 0 else ""
            print(f"  [{pr['probe']}] Δn_occ={sign}{delta}")
        report["comparison"] = {
            "mean_n_occ_delta": probe_summary["mean_n_occ"] - val_summary["mean_n_occ"],
            "nonempty_hits_delta": probe_summary["nonempty_hits"] - val_summary["nonempty_hits"],
        }
        print(
            f"  mean_n_occ Δ={report['comparison']['mean_n_occ_delta']:+.1f} | "
            f"nonempty Δ={report['comparison']['nonempty_hits_delta']:+d}"
        )

    if args.out:
        out_path = Path(args.out)
    else:
        tag = file_tag(probe_path, stamp)
        if val_path.exists():
            tag = f"{stamp}_probe_vs_{probe_path.stem}_and_{val_path.stem}"
        out_path = Path(f"../../weights/eval_reports/compare_{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告: {out_path.resolve()}")


if __name__ == "__main__":
    main()
