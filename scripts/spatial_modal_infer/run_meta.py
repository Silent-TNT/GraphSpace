"""生成时间与权重信息，用于输出文件命名与 JSON 元数据。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def weights_info(weights_path: str | Path) -> dict:
    p = Path(weights_path).resolve()
    return {
        "weights": str(p),
        "weights_file": p.name,
        "weights_stem": p.stem,
    }


def build_run_meta(weights_path: str | Path, **extra) -> dict:
    return {
        "generated_at": now_iso(),
        "generated_stamp": now_stamp(),
        **weights_info(weights_path),
        **extra,
    }


def file_tag(weights_path: str | Path, stamp: str | None = None) -> str:
    stamp = stamp or now_stamp()
    return f"{stamp}_{Path(weights_path).stem}"
