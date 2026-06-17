"""Lazy dataset for the minimal V5 supervised experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE2_DIR = ROOT / "data" / "phase2_v5" / "samples"
DEFAULT_PHASE5_DIR = ROOT / "data" / "phase5_instances" / "samples"
DEFAULT_SPLIT_PATH = ROOT / "data" / "phase1" / "split_v1.json"
COUNT_SCALE = 8.0


def load_split_ids(split_path: Path, split: str) -> list[str]:
    with split_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    split_key = {"val": "validation"}.get(split, split)
    if split_key not in value:
        raise KeyError("Unknown split: {}".format(split))
    return [str(item) for item in value[split_key]]


def site_extent(site_mask: np.ndarray) -> tuple[int, int]:
    cells = np.argwhere(site_mask > 0)
    if cells.size == 0:
        raise ValueError("site_mask is empty")
    return int(cells[:, 0].max()) + 1, int(cells[:, 1].max()) + 1


class V5LazyDataset(Dataset):
    """Load one Phase 2 and Phase 5 pair only when requested."""

    def __init__(
        self,
        house_ids: Iterable[str],
        phase2_dir: Path = DEFAULT_PHASE2_DIR,
        phase5_dir: Path = DEFAULT_PHASE5_DIR,
    ) -> None:
        self.house_ids = list(house_ids)
        self.phase2_dir = Path(phase2_dir)
        self.phase5_dir = Path(phase5_dir)
        missing = [
            house_id
            for house_id in self.house_ids
            if not (self.phase2_dir / "{}.npz".format(house_id)).is_file()
            or not (self.phase5_dir / "{}.npz".format(house_id)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing Phase 2/5 samples: {}".format(", ".join(missing[:5]))
            )

    def __len__(self) -> int:
        return len(self.house_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        house_id = self.house_ids[index]
        with np.load(self.phase2_dir / "{}.npz".format(house_id)) as phase2:
            site_mask = phase2["site_mask"].astype(np.float32, copy=True)
            class_grid = phase2["class_grid"].astype(np.int64, copy=True)
            instance_grid = phase2["instance_grid"].astype(np.int64, copy=True)
            building_mask = phase2["building_mask"].astype(np.float32, copy=True)
            cross_floor_mask = phase2["cross_floor_mask"].astype(
                np.float32, copy=True
            )
        with np.load(self.phase5_dir / "{}.npz".format(house_id)) as phase5:
            center_heatmap = phase5["center_heatmap"].astype(
                np.float32, copy=True
            )
            center_offset = phase5["center_offset"].astype(np.float32, copy=True)
            center_valid_mask = phase5["center_valid_mask"].astype(
                np.float32, copy=True
            )
            boundary_mask = phase5["boundary_mask"].astype(
                np.float32, copy=True
            )
            floor_counts = phase5["floor_instance_counts"].astype(
                np.float32, copy=True
            )
            class_counts = phase5["class_instance_counts"].astype(
                np.float32, copy=True
            )

        width, height = site_extent(site_mask)
        canvas_size = float(site_mask.shape[0])
        condition = np.concatenate(
            (
                np.asarray([width / canvas_size, height / canvas_size], np.float32),
                class_counts.reshape(-1) / COUNT_SCALE,
            )
        )
        return {
            "house_id": house_id,
            "condition": torch.from_numpy(condition),
            "site_mask": torch.from_numpy(site_mask[None]),
            "class_grid": torch.from_numpy(class_grid),
            "instance_grid": torch.from_numpy(instance_grid),
            "building_mask": torch.from_numpy(building_mask),
            "cross_floor_mask": torch.from_numpy(cross_floor_mask),
            "center_heatmap": torch.from_numpy(center_heatmap),
            "center_offset": torch.from_numpy(center_offset),
            "center_valid_mask": torch.from_numpy(center_valid_mask),
            "boundary_mask": torch.from_numpy(boundary_mask),
            "floor_instance_counts": torch.from_numpy(floor_counts),
            "class_instance_counts": torch.from_numpy(class_counts),
        }


def make_dataset(
    split: str,
    split_path: Path = DEFAULT_SPLIT_PATH,
    phase2_dir: Path = DEFAULT_PHASE2_DIR,
    phase5_dir: Path = DEFAULT_PHASE5_DIR,
    max_samples: int | None = None,
) -> V5LazyDataset:
    house_ids = load_split_ids(Path(split_path), split)
    if max_samples is not None:
        house_ids = house_ids[:max_samples]
    return V5LazyDataset(house_ids, phase2_dir, phase5_dir)
