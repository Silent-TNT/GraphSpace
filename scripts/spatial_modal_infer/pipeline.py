from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

try:
    from .config import CHANNEL_MAP, LATENT_DIM, NUM_CHANNELS, RES_X, RES_Y, RES_Z, VOXEL_SIZE
    from .graph_utils import forward_model, json_to_sample, prepare_graph_batch
    from .layout import (
        layout_rooms_from_program,
        program_floor_room_targets,
        request_to_house_json,
        snap_modulus,
    )
    from .model import SpatialModalCVAE
except ImportError:
    from config import CHANNEL_MAP, LATENT_DIM, NUM_CHANNELS, RES_X, RES_Y, RES_Z, VOXEL_SIZE
    from graph_utils import forward_model, json_to_sample, prepare_graph_batch
    from layout import (
        layout_rooms_from_program,
        program_floor_room_targets,
        request_to_house_json,
        snap_modulus,
    )
    from model import SpatialModalCVAE


def load_model(weights_path: str | Path, device: str | None = None) -> tuple[SpatialModalCVAE, torch.device]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SpatialModalCVAE().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


def _set_seed(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def infer_voxels(user_req, rooms, model, device, seed: int = 42, sample_k: int = 1, program_graph=None, program_edge_types=None):
    """
    以 seed 控制潜变量采样，使不同种子/功能数量产生不同体素布局。
    sample_k>1 时多次采样取非空体素最多的一次。
    """
    data = request_to_house_json(user_req, rooms)
    sample = json_to_sample(data, program_graph, program_edge_types)
    hg = sample["graph"]
    cond = sample["condition"]
    batch = prepare_graph_batch(hg, condition=cond).to(device)
    cond_dev = cond.unsqueeze(0).to(device) if cond.dim() == 1 else cond.to(device)
    model.eval()

    logits, mu, logvar = forward_model(model, batch, cond)
    std = torch.exp(0.5 * logvar)

    best_pred, best_n, best_mode = None, -1, "latent_sample"
    best_score = -1e18
    best_metrics = {}
    candidates = []

    k = max(1, int(sample_k))
    for i in range(k):
        _set_seed(seed + i * 9973, device)
        eps = torch.randn_like(std)
        z = mu + eps * std
        logits_z = model.decoder(z, cond_dev)
        pred = torch.argmax(logits_z[0], dim=0).cpu().numpy()
        n_occ = int((pred > 0).sum())
        score, metrics = _score_candidate(pred, rooms, user_req, seed=seed)
        candidates.append((f"latent_sample_{i}", pred, n_occ, score, metrics))
        if score > best_score or (score == best_score and n_occ > best_n):
            best_pred, best_n, best_mode = pred, n_occ, f"latent_sample_{i}"
            best_score, best_metrics = score, metrics

    if best_n <= 0:
        _set_seed(seed, device)
        pred_mu = torch.argmax(model.decoder(mu, cond_dev)[0], dim=0).cpu().numpy()
        n_mu = int((pred_mu > 0).sum())
        score, metrics = _score_candidate(pred_mu, rooms, user_req, seed=seed)
        candidates.append(("encoder_mu", pred_mu, n_mu, score, metrics))
        if score > best_score or (score == best_score and n_mu > best_n):
            best_pred, best_n, best_mode = pred_mu, n_mu, "encoder_mu"
            best_score, best_metrics = score, metrics

    if best_n <= 0:
        _set_seed(seed + 4049, device)
        z_rand = torch.randn(1, LATENT_DIM, device=device)
        pred_rand = torch.argmax(model.decoder(z_rand, cond_dev)[0], dim=0).cpu().numpy()
        n_rand = int((pred_rand > 0).sum())
        score, metrics = _score_candidate(pred_rand, rooms, user_req, seed=seed)
        candidates.append(("random_z_cond", pred_rand, n_rand, score, metrics))
        if score > best_score or (score == best_score and n_rand > best_n):
            best_pred, best_n, best_mode = pred_rand, n_rand, "random_z_cond"
            best_score, best_metrics = score, metrics

    if best_n <= 0:
        pred_graph = torch.argmax(logits[0], dim=0).cpu().numpy()
        best_pred = pred_graph
        best_n = int((pred_graph > 0).sum())
        best_mode = "graph_forward"
        best_score, best_metrics = _score_candidate(pred_graph, rooms, user_req, seed=seed)
        candidates.append((best_mode, pred_graph, best_n, best_score, best_metrics))

    return best_pred, best_n, best_mode, candidates, best_score, best_metrics


def filter_requested_room_types(display_rooms, user_req):
    allowed = set((user_req or {}).get("room_counts", {}).keys())
    if not allowed:
        return display_rooms
    return [r for r in display_rooms if r["type"] in allowed]


def count_rooms_by_type(rooms) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rooms:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    return counts


FLOOR_SLABS = {1: (0.0, 3000.0), 2: (3000.0, 6000.0)}


ROOM_AREA_LIMITS = {
    "entryway": (3.0, 18.0),
    "living_room": (16.0, 60.0),
    "dining_room": (8.0, 32.0),
    "kitchen": (6.0, 24.0),
    "bedroom": (8.0, 28.0),
    "bathroom": (3.0, 14.0),
    "corridor": (3.0, 22.0),
    "stairs": (6.0, 22.0),
    "utility": (3.0, 14.0),
    "balcony": (3.0, 18.0),
    "multi_purpose": (8.0, 36.0),
}


def _score_candidate(pred_cls: np.ndarray, rooms, user_req, seed: int = 42) -> tuple[float, dict]:
    """Score generated voxels by architectural plausibility, not just fill volume."""
    if pred_cls is None:
        return -1e9, {"reason": "empty"}

    voxel_area_m2 = (VOXEL_SIZE / 1000.0) ** 2
    room_counts = (user_req or {}).get("room_counts", {})
    targets = program_floor_room_targets(room_counts, seed=seed)
    layers = voxels_to_floor_layers(pred_cls, rooms)
    requested = set((user_req or {}).get("room_counts", {}).keys())
    type_names = {v: k for k, v in CHANNEL_MAP.items() if v > 0}

    score = 0.0
    penalties = {
        "missing": 0,
        "extra_area": 0.0,
        "area": 0.0,
        "fragment": 0.0,
        "shape": 0.0,
    }
    present_targets = 0
    total_targets = sum(sum(v.values()) for v in targets.values())

    for floor, floor_targets in targets.items():
        grid = layers[floor]["grid"]
        for rtype, target_count in floor_targets.items():
            cid = CHANNEL_MAP.get(rtype)
            if not cid:
                continue
            comps = [c for c in _find_2d_components(grid, cid) if c["area"] >= 2]
            total_cells = sum(c["area"] for c in comps)
            if total_cells <= 0:
                penalties["missing"] += target_count
                score -= 45.0 * target_count
                continue
            present_targets += min(len(comps), target_count)

            area_m2 = total_cells * voxel_area_m2
            min_area, max_area = ROOM_AREA_LIMITS.get(rtype, (4.0, 40.0))
            min_total = min_area * target_count
            max_total = max_area * target_count
            if area_m2 < min_total:
                p = (min_total - area_m2) / max(min_total, 1.0)
                penalties["area"] += p
                score -= 22.0 * p
            elif area_m2 > max_total:
                p = (area_m2 - max_total) / max(max_total, 1.0)
                penalties["area"] += p
                score -= 30.0 * p
            else:
                score += 8.0

            if len(comps) > max(target_count * 2, target_count + 2):
                p = len(comps) - max(target_count * 2, target_count + 2)
                penalties["fragment"] += p
                score -= 4.0 * p

            for comp in comps:
                arr = np.array(comp["cells"])
                span_x = int(arr[:, 0].max() - arr[:, 0].min() + 1)
                span_y = int(arr[:, 1].max() - arr[:, 1].min() + 1)
                bbox_area = max(1, span_x * span_y)
                fill = comp["area"] / bbox_area
                aspect = max(span_x / max(span_y, 1), span_y / max(span_x, 1))
                if fill < 0.48:
                    p = 0.48 - fill
                    penalties["shape"] += p
                    score -= 18.0 * p
                if rtype not in {"corridor", "balcony"} and aspect > 4.0:
                    p = aspect - 4.0
                    penalties["shape"] += p
                    score -= 3.0 * p

        for cid in range(1, NUM_CHANNELS):
            rtype = type_names[cid]
            if requested and rtype not in requested:
                extra_cells = int((grid == cid).sum())
                if extra_cells:
                    p = extra_cells * voxel_area_m2
                    penalties["extra_area"] += p
                    score -= min(25.0, p * 1.5)

    if total_targets:
        score += 35.0 * (present_targets / total_targets)

    non_empty = int((pred_cls > 0).sum())
    if non_empty <= 0:
        score -= 1e6
    elif non_empty < 250:
        score -= 80.0

    metrics = {
        "score": round(float(score), 3),
        "present_targets": int(present_targets),
        "total_targets": int(total_targets),
        "non_empty": non_empty,
        **{k: round(float(v), 3) for k, v in penalties.items()},
    }
    return float(score), metrics


def voxel_grid_phys_offset(rooms):
    if not rooms:
        return np.array([0.0, 0.0]), 0.0
    all_coords = np.array([r["box_min"] for r in rooms] + [r["box_max"] for r in rooms])
    build_min = all_coords.min(axis=0)
    build_max = all_coords.max(axis=0)
    phys_center_xy = (build_min[:2] + build_max[:2]) / 2.0
    offset_xy = np.array([RES_X * VOXEL_SIZE / 2, RES_Y * VOXEL_SIZE / 2]) - phys_center_xy
    return offset_xy, float(build_min[2])


def _iz_range_for_floor(z_min_phys, z_lo, z_hi):
    iz0 = int(np.floor((z_lo - z_min_phys) / VOXEL_SIZE + 1e-6))
    iz1 = int(np.ceil((z_hi - z_min_phys) / VOXEL_SIZE))
    return max(0, iz0), min(RES_Z, iz1)


def _majority_class_2d(slab: np.ndarray) -> np.ndarray:
    """每层竖向切片：每格取出现最多的房间类型（忽略 empty）。"""
    grid = np.zeros((slab.shape[0], slab.shape[1]), dtype=np.int16)
    for ix in range(slab.shape[0]):
        for iy in range(slab.shape[1]):
            counts = np.bincount(slab[ix, iy].ravel(), minlength=NUM_CHANNELS)
            counts[0] = 0
            if counts.max() > 0:
                grid[ix, iy] = int(counts.argmax())
    return grid


def voxels_to_floor_layers(pred_cls, rooms=None):
    """
    按楼层把 3D 体素压成 2D 占用图（一格一色，与训练数据俯视一致）。
    返回每层 grid、物理偏移与层高，供平面图 imshow / 3D 拉伸体块使用。
    """
    type_names = {v: k for k, v in CHANNEL_MAP.items() if v > 0}
    offset_xy, z_min_phys = voxel_grid_phys_offset(rooms) if rooms else (np.array([0.0, 0.0]), 0.0)
    layers = {}
    for floor, (z_lo, z_hi) in FLOOR_SLABS.items():
        iz0, iz1 = _iz_range_for_floor(z_min_phys, z_lo, z_hi)
        grid = np.zeros((RES_X, RES_Y), dtype=np.int16)
        if iz1 > iz0:
            grid = _majority_class_2d(pred_cls[:, :, iz0:iz1])
        layers[floor] = {
            "grid": grid,
            "offset_xy": offset_xy,
            "z_lo": z_lo,
            "z_hi": z_hi,
            "type_names": type_names,
        }
    return layers


def footprint_layers_to_rooms(layers):
    """把每层占用图拉成整层高的体块：一格一地砖，相邻格不重叠。"""
    rooms = []
    for floor, layer in layers.items():
        grid = layer["grid"]
        offset_xy = layer["offset_xy"]
        ox, oy = float(offset_xy[0]), float(offset_xy[1])
        z0, z1 = layer["z_lo"], layer["z_hi"]
        tnames = layer["type_names"]
        for ix in range(grid.shape[0]):
            for iy in range(grid.shape[1]):
                cid = int(grid[ix, iy])
                if cid <= 0:
                    continue
                rooms.append(
                    {
                        "type": tnames[cid],
                        "floor": floor,
                        "footprint": True,
                        "box_min": [
                            snap_modulus(ix * VOXEL_SIZE - ox),
                            snap_modulus(iy * VOXEL_SIZE - oy),
                            z0,
                        ],
                        "box_max": [
                            snap_modulus((ix + 1) * VOXEL_SIZE - ox),
                            snap_modulus((iy + 1) * VOXEL_SIZE - oy),
                            z1,
                        ],
                    }
                )
    return rooms


def _find_2d_components(grid: np.ndarray, cid: int) -> list[dict]:
    mask = grid == cid
    if not mask.any():
        return []
    visited = np.zeros(mask.shape, dtype=bool)
    comps = []
    xs, ys = np.where(mask)
    for x, y in zip(xs, ys):
        if visited[x, y]:
            continue
        stack = [(int(x), int(y))]
        cells = []
        visited[x, y] = True
        while stack:
            cx, cy = stack.pop()
            cells.append((cx, cy))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                    if mask[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        stack.append((nx, ny))
        arr = np.array(cells)
        comps.append({"cells": cells, "area": len(cells), "centroid": arr.mean(axis=0)})
    return comps


def _merge_components_to_target(components: list[dict], target: int) -> list[dict]:
    if target <= 0 or len(components) <= target:
        return components
    comps = [dict(c, cells=list(c["cells"])) for c in components]
    while len(comps) > target:
        comps.sort(key=lambda c: c["area"])
        small = comps.pop(0)
        sc = small["centroid"]
        best_i = min(
            range(len(comps)),
            key=lambda i: float(np.linalg.norm(comps[i]["centroid"] - sc)),
        )
        comps[best_i]["cells"].extend(small["cells"])
        arr = np.array(comps[best_i]["cells"])
        comps[best_i]["area"] = len(arr)
        comps[best_i]["centroid"] = arr.mean(axis=0)
    return comps


def _cells_to_rectangles(cells: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Greedy row-run rectangles: preserves stepped footprints better than one bbox."""
    if not cells:
        return []
    by_x: dict[int, list[int]] = {}
    for x, y in cells:
        by_x.setdefault(int(x), []).append(int(y))

    row_runs_by_x: dict[int, list[tuple[int, int]]] = {}
    for x in sorted(by_x):
        ys = sorted(set(by_x[x]))
        if not ys:
            continue
        start = prev = ys[0]
        for y in ys[1:]:
            if y == prev + 1:
                prev = y
            else:
                row_runs_by_x.setdefault(x, []).append((start, prev + 1))
                start = prev = y
        row_runs_by_x.setdefault(x, []).append((start, prev + 1))

    rectangles: list[list[int]] = []
    active: dict[tuple[int, int], list[int]] = {}
    expected_x = None
    for x in sorted(row_runs_by_x):
        if expected_x is not None and x != expected_x:
            rectangles.extend(active.values())
            active = {}
        next_active: dict[tuple[int, int], list[int]] = {}
        for y0, y1 in row_runs_by_x[x]:
            key = (y0, y1)
            if key in active and active[key][1] == x:
                rect = active[key]
                rect[1] = x + 1
                next_active[key] = rect
            else:
                next_active[key] = [x, x + 1, y0, y1]
        for key, rect in active.items():
            if key not in next_active:
                rectangles.append(rect)
        active = next_active
        expected_x = x + 1
    rectangles.extend(active.values())
    return [(x0, y0, x1, y1) for x0, x1, y0, y1 in rectangles]


def voxels_to_region_rooms(pred_cls, rooms=None, user_req=None, min_cells: int = 2, preserve_footprint: bool = True, seed: int = 42):
    """
    每层 2D 连通域 → 按用户需求合并碎片 → 每域一个紧贴外框并拉满层高。
    比旧版 3D 外接盒更少重叠、房间数更接近清单。
    """
    type_names = {v: k for k, v in CHANNEL_MAP.items() if v > 0}
    floor_targets = program_floor_room_targets((user_req or {}).get("room_counts", {}), seed=seed)
    layers = voxels_to_floor_layers(pred_cls, rooms)
    out = []
    for floor, layer in layers.items():
        grid = layer["grid"]
        offset_xy = layer["offset_xy"]
        ox, oy = float(offset_xy[0]), float(offset_xy[1])
        z0, z1 = layer["z_lo"], layer["z_hi"]
        for cid in range(1, NUM_CHANNELS):
            rtype = type_names[cid]
            comps = _find_2d_components(grid, cid)
            comps = [c for c in comps if c["area"] >= min_cells]
            if not comps:
                continue
            target = int(floor_targets.get(floor, {}).get(rtype, 0))
            if target > 0:
                comps = _merge_components_to_target(comps, target)
            elif rtype not in floor_targets.get(floor, {}):
                continue
            for comp in comps:
                if preserve_footprint:
                    rects = _cells_to_rectangles(comp["cells"])
                else:
                    arr = np.array(comp["cells"])
                    ix0, iy0 = arr.min(axis=0)
                    ix1, iy1 = arr.max(axis=0) + 1
                    rects = [(int(ix0), int(iy0), int(ix1), int(iy1))]
                for ix0, iy0, ix1, iy1 in rects:
                    out.append(
                        {
                            "type": rtype,
                            "floor": floor,
                            "footprint": bool(preserve_footprint),
                            "box_min": [
                                snap_modulus(ix0 * VOXEL_SIZE - ox),
                                snap_modulus(iy0 * VOXEL_SIZE - oy),
                                z0,
                            ],
                            "box_max": [
                                snap_modulus(ix1 * VOXEL_SIZE - ox),
                                snap_modulus(iy1 * VOXEL_SIZE - oy),
                                z1,
                            ],
                        }
                    )
    return out


def voxels_to_boxes(pred_cls, rooms=None):
    type_names = {v: k for k, v in CHANNEL_MAP.items() if v > 0}
    offset_xy, z_min_phys = voxel_grid_phys_offset(rooms) if rooms else (np.array([0.0, 0.0]), 0.0)
    boxes = []
    for cid in range(1, NUM_CHANNELS):
        mask = pred_cls == cid
        if not mask.any():
            continue
        visited = np.zeros(mask.shape, dtype=bool)
        xs, ys, zs = np.where(mask)
        for x, y, z in zip(xs, ys, zs):
            if visited[x, y, z]:
                continue
            stack = [(int(x), int(y), int(z))]
            comp = []
            visited[x, y, z] = True
            while stack:
                cx, cy, cz = stack.pop()
                comp.append((cx, cy, cz))
                for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nz < mask.shape[2]:
                        if mask[nx, ny, nz] and not visited[nx, ny, nz]:
                            visited[nx, ny, nz] = True
                            stack.append((nx, ny, nz))
            arr = np.array(comp)
            ix0, iy0, iz0 = arr.min(axis=0)
            ix1, iy1, iz1 = arr.max(axis=0) + 1
            rtype = type_names[cid]
            boxes.append(
                {
                    "type": rtype,
                    "box_min": [
                        snap_modulus(ix0 * VOXEL_SIZE - offset_xy[0]),
                        snap_modulus(iy0 * VOXEL_SIZE - offset_xy[1]),
                        snap_modulus(iz0 * VOXEL_SIZE + z_min_phys),
                    ],
                    "box_max": [
                        snap_modulus(ix1 * VOXEL_SIZE - offset_xy[0]),
                        snap_modulus(iy1 * VOXEL_SIZE - offset_xy[1]),
                        snap_modulus(iz1 * VOXEL_SIZE + z_min_phys),
                    ],
                }
            )
    return boxes


@torch.no_grad()
def generate_user_layout(
    user_req,
    model,
    device,
    seed=None,
    sample_k: int = 8,
    display_style: str = "regions",
):
    """
    display_style:
      - regions:   每层合并连通域，简洁 3D 体块（默认，推荐网页端）
      - footprint: 按体素真实占位（平面图最准，3D 会碎成大量小格）
      - boxes:     旧版 3D 连通域外接盒（易出现虚假重叠）
    """
    if seed is None:
        seed = random.randint(1, 999999)
    seed = int(seed)
    rooms, G, pos, edge_types = layout_rooms_from_program(user_req, seed=seed)
    pred, n_occ, mode, candidates, quality_score, quality_metrics = infer_voxels(
        user_req, rooms, model, device, seed=seed, sample_k=sample_k,
        program_graph=G, program_edge_types=edge_types,
    )
    floor_layers = None
    if n_occ > 0:
        floor_layers = voxels_to_floor_layers(pred, rooms)
        if display_style == "regions":
            display_rooms = voxels_to_region_rooms(pred, rooms, user_req, seed=seed)
            display_source = "model_regions"
        elif display_style == "boxes":
            display_rooms = voxels_to_boxes(pred, rooms)
            display_source = "model_boxes"
        else:
            display_rooms = footprint_layers_to_rooms(floor_layers)
            display_source = "model_footprint"
        display_rooms = filter_requested_room_types(display_rooms, user_req)
    else:
        display_rooms = rooms
        display_source = "layout_fallback"
    return {
        "rooms": rooms,
        "display_rooms": display_rooms,
        "display_source": display_source,
        "display_style": display_style,
        "floor_layers": floor_layers,
        "graph": G,
        "pos": pos,
        "edge_types": edge_types,
        "pred": pred,
        "n_occ": n_occ,
        "quality_score": quality_score,
        "quality_metrics": quality_metrics,
        "decode_mode": mode,
        "seed": seed,
        "candidates": candidates,
    }
