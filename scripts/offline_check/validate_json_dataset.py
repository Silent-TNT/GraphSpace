# -*- coding: utf-8 -*-
"""
批量校验 house_*.json 数据集（离线，无需 Rhino）

用法:
  python validate_json_dataset.py
  python validate_json_dataset.py --data-dir ../data
  python validate_json_dataset.py --strict-v14   # 仅接受 V14 导出字段
  python validate_json_dataset.py --no-tensor    # 跳过 json_to_sample 试转

输出（默认带日期）:
  qc_report_YYYYMMDD.csv    列: house_id, severity, error_type, detail, suggestion
  qc_summary_YYYYMMDD.json  汇总统计
"""
from __future__ import print_function

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import traceback
from collections import deque
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# 与 260524rhino-json-v14.py 对齐的 QC 常量
# ---------------------------------------------------------------------------
QC_VERSION = "validate_v1"
VOXEL_SIZE = 300
RES_X, RES_Y, RES_Z = 96, 96, 32
MAX_BUILDING_X = RES_X * VOXEL_SIZE
MAX_BUILDING_Y = RES_Y * VOXEL_SIZE
MAX_BUILDING_Z = 6000

TOL = 10
MODULUS = 300
COORD_SNAP_TOL = 1.0
GAP_MAX = 600
MIN_FACE_OVERLAP = MODULUS
STANDARD_FLOOR_HEIGHT = 3000
DOUBLE_HEIGHT_MIN = 5700
DOUBLE_HEIGHT_MAX = 6300

REQUIRED_ROOMS = [
    "entryway", "living_room", "bedroom", "dining_room", "bathroom", "stairs",
]

HEIGHT_3000_OR_6000_TYPES = frozenset([
    "living_room", "multi_purpose", "dining_room", "balcony",
])

TARGET_ROOMS = [
    "living_room", "bedroom", "dining_room", "bathroom",
    "kitchen", "corridor", "stairs", "utility",
    "balcony", "multi_purpose", "entryway",
]

ROOM_TYPES = TARGET_ROOMS  # 与训练 CHANNEL_MAP 一致（不含 empty）

LIGHTING_REVIEW_STRONG = frozenset(["living_room", "bedroom"])
LIGHTING_REVIEW_SOFT = frozenset(["dining_room", "multi_purpose", "kitchen"])

ROOM_TYPE_CN = {
    "living_room": "客厅", "bedroom": "卧室", "dining_room": "餐厅",
    "kitchen": "厨房", "bathroom": "卫生间", "corridor": "过道",
    "stairs": "楼梯", "utility": "家政/储藏", "balcony": "阳台/露台",
    "multi_purpose": "多功能房", "entryway": "玄关",
}

CHANNEL_MAP = {
    "entryway": 1, "living_room": 2, "dining_room": 3, "kitchen": 4,
    "bedroom": 5, "bathroom": 6, "corridor": 7, "stairs": 8,
    "utility": 9, "balcony": 10, "multi_purpose": 11,
}

NODE_IN_DIM = 8
LIGHTING_ACCESS_MAP = {"direct": 1.0, "indirect": 0.5, "none": 0.0}

ERROR_TYPE_RE = re.compile(r"【([^】]+)】")

SUGGESTION_MAP = {
    "schema": "补全 JSON 必需字段，参考 V14 导出格式",
    "JSON解析": "检查文件是否为合法 UTF-8 JSON",
    "体块重合": "在 Rhino 中分离重叠体块至共面贴邻（不体积重叠）",
    "体块缝隙": "移动体块闭合 10~600mm 非预期缝隙，或补 300mm 宽连接走道",
    "模数错误": "将体块长宽高调整为 300mm 整数倍",
    "坐标模数错误": "将角点坐标对齐 300mm 模数网格",
    "高度错误": "按功能类型修正体块高度（普通 3000 / 楼梯·挑空 6000）",
    "楼板界面": "对齐归一化 Z 至 0/3000/6000 楼板界面",
    "楼层判定失败": "修正体块 Z 标高，使 1F=0~3000、2F=3000~6000",
    "体素越界": "缩小建筑平面跨度或联系管理员调整训练栅格上限",
    "体素裁剪": "缩小户型或居中后仍超 96×96×32 栅格，需缩模或调整 RES",
    "功能缺失": "补全必需功能体块（玄关/客厅/卧室/餐厅/卫生间/楼梯）",
    "玄关规则": "至少一个玄关体块需有直接 outdoor 采光面",
    "拓扑孤岛": "用 corridor 等走道连接孤立体块，保证全屋单连通",
    "楼梯未跨层": "楼梯体块应 Z=0~6000 贯穿两层",
    "楼梯未连通": "楼梯需共面邻接一层与二层体块",
    "采光复核·重要": "为客厅/卧室增加采光面或确认平面是否合理，建模师复核",
    "采光复核": "确认该房间是否应为无采光暗房间，建模师复核",
    "tensor试转": "检查 rooms 字段与 box 坐标是否完整，或安装 torch/torch_geometric",
    "graph边缺失": "体块共面贴邻不足，检查 150mm 容差下训练图是否断边",
    "orientation缺失": "使用 260524rhino-json-v14.py 重新导出以写入朝向条件",
    "voxel网格不匹配": "metadata.constraints.voxel_grid 与当前训练 RES 不一致，建议重导出",
}


# ---------------------------------------------------------------------------
# Issue 记录
# ---------------------------------------------------------------------------
class Issue(object):
    __slots__ = ("house_id", "severity", "error_type", "detail", "suggestion")

    def __init__(self, house_id, severity, error_type, detail, suggestion=None):
        self.house_id = house_id
        self.severity = severity
        self.error_type = error_type
        self.detail = detail
        self.suggestion = suggestion or SUGGESTION_MAP.get(error_type, "请对照建模规范修正后重新导出")

    def to_row(self):
        return {
            "house_id": self.house_id,
            "severity": self.severity,
            "error_type": self.error_type,
            "detail": self.detail,
            "suggestion": self.suggestion,
        }


def _issue_from_msg(house_id, severity, msg):
    m = ERROR_TYPE_RE.search(msg)
    etype = m.group(1) if m else "其他"
    return Issue(house_id, severity, etype, msg.replace("\n", " / "))


# ---------------------------------------------------------------------------
# 几何 / QC 工具（纯 NumPy）
# ---------------------------------------------------------------------------
def _near_level(value, level, tol=COORD_SNAP_TOL):
    return abs(float(value) - float(level)) <= tol + 0.5


def _is_modulus_aligned(value, modulus=MODULUS, tol=COORD_SNAP_TOL):
    rem = float(value) % modulus
    return rem <= tol or rem >= modulus - tol


def _is_single_floor_height(dz):
    return (STANDARD_FLOOR_HEIGHT - 300) <= dz <= (STANDARD_FLOOR_HEIGHT + 300)


def _is_double_floor_height(dz):
    return DOUBLE_HEIGHT_MIN <= dz <= DOUBLE_HEIGHT_MAX


def _overlap_1d(a_min, a_max, b_min, b_max, tol=TOL):
    return a_max > b_min + tol and b_max > a_min + tol


def _boxes_have_volume_overlap(a_min, a_max, b_min, b_max, tol=TOL):
    for dim in range(3):
        if a_max[dim] <= b_min[dim] + tol or b_max[dim] <= a_min[dim] + tol:
            return False
    return True


def _boxes_share_face(a_min, a_max, b_min, b_max, direction):
    if direction == "x-":
        if abs(a_min[0] - b_max[0]) > TOL:
            return False
        return _overlap_1d(a_min[1], a_max[1], b_min[1], b_max[1]) and _overlap_1d(a_min[2], a_max[2], b_min[2], b_max[2])
    if direction == "x+":
        if abs(a_max[0] - b_min[0]) > TOL:
            return False
        return _overlap_1d(a_min[1], a_max[1], b_min[1], b_max[1]) and _overlap_1d(a_min[2], a_max[2], b_min[2], b_max[2])
    if direction == "y-":
        if abs(a_min[1] - b_max[1]) > TOL:
            return False
        return _overlap_1d(a_min[0], a_max[0], b_min[0], b_max[0]) and _overlap_1d(a_min[2], a_max[2], b_min[2], b_max[2])
    if direction == "y+":
        if abs(a_max[1] - b_min[1]) > TOL:
            return False
        return _overlap_1d(a_min[0], a_max[0], b_min[0], b_max[0]) and _overlap_1d(a_min[2], a_max[2], b_min[2], b_max[2])
    if direction == "z+":
        if abs(a_max[2] - b_min[2]) > TOL:
            return False
        return _overlap_1d(a_min[0], a_max[0], b_min[0], b_max[0]) and _overlap_1d(a_min[1], a_max[1], b_min[1], b_max[1])
    return False


def _build_adjacency_detail(rooms):
    directions = ("x-", "x+", "y-", "y+", "z+")
    reverse = {"x-": "x+", "x+": "x-", "y-": "y+", "y+": "y-", "z+": "z-"}
    adj = {r["id"]: [] for r in rooms}
    for i, ra in enumerate(rooms):
        amin, amax = ra["box_min"], ra["box_max"]
        for rb in rooms[i + 1:]:
            bmin, bmax = rb["box_min"], rb["box_max"]
            for direction in directions:
                if _boxes_share_face(amin, amax, bmin, bmax, direction):
                    adj[ra["id"]].append((rb["id"], direction))
                    adj[rb["id"]].append((ra["id"], reverse[direction]))
                    break
    return adj


def _get_floors(room):
    if isinstance(room.get("floors"), list) and room["floors"]:
        return [int(f) for f in room["floors"]]
    return [int(room.get("floor", 1))]


def _infer_floors_from_z(z0, z1, room_type):
    height = z1 - z0
    if room_type == "stairs":
        if _near_level(z0, 0) and _near_level(z1, 6000):
            return [1, 2]
        return None
    if room_type in HEIGHT_3000_OR_6000_TYPES and _is_double_floor_height(height):
        if _near_level(z0, 0) and _near_level(z1, 6000):
            return [1, 2]
        if _near_level(z0, 3000) and _near_level(z1, 6000):
            return [2]
        return None
    if _near_level(z0, 0) and _near_level(z1, 3000):
        return [1]
    if _near_level(z0, 3000) and _near_level(z1, 6000):
        return [2]
    return None


def _axis_overlap_len(a_min, a_max, b_min, b_max, axis):
    return min(a_max[axis], b_max[axis]) - max(a_min[axis], b_min[axis])


def _find_planar_gaps(a_min, a_max, b_min, b_max):
    gaps = []
    candidates = [
        (0, b_min[0] - a_max[0], "b_east"),
        (0, a_min[0] - b_max[0], "a_east"),
        (1, b_min[1] - a_max[1], "b_south"),
        (1, a_min[1] - b_max[1], "a_south"),
    ]
    seen = set()
    for axis, gap, mode in candidates:
        if gap <= TOL or gap > GAP_MAX:
            continue
        other = 1 if axis == 0 else 0
        if _axis_overlap_len(a_min, a_max, b_min, b_max, other) < MIN_FACE_OVERLAP:
            continue
        if _axis_overlap_len(a_min, a_max, b_min, b_max, 2) <= TOL:
            continue
        key = (axis, mode, round(gap, 1))
        if key in seen:
            continue
        seen.add(key)
        gaps.append({"axis": "X" if axis == 0 else "Y", "gap": round(gap, 1), "mode": mode, "axis_idx": axis})
    return gaps


def _gap_slab(a_min, a_max, b_min, b_max, axis, mode):
    if axis == 0 and mode == "b_east":
        return ([a_max[0], max(a_min[1], b_min[1]), max(a_min[2], b_min[2])],
                [b_min[0], min(a_max[1], b_max[1]), min(a_max[2], b_max[2])])
    if axis == 0 and mode == "a_east":
        return ([b_max[0], max(a_min[1], b_min[1]), max(a_min[2], b_min[2])],
                [a_min[0], min(a_max[1], b_max[1]), min(a_max[2], b_max[2])])
    if axis == 1 and mode == "b_south":
        return ([max(a_min[0], b_min[0]), a_max[1], max(a_min[2], b_min[2])],
                [min(a_max[0], b_max[0]), b_min[1], min(a_max[2], b_max[2])])
    if axis == 1 and mode == "a_south":
        return ([max(a_min[0], b_min[0]), b_max[1], max(a_min[2], b_min[2])],
                [min(a_max[0], b_max[0]), a_min[1], min(a_max[2], b_max[2])])
    return None


def _is_gap_bridged(slab_min, slab_max, entries, exclude):
    if any(slab_max[d] <= slab_min[d] + TOL for d in range(3)):
        return False
    for idx, e in enumerate(entries):
        if idx in exclude:
            continue
        if _boxes_have_volume_overlap(slab_min, slab_max, e["box_min"], e["box_max"]):
            return True
    return False


def _room_label(room):
    return room.get("layer_name") or room.get("id", "?")


# ---------------------------------------------------------------------------
# json_to_sample（与 MVP notebook 对齐，torch 可选）
# ---------------------------------------------------------------------------
def _get_node_supertype(room_type):
    if room_type in ["bedroom", "living_room", "dining_room", "multi_purpose"]:
        return "living"
    if room_type in ["kitchen", "bathroom", "utility", "balcony"]:
        return "service"
    return "circulation"


def _check_hetero_adjacency(r1, r2, tol=150, min_shared_len=300):
    if r1.get("floor", 1) != r2.get("floor", 1):
        if r1["type"] == "stairs" or r2["type"] == "stairs":
            ox = max(0, min(r1["box_max"][0], r2["box_max"][0]) - max(r1["box_min"][0], r2["box_min"][0]))
            oy = max(0, min(r1["box_max"][1], r2["box_max"][1]) - max(r1["box_min"][1], r2["box_min"][1]))
            if ox > 0 and oy > 0:
                return "vertical"
        return None
    ox = max(0, min(r1["box_max"][0], r2["box_max"][0]) - max(r1["box_min"][0], r2["box_min"][0]) + tol)
    oy = max(0, min(r1["box_max"][1], r2["box_max"][1]) - max(r1["box_min"][1], r2["box_min"][1]) + tol)
    if (ox > min_shared_len and oy > 0) or (oy > min_shared_len and ox > 0):
        return "horizontal"
    return None


def _effective_lighting_ratio(room):
    dx = room["box_max"][0] - room["box_min"][0]
    dy = room["box_max"][1] - room["box_min"][1]
    floor_area = max(dx * dy, 1.0)
    eff = room.get("effective_lighting", [])
    total = sum(float(e.get("area", 0)) * float(e.get("attenuation", 1.0)) for e in eff)
    return min(total / floor_area, 5.0) / 5.0


def _build_room_features(room, build_min, b_size):
    dx = room["box_max"][0] - room["box_min"][0]
    dy = room["box_max"][1] - room["box_min"][1]
    area = (dx * dy) / 1e6
    aspect = max(dx, dy) / max(min(dx, dy), 1.0)
    cx = ((room["box_min"][0] + room["box_max"][0]) / 2 - build_min[0]) / (b_size[0] + 1e-5)
    cy = ((room["box_min"][1] + room["box_max"][1]) / 2 - build_min[1]) / (b_size[1] + 1e-5)
    cz = ((room["box_min"][2] + room["box_max"][2]) / 2 - build_min[2]) / (b_size[2] + 1e-5)
    lighting_access = LIGHTING_ACCESS_MAP.get(room.get("lighting_access", "none"), 0.0)
    lighting_priority = float(room.get("lighting_priority", 0)) / 10.0
    lighting_ratio = _effective_lighting_ratio(room)
    return [area, aspect, cx, cy, cz, lighting_access, lighting_priority, lighting_ratio]


def json_to_sample_numpy(data, res_x=RES_X, res_y=RES_Y, res_z=RES_Z, voxel_size=VOXEL_SIZE):
    """NumPy 版体素化（不依赖 torch），用于基础 tensor 试转"""
    rooms = data.get("rooms", [])
    if not rooms:
        return None

    all_coords = np.array([r["box_min"] for r in rooms] + [r["box_max"] for r in rooms], dtype=float)
    build_min = all_coords.min(axis=0)
    build_max = all_coords.max(axis=0)

    phys_center_xy = (build_min[:2] + build_max[:2]) / 2.0
    offset_xy = np.array([res_x * voxel_size / 2.0, res_y * voxel_size / 2.0]) - phys_center_xy
    z_min_phys = build_min[2]

    grid = np.zeros((res_x, res_y, res_z), dtype=np.int8)
    edge_count = 0

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if _check_hetero_adjacency(rooms[i], rooms[j]):
                edge_count += 1

    for r in rooms:
        ch = CHANNEL_MAP.get(r["type"], 0)
        ix0 = int((r["box_min"][0] + offset_xy[0]) / voxel_size)
        ix1 = int((r["box_max"][0] + offset_xy[0]) / voxel_size)
        iy0 = int((r["box_min"][1] + offset_xy[1]) / voxel_size)
        iy1 = int((r["box_max"][1] + offset_xy[1]) / voxel_size)
        iz0 = int((r["box_min"][2] - z_min_phys) / voxel_size)
        iz1 = int((r["box_max"][2] - z_min_phys) / voxel_size)
        grid[max(0, ix0):min(res_x, ix1), max(0, iy0):min(res_y, iy1), max(0, iz0):min(res_z, iz1)] = ch

    one_hot = np.zeros((len(CHANNEL_MAP) + 1, res_x, res_y, res_z), dtype=np.float32)
    for c in range(len(CHANNEL_MAP) + 1):
        one_hot[c] = (grid == c).astype(np.float32)

    return {
        "voxel": one_hot,
        "occupied_voxels": int((grid > 0).sum()),
        "graph_edges": edge_count,
        "build_span": (build_max - build_min).tolist(),
    }


def json_to_sample_torch(data, res_x=RES_X, res_y=RES_Y, res_z=RES_Z, voxel_size=VOXEL_SIZE):
    """完整 torch 版（与 MVP notebook 一致）"""
    import torch
    from torch_geometric.data import HeteroData

    rooms = data.get("rooms", [])
    if not rooms:
        return None

    all_coords = np.array([r["box_min"] for r in rooms] + [r["box_max"] for r in rooms], dtype=float)
    build_min = all_coords.min(axis=0)
    build_max = all_coords.max(axis=0)
    b_size = build_max - build_min

    hetero = HeteroData()
    nodes_dict = {"living": [], "service": [], "circulation": []}
    id_to_idx = {}

    for r in rooms:
        ntype = _get_node_supertype(r["type"])
        id_to_idx[r["id"]] = (ntype, len(nodes_dict[ntype]))
        nodes_dict[ntype].append(_build_room_features(r, build_min, b_size))

    for ntype, feats in nodes_dict.items():
        hetero[ntype].x = torch.tensor(feats, dtype=torch.float32) if feats else torch.empty((0, NODE_IN_DIM))

    edges_dict = {}
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            r1, r2 = rooms[i], rooms[j]
            rel = _check_hetero_adjacency(r1, r2)
            if not rel:
                continue
            t1, idx1 = id_to_idx[r1["id"]]
            t2, idx2 = id_to_idx[r2["id"]]
            for src_t, dst_t, si, di in ((t1, t2, idx1, idx2), (t2, t1, idx2, idx1)):
                key = (src_t, rel, dst_t)
                edges_dict.setdefault(key, []).append([si, di])

    for key, elist in edges_dict.items():
        hetero[key].edge_index = torch.tensor(elist, dtype=torch.long).t().contiguous()

    grid = np.zeros((res_x, res_y, res_z), dtype=np.int8)
    phys_center_xy = (build_min[:2] + build_max[:2]) / 2.0
    offset_xy = np.array([res_x * voxel_size / 2.0, res_y * voxel_size / 2.0]) - phys_center_xy
    z_min_phys = build_min[2]

    for r in rooms:
        ch = CHANNEL_MAP.get(r["type"], 0)
        ix0 = int((r["box_min"][0] + offset_xy[0]) / voxel_size)
        ix1 = int((r["box_max"][0] + offset_xy[0]) / voxel_size)
        iy0 = int((r["box_min"][1] + offset_xy[1]) / voxel_size)
        iy1 = int((r["box_max"][1] + offset_xy[1]) / voxel_size)
        iz0 = int((r["box_min"][2] - z_min_phys) / voxel_size)
        iz1 = int((r["box_max"][2] - z_min_phys) / voxel_size)
        grid[max(0, ix0):min(res_x, ix1), max(0, iy0):min(res_y, iy1), max(0, iz0):min(res_z, iz1)] = ch

    num_channels = max(CHANNEL_MAP.values()) + 1
    one_hot = np.zeros((num_channels, res_x, res_y, res_z), dtype=np.float32)
    for c in range(num_channels):
        one_hot[c] = (grid == c).astype(np.float32)

    return {
        "graph": hetero,
        "voxel": torch.tensor(one_hot, dtype=torch.float32),
        "occupied_voxels": int((grid > 0).sum()),
        "graph_edge_keys": len(edges_dict),
    }


def try_json_to_sample(data, use_torch=True):
    if use_torch:
        try:
            return json_to_sample_torch(data), "torch"
        except ImportError:
            pass
        except Exception as exc:
            raise exc
    return json_to_sample_numpy(data), "numpy"


# ---------------------------------------------------------------------------
# 单文件校验
# ---------------------------------------------------------------------------
def _validate_schema(house_id, data, issues, strict_v14=False):
    if not isinstance(data, dict):
        issues.append(Issue(house_id, "error", "schema", "根对象不是 JSON object"))
        return False

    ok = True
    if "rooms" not in data:
        issues.append(Issue(house_id, "error", "schema", "缺少 rooms 字段"))
        ok = False
    if "metadata" not in data:
        issues.append(Issue(house_id, "error", "schema", "缺少 metadata 字段"))
        ok = False

    meta = data.get("metadata", {})
    if "building_size" not in meta:
        issues.append(Issue(house_id, "error", "schema", "metadata 缺少 building_size"))
        ok = False
    if "stats" not in meta:
        issues.append(Issue(house_id, "warning", "schema", "metadata 缺少 stats（建议补全）"))

    rooms = data.get("rooms", [])
    if not isinstance(rooms, list) or len(rooms) == 0:
        issues.append(Issue(house_id, "error", "schema", "rooms 为空或不是数组"))
        return False

    required_room_keys = ("id", "type", "box_min", "box_max")
    for idx, room in enumerate(rooms):
        for key in required_room_keys:
            if key not in room:
                issues.append(Issue(
                    house_id, "error", "schema",
                    "rooms[{}] 缺少字段 {}".format(idx, key),
                ))
                ok = False
        if room.get("type") not in TARGET_ROOMS:
            issues.append(Issue(
                house_id, "error", "schema",
                "rooms[{}] type='{}' 不在合法功能列表".format(idx, room.get("type")),
            ))
            ok = False
        for corner in ("box_min", "box_max"):
            val = room.get(corner)
            if val is None or not isinstance(val, list) or len(val) < 3:
                issues.append(Issue(house_id, "error", "schema", "rooms[{}].{} 无效".format(idx, corner)))
                ok = False

    if strict_v14:
        constraints = meta.get("constraints", {})
        if constraints.get("qc_version") != "v14":
            issues.append(Issue(
                house_id, "warning", "schema",
                "非 V14 导出（qc_version={}）".format(constraints.get("qc_version")),
            ))
        vg = constraints.get("voxel_grid", {})
        if vg.get("res") != [RES_X, RES_Y, RES_Z]:
            issues.append(Issue(
                house_id, "warning", "voxel网格不匹配",
                "metadata voxel_grid.res={}，当前训练期望 [{},{},{}]".format(
                    vg.get("res"), RES_X, RES_Y, RES_Z),
            ))
        if "orientation_qc" not in meta:
            issues.append(Issue(house_id, "warning", "orientation缺失", "metadata 缺少 orientation_qc"))

    return ok


def _prepare_entries(rooms):
    entries = []
    for room in rooms:
        bmin = [float(v) for v in room["box_min"]]
        bmax = [float(v) for v in room["box_max"]]
        entries.append({
            "id": room["id"],
            "type": room["type"],
            "layer_name": _room_label(room),
            "box_min": bmin,
            "box_max": bmax,
            "norm_min": bmin,
            "norm_max": bmax,
            "floor": int(room.get("floor", 1)),
            "floors": _get_floors(room),
        })
    return entries


def _check_overlaps(house_id, entries, issues):
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if not _boxes_have_volume_overlap(a["box_min"], a["box_max"], b["box_min"], b["box_max"]):
                continue
            msg = (
                "【体块重合】{}（{}）↔ {}（{}）".format(
                    a["layer_name"], ROOM_TYPE_CN.get(a["type"], a["type"]),
                    b["layer_name"], ROOM_TYPE_CN.get(b["type"], b["type"]),
                ))
            issues.append(_issue_from_msg(house_id, "error", msg))


def _check_gaps(house_id, entries, issues):
    reported = set()
    for i, a in enumerate(entries):
        for j, b in enumerate(entries[i + 1:], start=i + 1):
            gaps = _find_planar_gaps(a["box_min"], a["box_max"], b["box_min"], b["box_max"])
            for gap in gaps:
                slab = _gap_slab(
                    a["box_min"], a["box_max"], b["box_min"], b["box_max"],
                    gap["axis_idx"], gap["mode"],
                )
                if _is_gap_bridged(slab[0], slab[1], entries, {i, j}):
                    continue
                key = (i, b["layer_name"], gap["axis"], gap["gap"])
                if key in reported:
                    continue
                reported.add(key)
                msg = "【体块缝隙】{} ↔ {}，{} 方向 {} mm".format(
                    a["layer_name"], b["layer_name"], gap["axis"], gap["gap"])
                issues.append(_issue_from_msg(house_id, "error", msg))


def _check_modulus_and_slab(house_id, entries, building_size, issues):
    bx = float(building_size.get("x", 0))
    by = float(building_size.get("y", 0))
    bz = float(building_size.get("z", 0))

    for e in entries:
        layer = e["layer_name"]
        rtype = e["type"]
        bmin, bmax = e["box_min"], e["box_max"]
        dx = int(round(bmax[0] - bmin[0]))
        dy = int(round(bmax[1] - bmin[1]))
        dz = int(round(bmax[2] - bmin[2]))

        for dim, val, axis in ((dx, dx, "长(X)"), (dy, dy, "宽(Y)"), (dz, dz, "高(Z)")):
            if not _is_modulus_aligned(val):
                msg = "【模数错误】[{}] {} 尺寸 {}mm 不符合 300mm 模数".format(layer, axis, val)
                issues.append(_issue_from_msg(house_id, "error", msg))

        for val, label in zip(list(bmin) + list(bmax),
                              ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]):
            if not _is_modulus_aligned(val):
                nearest = round(float(val) / MODULUS) * MODULUS
                if abs(float(val) - nearest) > COORD_SNAP_TOL:
                    msg = "【坐标模数错误】[{}] {}={:.2f}，最近模数 {}".format(layer, label, val, int(nearest))
                    issues.append(_issue_from_msg(house_id, "error", msg))

        floors = _infer_floors_from_z(bmin[2], bmax[2], rtype)
        if floors is None:
            msg = "【楼层判定失败】[{}] Z {:.0f}~{:.0f} 未对齐 0/3000/6000".format(layer, bmin[2], bmax[2])
            issues.append(_issue_from_msg(house_id, "error", msg))
        else:
            e["floors"] = floors
            e["floor"] = floors[0]

        z0, z1 = bmin[2], bmax[2]
        type_cn = ROOM_TYPE_CN.get(rtype, rtype)

        if rtype == "stairs":
            if not _is_double_floor_height(dz):
                msg = "【高度错误】[{}] 楼梯高度 {}mm".format(layer, dz)
                issues.append(_issue_from_msg(house_id, "error", msg))
            if not (_near_level(z0, 0) and _near_level(z1, 6000)):
                msg = "【楼板界面】[{}] 楼梯 Z {:.0f}~{:.0f}".format(layer, z0, z1)
                issues.append(_issue_from_msg(house_id, "error", msg))
        elif rtype in HEIGHT_3000_OR_6000_TYPES:
            valid_z = (
                (_near_level(z0, 0) and _near_level(z1, 3000)) or
                (_near_level(z0, 3000) and _near_level(z1, 6000)) or
                (_near_level(z0, 0) and _near_level(z1, 6000))
            )
            if not valid_z:
                msg = "【楼板界面】[{}] {} Z {:.0f}~{:.0f}".format(layer, type_cn, z0, z1)
                issues.append(_issue_from_msg(house_id, "error", msg))
            if not (_is_single_floor_height(dz) or _is_double_floor_height(dz)):
                msg = "【高度错误】[{}] {} 高度 {}mm".format(layer, type_cn, dz)
                issues.append(_issue_from_msg(house_id, "error", msg))
        else:
            if not _is_single_floor_height(dz):
                msg = "【高度错误】[{}] {} 高度 {}mm，只允许 3000".format(layer, type_cn, dz)
                issues.append(_issue_from_msg(house_id, "error", msg))
            valid_z = (
                (_near_level(z0, 0) and _near_level(z1, 3000)) or
                (_near_level(z0, 3000) and _near_level(z1, 6000))
            )
            if not valid_z:
                msg = "【楼板界面】[{}] {} Z {:.0f}~{:.0f}".format(layer, type_cn, z0, z1)
                issues.append(_issue_from_msg(house_id, "error", msg))

    if bx > MAX_BUILDING_X + COORD_SNAP_TOL:
        msg = "【体素越界】X 跨度 {}mm > {}".format(int(round(bx)), MAX_BUILDING_X)
        issues.append(_issue_from_msg(house_id, "error", msg))
    if by > MAX_BUILDING_Y + COORD_SNAP_TOL:
        msg = "【体素越界】Y 跨度 {}mm > {}".format(int(round(by)), MAX_BUILDING_Y)
        issues.append(_issue_from_msg(house_id, "error", msg))
    if bz > MAX_BUILDING_Z + COORD_SNAP_TOL:
        msg = "【体素越界】Z 跨度 {}mm > {}".format(int(round(bz)), MAX_BUILDING_Z)
        issues.append(_issue_from_msg(house_id, "error", msg))

    if not entries:
        return

    all_mins = [e["box_min"] for e in entries]
    all_maxs = [e["box_max"] for e in entries]
    build_min = [min(c[i] for c in all_mins) for i in range(3)]
    build_max = [max(c[i] for c in all_maxs) for i in range(3)]
    phys_center_xy = [(build_min[0] + build_max[0]) / 2.0, (build_min[1] + build_max[1]) / 2.0]
    offset_xy = [RES_X * VOXEL_SIZE / 2.0 - phys_center_xy[0], RES_Y * VOXEL_SIZE / 2.0 - phys_center_xy[1]]
    z_min_phys = build_min[2]

    clipped = []
    for e in entries:
        nmin, nmax = e["box_min"], e["box_max"]
        ix0 = int((nmin[0] + offset_xy[0]) / VOXEL_SIZE)
        ix1 = int((nmax[0] + offset_xy[0]) / VOXEL_SIZE)
        iy0 = int((nmin[1] + offset_xy[1]) / VOXEL_SIZE)
        iy1 = int((nmax[1] + offset_xy[1]) / VOXEL_SIZE)
        iz0 = int((nmin[2] - z_min_phys) / VOXEL_SIZE)
        iz1 = int((nmax[2] - z_min_phys) / VOXEL_SIZE)
        if ix0 < 0 or iy0 < 0 or iz0 < 0 or ix1 > RES_X or iy1 > RES_Y or iz1 > RES_Z:
            clipped.append(e["layer_name"])

    if clipped:
        msg = "【体素裁剪】{}×{}×{} 栅格下被截断：{}".format(
            RES_X, RES_Y, RES_Z, "、".join(clipped[:8]))
        issues.append(_issue_from_msg(house_id, "error", msg))


def _check_required_and_entryway(house_id, rooms, stats, issues):
    missing = [r for r in REQUIRED_ROOMS if int(stats.get(r, 0)) <= 0]
    if missing:
        cn = "、".join(ROOM_TYPE_CN.get(m, m) for m in missing)
        msg = "【功能缺失】缺少必需空间：{}".format(cn)
        issues.append(_issue_from_msg(house_id, "error", msg))

    entryways = [r for r in rooms if r["type"] == "entryway"]
    if entryways:
        has_outdoor = False
        for e in entryways:
            for surf in e.get("direct_lighting_surfaces", []):
                if surf.get("exposure_type", "outdoor") == "outdoor":
                    has_outdoor = True
                    break
        if not has_outdoor:
            names = "、".join(_room_label(e) for e in entryways)
            msg = "【玄关规则】{} 均无 outdoor direct 暴露面".format(names)
            issues.append(_issue_from_msg(house_id, "error", msg))


def _check_topology(house_id, rooms, issues):
    if len(rooms) <= 1:
        return

    adj = _build_adjacency_detail(rooms)
    start = rooms[0]["id"]
    visited = set()
    queue = deque([start])
    while queue:
        rid = queue.popleft()
        if rid in visited:
            continue
        visited.add(rid)
        for nid, _ in adj.get(rid, []):
            if nid not in visited:
                queue.append(nid)

    if len(visited) < len(rooms):
        room_by_id = {r["id"]: r for r in rooms}
        for rid in room_by_id:
            if rid in visited:
                continue
            r = room_by_id[rid]
            msg = "【拓扑孤岛】[{}] {} 未与主体连通".format(
                _room_label(r), ROOM_TYPE_CN.get(r["type"], r["type"]))
            issues.append(_issue_from_msg(house_id, "error", msg))

    room_by_id = {r["id"]: r for r in rooms}
    for st in [r for r in rooms if r["type"] == "stairs"]:
        st_floors = set(_get_floors(st))
        bmin, bmax = st["box_min"], st["box_max"]
        if not (1 in st_floors and 2 in st_floors):
            if not (_near_level(bmin[2], 0) and _near_level(bmax[2], 6000)):
                msg = "【楼梯未跨层】[{}] floors={}".format(_room_label(st), sorted(st_floors))
                issues.append(_issue_from_msg(house_id, "error", msg))

        has_f1 = has_f2 = False
        for nid, _ in adj.get(st["id"], []):
            nf = set(_get_floors(room_by_id[nid]))
            if 1 in nf:
                has_f1 = True
            if 2 in nf:
                has_f2 = True
        if not has_f1 or not has_f2:
            msg = "【楼梯未连通】[{}] 一层{} 二层{}".format(
                _room_label(st), "✓" if has_f1 else "✗", "✓" if has_f2 else "✗")
            issues.append(_issue_from_msg(house_id, "error", msg))


def _check_lighting_warnings(house_id, rooms, issues):
    for room in rooms:
        if room.get("lighting_access", "none") != "none":
            continue
        rtype = room["type"]
        layer = _room_label(room)
        type_cn = ROOM_TYPE_CN.get(rtype, rtype)
        if rtype in LIGHTING_REVIEW_STRONG:
            msg = "【采光复核·重要】[{}] {} 无采光".format(layer, type_cn)
            issues.append(_issue_from_msg(house_id, "warning", msg))
        elif rtype in LIGHTING_REVIEW_SOFT:
            msg = "【采光复核】[{}] {} 无采光".format(layer, type_cn)
            issues.append(_issue_from_msg(house_id, "warning", msg))
        elif rtype == "balcony":
            msg = "【采光复核】[{}] 阳台无 outdoor 暴露".format(layer)
            issues.append(_issue_from_msg(house_id, "warning", msg))


def _check_graph_edges(house_id, rooms, issues):
    """QC 共面邻接 vs 训练图构建（150mm 容差）不一致预警"""
    qc_adj = _build_adjacency_detail(rooms)
    qc_pairs = set()
    for rid, nbs in qc_adj.items():
        for nid, _ in nbs:
            qc_pairs.add(tuple(sorted((rid, nid))))

    train_pairs = set()
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if _check_hetero_adjacency(rooms[i], rooms[j]):
                train_pairs.add(tuple(sorted((rooms[i]["id"], rooms[j]["id"]))))

    only_qc = qc_pairs - train_pairs
    if only_qc and len(train_pairs) == 0 and len(rooms) > 2:
        issues.append(Issue(
            house_id, "warning", "graph边缺失",
            "共面邻接 {} 对，但训练图边为 0（检查 floor 字段或体块贴邻）".format(len(qc_pairs)),
        ))


def _check_tensor(house_id, data, issues, use_torch=True):
    try:
        sample, backend = try_json_to_sample(data, use_torch=use_torch)
    except Exception as exc:
        issues.append(Issue(
            house_id, "error", "tensor试转",
            "json_to_sample 失败: {} / {}".format(type(exc).__name__, exc),
        ))
        return

    if sample is None:
        issues.append(Issue(house_id, "error", "tensor试转", "json_to_sample 返回 None（rooms 为空？）"))
        return

    occ = sample.get("occupied_voxels", 0)
    if occ <= 0:
        issues.append(Issue(
            house_id, "error", "tensor试转",
            "体素栅格 occupied_voxels=0，训练样本无效",
        ))


def validate_json_file(path, strict_v14=False, run_tensor=True, use_torch=True):
    house_id = os.path.splitext(os.path.basename(path))[0]
    issues = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        issues.append(Issue(
            house_id, "error", "JSON解析",
            "无法读取 {}: {}".format(path, exc),
        ))
        return issues

    file_hid = data.get("house_id", house_id)
    if file_hid:
        house_id = file_hid

    if not _validate_schema(house_id, data, issues, strict_v14=strict_v14):
        return issues

    meta = data.get("metadata", {})
    rooms = data.get("rooms", [])
    stats = meta.get("stats", {})
    building_size = meta.get("building_size", {})

    entries = _prepare_entries(rooms)

    _check_overlaps(house_id, entries, issues)
    _check_gaps(house_id, entries, issues)
    _check_modulus_and_slab(house_id, entries, building_size, issues)
    _check_required_and_entryway(house_id, rooms, stats, issues)
    _check_topology(house_id, rooms, issues)
    _check_lighting_warnings(house_id, rooms, issues)
    _check_graph_edges(house_id, rooms, issues)

    if run_tensor:
        _check_tensor(house_id, data, issues, use_torch=use_torch)

    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_data_dir(explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "data"),
        os.path.join(script_dir, "data"),
        os.path.join(os.getcwd(), "data"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            return path
    return os.path.abspath(candidates[0])


def _date_tag():
    return datetime.now().strftime("%Y%m%d")


def resolve_output_paths(output_arg=None, summary_arg=None, base_dir=None):
    """默认输出 qc_report_YYYYMMDD.csv / qc_summary_YYYYMMDD.json"""
    date_tag = _date_tag()
    root = os.path.abspath(base_dir or os.getcwd())

    if output_arg:
        output_path = os.path.abspath(output_arg)
    else:
        output_path = os.path.join(root, "qc_report_{}.csv".format(date_tag))

    if summary_arg:
        summary_path = os.path.abspath(summary_arg)
    else:
        summary_path = os.path.join(
            os.path.dirname(output_path),
            "qc_summary_{}.json".format(date_tag),
        )
    return output_path, summary_path


def write_report(all_issues, output_path):
    fieldnames = ["house_id", "severity", "error_type", "detail", "suggestion"]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for issue in all_issues:
            writer.writerow(issue.to_row())


def write_summary(all_issues, file_results, summary_path, data_dir):
    total_files = len(file_results)
    pass_files = sum(1 for r in file_results if r["error_count"] == 0)
    summary = {
        "validator": QC_VERSION,
        "data_dir": data_dir,
        "voxel_grid": [RES_X, RES_Y, RES_Z],
        "max_building_mm": [MAX_BUILDING_X, MAX_BUILDING_Y, MAX_BUILDING_Z],
        "total_files": total_files,
        "pass_files": pass_files,
        "fail_files": total_files - pass_files,
        "total_issues": len(all_issues),
        "error_issues": sum(1 for i in all_issues if i.severity == "error"),
        "warning_issues": sum(1 for i in all_issues if i.severity == "warning"),
        "files": file_results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="批量校验 house_*.json 数据集")
    parser.add_argument("--data-dir", default=None, help="JSON 目录，默认 ../data")
    parser.add_argument(
        "--output", default=None,
        help="CSV 报告路径（默认 qc_report_YYYYMMDD.csv，输出到当前目录）",
    )
    parser.add_argument(
        "--summary", default=None,
        help="JSON 汇总路径（默认 qc_summary_YYYYMMDD.json，与 CSV 同目录）",
    )
    parser.add_argument("--strict-v14", action="store_true", help="严格检查 V14 metadata 字段")
    parser.add_argument("--no-tensor", action="store_true", help="跳过 json_to_sample 试转")
    parser.add_argument("--numpy-only", action="store_true", help="tensor 试转仅用 NumPy（不依赖 torch）")
    args = parser.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    pattern = os.path.join(data_dir, "house_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print("未找到 JSON 文件: {}".format(pattern))
        return 1

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path, summary_path = resolve_output_paths(
        args.output, args.summary, base_dir=script_dir,
    )

    all_issues = []
    file_results = []

    print("校验目录: {}".format(data_dir))
    print("文件数量: {}".format(len(files)))
    print("体素栅格: {}×{}×{} @ {}mm".format(RES_X, RES_Y, RES_Z, VOXEL_SIZE))
    print("-" * 60)

    for path in files:
        house_id = os.path.splitext(os.path.basename(path))[0]
        try:
            issues = validate_json_file(
                path,
                strict_v14=args.strict_v14,
                run_tensor=not args.no_tensor,
                use_torch=not args.numpy_only,
            )
        except Exception:
            tb = traceback.format_exc()
            issues = [Issue(house_id, "error", "内部错误", tb.replace("\n", " / "))]

        err_n = sum(1 for i in issues if i.severity == "error")
        warn_n = sum(1 for i in issues if i.severity == "warning")
        status = "PASS" if err_n == 0 else "FAIL"
        print("{:8} {}  errors={} warnings={}".format(status, house_id, err_n, warn_n))

        all_issues.extend(issues)
        file_results.append({
            "house_id": house_id,
            "path": path,
            "status": status,
            "error_count": err_n,
            "warning_count": warn_n,
        })

    write_report(all_issues, output_path)
    write_summary(all_issues, file_results, summary_path, data_dir)

    pass_n = sum(1 for r in file_results if r["status"] == "PASS")
    print("-" * 60)
    print("完成: {}/{} 通过".format(pass_n, len(files)))
    print("报告: {}".format(output_path))
    print("汇总: {}".format(summary_path))

    return 0 if pass_n == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
