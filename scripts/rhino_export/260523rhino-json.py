# -*- coding: utf-8 -*-
import json
import re
import math
from collections import deque

try:
    import rhinoscriptsyntax as rs
except ImportError:
    rs = None

# ==========================================
# 采光分析常量 (Phase 1-3)
# ==========================================
TOL = 10
CELL_SIZE = 300
MAX_LIGHT_HOPS = 8
MIN_EFFECTIVE_ATTENUATION = 0.08
SINGLE_FLOOR_HEIGHT = 3600
COORD_SNAP_TOL = 1.0
MODULUS = 300
GAP_MAX = 600              # 超过 600mm 的缝通常是有意留空，不报警
MIN_FACE_OVERLAP = MODULUS  # 共面邻接判定：垂直于缝方向至少 300mm 贴边

TARGET_ROOMS = [
    "living_room", "bedroom", "dining_room", "bathroom",
    "kitchen", "corridor", "stairs", "utility",
    "balcony", "multi_purpose", "entryway",
]

TRANSIT_TYPES = frozenset(["entryway", "corridor", "balcony", "stairs"])
BLOCKER_TYPES = frozenset(["bathroom", "utility"])

ATTENUATION = {
    "entryway": 0.70,
    "corridor": 0.50,
    "balcony": 1.00,
    "stairs": 0.60,
}

LIGHTING_PRIORITY = {
    "living_room": 10,
    "bedroom": 8,
    "multi_purpose": 6,
    "dining_room": 5,
    "kitchen": 4,
    "utility": 2,
    "bathroom": 2,
    "entryway": 1,
    "stairs": 1,
    "corridor": 0,
    "balcony": 0,
}

ROOM_TYPE_CN = {
    "living_room": "客厅",
    "bedroom": "卧室",
    "dining_room": "餐厅",
    "kitchen": "厨房",
    "bathroom": "卫生间",
    "corridor": "过道",
    "stairs": "楼梯",
    "utility": "家政/储藏",
    "balcony": "阳台/露台",
    "multi_purpose": "多功能房",
    "entryway": "玄关",
}

CORNER_QC_META = {
    "min_x": {"corner": "西侧垂直边", "axis": "X", "pos_dir": "东", "neg_dir": "西"},
    "max_x": {"corner": "东侧垂直边", "axis": "X", "pos_dir": "东", "neg_dir": "西"},
    "min_y": {"corner": "南侧垂直边", "axis": "Y", "pos_dir": "北", "neg_dir": "南"},
    "max_y": {"corner": "北侧垂直边", "axis": "Y", "pos_dir": "北", "neg_dir": "南"},
    "min_z": {"corner": "底面水平边", "axis": "Z", "pos_dir": "上", "neg_dir": "下"},
    "max_z": {"corner": "顶面水平边", "axis": "Z", "pos_dir": "上", "neg_dir": "下"},
}


def _parse_layer_room_type(layer_name):
    """
    从图层名解析功能类型。
    支持：06bedroom-卧室 | multi_purpose-多功能室 | multi_purpose多功能室
    """
    base = re.sub(r"^\d+", "", layer_name.split("::")[-1].strip())
    for room_type in sorted(TARGET_ROOMS, key=len, reverse=True):
        if base.startswith(room_type):
            return room_type
    head = re.sub(r"^\d+", "", base.split("-")[0].strip())
    return head if head in TARGET_ROOMS else None


def _layer_looks_like_function_space(layer_name):
    lower = layer_name.lower()
    return any(room_type in lower for room_type in TARGET_ROOMS)


def _overlap_1d(a_min, a_max, b_min, b_max, tol=TOL):
    return a_max > b_min + tol and b_max > a_min + tol


def _boxes_share_face(room_a, room_b, direction):
    """检测两房间是否在指定方向共面邻接。direction: x-, x+, y-, y+, z+"""
    a_min, a_max = room_a["_abs_min"], room_a["_abs_max"]
    b_min, b_max = room_b["_abs_min"], room_b["_abs_max"]

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


def _face_has_neighbor(room, direction, all_rooms):
    for other in all_rooms:
        if other["id"] == room["id"]:
            continue
        if _boxes_share_face(room, other, direction):
            return True
    return False


def _face_geometry(room, direction):
    """返回 (face_center_xyz, normal, area)"""
    a_min, a_max = room["_abs_min"], room["_abs_max"]
    dx = a_max[0] - a_min[0]
    dy = a_max[1] - a_min[1]
    dz = a_max[2] - a_min[2]

    if direction == "x-":
        return ([a_min[0], (a_min[1] + a_max[1]) / 2.0, (a_min[2] + a_max[2]) / 2.0], [-1.0, 0.0, 0.0], dy * dz)
    if direction == "x+":
        return ([a_max[0], (a_min[1] + a_max[1]) / 2.0, (a_min[2] + a_max[2]) / 2.0], [1.0, 0.0, 0.0], dy * dz)
    if direction == "y-":
        return ([(a_min[0] + a_max[0]) / 2.0, a_min[1], (a_min[2] + a_max[2]) / 2.0], [0.0, -1.0, 0.0], dx * dz)
    if direction == "y+":
        return ([(a_min[0] + a_max[0]) / 2.0, a_max[1], (a_min[2] + a_max[2]) / 2.0], [0.0, 1.0, 0.0], dx * dz)
    if direction == "z+":
        return ([(a_min[0] + a_max[0]) / 2.0, (a_min[1] + a_max[1]) / 2.0, a_max[2]], [0.0, 0.0, 1.0], dx * dy)
    return None


def _point_in_room_xy(x, y, room):
    a_min, a_max = room["_abs_min"], room["_abs_max"]
    return a_min[0] <= x <= a_max[0] and a_min[1] <= y <= a_max[1]


def _build_floor_grid(rooms_on_floor, padding_cells=2):
    """Phase 2: 2D 占用栅格 + 室外 flood-fill + 内庭院 void 检测"""
    if not rooms_on_floor:
        return None

    min_x = min(r["_abs_min"][0] for r in rooms_on_floor)
    max_x = max(r["_abs_max"][0] for r in rooms_on_floor)
    min_y = min(r["_abs_min"][1] for r in rooms_on_floor)
    max_y = max(r["_abs_max"][1] for r in rooms_on_floor)

    origin_x = min_x - padding_cells * CELL_SIZE
    origin_y = min_y - padding_cells * CELL_SIZE
    nx = int(math.ceil((max_x - min_x) / float(CELL_SIZE))) + 2 * padding_cells
    ny = int(math.ceil((max_y - min_y) / float(CELL_SIZE))) + 2 * padding_cells

    grid = [[None for _ in range(ny)] for _ in range(nx)]

    for i in range(nx):
        cx = origin_x + (i + 0.5) * CELL_SIZE
        for j in range(ny):
            cy = origin_y + (j + 0.5) * CELL_SIZE
            for room in rooms_on_floor:
                if _point_in_room_xy(cx, cy, room):
                    grid[i][j] = "occupied"
                    break

    queue = deque()
    for i in range(nx):
        for j in range(ny):
            on_border = i == 0 or i == nx - 1 or j == 0 or j == ny - 1
            if on_border and grid[i][j] is None:
                grid[i][j] = "outdoor"
                queue.append((i, j))

    while queue:
        i, j = queue.popleft()
        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ni, nj = i + di, j + dj
            if 0 <= ni < nx and 0 <= nj < ny and grid[ni][nj] is None:
                grid[ni][nj] = "outdoor"
                queue.append((ni, nj))

    for i in range(nx):
        for j in range(ny):
            if grid[i][j] is None:
                grid[i][j] = "courtyard"

    return {
        "grid": grid,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "cell": CELL_SIZE,
        "nx": nx,
        "ny": ny,
    }


def _probe_exposure_type(grid_info, center, normal):
    """沿法向探测面外侧是室外还是庭院"""
    if abs(normal[2]) > 0.9:
        return "outdoor"

    if grid_info is None:
        return "outdoor"

    step = CELL_SIZE * 0.6
    px = center[0] + normal[0] * step
    py = center[1] + normal[1] * step
    i = int((px - grid_info["origin_x"]) / grid_info["cell"])
    j = int((py - grid_info["origin_y"]) / grid_info["cell"])

    if i < 0 or j < 0 or i >= grid_info["nx"] or j >= grid_info["ny"]:
        return "outdoor"

    cell_type = grid_info["grid"][i][j]
    if cell_type == "courtyard":
        return "courtyard"
    return "outdoor"


def _nearest_modulus_value(value, modulus=MODULUS):
    return round(value / float(modulus)) * modulus


def _snap_mm(value, modulus=MODULUS):
    """距最近 300mm 模数 ≤ 1mm 则吸附，如 5099.43 → 5100"""
    nearest = _nearest_modulus_value(value, modulus)
    if abs(value - nearest) <= COORD_SNAP_TOL:
        return float(nearest)
    return round(value, 2)


def _is_modulus_aligned(value, modulus=MODULUS, tol=COORD_SNAP_TOL):
    rem = value % modulus
    return rem <= tol or rem >= modulus - tol


def _describe_plan_region(norm_min, norm_max, building_size):
    """平面九宫格方位：东南/西北等"""
    cx = (norm_min[0] + norm_max[0]) / 2.0
    cy = (norm_min[1] + norm_max[1]) / 2.0
    bx = building_size.get("x", 0)
    by = building_size.get("y", 0)
    x_pct = (cx / bx * 100.0) if bx > 0 else 50.0
    y_pct = (cy / by * 100.0) if by > 0 else 50.0

    def _band(pct, high, low):
        if pct >= 66:
            return high
        if pct <= 33:
            return low
        return ""

    ns = _band(y_pct, "北", "南")
    ew = _band(x_pct, "东", "西")

    if ns and ew:
        return "{}{}侧".format(ew, ns)
    if ns:
        return "{}侧".format(ns)
    if ew:
        return "{}侧".format(ew)
    return "中部"


def _format_move_hint(label, val, nearest):
    meta = CORNER_QC_META[label]
    delta = nearest - val
    if abs(delta) < 1e-9:
        return None
    direction = meta["pos_dir"] if delta > 0 else meta["neg_dir"]
    return {
        "label": label,
        "axis": meta["axis"],
        "direction": direction,
        "delta": delta,
        "val": val,
        "nearest": int(nearest),
        "text": "{} {}→{}".format(label, round(val, 2), int(nearest)),
    }


def _summarize_move_hints(issues):
    """同轴等距偏移 → 合并为「整体移动」；否则逐条列出"""
    by_axis = {}
    for item in issues:
        by_axis.setdefault(item["axis"], []).append(item)

    parts = []
    for axis in ("X", "Y", "Z"):
        items = by_axis.get(axis)
        if not items:
            continue
        deltas = [round(it["delta"], 1) for it in items]
        if len(set(deltas)) == 1:
            d = items[0]
            parts.append("整体沿 {} 轴向{} {} mm（{}）".format(
                axis, d["direction"], abs(d["delta"]),
                "，".join(it["text"] for it in items)))
        else:
            for it in items:
                parts.append("沿 {} 轴向{} {} mm（{}）".format(
                    axis, it["direction"], abs(it["delta"]), it["text"]))
    return parts


def _check_coordinate_modulus(norm_min, norm_max, layer_name, room_type, floor_num, building_size, errors, warnings):
    """角点坐标模数 QC：同体块合并为一条报错"""
    labels = ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]
    raw_coords = list(norm_min) + list(norm_max)
    type_cn = ROOM_TYPE_CN.get(room_type, room_type)
    region = _describe_plan_region(norm_min, norm_max, building_size)

    float_issues = []
    hard_issues = []

    for val, label in zip(raw_coords, labels):
        if _is_modulus_aligned(val):
            continue
        nearest = _nearest_modulus_value(val)
        drift = abs(val - nearest)
        if drift <= COORD_SNAP_TOL:
            float_issues.append("{} {:.2f}→{}".format(label, val, int(nearest)))
        else:
            hint = _format_move_hint(label, val, nearest)
            if hint:
                hard_issues.append(hint)

    for msg in float_issues:
        warnings.append("【坐标浮点误差】[{}] {}（已自动吸附）".format(layer_name, msg))

    if not hard_issues:
        return

    w = int(round(norm_max[0] - norm_min[0]))
    d = int(round(norm_max[1] - norm_min[1]))
    h = int(round(norm_max[2] - norm_min[2]))
    move_parts = _summarize_move_hints(hard_issues)

    errors.append(
        "【坐标模数错误】第{}层 · {}（{}）· {} · {}×{}×{} mm\n".format(
            floor_num, layer_name, type_cn, region, w, d, h)
        + "\n".join("  → " + p for p in move_parts)
    )


def _axis_overlap_len(a_min, a_max, b_min, b_max, axis):
    return min(a_max[axis], b_max[axis]) - max(a_min[axis], b_min[axis])


def _find_planar_gaps(a_min, a_max, b_min, b_max):
    """检测两 Box 在 X/Y 方向上本应贴邻却出现的小缝隙"""
    gaps = []
    candidates = [
        (0, b_min[0] - a_max[0], "东", "西", "b_east"),
        (0, a_min[0] - b_max[0], "西", "东", "a_east"),
        (1, b_min[1] - a_max[1], "北", "南", "b_south"),
        (1, a_min[1] - b_max[1], "南", "北", "a_south"),
    ]
    seen = set()
    for axis, gap, move_a, move_b, mode in candidates:
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
        axis_name = "X" if axis == 0 else "Y"
        gaps.append({
            "axis": axis_name,
            "axis_idx": axis,
            "mode": mode,
            "gap": round(gap, 1),
            "move_a": move_a,
            "move_b": move_b,
        })
    return gaps


def _gap_slab(a_min, a_max, b_min, b_max, axis, mode):
    """两 Box 之间缝隙区域（可能被细走道等中间体块填充）"""
    if axis == 0 and mode == "b_east":
        return (
            [a_max[0], max(a_min[1], b_min[1]), max(a_min[2], b_min[2])],
            [b_min[0], min(a_max[1], b_max[1]), min(a_max[2], b_max[2])],
        )
    if axis == 0 and mode == "a_east":
        return (
            [b_max[0], max(a_min[1], b_min[1]), max(a_min[2], b_min[2])],
            [a_min[0], min(a_max[1], b_max[1]), min(a_max[2], b_max[2])],
        )
    if axis == 1 and mode == "b_south":
        return (
            [max(a_min[0], b_min[0]), a_max[1], max(a_min[2], b_min[2])],
            [min(a_max[0], b_max[0]), b_min[1], min(a_max[2], b_max[2])],
        )
    if axis == 1 and mode == "a_south":
        return (
            [max(a_min[0], b_min[0]), b_max[1], max(a_min[2], b_min[2])],
            [min(a_max[0], b_max[0]), a_min[1], min(a_max[2], b_max[2])],
        )
    return None


def _is_gap_bridged(slab_min, slab_max, room_entries, exclude_idxs):
    """缝隙区域是否已有第三个体块（如 300mm 宽连接走道）填充"""
    if slab_min is None or slab_max is None:
        return False
    if any(slab_max[d] <= slab_min[d] + TOL for d in range(3)):
        return False
    for idx, room in enumerate(room_entries):
        if idx in exclude_idxs:
            continue
        if _boxes_have_volume_overlap(slab_min, slab_max, room["abs_min"], room["abs_max"]):
            return True
    return False


def _room_region(entry, building_size):
    return _describe_plan_region(entry["norm_min"], entry["norm_max"], building_size)


def _check_room_gaps(room_entries, building_size, errors):
    """检测本应收口贴邻、却因误移产生的小缝隙"""
    reported = set()
    for i, room_a in enumerate(room_entries):
        for j, room_b in enumerate(room_entries[i + 1:], start=i + 1):
            gaps = _find_planar_gaps(
                room_a["abs_min"], room_a["abs_max"],
                room_b["abs_min"], room_b["abs_max"],
            )
            for gap in gaps:
                slab = _gap_slab(
                    room_a["abs_min"], room_a["abs_max"],
                    room_b["abs_min"], room_b["abs_max"],
                    gap["axis_idx"], gap["mode"],
                )
                if _is_gap_bridged(slab[0], slab[1], room_entries, {i, j}):
                    continue

                pair_key = (i, room_b["layer_name"], gap["axis"], gap["gap"])
                if pair_key in reported:
                    continue
                reported.add(pair_key)

                type_a = ROOM_TYPE_CN.get(room_a["type"], room_a["type"])
                type_b = ROOM_TYPE_CN.get(room_b["type"], room_b["type"])
                region_a = _room_region(room_a, building_size)
                region_b = _room_region(room_b, building_size)
                g = gap["gap"]

                errors.append(
                    "【体块缝隙】第{}层 · {}（{}·{}）↔ {}（{}·{}）\n"
                    "  → {} 方向缝隙 {} mm\n"
                    "  → 将 {} 向{} {} mm，或将 {} 向{} {} mm，使贴面闭合".format(
                        room_a["floor"],
                        room_a["layer_name"], type_a, region_a,
                        room_b["layer_name"], type_b, region_b,
                        gap["axis"], g,
                        room_a["layer_name"], gap["move_a"], g,
                        room_b["layer_name"], gap["move_b"], g,
                    ))


def _boxes_have_volume_overlap(a_min, a_max, b_min, b_max, tol=TOL):
    """两 Box 是否存在体积重叠（共面贴邻不算重叠）"""
    for dim in range(3):
        if a_max[dim] <= b_min[dim] + tol or b_max[dim] <= a_min[dim] + tol:
            return False
    return True


def _overlap_intersection(a_min, a_max, b_min, b_max):
    """计算两 Box 交叠区域在各轴上的尺寸"""
    return {
        "size_x": int(round(min(a_max[0], b_max[0]) - max(a_min[0], b_min[0]))),
        "size_y": int(round(min(a_max[1], b_max[1]) - max(a_min[1], b_min[1]))),
        "size_z": int(round(min(a_max[2], b_max[2]) - max(a_min[2], b_min[2]))),
    }


def _separate_hint_x(a_min, a_max, b_min, b_max, amount, room_a, room_b):
    a_cx = (a_min[0] + a_max[0]) / 2.0
    b_cx = (b_min[0] + b_max[0]) / 2.0
    if a_cx <= b_cx:
        west_name, east_name = room_a["layer_name"], room_b["layer_name"]
    else:
        west_name, east_name = room_b["layer_name"], room_a["layer_name"]
    return "沿东西向分离 {} mm（{} 向西 或 {} 向东）".format(amount, west_name, east_name)


def _separate_hint_y(a_min, a_max, b_min, b_max, amount, room_a, room_b):
    a_cy = (a_min[1] + a_max[1]) / 2.0
    b_cy = (b_min[1] + b_max[1]) / 2.0
    if a_cy <= b_cy:
        south_name, north_name = room_a["layer_name"], room_b["layer_name"]
    else:
        south_name, north_name = room_b["layer_name"], room_a["layer_name"]
    return "沿南北向分离 {} mm（{} 向南 或 {} 向北）".format(amount, north_name, south_name)


def _format_overlap_detail(intersection, room_a, room_b):
    sx = intersection["size_x"]
    sy = intersection["size_y"]
    sz = intersection["size_z"]
    a_min, a_max = room_a["abs_min"], room_a["abs_max"]
    b_min, b_max = room_b["abs_min"], room_b["abs_max"]

    lines = ["重叠量：东西向 {} mm；南北向 {} mm；竖向 {} mm".format(sx, sy, sz)]

    if sx > TOL and sy > TOL:
        if sx <= sy:
            lines.append(_separate_hint_x(a_min, a_max, b_min, b_max, sx, room_a, room_b))
        else:
            lines.append(_separate_hint_y(a_min, a_max, b_min, b_max, sy, room_a, room_b))
    elif sx > TOL:
        lines.append(_separate_hint_x(a_min, a_max, b_min, b_max, sx, room_a, room_b))
    elif sy > TOL:
        lines.append(_separate_hint_y(a_min, a_max, b_min, b_max, sy, room_a, room_b))
    elif sz > TOL:
        lines.append("沿竖向分离 {} mm（调整 Z 高度或上下错层）".format(sz))

    return "\n  → ".join(lines)


def _check_room_overlaps(room_entries, building_size, errors):
    """检测体块体积重合"""
    for i, room_a in enumerate(room_entries):
        for room_b in room_entries[i + 1:]:
            if not _boxes_have_volume_overlap(
                room_a["abs_min"], room_a["abs_max"],
                room_b["abs_min"], room_b["abs_max"],
            ):
                continue
            type_a = ROOM_TYPE_CN.get(room_a["type"], room_a["type"])
            type_b = ROOM_TYPE_CN.get(room_b["type"], room_b["type"])
            intersection = _overlap_intersection(
                room_a["abs_min"], room_a["abs_max"],
                room_b["abs_min"], room_b["abs_max"],
            )
            region_a = _room_region(room_a, building_size)
            region_b = _room_region(room_b, building_size)
            overlap_detail = _format_overlap_detail(intersection, room_a, room_b)
            errors.append(
                "【体块重合】第{}层 · {}（{}·{}）↔ 第{}层 · {}（{}·{}）\n"
                "  → {}".format(
                    room_a["floor"], room_a["layer_name"], type_a, region_a,
                    room_b["floor"], room_b["layer_name"], type_b, region_b,
                    overlap_detail,
                ))


def _snap_box(box):
    return [_snap_mm(v) for v in box]


def _room_height(room):
    a_min, a_max = room["_abs_min"], room["_abs_max"]
    return a_max[2] - a_min[2]


def _room_is_double_height(room):
    return _room_height(room) > SINGLE_FLOOR_HEIGHT + TOL


def _is_vertical_normal(normal):
    return abs(normal[2]) > 0.9


def _build_adjacency_detail(rooms):
    """room_id -> [(neighbor_id, outward_direction), ...]"""
    directions = ("x-", "x+", "y-", "y+", "z+")
    reverse = {"x-": "x+", "x+": "x-", "y-": "y+", "y+": "y-", "z+": "z-"}
    adj = {r["id"]: [] for r in rooms}

    for i, room_a in enumerate(rooms):
        for room_b in rooms[i + 1:]:
            for direction in directions:
                if _boxes_share_face(room_a, room_b, direction):
                    adj[room_a["id"]].append((room_b["id"], direction))
                    adj[room_b["id"]].append((room_a["id"], reverse[direction]))
                    break

    return adj


def _allow_vertical_light_passage(room_a, room_b):
    if room_a["type"] == "stairs" or room_b["type"] == "stairs":
        return True
    if _room_is_double_height(room_a) or _room_is_double_height(room_b):
        return True
    return False


def _can_propagate_across_edge(from_room, to_room, surf, edge_direction):
    """传播是否允许穿越该邻接边（建筑物理约束）"""
    if _is_vertical_normal(surf["normal"]):
        return False

    if edge_direction in ("z+", "z-"):
        return _allow_vertical_light_passage(from_room, to_room)

    return True


def _build_adjacency(rooms):
    detail = _build_adjacency_detail(rooms)
    return {k: [n for n, _ in edges] for k, edges in detail.items()}


def _edge_direction_map(adj_detail):
    return {(rid, nid): direction for rid, edges in adj_detail.items() for nid, direction in edges}


def _normal_key(normal):
    return tuple(round(c, 1) for c in normal)


def _merge_surfaces_by_normal(surfaces):
    """合并同法向的采光面，保留最大衰减路径信息"""
    merged = {}
    for surf in surfaces:
        key = _normal_key(surf["normal"])
        if key not in merged:
            merged[key] = dict(surf)
            continue
        merged[key]["area"] = round(merged[key]["area"] + surf["area"], 2)
        if surf.get("attenuation", 0) > merged[key].get("attenuation", 0):
            merged[key]["source"] = surf.get("source")
            merged[key]["hops"] = surf.get("hops", 0)
            merged[key]["attenuation"] = surf.get("attenuation", 1.0)
            merged[key]["path"] = surf.get("path", [])
    return list(merged.values())


def compute_direct_lighting(rooms):
    """Phase 1 + 2: 邻接面剔除 + 庭院 void 判定"""
    floor_groups = {}
    for room in rooms:
        floor_groups.setdefault(room["floor"], []).append(room)

    floor_grids = {}
    for floor, floor_rooms in floor_groups.items():
        floor_grids[floor] = _build_floor_grid(floor_rooms)

    face_directions = ("x-", "x+", "y-", "y+", "z+")

    for room in rooms:
        direct_surfaces = []
        grid_info = floor_grids.get(room["floor"])

        for direction in face_directions:
            if _face_has_neighbor(room, direction, rooms):
                continue

            geom = _face_geometry(room, direction)
            if not geom:
                continue

            center, normal, area = geom
            if area <= 0:
                continue

            exposure_type = _probe_exposure_type(grid_info, center, normal)
            direct_surfaces.append({
                "normal": normal,
                "area": round(area, 2),
                "exposure_type": exposure_type,
            })

        room["direct_lighting_surfaces"] = direct_surfaces
        room["lighting_surfaces"] = direct_surfaces


def _can_traverse(room):
    return room["type"] in TRANSIT_TYPES


def _can_receive_indirect(room):
    return room["type"] not in BLOCKER_TYPES and room["lighting_priority"] > 0


def propagate_effective_lighting(rooms):
    """Phase 3: 经玄关/走道/阳台的间接采光传播（含垂直/屋顶约束）"""
    room_by_id = {r["id"]: r for r in rooms}
    adj_detail = _build_adjacency_detail(rooms)
    adjacency = {k: [n for n, _ in edges] for k, edges in adj_detail.items()}
    edge_dirs = _edge_direction_map(adj_detail)

    for room in rooms:
        room["effective_lighting"] = []
        room["lighting_access"] = "none"

    for room in rooms:
        if room["direct_lighting_surfaces"]:
            room["lighting_access"] = "direct"
            room["effective_lighting"] = [
                {
                    "normal": surf["normal"],
                    "area": surf["area"],
                    "exposure_type": surf.get("exposure_type", "outdoor"),
                    "source": room["id"],
                    "hops": 0,
                    "attenuation": 1.0,
                    "path": [room["id"]],
                }
                for surf in room["direct_lighting_surfaces"]
            ]

    queue = deque()
    for room in rooms:
        if not room["direct_lighting_surfaces"]:
            continue
        for surf in room["direct_lighting_surfaces"]:
            if _is_vertical_normal(surf["normal"]):
                continue
            queue.append((room["id"], [room["id"]], 1.0, surf))

    best_indirect = {}

    while queue:
        current_id, path, atten, surf = queue.popleft()
        if len(path) > MAX_LIGHT_HOPS:
            continue

        current = room_by_id[current_id]

        for neighbor_id in adjacency.get(current_id, []):
            if neighbor_id in path:
                continue

            neighbor = room_by_id[neighbor_id]
            if neighbor["type"] in BLOCKER_TYPES:
                continue

            edge_dir = edge_dirs.get((current_id, neighbor_id))
            if edge_dir is None:
                continue
            if not _can_propagate_across_edge(current, neighbor, surf, edge_dir):
                continue

            if _can_traverse(neighbor):
                hop_factor = ATTENUATION.get(neighbor["type"], 0.5)
                new_atten = atten * hop_factor
                if new_atten >= MIN_EFFECTIVE_ATTENUATION:
                    queue.append((neighbor_id, path + [neighbor_id], new_atten, surf))

            if _can_receive_indirect(neighbor):
                if len(path) == 1 and _can_traverse(current):
                    deliver_atten = atten * ATTENUATION.get(current["type"], 0.7)
                elif _can_traverse(current):
                    deliver_atten = atten
                else:
                    continue

                if deliver_atten < MIN_EFFECTIVE_ATTENUATION:
                    continue

                new_path = path + [neighbor_id]
                if len(new_path) <= 1:
                    continue

                key = (neighbor_id, _normal_key(surf["normal"]))
                candidate = {
                    "normal": surf["normal"],
                    "area": round(surf["area"] * deliver_atten, 2),
                    "exposure_type": surf.get("exposure_type", "outdoor"),
                    "source": path[0],
                    "hops": len(new_path) - 1,
                    "attenuation": round(deliver_atten, 3),
                    "path": new_path,
                }
                prev = best_indirect.get(key)
                if prev is None or candidate["attenuation"] > prev["attenuation"]:
                    best_indirect[key] = candidate

    for (room_id, normal_key), surf in best_indirect.items():
        room = room_by_id[room_id]
        has_direct_same_normal = any(
            _normal_key(s["normal"]) == normal_key for s in room.get("direct_lighting_surfaces", [])
        )
        if has_direct_same_normal:
            continue

        if room["lighting_access"] == "none":
            room["lighting_access"] = "indirect"
        room["effective_lighting"].append(surf)

    for room in rooms:
        room["effective_lighting"] = _merge_surfaces_by_normal(room["effective_lighting"])


def analyze_lighting(rooms):
    compute_direct_lighting(rooms)
    propagate_effective_lighting(rooms)


def _all_lighting_surfaces(room):
    """合并 direct + effective，供朝向 QC 使用"""
    surfaces = list(room.get("direct_lighting_surfaces", []))
    for surf in room.get("effective_lighting", []):
        if surf.get("hops", 0) > 0:
            surfaces.append(surf)
    return surfaces


def export_and_qc_rhino_dataset():
    if rs is None:
        raise RuntimeError("此脚本需在 Rhino 环境中运行（缺少 rhinoscriptsyntax）")

    objects = rs.NormalObjects()
    if not objects:
        rs.MessageBox("模型中没有可视对象！请确保体块未被隐藏或锁定。", 0, "错误")
        return

    errors = []
    warnings = []
    rooms = []
    room_counts = {}

    valid_objs = []
    has_bad_geometry = False
    bad_reasons = []
    skipped_function_layers = {}

    for obj in objects:
        layer_full_name = rs.ObjectLayer(obj)
        layer_name = layer_full_name.split("::")[-1]
        clean_type = _parse_layer_room_type(layer_name)

        if clean_type is None:
            if _layer_looks_like_function_space(layer_name):
                skipped_function_layers[layer_name] = skipped_function_layers.get(layer_name, 0) + 1
            continue

        obj_type = rs.ObjectType(obj)
        if obj_type in [16, 32, 1073741824]:
            if rs.IsObjectSolid(obj):
                valid_objs.append(obj)
            else:
                has_bad_geometry = True
                bad_reasons.append("【{}】层有空心体/未盖面".format(layer_name))
        else:
            has_bad_geometry = True
            bad_reasons.append("【{}】层混入了散线/单面".format(layer_name))

    for layer_name, count in skipped_function_layers.items():
        errors.append(
            "【图层未识别】[{}] 含 {} 个体块未被导出\n"
            "  → 当前图层名无法解析，推荐格式：11multi_purpose-多功能室 或 11multi_purpose多功能室".format(
                layer_name, count))

    if has_bad_geometry:
        msg = "⛔ 严重错误：功能图层内存在杂物！\n\n" + "\n".join(list(set(bad_reasons))) + "\n\n请加盖(_Cap)或清理散线。"
        rs.MessageBox(msg, 0, "体块不合格")
        return

    if not valid_objs:
        rs.MessageBox("没有找到任何有效的功能体块！", 0, "错误")
        return

    overall_bbox = rs.BoundingBox(valid_objs)
    global_min_x, global_min_y, global_min_z = overall_bbox[0].X, overall_bbox[0].Y, overall_bbox[0].Z
    global_max_x, global_max_y, global_max_z = overall_bbox[6].X, overall_bbox[6].Y, overall_bbox[6].Z
    building_size = {
        "x": global_max_x - global_min_x,
        "y": global_max_y - global_min_y,
        "z": global_max_z - global_min_z,
    }
    room_entries = []

    for i, obj in enumerate(valid_objs):
        layer_full_name = rs.ObjectLayer(obj)
        layer_name = layer_full_name.split("::")[-1]
        clean_type = _parse_layer_room_type(layer_name)
        if clean_type is None:
            continue

        if clean_type not in room_counts:
            room_counts[clean_type] = 0
        room_counts[clean_type] += 1

        bbox = rs.BoundingBox(obj)
        pt_min, pt_max = bbox[0], bbox[6]

        dx = round(pt_max.X - pt_min.X)
        dy = round(pt_max.Y - pt_min.Y)
        dz = round(pt_max.Z - pt_min.Z)

        bbox_vol = dx * dy * dz
        if bbox_vol > 0:
            try:
                if rs.IsMesh(obj):
                    actual_vol_data = rs.MeshVolume(obj)
                    actual_vol = actual_vol_data[1] if actual_vol_data else None
                else:
                    actual_vol_data = rs.SurfaceVolume(obj)
                    actual_vol = actual_vol_data[0] if actual_vol_data else None

                if actual_vol and (actual_vol / bbox_vol < 0.95):
                    errors.append("【异型错误】[{}] 图层检测到 L/T型 或挖洞体块！\n请严格像切豆腐一样把它切分为纯正的矩形 Box！".format(layer_name))
            except Exception:
                pass

        for dim, axis in zip([dx, dy, dz], ["长(X)", "宽(Y)", "高(Z)"]):
            if not _is_modulus_aligned(dim):
                errors.append("【模数错误】[{}] 的 {} 尺寸 {}mm 不符合 300mm 模数！".format(layer_name, axis, dim))

        if clean_type == "stairs" and not (5700 <= dz <= 6300):
            errors.append("【高度错误】楼梯高度为 {}mm (强制要求 5700~6300mm)！若是单层起步台阶，请降级移至 [过道] 图层！".format(dz))
        if clean_type == "corridor" and dz > 3600:
            errors.append("【高度错误】过道高度检测为 {}mm！(单层空间不能超过 3600mm，请切分！)".format(dz))
        if clean_type == "balcony" and dz > 3600:
            errors.append("【高度错误】阳台/露台高度检测为 {}mm！不允许超过 3600mm。".format(dz))

        floor_num = 2 if (pt_min.Z - global_min_z) > 1500 else 1

        raw_norm_min = [pt_min.X - global_min_x, pt_min.Y - global_min_y, pt_min.Z - global_min_z]
        raw_norm_max = [pt_max.X - global_min_x, pt_max.Y - global_min_y, pt_max.Z - global_min_z]
        abs_min = _snap_box([pt_min.X, pt_min.Y, pt_min.Z])
        abs_max = _snap_box([pt_max.X, pt_max.Y, pt_max.Z])
        norm_min = _snap_box(raw_norm_min)
        norm_max = _snap_box(raw_norm_max)

        _check_coordinate_modulus(
            raw_norm_min, raw_norm_max, layer_name, clean_type, floor_num, building_size, errors, warnings
        )

        room_entries.append({
            "layer_name": layer_name,
            "type": clean_type,
            "floor": floor_num,
            "abs_min": abs_min,
            "abs_max": abs_max,
            "norm_min": raw_norm_min,
            "norm_max": raw_norm_max,
        })

        rooms.append({
            "id": "room_{}".format(i),
            "type": clean_type,
            "floor": floor_num,
            "lighting_priority": LIGHTING_PRIORITY.get(clean_type, 0),
            "_abs_min": abs_min,
            "_abs_max": abs_max,
            "box_min": norm_min,
            "box_max": norm_max,
        })

    _check_room_overlaps(room_entries, building_size, errors)
    _check_room_gaps(room_entries, building_size, errors)

    required = ["living_room", "bedroom", "dining_room", "bathroom"]
    missing = [r for r in required if r not in room_counts]
    if missing:
        errors.append("【功能缺失】缺少必需空间：{}。".format(", ".join(missing)))

    if errors:
        msg = "⛔ 严重错误：请修正后再导出！(已拦截)\n\n" + "\n\n".join(errors[:8])
        if len(errors) > 8:
            msg += "\n\n...(还有更多错误，请先修改上述问题)"
        rs.MessageBox(msg, 0, "AI 严苛质检 V13.7")
        return

    analyze_lighting(rooms)

    courtyard_count = 0
    indirect_count = 0
    for room in rooms:
        for surf in room.get("direct_lighting_surfaces", []):
            if surf.get("exposure_type") == "courtyard":
                courtyard_count += 1
        if room.get("lighting_access") == "indirect":
            indirect_count += 1
        del room["_abs_min"]
        del room["_abs_max"]

    orientation_msg = "未知"
    if "living_room" in room_counts:
        for r in rooms:
            if r["type"] == "living_room":
                all_surf = _all_lighting_surfaces(r)
                has_south_direct = any(s["normal"][1] == -1.0 for s in r.get("direct_lighting_surfaces", []))
                has_south_any = any(s["normal"][1] == -1.0 for s in all_surf)
                access = r.get("lighting_access", "none")

                if has_south_direct:
                    orientation_msg = "南向基准确立 (客厅有直接南向采光) ☀️"
                elif has_south_any and access == "indirect":
                    orientation_msg = "南向基准间接成立 (客厅经玄关/走道获得南向采光) 🌤️"
                else:
                    orientation_msg = "非正南朝向或南向采光不足 ❄️"
                break

    ordered_keys = [
        "entryway", "living_room", "dining_room", "kitchen",
        "bathroom", "bedroom", "corridor", "stairs",
        "balcony", "utility", "multi_purpose"
    ]

    name_map = {
        "entryway": "01 玄关", "living_room": "02 客厅", "dining_room": "03 餐厅",
        "kitchen": "04 厨房", "bathroom": "05 卫生间", "bedroom": "06 卧室",
        "corridor": "07 过道", "stairs": "08 楼梯", "balcony": "09 阳台/露台",
        "utility": "10 储藏/设备", "multi_purpose": "11 多功能房"
    }

    count_strs = []
    for key in ordered_keys:
        if key in room_counts:
            count_strs.append("   - {}: {} 个".format(name_map[key], room_counts[key]))

    count_display = "\n".join(count_strs)

    confirm_msg = "✅ 质检通过！准备导出数据。\n\n"
    confirm_msg += "🧭 {}\n\n".format(orientation_msg)
    confirm_msg += "💡 采光分析: {} 间间接采光, {} 面庭院采光\n\n".format(indirect_count, courtyard_count)
    if warnings:
        confirm_msg += "⚠️ 坐标浮点吸附 {} 条（≤1mm 误差已自动修正）\n\n".format(len(set(warnings)))
    confirm_msg += "📊 请核对当前识别到的功能体块（共 {} 个）：\n{}\n\n".format(len(rooms), count_display)
    confirm_msg += "👉 数量如果正确，是否确认导出 JSON 文件？"

    response = rs.MessageBox(confirm_msg, 4 | 32, "确认导出清单")
    if response != 6:
        return

    doc_name = rs.DocumentName()
    if doc_name:
        base_name = doc_name.split('.')[0]
        match = re.search(r'\d+', base_name)
        if match:
            suggested_name = "house_{}.json".format(match.group())
        else:
            suggested_name = "house_{}.json".format(base_name)
    else:
        suggested_name = "house_001.json"

    metadata = {
        "total_rooms": len(rooms),
        "stats": room_counts,
        "constraints": {
            "pure_box_enforced": True,
            "origin_aligned_auto": True,
            "modulus": 300,
            "lighting_analysis": "adjacency_void_propagation_v2",
            "coord_modulus_qc": True,
            "overlap_qc": True,
            "gap_qc": True,
        },
        "building_size": {
            "x": round(global_max_x - global_min_x),
            "y": round(global_max_y - global_min_y),
            "z": round(global_max_z - global_min_z)
        }
    }

    output = {"house_id": suggested_name.replace('.json', ''), "metadata": metadata, "rooms": rooms}

    save_path = rs.SaveFileName("保存合格的 AI 数据集", "JSON Files (*.json)|*.json||", filename=suggested_name)

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)
        rs.MessageBox("🎉 JSON 数据已成功保存！", 0, "导出完成")


if __name__ == "__main__":
    export_and_qc_rhino_dataset()
