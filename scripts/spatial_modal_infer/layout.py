from __future__ import annotations

import random

import networkx as nx
import numpy as np

try:
    from .config import DEFAULT_ROOM_SIZE, ROOM_TYPES, VOXEL_SIZE
except ImportError:
    from config import DEFAULT_ROOM_SIZE, ROOM_TYPES, VOXEL_SIZE


# 固定楼层的房间类型
FIXED_FLOOR_1 = {"entryway", "living_room", "dining_room", "kitchen"}
FIXED_FLOOR_2 = {"bedroom", "balcony"}
FLEXIBLE_TYPES = {"bathroom", "corridor", "utility", "multi_purpose"}

# 互补房间对——这些房间类型之间更倾向有直接边
COMPLEMENTARY_PAIRS = {
    ("bedroom", "bathroom"), ("kitchen", "dining_room"),
    ("living_room", "dining_room"), ("living_room", "entryway"),
    ("bedroom", "balcony"), ("kitchen", "utility"),
    ("corridor", "living_room"), ("corridor", "bedroom"),
}


def _are_complementary(t1: str, t2: str) -> bool:
    return (t1, t2) in COMPLEMENTARY_PAIRS or (t2, t1) in COMPLEMENTARY_PAIRS


def _assign_floors(room_counts: dict, seed: int) -> list[tuple[str, int, str]]:
    """
    基于种子的楼层分配，返回 [(nid, room_type, floor), ...]。
    种子不同 → 灵活房间的楼层分布不同 → 图结构不同。
    """
    rng = random.Random(seed)
    entries: list[tuple[str, int, str]] = []

    # 固定楼层房间
    for r_type in FIXED_FLOOR_1:
        for i in range(room_counts.get(r_type, 0)):
            entries.append((f"{r_type}_{i}", r_type, 1))

    for r_type in FIXED_FLOOR_2:
        for i in range(room_counts.get(r_type, 0)):
            entries.append((f"{r_type}_{i}", r_type, 2))

    # 灵活楼层房间：用种子 shuffle 分配
    flexible: list[tuple[str, int]] = []
    for r_type in FLEXIBLE_TYPES:
        for i in range(room_counts.get(r_type, 0)):
            flexible.append((r_type, i))
    rng.shuffle(flexible)

    mid = len(flexible) // 2
    for j, (r_type, i) in enumerate(flexible):
        floor = 1 if j < mid else 2
        entries.append((f"{r_type}_{i}", r_type, floor))

    # 楼梯
    for i in range(room_counts.get("stairs", 0)):
        entries.append((f"stairs_{i}", "stairs", "1&2"))

    return entries


def snap_modulus(v: float) -> float:
    return round(float(v) / VOXEL_SIZE) * VOXEL_SIZE


def build_user_request(site_x, site_y, room_counts, site_z=6000):
    counts = {k: int(v) for k, v in room_counts.items() if int(v) > 0}
    return {
        "site_x": float(site_x),
        "site_y": float(site_y),
        "site_z": float(site_z),
        "room_counts": counts,
    }


def program_floor_room_targets(room_counts: dict, seed: int = 42) -> dict[int, dict[str, int]]:
    """
    按种子分配楼层，统计每层各房间类型的目标数量。
    与 build_program_topology 使用相同的 _assign_floors 逻辑，
    保证生成和评分对同一楼层分布的认知一致。
    """
    targets: dict[int, dict[str, int]] = {1: {}, 2: {}}
    for _nid, r_type, floor in _assign_floors(room_counts, seed):
        if r_type == "stairs":
            targets[1][r_type] = targets[1].get(r_type, 0) + 1
        elif isinstance(floor, int):
            targets[floor][r_type] = targets[floor].get(r_type, 0) + 1
    return targets


def build_program_topology(room_counts, seed=42):
    """
    基于种子构建异构图拓扑。
    种子决定：楼层分配、hub 选择、额外连边 → 不同种子产生真正的不同图结构。
    """
    rng = random.Random(seed)
    G = nx.Graph()

    # Step 1: 种子控制的楼层分配
    entries = _assign_floors(room_counts, seed)
    nodes = [(nid, r_type, floor) for nid, r_type, floor in entries]
    for nid, r_type, floor in entries:
        G.add_node(nid, type=r_type, floor=floor)

    # Step 2: 种子控制的 hub 选择（不同种子可能选不同房间做 hub）
    f1_hub_candidates = [
        n for n, t, f in entries
        if f == 1 and t in ("corridor", "living_room", "entryway")
    ]
    f2_hub_candidates = [
        n for n, t, f in entries
        if f == 2 and t in ("corridor", "bedroom")
    ]

    if f1_hub_candidates:
        f1_hub = rng.choice(f1_hub_candidates)
    else:
        f1_hub = next((n for n, t, f in entries if f == 1), entries[0][0])

    if f2_hub_candidates:
        f2_hub = rng.choice(f2_hub_candidates)
    else:
        f2_hub = next((n for n, t, f in entries if f == 2), entries[-1][0])

    # Step 3: 种子控制图拓扑形态
    # 不同种子使用不同拓扑模式：star / tree / small-world
    edge_types = {}
    topology_mode = seed % 3  # 0=star, 1=tree, 2=small-world

    f1_rooms = [(n, t) for n, t, f in entries if f == 1 and t != "stairs"]
    f2_rooms = [(n, t) for n, t, f in entries if f == 2 and t != "stairs"]

    for floor_label, floor_rooms, hub in [("F1", f1_rooms, f1_hub), ("F2", f2_rooms, f2_hub)]:
        nids = [n for n, t in floor_rooms]
        if len(nids) <= 1:
            continue

        if topology_mode == 0:
            # Star: 所有房间连到 hub（原始行为）
            for nid in nids:
                if nid != hub:
                    G.add_edge(nid, hub)
                    edge_types[(nid, hub)] = "horizontal"
                    edge_types[(hub, nid)] = "horizontal"

        elif topology_mode == 1:
            # Tree: hub 作为根，种子控制 BFS 顺序生成树
            rng.shuffle(nids)
            connected = {hub}
            remaining = [n for n in nids if n != hub]
            rng.shuffle(remaining)
            for nid in remaining:
                parent = rng.choice(list(connected))
                G.add_edge(nid, parent)
                edge_types[(nid, parent)] = "horizontal"
                edge_types[(parent, nid)] = "horizontal"
                connected.add(nid)

        else:  # topology_mode == 2
            # Small-world: 环 + 种子控制随机捷径
            shuffled = list(nids)
            rng.shuffle(shuffled)
            for i in range(len(shuffled)):
                n1 = shuffled[i]
                n2 = shuffled[(i + 1) % len(shuffled)]
                if n1 != n2:
                    G.add_edge(n1, n2)
                    edge_types[(n1, n2)] = "horizontal"
                    edge_types[(n2, n1)] = "horizontal"
            # 随机捷径
            for i in range(len(shuffled)):
                for j in range(i + 2, len(shuffled)):
                    if rng.random() < 0.25:
                        n1, n2 = shuffled[i], shuffled[j]
                        if not G.has_edge(n1, n2):
                            G.add_edge(n1, n2)
                            edge_types[(n1, n2)] = "horizontal"
                            edge_types[(n2, n1)] = "horizontal"

        # 无论哪种拓扑，互补房间获得额外边
        for i in range(len(floor_rooms)):
            for j in range(i + 1, len(floor_rooms)):
                n1, t1 = floor_rooms[i]
                n2, t2 = floor_rooms[j]
                if _are_complementary(t1, t2) and rng.random() < 0.5:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
                        edge_types[(n1, n2)] = "horizontal"
                        edge_types[(n2, n1)] = "horizontal"

    # 楼梯跨层连接
    stairs = [n for n, t, f in entries if t == "stairs"]
    if stairs:
        st = stairs[0]
        if f1_hub != st:
            G.add_edge(st, f1_hub)
            edge_types[(st, f1_hub)] = "vertical"
        if f2_hub != st and f2_hub != f1_hub:
            G.add_edge(st, f2_hub)
            edge_types[(st, f2_hub)] = "vertical"

    pos = nx.spring_layout(G, seed=seed, k=1.2)
    return G, pos, nodes, edge_types


def layout_rooms_from_program(user_req, seed=42):
    """
    基于种子生成房间布局。
    种子控制：图拓扑(楼层/hub/边)、spring_layout 位置、抖动幅度、房间尺寸微调。
    """
    rng = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))

    G, pos, nodes, edge_types = build_program_topology(user_req["room_counts"], seed=seed)
    sx, sy = user_req["site_x"], user_req["site_y"]
    rooms = []

    for nid, r_type, floor in nodes:
        base_w, base_d, base_h = DEFAULT_ROOM_SIZE.get(r_type, (3600, 3600, 3000))

        # 种子控制的房间尺寸微调 (+-25%, 比原来的 0% 变化更大)
        w_scale = 1.0 + rng.uniform(-0.25, 0.25)
        d_scale = 1.0 + rng.uniform(-0.25, 0.25)
        w = base_w * w_scale
        d = base_d * d_scale
        h = base_h

        px, py = pos[nid]
        # 种子控制的抖动（+-0.30 比原来 +-0.12 更显著）
        jx = rng.uniform(-0.30, 0.30)
        jy = rng.uniform(-0.30, 0.30)
        cx = (px + 1) / 2 * (sx * 0.7) + sx * 0.15 + jx * sx * 0.12
        cy = (py + 1) / 2 * (sy * 0.7) + sy * 0.15 + jy * sy * 0.12
        cx, cy = snap_modulus(cx), snap_modulus(cy)
        w, d = snap_modulus(w), snap_modulus(d)
        z0 = 0 if floor == 1 or floor == "1&2" else 3000
        z1 = z0 + h

        rooms.append(
            {
                "id": nid,
                "type": r_type,
                "floor": 1 if floor == "1&2" else floor,
                "box_min": [max(0, cx - w / 2), max(0, cy - d / 2), z0],
                "box_max": [min(sx, cx + w / 2), min(sy, cy + d / 2), z1],
                "lighting_access": "direct" if r_type in ["living_room", "bedroom", "balcony"] else "indirect",
                "lighting_priority": 8 if r_type in ["living_room", "bedroom"] else 4,
                "effective_lighting": [],
            }
        )
    return rooms, G, pos, edge_types


def request_to_house_json(user_req, rooms):
    stats = {t: 0 for t in ROOM_TYPES}
    for r in rooms:
        stats[r["type"]] = stats.get(r["type"], 0) + 1
    return {
        "metadata": {
            "stats": stats,
            "total_rooms": len(rooms),
            "building_size": {
                "x": user_req["site_x"],
                "y": user_req["site_y"],
                "z": user_req["site_z"],
            },
            "constraints": {"modulus": int(VOXEL_SIZE)},
        },
        "rooms": rooms,
    }
