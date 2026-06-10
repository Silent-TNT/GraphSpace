from __future__ import annotations

import random

import networkx as nx
import numpy as np

try:
    from .config import DEFAULT_ROOM_SIZE, ROOM_TYPES, VOXEL_SIZE
except ImportError:
    from config import DEFAULT_ROOM_SIZE, ROOM_TYPES, VOXEL_SIZE


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


def program_floor_room_targets(room_counts: dict) -> dict[int, dict[str, int]]:
    """按程序拓扑规则，统计每层各房间类型的目标数量。"""
    targets: dict[int, dict[str, int]] = {1: {}, 2: {}}
    bath_i = corr_i = 0
    for r_type, count in room_counts.items():
        n = int(count)
        if n <= 0:
            continue
        if r_type == "stairs":
            targets[1][r_type] = targets[1].get(r_type, 0) + n
            continue
        for _ in range(n):
            if r_type in ["entryway", "living_room", "dining_room", "kitchen"]:
                floor = 1
            elif r_type in ["bedroom", "balcony"]:
                floor = 2
            elif r_type == "bathroom":
                floor = 1 if bath_i % 2 == 0 else 2
                bath_i += 1
            elif r_type == "corridor":
                floor = 1 if corr_i % 2 == 0 else 2
                corr_i += 1
            else:
                floor = 1
            targets[floor][r_type] = targets[floor].get(r_type, 0) + 1
    return targets


def build_program_topology(room_counts, seed=42):
    G = nx.Graph()
    nodes = []
    bath_i = corr_i = 0
    for r_type, count in room_counts.items():
        for i in range(count):
            nid = f"{r_type}_{i}"
            if r_type in ["entryway", "living_room", "dining_room", "kitchen"]:
                floor = 1
            elif r_type in ["bedroom", "balcony"]:
                floor = 2
            elif r_type == "bathroom":
                floor = 1 if bath_i % 2 == 0 else 2
                bath_i += 1
            elif r_type == "corridor":
                floor = 1 if corr_i % 2 == 0 else 2
                corr_i += 1
            elif r_type == "stairs":
                floor = "1&2"
            else:
                floor = 1
            nodes.append((nid, r_type, floor))
            G.add_node(nid, type=r_type, floor=floor)

    f1_corr = [n for n, t, f in nodes if t == "corridor" and f == 1]
    f2_corr = [n for n, t, f in nodes if t == "corridor" and f == 2]
    f1_hub = f1_corr[0] if f1_corr else next((n for n, t, f in nodes if t == "living_room"), nodes[0][0])
    f2_hub = f2_corr[0] if f2_corr else next((n for n, t, f in nodes if t == "bedroom"), nodes[-1][0])

    edge_types = {}
    for nid, r_type, floor in nodes:
        if r_type == "stairs":
            continue
        hub = f1_hub if floor == 1 else f2_hub
        if nid != hub:
            G.add_edge(nid, hub)
            edge_types[(nid, hub)] = "horizontal"
            edge_types[(hub, nid)] = "horizontal"

    stairs = [n for n, t, f in nodes if t == "stairs"]
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
    rng = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))
    G, pos, nodes, edge_types = build_program_topology(user_req["room_counts"], seed=seed)
    sx, sy = user_req["site_x"], user_req["site_y"]
    rooms = []
    for nid, r_type, floor in nodes:
        w, d, h = DEFAULT_ROOM_SIZE.get(r_type, (3600, 3600, 3000))
        px, py = pos[nid]
        jx = rng.uniform(-0.12, 0.12)
        jy = rng.uniform(-0.12, 0.12)
        cx = (px + 1) / 2 * (sx * 0.7) + sx * 0.15 + jx * sx * 0.08
        cy = (py + 1) / 2 * (sy * 0.7) + sy * 0.15 + jy * sy * 0.08
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
