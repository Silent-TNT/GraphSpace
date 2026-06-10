from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GraphDataLoader

try:
    from .config import (
        CHANNEL_MAP, COND_COUNT_SCALE, COND_DIM, LIGHTING_ACCESS_MAP, NODE_IN_DIM, NUM_CHANNELS,
        RES_X, RES_Y, RES_Z, ROOM_TYPES, VOXEL_SIZE,
    )
except ImportError:
    from config import (
        CHANNEL_MAP, COND_COUNT_SCALE, COND_DIM, LIGHTING_ACCESS_MAP, NODE_IN_DIM, NUM_CHANNELS,
        RES_X, RES_Y, RES_Z, ROOM_TYPES, VOXEL_SIZE,
    )


def get_node_supertype(room_type: str) -> str:
    if room_type in ["bedroom", "living_room", "dining_room", "multi_purpose"]:
        return "living"
    if room_type in ["kitchen", "bathroom", "utility", "balcony"]:
        return "service"
    return "circulation"


def check_hetero_adjacency(r1, r2, tol=150, min_shared_len=300):
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


def effective_lighting_ratio(room) -> float:
    dx = room["box_max"][0] - room["box_min"][0]
    dy = room["box_max"][1] - room["box_min"][1]
    floor_area = max(dx * dy, 1.0)
    eff = room.get("effective_lighting", [])
    total = sum(float(e.get("area", 0)) * float(e.get("attenuation", 1.0)) for e in eff)
    return min(total / floor_area, 5.0) / 5.0


def build_room_features(room, build_min, b_size):
    dx = room["box_max"][0] - room["box_min"][0]
    dy = room["box_max"][1] - room["box_min"][1]
    area = (dx * dy) / 1e6
    aspect = max(dx, dy) / max(min(dx, dy), 1.0)
    cx = ((room["box_min"][0] + room["box_max"][0]) / 2 - build_min[0]) / (b_size[0] + 1e-5)
    cy = ((room["box_min"][1] + room["box_max"][1]) / 2 - build_min[1]) / (b_size[1] + 1e-5)
    cz = ((room["box_min"][2] + room["box_max"][2]) / 2 - build_min[2]) / (b_size[2] + 1e-5)
    lighting_access = LIGHTING_ACCESS_MAP.get(room.get("lighting_access", "none"), 0.0)
    lighting_priority = float(room.get("lighting_priority", 0)) / 10.0
    lighting_ratio = effective_lighting_ratio(room)
    return [area, aspect, cx, cy, cz, lighting_access, lighting_priority, lighting_ratio]


def build_condition_vector(data) -> list[float]:
    meta = data.get("metadata", {})
    stats = meta.get("stats", {})
    bsize = meta.get("building_size", {"x": 1.0, "y": 1.0, "z": 1.0})
    rooms = data.get("rooms", [])
    cond = [
        float(bsize.get("x", 1.0)) / 30000.0,
        float(bsize.get("y", 1.0)) / 30000.0,
        float(bsize.get("z", 1.0)) / 9000.0,
    ]
    for rt in ROOM_TYPES:
        cond.append(float(stats.get(rt, 0)) / COND_COUNT_SCALE)
    direct = indirect = none = 0
    for r in rooms:
        acc = r.get("lighting_access", "none")
        if acc == "direct":
            direct += 1
        elif acc == "indirect":
            indirect += 1
        else:
            none += 1
    cond.extend([
        direct / COND_COUNT_SCALE,
        indirect / COND_COUNT_SCALE,
        none / COND_COUNT_SCALE,
    ])
    return cond


def _append_hetero_edge(edges_dict, src_t, rel, dst_t, si, di):
    key = (src_t, rel, dst_t)
    edges_dict.setdefault(key, []).append([si, di])


def merge_program_topology_edges(rooms, program_graph, program_edge_types, id_to_idx, edges_dict):
    """用户条件生成时，用程序拓扑补全几何邻接缺失的边，避免孤立节点。"""
    if program_graph is None:
        return
    for u, v in program_graph.edges:
        if u not in id_to_idx or v not in id_to_idx:
            continue
        t1, idx1 = id_to_idx[u]
        t2, idx2 = id_to_idx[v]
        rel = (program_edge_types or {}).get((u, v), "horizontal")
        if rel not in ("horizontal", "vertical"):
            rel = "horizontal"
        for src_t, dst_t, si, di in ((t1, t2, idx1, idx2), (t2, t1, idx2, idx1)):
            _append_hetero_edge(edges_dict, src_t, rel, dst_t, si, di)


def json_to_sample(data, program_graph=None, program_edge_types=None):
    rooms = data.get("rooms", [])
    if not rooms:
        return None
    all_coords = np.array([r["box_min"] for r in rooms] + [r["box_max"] for r in rooms])
    build_min = all_coords.min(axis=0)
    build_max = all_coords.max(axis=0)
    b_size = build_max - build_min

    hetero = HeteroData()
    nodes_dict = {"living": [], "service": [], "circulation": []}
    id_to_idx = {}
    for r in rooms:
        ntype = get_node_supertype(r["type"])
        id_to_idx[r["id"]] = (ntype, len(nodes_dict[ntype]))
        nodes_dict[ntype].append(build_room_features(r, build_min, b_size))
    for ntype, feats in nodes_dict.items():
        hetero[ntype].x = torch.tensor(feats, dtype=torch.float32) if feats else torch.empty((0, NODE_IN_DIM))

    edges_dict = {}
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            r1, r2 = rooms[i], rooms[j]
            rel = check_hetero_adjacency(r1, r2)
            if not rel:
                continue
            t1, idx1 = id_to_idx[r1["id"]]
            t2, idx2 = id_to_idx[r2["id"]]
            for src_t, dst_t, si, di in ((t1, t2, idx1, idx2), (t2, t1, idx2, idx1)):
                key = (src_t, rel, dst_t)
                edges_dict.setdefault(key, []).append([si, di])
    merge_program_topology_edges(
        rooms, program_graph, program_edge_types, id_to_idx, edges_dict
    )
    for key, elist in edges_dict.items():
        hetero[key].edge_index = torch.tensor(elist, dtype=torch.long).t().contiguous()

    grid = np.zeros((RES_X, RES_Y, RES_Z), dtype=np.int8)
    phys_center_xy = (build_min[:2] + build_max[:2]) / 2.0
    offset_xy = np.array([RES_X * VOXEL_SIZE / 2, RES_Y * VOXEL_SIZE / 2]) - phys_center_xy
    z_min_phys = build_min[2]
    for r in rooms:
        ch = CHANNEL_MAP.get(r["type"], 0)
        ix_min = int((r["box_min"][0] + offset_xy[0]) / VOXEL_SIZE)
        ix_max = int((r["box_max"][0] + offset_xy[0]) / VOXEL_SIZE)
        iy_min = int((r["box_min"][1] + offset_xy[1]) / VOXEL_SIZE)
        iy_max = int((r["box_max"][1] + offset_xy[1]) / VOXEL_SIZE)
        iz_min = int((r["box_min"][2] - z_min_phys) / VOXEL_SIZE)
        iz_max = int((r["box_max"][2] - z_min_phys) / VOXEL_SIZE)
        grid[
            max(0, ix_min) : min(RES_X, ix_max),
            max(0, iy_min) : min(RES_Y, iy_max),
            max(0, iz_min) : min(RES_Z, iz_max),
        ] = ch

    one_hot = np.zeros((NUM_CHANNELS, RES_X, RES_Y, RES_Z), dtype=np.float32)
    for c in range(NUM_CHANNELS):
        one_hot[c] = (grid == c).astype(np.float32)
    cond = torch.tensor(build_condition_vector(data), dtype=torch.float32)
    return {"graph": hetero, "voxel": torch.tensor(one_hot, dtype=torch.float32), "condition": cond}


def graph_batch_size(batch) -> int:
    return int(getattr(batch, "num_graphs", 1))


def graph_batch_dict(batch):
    if hasattr(batch, "batch_dict"):
        return batch.batch_dict
    bd = {}
    for ntype in batch.node_types:
        if batch[ntype].x.size(0) > 0:
            if hasattr(batch[ntype], "batch"):
                bd[ntype] = batch[ntype].batch
            else:
                bd[ntype] = torch.zeros(batch[ntype].x.size(0), dtype=torch.long, device=batch[ntype].x.device)
    return bd


def graph_condition(batch):
    bs = graph_batch_size(batch)
    if not hasattr(batch, "condition"):
        raise AttributeError("batch 缺少 condition")
    cond = batch.condition
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    return cond.view(bs, -1)


def prepare_graph_batch(hg, condition=None):
    if condition is not None:
        cond = condition.unsqueeze(0) if condition.dim() == 1 else condition
        hg.condition = cond
    batch = next(iter(GraphDataLoader([hg], batch_size=1)))
    if condition is not None and not hasattr(batch, "condition"):
        batch.condition = hg.condition
    return batch


def forward_model(model, batch, condition=None):
    cond = condition if condition is not None else graph_condition(batch)
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    mu, logvar = model.encoder(batch.x_dict, batch.edge_index_dict, graph_batch_dict(batch))
    z = model.reparameterize(mu, logvar)
    logits = model.decoder(z, cond.to(z.device))
    return logits, mu, logvar
