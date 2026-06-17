from __future__ import annotations

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from .config import (
        CHANNEL_MAP,
        CN_NAMES,
        ROOM_TYPES,
        TYPE_COLOR_DICT,
        VOXEL_SIZE,
    )
except ImportError:
    from config import (
        CHANNEL_MAP,
        CN_NAMES,
        ROOM_TYPES,
        TYPE_COLOR_DICT,
        VOXEL_SIZE,
    )

FLOOR_SLABS = {
    1: (0.0, 3000.0),
    2: (3000.0, 6000.0),
}


def _setup_matplotlib_cjk():
    from matplotlib import font_manager

    for name in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
    ):
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _hex_to_rgba(hex_color, alpha=0.48):
    return mcolors.to_rgba(hex_color, alpha=alpha)


def _darken(hex_color, factor=0.72):
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex((r * factor, g * factor, b * factor))


def _cuboid_mesh_data(x0, y0, z0, x1, y1, z1):
    """返回 Plotly Mesh3d 所需顶点和三角面索引（12 面 / 6 个矩形面）。"""
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    # 6 面 × 2 三角 = 12 个三角面
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return x, y, z, i, j, k


def _cuboid_edge_segments(x0, y0, z0, x1, y1, z1):
    """返回 12 条棱的 x/y/z 数组（None 分隔不连续线段），供 Scatter3d 描边。"""
    def seg(p, q):
        return ([p[0], q[0], None], [p[1], q[1], None], [p[2], q[2], None])
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # 底面 0-3
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # 顶面 4-7
    ]
    edges = [seg(v[a], v[b]) for a, b in [
        (0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4),  # 底+顶
        (0,4),(1,5),(2,6),(3,7),                           # 竖边
    ]]
    xs = sum((e[0] for e in edges), [])
    ys = sum((e[1] for e in edges), [])
    zs = sum((e[2] for e in edges), [])
    return xs, ys, zs


def _room_centroid(room):
    bm, bx = room["box_min"], room["box_max"]
    return ((bm[0] + bx[0]) / 2, (bm[1] + bx[1]) / 2, (bm[2] + bx[2]) / 2)


def _room_intersects_floor(room, floor: int) -> bool:
    z0, z1 = room["box_min"][2], room["box_max"][2]
    lo, hi = FLOOR_SLABS[floor]
    return z0 < hi and z1 > lo


def _clip_room_to_floor_slab(room, floor: int) -> dict | None:
    if not _room_intersects_floor(room, floor):
        return None
    lo, hi = FLOOR_SLABS[floor]
    r = {
        "type": room["type"],
        "box_min": list(room["box_min"]),
        "box_max": list(room["box_max"]),
    }
    if "id" in room:
        r["id"] = room["id"]
    if "floor" in room:
        r["floor"] = room["floor"]
    r["box_min"][2] = max(r["box_min"][2], lo)
    r["box_max"][2] = min(r["box_max"][2], hi)
    if r["box_max"][2] <= r["box_min"][2]:
        return None
    return r


def filter_rooms_for_floor(rooms, floor: int) -> list:
    out = []
    for r in rooms:
        clipped = _clip_room_to_floor_slab(r, floor)
        if clipped is not None:
            out.append(clipped)
    return out


def _site_boundary_trace(site_x, site_y):
    """用地边界线（Plotly Scatter3d）。"""
    xs = [0, site_x, site_x, 0, 0]
    ys = [0, 0, site_y, site_y, 0]
    zs = [0, 0, 0, 0, 0]
    return dict(
        type="scatter3d", x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="#999999", width=2, dash="dash"),
        name="用地边界", showlegend=True, hoverinfo="skip",
    )


def plot_3d_layout(
    rooms,
    site_x=None,
    site_y=None,
    title="AI 生成的 3D 功能体块布局图",
    graph=None,
    edge_types=None,
    layout_rooms=None,
    color_map=None,
    site_z=9000,
    show_topology=True,
):
    """交互式 3D 体块（Plotly）：可旋转缩放 + hover 房间信息 + 拓扑叠加。

    返回 plotly.graph_objects.Figure，Gradio 用 gr.Plot 直接渲染，
    Notebook 用 fig.show()，保存用 fig.write_html() / fig.write_image()。
    """
    import plotly.graph_objects as go

    cmap = color_map or TYPE_COLOR_DICT
    fig = go.Figure()
    max_extent = max(
        float(site_x or 1.0),
        float(site_y or 1.0),
        float(site_z or 1.0),
    )

    # ---- 房间体块 ----
    grouped = {}
    for r in rooms:
        x0, y0, z0 = r["box_min"]
        x1, y1, z1 = r["box_max"]
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue
        rtype = r["type"]
        cn_name = CN_NAMES.get(rtype, rtype)
        hover = f"{cn_name}<br>{x1 - x0:.0f}×{y1 - y0:.0f}×{z1 - z0:.0f} mm<br>Z: {z0:.0f}–{z1:.0f}"
        g = grouped.setdefault(rtype, {
            "x": [], "y": [], "z": [], "i": [], "j": [], "k": [],
            "ex": [], "ey": [], "ez": [], "hover": [],
        })
        mx, my, mz, mi, mj, mk = _cuboid_mesh_data(x0, y0, z0, x1, y1, z1)
        base = len(g["x"])
        g["x"].extend(mx); g["y"].extend(my); g["z"].extend(mz)
        g["i"].extend([v + base for v in mi])
        g["j"].extend([v + base for v in mj])
        g["k"].extend([v + base for v in mk])
        g["hover"].extend([hover] * len(mx))
        ex, ey, ez = _cuboid_edge_segments(x0, y0, z0, x1, y1, z1)
        g["ex"].extend(ex); g["ey"].extend(ey); g["ez"].extend(ez)

    for rtype, g in grouped.items():
        color = cmap.get(rtype, "#CCCCCC")
        cn_name = CN_NAMES.get(rtype, rtype)
        fig.add_trace(go.Mesh3d(
            x=g["x"], y=g["y"], z=g["z"], i=g["i"], j=g["j"], k=g["k"],
            color=color, opacity=0.42, flatshading=True,
            name=cn_name, legendgroup=rtype, showlegend=True,
            hovertext=g["hover"], hoverinfo="text",
        ))
        fig.add_trace(go.Scatter3d(
            x=g["ex"], y=g["ey"], z=g["ez"], mode="lines",
            line=dict(color=color, width=1.7),
            name=cn_name, legendgroup=rtype, showlegend=False,
            hoverinfo="skip",
        ))

    # ---- 用地边界 ----
    if site_x and site_y:
        fig.add_trace(go.Scatter3d(**_site_boundary_trace(site_x, site_y)))

        # Floor references are line frames. Filled transparent planes can hide
        # room meshes in Plotly static export when many room blocks are present.
        for z_floor, fcolor, fname in [
            (0.0, "#D8E2EF", "1F 地面"),
            (3000.0, "#EAD7D7", "2F 楼面"),
        ]:
            fig.add_trace(go.Scatter3d(
                x=[0, site_x, site_x, 0, 0],
                y=[0, 0, site_y, site_y, 0],
                z=[z_floor] * 5,
                mode="lines",
                line=dict(color=fcolor, width=1.4),
                name=fname, showlegend=True, hoverinfo="skip",
            ))

    # ---- 拓扑叠加 ----
    if show_topology and graph is not None:
        topo = layout_rooms if layout_rooms is not None else rooms
        centroids = {r["id"]: _room_centroid(r) for r in topo if "id" in r}
        if centroids:
            # 节点
            nx_vals, ny_vals, nz_vals, ncolors, nhovers = [], [], [], [], []
            for nid in graph.nodes:
                if nid not in centroids:
                    continue
                cx, cy, cz = centroids[nid]
                nx_vals.append(cx); ny_vals.append(cy); nz_vals.append(cz)
                ntype = graph.nodes[nid].get("type", "corridor")
                ncolors.append(cmap.get(ntype, "#888888"))
                nhovers.append(f"{CN_NAMES.get(ntype, ntype)}<br>({nid})")
            if nx_vals:
                fig.add_trace(go.Scatter3d(
                    x=nx_vals, y=ny_vals, z=nz_vals, mode="markers",
                    marker=dict(size=5, color=ncolors, line=dict(color="white", width=1.2)),
                    name="房间节点", showlegend=True, hovertext=nhovers, hoverinfo="text",
                ))

            # 边：水平/垂直分两组
            hx, hy, hz = [], [], []
            vx, vy, vz = [], [], []
            for u, v in graph.edges:
                if u not in centroids or v not in centroids:
                    continue
                p0, p1 = centroids[u], centroids[v]
                et = (edge_types or {}).get((u, v), "horizontal")
                target = (hx, hy, hz) if et == "horizontal" else (vx, vy, vz)
                target[0].extend([p0[0], p1[0], None])
                target[1].extend([p0[1], p1[1], None])
                target[2].extend([p0[2], p1[2], None])
            if hx:
                fig.add_trace(go.Scatter3d(
                    x=hx, y=hy, z=hz, mode="lines",
                    line=dict(color="#B03060", width=2, dash="dash"),
                    name="水平连接", showlegend=True, hoverinfo="skip",
                ))
            if vx:
                fig.add_trace(go.Scatter3d(
                    x=vx, y=vy, z=vz, mode="lines",
                    line=dict(color="#E94F37", width=2.8, dash="dot"),
                    name="垂直连接", showlegend=True, hoverinfo="skip",
                ))

    # ---- 布局 ----
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        scene=dict(
            xaxis=dict(title="X (mm)", range=[0, site_x] if site_x else None, gridcolor="#e8e8e8"),
            yaxis=dict(title="Y (mm)", range=[0, site_y] if site_y else None, gridcolor="#e8e8e8"),
            zaxis=dict(title="Z (mm)", range=[0, site_z], gridcolor="#e8e8e8"),
            aspectmode="manual" if site_x and site_y else "data",
            aspectratio=dict(
                x=float(site_x) / max_extent if site_x else 1,
                y=float(site_y) / max_extent if site_y else 1,
                z=float(site_z) / max_extent if site_z else 1,
            ) if site_x and site_y else None,
            camera=dict(
                center=dict(x=0.0, y=0.0, z=-0.04),
                eye=dict(x=1.65, y=1.65, z=1.25),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
            dragmode="orbit",
        ),
        annotations=(
            [
                dict(
                    text=(
                        f"用户输入用地：{float(site_x):,.0f} × "
                        f"{float(site_y):,.0f} mm"
                    ),
                    x=0.5,
                    y=1.02,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="#444444"),
                )
            ]
            if site_x and site_y
            else []
        ),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, font=dict(size=10)),
        margin=dict(l=10, r=10, b=10, t=50),
        template="plotly_white",
    )
    return fig


def _cuboid_faces(x0, y0, z0, x1, y1, z1):
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
        [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
    ]


def plot_3d_layout_static(
    rooms,
    site_x=None,
    site_y=None,
    title="AI 生成的 3D 功能体块布局图",
    color_map=None,
    site_z=9000,
):
    """Matplotlib fallback for reliable nonblank PNG export."""
    _setup_matplotlib_cjk()
    cmap = color_map or TYPE_COLOR_DICT
    fig = plt.figure(figsize=(12, 8), facecolor="#fbfcfe")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#fbfcfe")

    legend_handles = {}
    for r in rooms:
        x0, y0, z0 = r["box_min"]
        x1, y1, z1 = r["box_max"]
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue
        rtype = r["type"]
        color = cmap.get(rtype, "#CCCCCC")
        faces = _cuboid_faces(x0, y0, z0, x1, y1, z1)
        poly = Poly3DCollection(
            faces,
            facecolors=[_hex_to_rgba(color, 0.42)] * len(faces),
            edgecolors=_darken(color, 0.55),
            linewidths=0.28,
        )
        ax.add_collection3d(poly)
        if rtype not in legend_handles:
            legend_handles[rtype] = Line2D(
                [0], [0], color=color, lw=6, label=CN_NAMES.get(rtype, rtype)
            )

    if site_x and site_y:
        for z, color in [(0.0, "#9AA6B2"), (3000.0, "#B8A0A0")]:
            xs = [0, site_x, site_x, 0, 0]
            ys = [0, 0, site_y, site_y, 0]
            zs = [z] * 5
            ax.plot(xs, ys, zs, color=color, linewidth=0.9, linestyle="--", alpha=0.65)
        ax.set_xlim(0, site_x)
        ax.set_ylim(0, site_y)
    else:
        xs = [v for r in rooms for v in (r["box_min"][0], r["box_max"][0])]
        ys = [v for r in rooms for v in (r["box_min"][1], r["box_max"][1])]
        if xs and ys:
            ax.set_xlim(min(xs), max(xs))
            ax.set_ylim(min(ys), max(ys))
    ax.set_zlim(0, site_z)
    if site_x and site_y:
        ax.set_box_aspect((site_x, site_y, site_z * 0.75))
    ax.view_init(elev=30, azim=-58)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title, fontsize=13, pad=14)
    if site_x and site_y:
        fig.text(
            0.5,
            0.94,
            f"用户输入用地：{float(site_x):,.0f} × {float(site_y):,.0f} mm",
            ha="center",
            va="center",
            fontsize=10,
            color="#444444",
        )
    ax.grid(True, linestyle=":", alpha=0.25)
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()), loc="upper left", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    return fig


def _build_floor_imshow_rgba(grid, layer, color_map, alpha=0.72):
    """体素占用图 → RGBA 图像（一格一色，无重叠）。"""
    tnames = layer["type_names"]
    h, w = grid.shape
    rgba = np.ones((h, w, 4), dtype=float)
    rgba[..., :] = (0.97, 0.97, 0.98, 1.0)
    for cid in range(1, len(CHANNEL_MAP)):
        mask = grid == cid
        if not mask.any():
            continue
        rtype = tnames.get(cid, "empty")
        rgb = mcolors.to_rgb(color_map.get(rtype, "#CCCCCC"))
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = alpha
    return rgba


def _grid_extent_mm(grid, layer):
    offset_xy = layer["offset_xy"]
    ox, oy = float(offset_xy[0]), float(offset_xy[1])
    h, w = grid.shape
    x0 = -ox
    y0 = -oy
    x1 = x0 + h * VOXEL_SIZE
    y1 = y0 + w * VOXEL_SIZE
    return [x0, x1, y0, y1]


def plot_floor_plan(
    rooms,
    floor: int,
    site_x=None,
    site_y=None,
    title=None,
    color_map=None,
    floor_layers=None,
):
    """单层平面俯视。优先用体素占用图（像真户型栅格），否则回退矩形体块。"""
    _setup_matplotlib_cjk()
    cmap = color_map or TYPE_COLOR_DICT
    lo, hi = FLOOR_SLABS[floor]
    title = title or f"{floor}层平面功能布局 ({lo:.0f}–{hi:.0f} mm)"

    fig, ax = plt.subplots(figsize=(9, 7.2), facecolor="#fafbfc")
    ax.set_facecolor("#fafbfc")
    legend_handles = {}

    if floor_layers and floor in floor_layers:
        layer = floor_layers[floor]
        grid = layer["grid"]
        rgba = _build_floor_imshow_rgba(grid, layer, cmap)
        extent = _grid_extent_mm(grid, layer)
        ax.imshow(
            rgba,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            zorder=2,
        )
        tnames = layer["type_names"]
        seen = set()
        for cid in range(1, len(CHANNEL_MAP)):
            if not (grid == cid).any():
                continue
            rtype = tnames[cid]
            if rtype in seen:
                continue
            seen.add(rtype)
            legend_handles[rtype] = Line2D(
                [0], [0], color=cmap.get(rtype, "#CCCCCC"), lw=6,
                label=CN_NAMES.get(rtype, rtype),
            )
    else:
        slab_rooms = filter_rooms_for_floor(rooms, floor)
        for r in slab_rooms:
            x0, y0 = r["box_min"][0], r["box_min"][1]
            x1, y1 = r["box_max"][0], r["box_max"][1]
            if x1 <= x0 or y1 <= y0:
                continue
            color = cmap.get(r["type"], "#CCCCCC")
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=_hex_to_rgba(color, alpha=0.62),
                    edgecolor=_darken(color, 0.5),
                    linewidth=1.1,
                    zorder=2,
                )
            )
            label = CN_NAMES.get(r["type"], r["type"])
            if r["type"] not in legend_handles:
                legend_handles[r["type"]] = Line2D(
                    [0], [0], color=color, lw=6, label=label
                )

    if site_x and site_y:
        ax.add_patch(
            Rectangle(
                (0, 0), site_x, site_y,
                fill=False, edgecolor="#666666", linewidth=1.4, linestyle="--", zorder=1,
            )
        )
        ax.set_xlim(0, site_x)
        ax.set_ylim(0, site_y)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title, fontsize=12)
    if site_x and site_y:
        ax.text(
            0.01,
            0.99,
            f"用户输入用地：{float(site_x):,.0f} × {float(site_y):,.0f} mm",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#444444",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#BBBBBB",
                alpha=0.9,
            ),
            zorder=5,
        )
    ax.grid(True, linestyle=":", alpha=0.4)
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()), loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    return fig


def plot_topology_graph(
    graph,
    pos,
    edge_types=None,
    title="功能拓扑连接图",
    color_map=None,
    show_node_ids=True,
):
    """纯拓扑图：节点 + 连边，无体块。"""
    _setup_matplotlib_cjk()
    cmap = color_map or TYPE_COLOR_DICT
    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor="#fafbfc")
    ax.set_facecolor("#fafbfc")
    ax.axis("off")

    if graph is None or pos is None:
        ax.text(0.5, 0.5, "无拓扑数据", ha="center", va="center", transform=ax.transAxes)
        return fig

    edge_handles = []
    seen = set()
    for u, v in graph.edges:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        et = (edge_types or {}).get((u, v), "horizontal")
        if et == "vertical":
            color, ls, lw, tag = "#E94F37", "--", 2.4, "垂直连接"
        else:
            color, ls, lw, tag = "#B03060", "-", 2.0, "水平连接"
        ax.plot([x0, x1], [y0, y1], color=color, linestyle=ls, linewidth=lw, alpha=0.85, zorder=1)
        if tag not in seen:
            edge_handles.append(Line2D([0], [0], color=color, lw=lw, linestyle=ls, label=tag))
            seen.add(tag)

    for nid in graph.nodes:
        if nid not in pos:
            continue
        x, y = pos[nid]
        ntype = graph.nodes[nid].get("type", "corridor")
        color = cmap.get(ntype, "#888888")
        ax.scatter(x, y, s=520, c=[color], edgecolors="white", linewidths=1.2, zorder=3)
        floor = graph.nodes[nid].get("floor", "")
        name = CN_NAMES.get(ntype, ntype)
        label = f"{name}\n({nid})" if show_node_ids else name
        ax.annotate(
            label,
            (x, y), textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=7, color="#333", zorder=4,
        )
        if floor:
            ax.annotate(
                f"F{floor}", (x, y), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=6, color="#666", zorder=4,
            )

    ax.set_title(title, fontsize=12, pad=10)
    type_counts = Counter(
        graph.nodes[nid].get("type", "unknown") for nid in graph.nodes
    )
    type_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=cmap.get(room_type, "#888888"),
            markeredgecolor="white",
            markersize=9,
            label=f"{CN_NAMES.get(room_type, room_type)} × {type_counts[room_type]}",
        )
        for room_type in ROOM_TYPES
        if type_counts.get(room_type, 0) > 0
    ]
    if type_handles:
        type_legend = ax.legend(
            handles=type_handles,
            title=f"功能数量（共 {sum(type_counts.values())} 个）",
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.94,
        )
        ax.add_artist(type_legend)
    if edge_handles:
        ax.legend(
            handles=edge_handles,
            title="关系类型",
            loc="lower right",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.92,
        )
    fig.tight_layout()
    return fig


def save_layout_figures(
    result,
    user_req,
    out_dir,
    prefix="layout",
    weights_path=None,
    run_meta=None,
    extra_meta=None,
):
    import json
    from pathlib import Path

    try:
        from .run_meta import build_run_meta, file_tag, now_stamp
    except ImportError:
        from run_meta import build_run_meta, file_tag, now_stamp

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_rooms = result.get("display_rooms", result["rooms"])
    seed = result["seed"]
    sx, sy = user_req["site_x"], user_req["site_y"]
    src = result.get("display_source", "rooms")

    meta = dict(run_meta or {})
    if weights_path and not meta.get("weights"):
        meta = build_run_meta(weights_path, **meta)
    stamp = meta.get("generated_stamp") or now_stamp()
    tag = file_tag(weights_path, stamp) if weights_path else f"{stamp}_seed{seed}"
    wt_label = meta.get("weights_file", "")

    pk3d = dict(
        graph=result.get("graph"),
        edge_types=result.get("edge_types"),
        layout_rooms=result.get("rooms"),
        show_topology=False,
    )
    paths = {}
    title_suffix = f"  {wt_label}  {meta.get('generated_at', '')}" if wt_label else ""

    fig3d = plot_3d_layout(
        plot_rooms, sx, sy,
        title=f"3D 功能体块  seed={seed}  ({src}){title_suffix}",
        **pk3d,
    )
    # Plotly 3D：保存交互式 HTML + 静态 PNG（如有 kaleido）
    html3d = out / f"{prefix}_{tag}_3d_seed{seed}.html"
    fig3d.write_html(str(html3d), include_plotlyjs="cdn")
    paths["3d_html"] = html3d
    try:
        png3d = out / f"{prefix}_{tag}_3d_seed{seed}.png"
        fig3d_static = plot_3d_layout_static(
            plot_rooms, sx, sy,
            title=f"3D 功能体块  seed={seed}  ({src}){title_suffix}",
        )
        fig3d_static.savefig(png3d, dpi=160, bbox_inches="tight", facecolor=fig3d_static.get_facecolor())
        plt.close(fig3d_static)
        paths["3d"] = png3d
    except Exception:
        pass

    floor_layers = result.get("floor_layers")
    for floor in (1, 2):
        figf = plot_floor_plan(
            plot_rooms, floor, sx, sy,
            title=f"{floor}层平面  seed={seed}{title_suffix}",
            floor_layers=floor_layers,
        )
        paths[f"floor{floor}"] = out / f"{prefix}_{tag}_floor{floor}_seed{seed}.png"
        figf.savefig(paths[f"floor{floor}"], dpi=160, bbox_inches="tight", facecolor=figf.get_facecolor())
        plt.close(figf)

    figt = plot_topology_graph(
        result.get("graph"),
        result.get("pos"),
        result.get("edge_types"),
        title=f"功能拓扑  seed={seed}{title_suffix}",
    )
    paths["topology"] = out / f"{prefix}_{tag}_topology_seed{seed}.png"
    figt.savefig(paths["topology"], dpi=160, bbox_inches="tight", facecolor=figt.get_facecolor())
    plt.close(figt)

    if weights_path or run_meta or extra_meta:
        record = {
            **meta,
            "seed": seed,
            "decode_mode": result.get("decode_mode"),
            "n_occ": result.get("n_occ"),
            "quality_score": result.get("quality_score"),
            "quality_metrics": result.get("quality_metrics"),
            "display_source": src,
            "display_style": result.get("display_style"),
            "room_counts": user_req.get("room_counts"),
            "site": [sx, sy],
            "outputs": {k: str(v) for k, v in paths.items()},
            **(extra_meta or {}),
        }
        meta_path = out / f"result_{tag}_seed{seed}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        paths["meta"] = meta_path

    return paths


def show_layout_fig(fig):
    from IPython.display import display

    if isinstance(fig, (list, tuple)):
        for f in fig:
            display(f)
    else:
        display(fig)
