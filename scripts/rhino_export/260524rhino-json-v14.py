# -*- coding: utf-8 -*-
"""
Rhino 户型数据集 QC + JSON 导出 (V14)

基于 260523rhino-json.py，新增：
  - 拓扑连通 / 楼梯跨层检查
  - Z 区间楼层判定 + 楼板界面对齐 + 高度白名单
  - 玄关 / 必需功能 / 体素包络 / 采光复核 / 朝向 metadata / 底部架空二层漂浮检查
  - Eto 可滚动错误弹窗（已内置，无需额外文件）
"""
import json
import os
import re
import math
from collections import deque

try:
    import rhinoscriptsyntax as rs
except ImportError:
    rs = None


# ==========================================
# Eto 弹窗（内置于本文件，绑定 Rhino 主窗口 + SemiModal）
# ==========================================
def _dlg_u(text):
    if text is None:
        return u""
    if isinstance(text, unicode):
        return text
    try:
        return unicode(text, "utf-8", "replace")
    except Exception:
        return unicode(text)


def _dlg_get_parent():
    try:
        import Rhino
        import scriptcontext as sc
        doc = sc.doc or Rhino.RhinoDoc.ActiveDoc
        if doc is not None:
            return Rhino.UI.RhinoEtoApp.MainWindowForDocument(doc)
    except Exception:
        pass
    try:
        import Rhino
        return Rhino.UI.RhinoEtoApp.MainWindow
    except Exception:
        return None


def _dlg_focus_rhino():
    try:
        import Rhino
        Rhino.RhinoApp.SetFocusToMainWindow()
    except Exception:
        pass


def _dlg_hidden_abort_button():
    import Eto.Forms as forms
    btn = forms.Button(Text=u"取消")
    btn.Visible = False
    return btn


def _dlg_show_modal(dialog):
    import Eto.Forms as forms
    _dlg_focus_rhino()
    parent = _dlg_get_parent()
    try:
        dialog.Topmost = True
    except Exception:
        pass
    try:
        import Rhino
        import scriptcontext as sc
        doc = sc.doc or Rhino.RhinoDoc.ActiveDoc
        if doc is not None and hasattr(Rhino.UI, "EtoExtensions"):
            if dialog.DefaultButton is None:
                dialog.DefaultButton = forms.Button(Text=u"确定")
            if dialog.AbortButton is None:
                dialog.AbortButton = _dlg_hidden_abort_button()
            Rhino.UI.EtoExtensions.ShowSemiModal(dialog, doc, parent)
            return
    except Exception:
        pass
    if parent is not None:
        dialog.ShowModal(parent)
    else:
        dialog.ShowModal()


def _dlg_fallback_message(text, title, buttons=0):
    if rs is None:
        print(_dlg_u(title) + u": " + _dlg_u(text))
        return 1 if buttons else 0
    return rs.MessageBox(_dlg_u(text), buttons, _dlg_u(title))


def _dlg_estimate_message_size(text, max_width=560, min_width=320):
    import Eto.Drawing as drawing
    lines = _dlg_u(text).split(u"\n")
    line_count = max(1, len(lines))
    longest = max(len(line) for line in lines) if lines else 0
    width = min(max_width, max(min_width, 48 + longest * 9))
    height = min(480, max(140, 72 + line_count * 20))
    scroll_h = max(60, height - 72)
    return drawing.Size(width, height), drawing.Size(width - 32, scroll_h)


def _dlg_make_button(text, width=88):
    import Eto.Drawing as drawing
    import Eto.Forms as forms
    btn = forms.Button(Text=_dlg_u(text))
    btn.Size = drawing.Size(width, 28)
    return btn


def _dlg_button_row(*buttons):
    import Eto.Drawing as drawing
    import Eto.Forms as forms
    row = forms.TableLayout()
    row.Spacing = drawing.Size(8, 0)
    cells = [None]
    cells.extend(buttons)
    row.Rows.Add(forms.TableRow(*cells))
    return row


def ui_show_message(text, title=u"提示", width=None, height=None):
    """替代 rs.MessageBox(..., 0, title)"""
    try:
        import Eto.Drawing as drawing
        import Eto.Forms as forms
        text = _dlg_u(text)
        if width is None or height is None:
            client_size, scroll_size = _dlg_estimate_message_size(text)
        else:
            client_size = drawing.Size(width, height)
            scroll_size = drawing.Size(max(200, width - 32), max(60, height - 72))
        dialog = forms.Dialog()
        dialog.Title = _dlg_u(title)
        dialog.ClientSize = client_size
        dialog.MinimumSize = client_size
        dialog.Resizable = False
        body = forms.TextArea()
        body.ReadOnly = True
        body.Wrap = True
        body.Text = text
        scroll = forms.Scrollable()
        scroll.Content = body
        scroll.Border = forms.BorderType.Bezel
        scroll.Size = scroll_size
        btn = _dlg_make_button(u"确定")
        btn.Click += lambda sender, e: dialog.Close()
        dialog.DefaultButton = btn
        dialog.AbortButton = _dlg_hidden_abort_button()
        layout = forms.DynamicLayout()
        layout.Padding = drawing.Padding(12)
        layout.Spacing = drawing.Size(6, 6)
        layout.AddRow(scroll)
        layout.AddRow(_dlg_button_row(btn))
        dialog.Content = layout
        _dlg_show_modal(dialog)
    except Exception:
        _dlg_fallback_message(text, title, buttons=0)


def ui_show_confirm_scrollable(text, title=u"确认", yes_text=u"确认", no_text=u"取消",
                               width=560, height=400):
    """可滚动确认框，返回 True=确认 / False=取消"""
    try:
        import Eto.Drawing as drawing
        import Eto.Forms as forms
        dialog = forms.Dialog()
        dialog.Title = _dlg_u(title)
        dialog.ClientSize = drawing.Size(width, height)
        dialog.MinimumSize = drawing.Size(400, 260)
        dialog.Resizable = True
        body = forms.TextArea()
        body.ReadOnly = True
        body.Wrap = True
        body.Text = _dlg_u(text)
        scroll = forms.Scrollable()
        scroll.Content = body
        scroll.Border = forms.BorderType.Bezel
        scroll.Size = drawing.Size(width - 32, height - 88)
        result = {"confirmed": False}

        def on_yes(sender, e):
            result["confirmed"] = True
            dialog.Close()

        def on_no(sender, e):
            dialog.Close()

        btn_yes = _dlg_make_button(yes_text, width=96)
        btn_no = _dlg_make_button(no_text, width=72)
        btn_yes.Click += on_yes
        btn_no.Click += on_no
        dialog.DefaultButton = btn_yes
        dialog.AbortButton = btn_no
        layout = forms.DynamicLayout()
        layout.Padding = drawing.Padding(12)
        layout.Spacing = drawing.Size(6, 6)
        layout.AddRow(scroll)
        layout.AddRow(_dlg_button_row(btn_yes, btn_no))
        dialog.Content = layout
        _dlg_show_modal(dialog)
        return result["confirmed"]
    except Exception:
        return _dlg_fallback_message(text, title, buttons=4 | 32) == 6


def ui_show_report(lines, title=u"报告", is_warning=False, width=640, height=480):
    """可滚动报告（QC 错误/警告）"""
    prefix = u"⚠️ " if is_warning else u"⛔ "
    full_title = prefix + _dlg_u(title)
    text = u"\n\n".join(_dlg_u(line) for line in lines) if lines else u"（无内容）"
    header = u"共 {} 条{}。".format(
        len(lines) if lines else 0,
        u"警告" if is_warning else u"错误",
    )
    try:
        import Eto.Drawing as drawing
        import Eto.Forms as forms
        dialog = forms.Dialog()
        dialog.Title = full_title
        dialog.ClientSize = drawing.Size(width, height)
        dialog.MinimumSize = drawing.Size(480, 320)
        dialog.Resizable = True
        header_lbl = forms.Label()
        header_lbl.Text = header
        body = forms.TextArea()
        body.ReadOnly = True
        body.Wrap = True
        body.Text = text
        scroll = forms.Scrollable()
        scroll.Content = body
        scroll.Border = forms.BorderType.Bezel
        scroll.Size = drawing.Size(width - 32, height - 120)
        btn = _dlg_make_button(u"关闭")
        btn.Click += lambda sender, e: dialog.Close()
        dialog.DefaultButton = btn
        dialog.AbortButton = _dlg_hidden_abort_button()
        layout = forms.DynamicLayout()
        layout.Padding = drawing.Padding(12)
        layout.Spacing = drawing.Size(6, 6)
        layout.AddRow(header_lbl)
        layout.AddRow(scroll)
        layout.AddRow(_dlg_button_row(btn))
        dialog.Content = layout
        _dlg_show_modal(dialog)
    except Exception:
        _dlg_fallback_message(text, full_title, buttons=0)


def show_qc_report_dialog(lines, title="AI 严苛质检 V14", is_warning=False):
    """建模时显示全部 QC 信息（可滚动）"""
    ui_show_report(lines, title=title, is_warning=is_warning)


def show_confirm_export_dialog(orientation_msg, warning_lines, room_count, count_display,
                               courtyard_count, indirect_count, float_snap_count):
    """导出确认（含警告时可滚动）"""
    header_lines = [
        "✅ 质检通过！准备导出数据。",
        "",
        "🧭 " + orientation_msg,
        "",
        "💡 采光分析: {} 间间接采光, {} 面庭院采光".format(indirect_count, courtyard_count),
    ]
    if float_snap_count:
        header_lines.append("⚠️ 坐标浮点吸附 {} 条（≤1mm 误差已自动修正）".format(float_snap_count))
    header_lines.extend([
        "",
        "📊 请核对当前识别到的功能体块（共 {} 个）：".format(room_count),
        count_display,
        "",
        "👉 数量如果正确，是否确认导出 JSON 文件？",
    ])
    if warning_lines:
        header_lines.extend(["", "—— 以下项目请建模师复核 ——", ""] + warning_lines)
    return ui_show_confirm_scrollable(
        "\n".join(header_lines), title="确认导出清单 V14", yes_text="确认导出", no_text="取消",
    )


# ==========================================
# QC V14 常量
# ==========================================
QC_VERSION = "v14"
TOL = 10
CELL_SIZE = 300
VOXEL_SIZE = 300
RES_X, RES_Y, RES_Z = 96, 96, 32
MAX_BUILDING_X = RES_X * VOXEL_SIZE   # 28800
MAX_BUILDING_Y = RES_Y * VOXEL_SIZE   # 28800
MAX_BUILDING_Z = 6000
GRID_Z_PHYS = RES_Z * VOXEL_SIZE      # 9600（体素网格高度，实际建筑 6000）

MAX_LIGHT_HOPS = 8
MIN_EFFECTIVE_ATTENUATION = 0.08
SINGLE_FLOOR_HEIGHT = 3600
COORD_SNAP_TOL = 1.0
MODULUS = 300
GAP_MAX = 600
MIN_FACE_OVERLAP = MODULUS

STANDARD_FLOOR_HEIGHT = 3000
DOUBLE_HEIGHT_MIN = 5700
DOUBLE_HEIGHT_MAX = 6300

REQUIRED_ROOMS = [
    "entryway", "living_room", "bedroom", "dining_room", "bathroom", "stairs",
]

HEIGHT_6000_TYPES = frozenset(["stairs"])
HEIGHT_3000_OR_6000_TYPES = frozenset([
    "living_room", "multi_purpose", "dining_room", "balcony",
    "entryway",
])
HEIGHT_6000_ALLOWED = HEIGHT_6000_TYPES | HEIGHT_3000_OR_6000_TYPES

# 底部架空：纯 2F 体块在 1F (Z=0~3000) 下方缺少足够支承平面
MIN_2F_SLAB_SUPPORT_RATIO = 0.25   # 二层底面至少 25% 面积下方有 1F 体块顶板
MAX_FLOATING_2F_ROOMS = 2          # 允许最多 2 个「漂浮」二层体块，超出则拦截导出
FLOATING_2F_EXEMPT_TYPES = frozenset(["balcony"])  # 阳台允许挑出，不计入漂浮

LIGHTING_REVIEW_STRONG = frozenset(["living_room", "bedroom"])
LIGHTING_REVIEW_SOFT = frozenset(["dining_room", "multi_purpose", "kitchen"])

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


# ==========================================
# 工具函数（与原脚本一致 + V14 扩展）
# ==========================================
_LAYER_PATTERN = re.compile(r"\[([^\[\]]+?)\]")

_GLOBAL_LAYER_TOKENS = frozenset([
    "功能缺失", "体素越界", "图层未识别", "底部架空",
])


def _sort_messages_by_layer(items):
    """按错误/警告中提取的第一个 [token] 排序。

    - 带 layer_name 的逐图层排到前面（便于在 Rhino 里依次定位）
    - 全局类别（功能缺失/体素越界 等）排到末尾
    - 完全无 [..] 标记的兜底排最后
    """
    def sort_key(msg):
        m = _LAYER_PATTERN.search(msg)
        if not m:
            return (2, "", msg)
        token = m.group(1)
        if token in _GLOBAL_LAYER_TOKENS:
            return (1, token, msg)
        return (0, token, msg)
    return sorted(items, key=sort_key)


def _parse_layer_room_type(layer_name):
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


def _near_level(value, level, tol=COORD_SNAP_TOL):
    return abs(float(value) - float(level)) <= tol + 0.5


def _is_single_floor_height(dz):
    return (STANDARD_FLOOR_HEIGHT - 300) <= dz <= (STANDARD_FLOOR_HEIGHT + 300)


def _is_double_floor_height(dz):
    return DOUBLE_HEIGHT_MIN <= dz <= DOUBLE_HEIGHT_MAX


def _infer_floors_from_z(norm_min_z, norm_max_z, room_type):
    """根据归一化 Z 区间返回房间实际占据的楼层列表。

    - 单层 1F: [1]      (Z 区间 0~3000)
    - 单层 2F: [2]      (Z 区间 3000~6000)
    - 跨层:    [1, 2]   (Z 区间 0~6000；stairs / 挑空 living_room / 玄关等允许)
    返回 None 表示楼层判定失败（Z 未对齐 0/3000/6000 楼板）。

    主层（floor 兼容字段）= floors[0]。
    """
    z0 = float(norm_min_z)
    z1 = float(norm_max_z)
    height = z1 - z0

    if room_type == "stairs":
        if _near_level(z0, 0) and _near_level(z1, 6000):
            return [1, 2]
        return None

    if room_type in HEIGHT_6000_ALLOWED and _is_double_floor_height(height):
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


def _main_floor(floors):
    """主层（floors[0]），用于兼容旧 floor 字段与 stats 分组"""
    return floors[0] if floors else 0


def _check_floor_slab_and_height(room_type, norm_min, norm_max, dz, layer_name, errors):
    """楼板界面 (0/3000/6000) + 功能高度白名单"""
    z0 = float(norm_min[2])
    z1 = float(norm_max[2])
    type_cn = ROOM_TYPE_CN.get(room_type, room_type)

    if room_type == "stairs":
        if not _is_double_floor_height(dz):
            errors.append(
                "【高度错误】[{}] 楼梯高度 {}mm，要求 {}~{}mm".format(
                    layer_name, dz, DOUBLE_HEIGHT_MIN, DOUBLE_HEIGHT_MAX))
        if not (_near_level(z0, 0) and _near_level(z1, 6000)):
            errors.append(
                "【楼板界面】[{}] 楼梯底/顶应为 Z=0/6000，当前 {:.0f}~{:.0f}".format(layer_name, z0, z1))
        return

    if room_type in HEIGHT_3000_OR_6000_TYPES:
        valid_z = (
            (_near_level(z0, 0) and _near_level(z1, 3000)) or
            (_near_level(z0, 3000) and _near_level(z1, 6000)) or
            (_near_level(z0, 0) and _near_level(z1, 6000))
        )
        valid_h = _is_single_floor_height(dz) or _is_double_floor_height(dz)
        if not valid_z:
            errors.append(
                "【楼板界面】[{}] {} 底/顶应为 0/3000、3000/6000 或 0/6000，当前 {:.0f}~{:.0f}".format(
                    layer_name, type_cn, z0, z1))
        if not valid_h:
            errors.append(
                "【高度错误】[{}] {} 高度 {}mm，只允许 3000 或 6000（±300）".format(
                    layer_name, type_cn, dz))
        return

    if not _is_single_floor_height(dz):
        errors.append(
            "【高度错误】[{}] {} 高度 {}mm，非挑空/楼梯类型只允许 3000mm（±300）".format(
                layer_name, type_cn, dz))

    valid_z = (
        (_near_level(z0, 0) and _near_level(z1, 3000)) or
        (_near_level(z0, 3000) and _near_level(z1, 6000))
    )
    if not valid_z:
        errors.append(
            "【楼板界面】[{}] {} 底/顶应为 0/3000 或 3000/6000，当前 {:.0f}~{:.0f}".format(
                layer_name, type_cn, z0, z1))


def _check_voxel_grid_bounds(room_entries, building_size, errors):
    """体素包络 28800×28800×9600（居中栅格）边界检查"""
    bx = float(building_size.get("x", 0))
    by = float(building_size.get("y", 0))
    bz = float(building_size.get("z", 0))

    if bx > MAX_BUILDING_X + COORD_SNAP_TOL:
        errors.append(
            "【体素越界】建筑 X 向跨度 {}mm 超过 {}mm（{} 个体素）".format(
                int(round(bx)), MAX_BUILDING_X, RES_X))
    if by > MAX_BUILDING_Y + COORD_SNAP_TOL:
        errors.append(
            "【体素越界】建筑 Y 向跨度 {}mm 超过 {}mm（{} 个体素）".format(
                int(round(by)), MAX_BUILDING_Y, RES_Y))
    if bz > MAX_BUILDING_Z + COORD_SNAP_TOL:
        errors.append(
            "【体素越界】建筑 Z 向跨度 {}mm 超过 {}mm（两层标准高度）".format(
                int(round(bz)), MAX_BUILDING_Z))

    if not room_entries:
        return

    all_mins = [e["norm_min"] for e in room_entries]
    all_maxs = [e["norm_max"] for e in room_entries]
    build_min = [min(c[i] for c in all_mins) for i in range(3)]
    build_max = [max(c[i] for c in all_maxs) for i in range(3)]
    phys_center_xy = [(build_min[0] + build_max[0]) / 2.0, (build_min[1] + build_max[1]) / 2.0]
    offset_xy = [
        RES_X * VOXEL_SIZE / 2.0 - phys_center_xy[0],
        RES_Y * VOXEL_SIZE / 2.0 - phys_center_xy[1],
    ]
    z_min_phys = build_min[2]

    clipped = []
    for entry in room_entries:
        nmin, nmax = entry["norm_min"], entry["norm_max"]
        ix0 = int((nmin[0] + offset_xy[0]) / VOXEL_SIZE)
        ix1 = int((nmax[0] + offset_xy[0]) / VOXEL_SIZE)
        iy0 = int((nmin[1] + offset_xy[1]) / VOXEL_SIZE)
        iy1 = int((nmax[1] + offset_xy[1]) / VOXEL_SIZE)
        iz0 = int((nmin[2] - z_min_phys) / VOXEL_SIZE)
        iz1 = int((nmax[2] - z_min_phys) / VOXEL_SIZE)
        if ix0 < 0 or iy0 < 0 or iz0 < 0 or ix1 > RES_X or iy1 > RES_Y or iz1 > RES_Z:
            clipped.append(entry["layer_name"])

    if clipped:
        errors.append(
            "【体素裁剪】以下体块在 {}×{}×{} 居中栅格下会被截断：{}".format(
                RES_X, RES_Y, RES_Z, "、".join(clipped[:6]) +
                (" 等{}个".format(len(clipped)) if len(clipped) > 6 else "")))


def _check_topology_connectivity(rooms, errors):
    """整体拓扑连通：不允许孤岛房间"""
    if len(rooms) <= 1:
        return

    adj_detail = _build_adjacency_detail(rooms)
    start_id = rooms[0]["id"]
    visited = set()
    queue = deque([start_id])

    while queue:
        rid = queue.popleft()
        if rid in visited:
            continue
        visited.add(rid)
        for nid, _ in adj_detail.get(rid, []):
            if nid not in visited:
                queue.append(nid)

    if len(visited) >= len(rooms):
        return

    for room in rooms:
        if room["id"] in visited:
            continue
        errors.append(
            "【拓扑孤岛】[{}] {}（{}）未与主体连通，请检查是否遗漏走道/门洞切分".format(
                room.get("layer_name", room["id"]),
                ROOM_TYPE_CN.get(room["type"], room["type"]),
                room["id"],
            ))


def _check_stairs_connectivity(rooms, errors):
    """楼梯必须跨层占据 1F+2F，且水平邻接的房间须同时覆盖 1F 与 2F（用 floors 集合判定）"""
    stairs_list = [r for r in rooms if r["type"] == "stairs"]
    if not stairs_list:
        return

    room_by_id = {r["id"]: r for r in rooms}
    adj_detail = _build_adjacency_detail(rooms)

    for st in stairs_list:
        st_floors = set(st.get("floors", [st.get("floor", 0)]))
        if not (1 in st_floors and 2 in st_floors):
            errors.append(
                "【楼梯未跨层】[{}] 应同时占据 1F 与 2F（当前 floors={}）".format(
                    st.get("layer_name", st["id"]), sorted(st_floors)))

        has_f1 = False
        has_f2 = False
        for nid, _ in adj_detail.get(st["id"], []):
            neighbor = room_by_id.get(nid)
            if neighbor is None:
                continue
            n_floors = set(neighbor.get("floors", [neighbor.get("floor", 0)]))
            if 1 in n_floors:
                has_f1 = True
            if 2 in n_floors:
                has_f2 = True

        if not has_f1 or not has_f2:
            errors.append(
                "【楼梯未连通】[{}] 需同时邻接一层与二层体块（当前：一层{} 二层{}）".format(
                    st.get("layer_name", st["id"]),
                    "✓" if has_f1 else "✗",
                    "✓" if has_f2 else "✗",
                ))


def _room_floors_set(room):
    if isinstance(room.get("floors"), list) and room["floors"]:
        return set(int(f) for f in room["floors"])
    return {int(room.get("floor", 1))}


def _is_pure_second_floor_room(room):
    """仅占据 2F (Z≈3000~6000)，非楼梯/挑空跨层体块"""
    return _room_floors_set(room) == {2}


def _xy_overlap_area(a_min, a_max, b_min, b_max):
    ox = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0])
    oy = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1])
    if ox <= TOL or oy <= TOL:
        return 0.0
    return ox * oy


def _is_first_floor_slab_supporter(room_below):
    """可作为二层下方的 1F 楼板支承（顶面在 Z=3000）"""
    floors = _room_floors_set(room_below)
    if 1 not in floors:
        return False
    if room_below["type"] == "stairs":
        return True
    o_min, o_max = room_below["_abs_min"], room_below["_abs_max"]
    # 跨层挑空/楼梯：Z=0~6000，不提供二层楼板支承
    if 2 in floors and _near_level(o_min[2], 0) and _near_level(o_max[2], 6000):
        return False
    return _near_level(o_max[2], 3000)


def _second_floor_slab_support_ratio(room_2f, rooms):
    """纯 2F 体块底面 (Z=3000) 被 1F 体块顶板覆盖的面积比例"""
    if not _is_pure_second_floor_room(room_2f):
        return 1.0

    b_min, b_max = room_2f["_abs_min"], room_2f["_abs_max"]
    if not _near_level(b_min[2], 3000):
        return 1.0

    plan = _room_plan_area(room_2f)
    if plan <= 0:
        return 1.0

    supported = 0.0
    for other in rooms:
        if other["id"] == room_2f["id"]:
            continue
        if not _is_first_floor_slab_supporter(other):
            continue
        if not _boxes_share_face(other, room_2f, "z+"):
            continue
        o_min, o_max = other["_abs_min"], other["_abs_max"]
        supported += _xy_overlap_area(b_min, b_max, o_min, o_max)

    return min(1.0, supported / plan)


def _is_floating_second_floor_room(room, rooms):
    """底部架空导致二层体块「漂浮」：2F 体块下方缺少足够 1F 支承"""
    if not _is_pure_second_floor_room(room):
        return False
    if room["type"] in FLOATING_2F_EXEMPT_TYPES:
        return False
    return _second_floor_slab_support_ratio(room, rooms) < MIN_2F_SLAB_SUPPORT_RATIO


def _check_pilotis_floating_2f(rooms, errors):
    """底部架空：过多二层体块下方无 1F 支承时禁止导出"""
    pure_2f = [r for r in rooms if _is_pure_second_floor_room(r)]
    if not pure_2f:
        return

    floating = [r for r in pure_2f if _is_floating_second_floor_room(r, rooms)]
    if len(floating) <= MAX_FLOATING_2F_ROOMS:
        return

    lines = [
        "【底部架空】检测到 {} 个二层漂浮体块（允许 ≤{} 个）".format(
            len(floating), MAX_FLOATING_2F_ROOMS),
        "  → 一层 Z=0~3000 下方缺少足够体块顶板（常见于底部架空/挑空过多）",
        "  → 请在架空区补 1F 体块，或将二层功能移至有下层支承的位置",
    ]
    for room in floating[:10]:
        ratio = _second_floor_slab_support_ratio(room, rooms)
        type_cn = ROOM_TYPE_CN.get(room["type"], room["type"])
        lines.append(
            "  · [{}] {} — 下层支承覆盖率 {:.0f}%（要求 ≥{:.0f}%）".format(
                room.get("layer_name", room["id"]),
                type_cn,
                ratio * 100.0,
                MIN_2F_SLAB_SUPPORT_RATIO * 100.0,
            ))
    if len(floating) > 10:
        lines.append("  · … 另有 {} 个漂浮体块".format(len(floating) - 10))
    errors.append("\n".join(lines))


def _entryway_has_lighting(room):
    """玄关有采光即可（含经走道间接传导，不要求 direct outdoor）"""
    return room.get("lighting_access", "none") != "none"


def _check_entryway_rules(rooms, errors):
    entryways = [r for r in rooms if r["type"] == "entryway"]
    if not entryways:
        errors.append("【功能缺失】缺少必需空间：entryway（玄关）。")
        return

    if not any(_entryway_has_lighting(e) for e in entryways):
        names = "、".join(e.get("layer_name", e["id"]) for e in entryways)
        errors.append(
            "【玄关规则】{} 无任何采光，至少一个玄关须具备 direct/indirect 采光（含经走道传导）".format(names))


def _check_required_rooms(room_counts, errors):
    missing = [r for r in REQUIRED_ROOMS if r not in room_counts]
    if missing:
        cn = [ROOM_TYPE_CN.get(m, m) for m in missing]
        errors.append("【功能缺失】缺少必需空间：{}。".format("、".join(cn)))


def _collect_lighting_review_warnings(rooms):
    """需采光功能无采光 → 建模师复核（警告，不硬拦）"""
    warnings = []
    for room in rooms:
        rtype = room["type"]
        access = room.get("lighting_access", "none")
        if access != "none":
            continue

        layer = room.get("layer_name", room["id"])
        type_cn = ROOM_TYPE_CN.get(rtype, rtype)

        if rtype in LIGHTING_REVIEW_STRONG:
            warnings.append(
                "【采光复核·重要】[{}] {} 无任何采光（direct/indirect 均为 none）".format(layer, type_cn))
        elif rtype in LIGHTING_REVIEW_SOFT:
            warnings.append(
                "【采光复核】[{}] {} 无采光，请确认是否为内区或暗房间".format(layer, type_cn))
        elif rtype == "balcony":
            warnings.append(
                "【采光复核】[{}] 阳台/露台无 outdoor 暴露，请确认是否误标".format(layer))
    return warnings


def _room_plan_area(room):
    bmin, bmax = room["box_min"], room["box_max"]
    return max(0.0, bmax[0] - bmin[0]) * max(0.0, bmax[1] - bmin[1])


def _has_south_direct(room):
    for surf in room.get("direct_lighting_surfaces", []):
        if surf.get("normal", [0, 0, 0])[1] == -1.0:
            return True
    return False


def _has_south_any(room):
    for surf in _all_lighting_surfaces(room):
        if surf.get("normal", [0, 0, 0])[1] == -1.0:
            return True
    return False


def _compute_orientation_qc(rooms):
    """朝向 QC + 供训练条件向量使用的结构化字段"""
    orientation = {
        "living_room_south_direct": False,
        "living_room_south_indirect": False,
        "main_bedroom_south_direct": False,
        "facade_exposed_sides": [],
    }

    living = next((r for r in rooms if r["type"] == "living_room"), None)
    if living is not None:
        if _has_south_direct(living):
            orientation["living_room_south_direct"] = True
        elif _has_south_any(living) and living.get("lighting_access") == "indirect":
            orientation["living_room_south_indirect"] = True

    bedrooms = [r for r in rooms if r["type"] == "bedroom"]
    if bedrooms:
        main_bed = max(bedrooms, key=_room_plan_area)
        orientation["main_bedroom_south_direct"] = _has_south_direct(main_bed)

    side_set = set()
    for room in rooms:
        for surf in room.get("direct_lighting_surfaces", []):
            if surf.get("exposure_type") != "outdoor":
                continue
            n = surf.get("normal", [0, 0, 0])
            if n[1] == 1.0:
                side_set.add("N")
            elif n[1] == -1.0:
                side_set.add("S")
            elif n[0] == 1.0:
                side_set.add("E")
            elif n[0] == -1.0:
                side_set.add("W")

    order = {"N": 0, "S": 1, "E": 2, "W": 3}
    orientation["facade_exposed_sides"] = sorted(side_set, key=lambda s: order.get(s, 9))
    return orientation


def _build_orientation_condition_vector(orientation_qc):
    """朝向条件向量（7 维，写入 metadata 供训练 notebook 拼接）"""
    sides = set(orientation_qc.get("facade_exposed_sides", []))
    return [
        1.0 if orientation_qc.get("living_room_south_direct") else 0.0,
        1.0 if orientation_qc.get("living_room_south_indirect") else 0.0,
        1.0 if orientation_qc.get("main_bedroom_south_direct") else 0.0,
        1.0 if "N" in sides else 0.0,
        1.0 if "S" in sides else 0.0,
        1.0 if "E" in sides else 0.0,
        1.0 if "W" in sides else 0.0,
    ]


def _orientation_summary_msg(orientation_qc):
    if orientation_qc.get("living_room_south_direct"):
        return "南向基准确立 (客厅有直接南向采光) ☀️"
    if orientation_qc.get("living_room_south_indirect"):
        return "南向基准间接成立 (客厅经玄关/走道获得南向采光) 🌤️"
    return "非正南朝向或南向采光不足 ❄️"


# ---- 以下与原 260523rhino-json.py 保持一致 ----

def _boxes_share_face(room_a, room_b, direction):
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
        return ([(a_min[0] + a_max[0]) / 2.0, a_max[1], (a_min[2] + a_max[2]) / 2.0], [0.0, 1.0, 0.0], dx * dy)
    if direction == "z+":
        return ([(a_min[0] + a_max[0]) / 2.0, (a_min[1] + a_max[1]) / 2.0, a_max[2]], [0.0, 0.0, 1.0], dx * dy)
    return None


def _point_in_room_xy(x, y, room):
    a_min, a_max = room["_abs_min"], room["_abs_max"]
    return a_min[0] <= x <= a_max[0] and a_min[1] <= y <= a_max[1]


def _build_floor_grid(rooms_on_floor, padding_cells=2):
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

    if grid_info["grid"][i][j] == "courtyard":
        return "courtyard"
    return "outdoor"


def _nearest_modulus_value(value, modulus=MODULUS):
    return round(value / float(modulus)) * modulus


def _snap_mm(value, modulus=MODULUS):
    nearest = _nearest_modulus_value(value, modulus)
    if abs(value - nearest) <= COORD_SNAP_TOL:
        return float(nearest)
    return round(value, 2)


def _is_modulus_aligned(value, modulus=MODULUS, tol=COORD_SNAP_TOL):
    rem = value % modulus
    return rem <= tol or rem >= modulus - tol


def _describe_plan_region(norm_min, norm_max, building_size):
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
    for dim in range(3):
        if a_max[dim] <= b_min[dim] + tol or b_max[dim] <= a_min[dim] + tol:
            return False
    return True


def _overlap_intersection(a_min, a_max, b_min, b_max):
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
    if _is_vertical_normal(surf["normal"]):
        return False
    if edge_direction in ("z+", "z-"):
        return _allow_vertical_light_passage(from_room, to_room)
    return True


def _normal_key(normal):
    return tuple(round(c, 1) for c in normal)


def _merge_surfaces_by_normal(surfaces):
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
    """直接采光面检测。

    跨层房间（stairs / 挑空 living_room 等 floors=[1,2]）会同时加入
    相关楼层的占用栅格，避免另一层的内庭院 flood-fill 在挑空位置
    误判为 outdoor。face exposure 仍按面中心 Z 落在的楼层栅格判定。
    """
    floor_groups = {}
    for room in rooms:
        for f in room.get("floors", [room.get("floor", 1)]):
            floor_groups.setdefault(f, []).append(room)

    floor_grids = {}
    for floor, floor_rooms in floor_groups.items():
        floor_grids[floor] = _build_floor_grid(floor_rooms)

    face_directions = ("x-", "x+", "y-", "y+", "z+")

    for room in rooms:
        direct_surfaces = []
        room_floors = room.get("floors", [room.get("floor", 1)])

        for direction in face_directions:
            if _face_has_neighbor(room, direction, rooms):
                continue

            geom = _face_geometry(room, direction)
            if not geom:
                continue

            center, normal, area = geom
            if area <= 0:
                continue

            if direction == "z+":
                target_floor = room_floors[-1]
            else:
                cz = center[2]
                inferred = 2 if cz >= 3000 else 1
                target_floor = inferred if inferred in room_floors else room_floors[0]
            grid_info = floor_grids.get(target_floor)

            exposure_type = _probe_exposure_type(grid_info, center, normal)
            direct_surfaces.append({
                "normal": normal,
                "area": round(area, 2),
                "exposure_type": exposure_type,
            })

        room["direct_lighting_surfaces"] = direct_surfaces


def _can_traverse(room):
    return room["type"] in TRANSIT_TYPES


def _can_receive_indirect(room):
    return room["type"] not in BLOCKER_TYPES and room["lighting_priority"] > 0


def propagate_effective_lighting(rooms):
    room_by_id = {r["id"]: r for r in rooms}
    adj_detail = _build_adjacency_detail(rooms)
    adjacency = {k: [n for n, _ in edges] for k, edges in adj_detail.items()}
    edge_dirs = {(rid, nid): direction for rid, edges in adj_detail.items() for nid, direction in edges}

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
    surfaces = list(room.get("direct_lighting_surfaces", []))
    for surf in room.get("effective_lighting", []):
        if surf.get("hops", 0) > 0:
            surfaces.append(surf)
    return surfaces


# ==========================================
# 主导出
# ==========================================
def _suggested_json_name(doc_name=None):
    """从文档名推断 house_*.json 文件名"""
    name = doc_name if doc_name is not None else rs.DocumentName()
    if name:
        base_name = name.split(".")[0]
        match = re.search(r"\d+", base_name)
        return "house_{}.json".format(match.group() if match else base_name)
    return "house_001.json"


def export_and_qc_rhino_dataset(interactive=True, save_path=None):
    """导出当前文档 JSON。

    interactive=False 时不弹窗（供批量脚本使用）；save_path 指定时跳过另存为对话框。
    返回 {"ok": True/False, ...} 结果字典。
    """
    if rs is None:
        raise RuntimeError("此脚本需在 Rhino 环境中运行（缺少 rhinoscriptsyntax）")

    def _fail(stage, errors_list=None, **extra):
        payload = {"ok": False, "stage": stage}
        if errors_list is not None:
            payload["errors"] = list(errors_list)
        payload.update(extra)
        return payload

    objects = rs.NormalObjects()
    if not objects:
        msg = "模型中没有可视对象！请确保体块未被隐藏或锁定。"
        if interactive:
            ui_show_message(msg, title="错误")
        return _fail("empty", [msg])

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
            "  → 推荐格式：11multi_purpose-多功能室".format(layer_name, count))

    if has_bad_geometry:
        bad_msgs = _sort_messages_by_layer(list(set(bad_reasons))) + ["", "请加盖(_Cap)或清理散线。"]
        if interactive:
            show_qc_report_dialog(bad_msgs, title="体块不合格")
        return _fail("geometry", bad_msgs)

    if not valid_objs:
        msg = "没有找到任何有效的功能体块！"
        if interactive:
            ui_show_message(msg, title="错误")
        return _fail("no_rooms", [msg])

    overall_bbox = rs.BoundingBox(valid_objs)
    global_min_x = overall_bbox[0].X
    global_min_y = overall_bbox[0].Y
    global_min_z = overall_bbox[0].Z
    global_max_x = overall_bbox[6].X
    global_max_y = overall_bbox[6].Y
    global_max_z = overall_bbox[6].Z

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

        room_counts[clean_type] = room_counts.get(clean_type, 0) + 1

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
                    errors.append(
                        "【异型错误】[{}] 检测到 L/T 型或挖洞体块！请切分为纯正矩形 Box。".format(layer_name))
            except Exception:
                pass

        for dim, axis in zip([dx, dy, dz], ["长(X)", "宽(Y)", "高(Z)"]):
            if not _is_modulus_aligned(dim):
                errors.append("【模数错误】[{}] 的 {} 尺寸 {}mm 不符合 300mm 模数！".format(layer_name, axis, dim))

        raw_norm_min = [
            pt_min.X - global_min_x,
            pt_min.Y - global_min_y,
            pt_min.Z - global_min_z,
        ]
        raw_norm_max = [
            pt_max.X - global_min_x,
            pt_max.Y - global_min_y,
            pt_max.Z - global_min_z,
        ]
        abs_min = _snap_box([pt_min.X, pt_min.Y, pt_min.Z])
        abs_max = _snap_box([pt_max.X, pt_max.Y, pt_max.Z])
        norm_min = _snap_box(raw_norm_min)
        norm_max = _snap_box(raw_norm_max)

        floors = _infer_floors_from_z(raw_norm_min[2], raw_norm_max[2], clean_type)
        if floors is None:
            errors.append(
                "【楼层判定失败】[{}] Z 区间 {:.0f}~{:.0f} 无法识别为 1F/2F/跨层（应对齐 0/3000/6000 楼板）".format(
                    layer_name, raw_norm_min[2], raw_norm_max[2]))
            room_floors = []
        else:
            room_floors = list(floors)

        main_floor = _main_floor(room_floors)
        floor_label = "+".join(str(f) for f in room_floors) if room_floors else "?"

        _check_floor_slab_and_height(clean_type, raw_norm_min, raw_norm_max, dz, layer_name, errors)

        _check_coordinate_modulus(
            raw_norm_min, raw_norm_max, layer_name, clean_type,
            floor_label, building_size, errors, warnings,
        )

        room_entries.append({
            "layer_name": layer_name,
            "type": clean_type,
            "floor": main_floor,
            "floors": room_floors,
            "abs_min": abs_min,
            "abs_max": abs_max,
            "norm_min": raw_norm_min,
            "norm_max": raw_norm_max,
        })

        rooms.append({
            "id": "room_{}".format(i),
            "type": clean_type,
            "floor": main_floor,
            "floors": room_floors,
            "layer_name": layer_name,
            "lighting_priority": LIGHTING_PRIORITY.get(clean_type, 0),
            "_abs_min": abs_min,
            "_abs_max": abs_max,
            "box_min": norm_min,
            "box_max": norm_max,
        })

    _check_room_overlaps(room_entries, building_size, errors)
    _check_room_gaps(room_entries, building_size, errors)
    _check_voxel_grid_bounds(room_entries, building_size, errors)
    _check_required_rooms(room_counts, errors)
    _check_topology_connectivity(rooms, errors)
    _check_stairs_connectivity(rooms, errors)
    _check_pilotis_floating_2f(rooms, errors)

    if errors:
        qc_msgs = ["⛔ 严重错误：请修正后再导出！(已拦截)", ""] + _sort_messages_by_layer(errors)
        if interactive:
            show_qc_report_dialog(qc_msgs, title="AI 严苛质检 V14")
        return _fail("qc", qc_msgs)

    analyze_lighting(rooms)

    _check_entryway_rules(rooms, errors)

    if errors:
        qc_msgs = ["⛔ 严重错误：请修正后再导出！(已拦截)", ""] + _sort_messages_by_layer(errors)
        if interactive:
            show_qc_report_dialog(qc_msgs, title="AI 严苛质检 V14")
        return _fail("qc", qc_msgs)

    lighting_warnings = _collect_lighting_review_warnings(rooms)
    warnings.extend(lighting_warnings)

    orientation_qc = _compute_orientation_qc(rooms)
    orientation_vector = _build_orientation_condition_vector(orientation_qc)
    orientation_msg = _orientation_summary_msg(orientation_qc)

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

    ordered_keys = [
        "entryway", "living_room", "dining_room", "kitchen",
        "bathroom", "bedroom", "corridor", "stairs",
        "balcony", "utility", "multi_purpose",
    ]
    name_map = {
        "entryway": "01 玄关", "living_room": "02 客厅", "dining_room": "03 餐厅",
        "kitchen": "04 厨房", "bathroom": "05 卫生间", "bedroom": "06 卧室",
        "corridor": "07 过道", "stairs": "08 楼梯", "balcony": "09 阳台/露台",
        "utility": "10 储藏/设备", "multi_purpose": "11 多功能房",
    }
    count_strs = []
    for key in ordered_keys:
        if key in room_counts:
            count_strs.append("   - {}: {} 个".format(name_map[key], room_counts[key]))
    count_display = "\n".join(count_strs)

    float_snap_count = len([w for w in warnings if w.startswith("【坐标浮点误差】")])
    review_warnings = _sort_messages_by_layer(
        [w for w in warnings if not w.startswith("【坐标浮点误差】")]
    )

    if interactive:
        if not show_confirm_export_dialog(
            orientation_msg, review_warnings, len(rooms), count_display,
            courtyard_count, indirect_count, float_snap_count,
        ):
            return _fail("cancelled")

    suggested_name = _suggested_json_name()

    metadata = {
        "total_rooms": len(rooms),
        "stats": room_counts,
        "constraints": {
            "qc_version": QC_VERSION,
            "pure_box_enforced": True,
            "origin_aligned_auto": True,
            "modulus": MODULUS,
            "lighting_analysis": "adjacency_void_propagation_v2",
            "coord_modulus_qc": True,
            "overlap_qc": True,
            "gap_qc": True,
            "topology_qc": True,
            "pilotis_floating_2f_qc": True,
            "slab_interface_qc": True,
            "floors_field": "rooms[i].floors=[1] / [2] / [1,2]; floor=floors[0] 兼容字段（跨层=stairs+挑空 living_room）",
            "lighting_field": "direct_lighting_surfaces + effective_lighting；effective 含 hops=0 直接采光副本，便于按法向聚合",
            "voxel_grid": {
                "res": [RES_X, RES_Y, RES_Z],
                "voxel_size": VOXEL_SIZE,
                "max_building_mm": [MAX_BUILDING_X, MAX_BUILDING_Y, MAX_BUILDING_Z],
                "grid_z_phys": GRID_Z_PHYS,
            },
        },
        "building_size": {
            "x": round(global_max_x - global_min_x),
            "y": round(global_max_y - global_min_y),
            "z": round(global_max_z - global_min_z),
        },
        "orientation_qc": orientation_qc,
        "condition_vector_extensions": {
            "schema": "orientation_v1",
            "orientation": orientation_vector,
            "labels": [
                "living_room_south_direct",
                "living_room_south_indirect",
                "main_bedroom_south_direct",
                "facade_N", "facade_S", "facade_E", "facade_W",
            ],
        },
        "lighting_qc": {
            "review_warnings": review_warnings,
        },
    }

    output = {
        "house_id": suggested_name.replace(".json", ""),
        "metadata": metadata,
        "rooms": rooms,
    }

    if save_path is None:
        save_path = rs.SaveFileName(
            "保存合格的 AI 数据集",
            "JSON Files (*.json)|*.json||",
            filename=suggested_name,
        )

    if not save_path:
        return _fail("no_save_path")

    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    if interactive:
        ui_show_message(
            "🎉 JSON 数据已成功保存！\n(QC {})".format(QC_VERSION),
            title="导出完成",
        )

    return {
        "ok": True,
        "save_path": save_path,
        "house_id": output["house_id"],
        "total_rooms": len(rooms),
        "warnings": review_warnings,
        "qc_version": QC_VERSION,
    }


if __name__ == "__main__":
    export_and_qc_rhino_dataset()
