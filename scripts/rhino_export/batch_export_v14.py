# -*- coding: utf-8 -*-
"""
Rhino EditPythonScript — 批量导出 JSON (V14 QC)

用法（Rhino 7/8）：
  1. 修改下方「路径配置」中的 BATCH_FOLDER（或 RAW_DIR / OUT_DIR）
  2. Rhino → EditPythonScript → 打开本文件 → Run
  3. JSON 输出至 OUT_DIR（house_*.json），与其它批次目录隔离
  4. 日志：OUT_DIR/batch_export_log.txt

依赖同目录下的 260524rhino-json-v14.py（已支持 interactive=False）。
"""
from __future__ import print_function

import os
import re
import sys
import shutil
import traceback
import codecs
from datetime import datetime

import rhinoscriptsyntax as rs
import Rhino
import scriptcontext as sc

# ========== 路径配置（只改这里） ==========
ROOT = r"E:\Documents\GraphSpace"

# 批次文件夹名：.3dm 在 data/raw/<名>，JSON 输出到 data/processed/<名>
# 示例：first_ | second_94 | third_299
BATCH_FOLDER = "third_299"

RAW_DIR = os.path.join(ROOT, "data", "raw", BATCH_FOLDER)
OUT_DIR = os.path.join(ROOT, "data", "processed", BATCH_FOLDER)

# 若目录结构不同，可直接覆盖上面两行，例如：
# RAW_DIR = r"E:\Documents\GraphSpace\data\raw\third_299"
# OUT_DIR = r"E:\Documents\GraphSpace\data\processed\third_299"

V14_SCRIPT = os.path.join(ROOT, "scripts", "rhino_export", "260524rhino-json-v14.py")
LOG_PATH = os.path.join(OUT_DIR, "batch_export_log.txt")

# True = 仅处理尚未存在对应 JSON 的文件；False = 全部覆盖重导
SKIP_EXISTING = False

# True = 每个文件完成后立即写日志（中断可保留进度）
FLUSH_LOG_EACH = True

# True = 关闭 .3dm 后删除同目录下 Rhino 解压的 *_embedded_files 文件夹
CLEAN_EMBEDDED_FOLDERS = True


def _load_v14_module():
    """IronPython 兼容：加载 V14 导出模块"""
    if not os.path.isfile(V14_SCRIPT):
        raise IOError("找不到 V14 脚本：{}".format(V14_SCRIPT))

    mod_name = "rhino_json_v14_batch"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    try:
        import imp
        mod = imp.load_source(mod_name, V14_SCRIPT)
        sys.modules[mod_name] = mod
        return mod
    except ImportError:
        namespace = {"__name__": mod_name, "__file__": V14_SCRIPT}
        with codecs.open(V14_SCRIPT, "r", "utf-8") as f:
            code = compile(f.read(), V14_SCRIPT, "exec")
        exec(code, namespace)
        mod = type(sys)(mod_name)
        mod.__dict__.update(namespace)
        sys.modules[mod_name] = mod
        return mod


def _iter_3dm_files(root_dir):
    if not os.path.isdir(root_dir):
        return []
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".3dm"):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def _json_name_from_3dm(path):
    base = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"\d+", base)
    if match:
        return "house_{}.json".format(match.group())
    safe = re.sub(r'[^\w\-]+', "_", base).strip("_") or "unknown"
    return "house_{}.json".format(safe)


def _open_3dm(path):
    """打开 .3dm 并设为当前 scriptcontext 文档"""
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isfile(path):
        return False, "文件不存在"

    try:
        import System
        was_open = System.Boolean(False)
        doc = Rhino.RhinoDoc.Open(path, was_open)
        if doc is not None:
            sc.doc = doc
            return True, None
    except Exception:
        pass

    quoted = '"{}"'.format(path)
    if rs.Command("_-Open {} _Enter".format(quoted), False):
        return True, None
    return False, "RhinoDoc.Open 与 _Open 命令均失败"


def _close_active_doc(save=False):
    try:
        doc = sc.doc
        if doc is None:
            return
        Rhino.RhinoDoc.Close(doc, save)
    except Exception:
        pass


def _embedded_folder_candidates(three_dm_path):
    """Rhino 打开 .3dm 时可能在同目录解压的内嵌资源文件夹名"""
    directory = os.path.dirname(os.path.abspath(three_dm_path))
    base = os.path.splitext(os.path.basename(three_dm_path))[0]
    suffixes = (
        "_embedded_files",
        "-embedded_files",
        "_embedded-files",
        "-embedded-files",
    )
    return [os.path.join(directory, base + suffix) for suffix in suffixes]


def _remove_tree(path):
    if not os.path.isdir(path):
        return False
    try:
        shutil.rmtree(path)
        return True
    except Exception:
        pass
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                fp = os.path.join(root, name)
                try:
                    os.chmod(fp, 0777)
                except Exception:
                    pass
                try:
                    os.remove(fp)
                except Exception:
                    return False
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except Exception:
                    return False
        os.rmdir(path)
        return True
    except Exception:
        return False


def _remove_embedded_files_folder(three_dm_path):
    """删除 Rhino 打开该 .3dm 时生成的 embedded_files 目录"""
    if not CLEAN_EMBEDDED_FOLDERS:
        return []

    three_dm_dir = os.path.dirname(os.path.abspath(three_dm_path))
    removed = []
    failed = []

    for folder in _embedded_folder_candidates(three_dm_path):
        if not os.path.isdir(folder):
            continue
        if os.path.dirname(os.path.abspath(folder)) != three_dm_dir:
            continue
        if _remove_tree(folder):
            removed.append(folder)
        else:
            failed.append(folder)

    if failed:
        rel_fail = [os.path.relpath(p, RAW_DIR) for p in failed]
        _append_log([
            "WARN  未能删除 embedded 文件夹: {}".format(", ".join(rel_fail)),
        ], flush=FLUSH_LOG_EACH)

    return removed


def _append_log(lines, flush=True):
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    with codecs.open(LOG_PATH, "a", "utf-8") as f:
        for line in lines:
            if not isinstance(line, unicode):
                line = unicode(line, "utf-8", "replace")
            f.write(line + u"\n")
        if flush:
            f.flush()


def _format_errors(result, max_lines=8):
    errors = result.get("errors") or []
    if not errors:
        return result.get("stage", "unknown")
    head = errors[:max_lines]
    text = " | ".join(e.replace("\n", " ") for e in head)
    if len(errors) > max_lines:
        text += " ... (+{} 条)".format(len(errors) - max_lines)
    return text


def batch_export():
    original_doc = sc.doc
    v14 = _load_v14_module()
    export_fn = v14.export_and_qc_rhino_dataset
    suggested_name_fn = v14._suggested_json_name

    three_dm_files = _iter_3dm_files(RAW_DIR)
    if not three_dm_files:
        msg = "在以下目录未找到任何 .3dm 文件：\n{}".format(RAW_DIR)
        v14.ui_show_message(msg, title="批量导出 V14")
        return

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    started = datetime.now()
    _append_log([
        "",
        "=" * 72,
        "批量导出开始 {}".format(started.strftime("%Y-%m-%d %H:%M:%S")),
        "BATCH_FOLDER: {}".format(BATCH_FOLDER),
        "RAW: {}".format(RAW_DIR),
        "OUT: {}".format(OUT_DIR),
        "共 {} 个 .3dm".format(len(three_dm_files)),
    ])

    ok_list = []
    skip_list = []
    fail_list = []

    total = len(three_dm_files)
    for idx, three_dm_path in enumerate(three_dm_files, start=1):
        json_name = _json_name_from_3dm(three_dm_path)
        out_path = os.path.join(OUT_DIR, json_name)
        rel_raw = os.path.relpath(three_dm_path, RAW_DIR)

        print("[{}/{}] {}".format(idx, total, rel_raw))

        if SKIP_EXISTING and os.path.isfile(out_path):
            skip_list.append((rel_raw, json_name, "JSON 已存在"))
            _append_log(["SKIP  {}  ->  {} (已存在)".format(rel_raw, json_name)], flush=FLUSH_LOG_EACH)
            continue

        opened, open_err = _open_3dm(three_dm_path)
        if not opened:
            fail_list.append((rel_raw, open_err))
            _append_log(["FAIL  {}  ->  打开失败: {}".format(rel_raw, open_err)], flush=FLUSH_LOG_EACH)
            continue

        result = None
        try:
            doc_base = os.path.basename(three_dm_path)
            suggested = suggested_name_fn(doc_base)
            if suggested != json_name:
                json_name = suggested
                out_path = os.path.join(OUT_DIR, json_name)

            result = export_fn(interactive=False, save_path=out_path)
        except Exception as exc:
            fail_list.append((rel_raw, "{}: {}".format(type(exc).__name__, exc)))
            _append_log([
                "FAIL  {}  ->  异常: {}".format(rel_raw, exc),
                traceback.format_exc(),
            ], flush=FLUSH_LOG_EACH)
            continue
        finally:
            _close_active_doc(save=False)
            _remove_embedded_files_folder(three_dm_path)

        if result and result.get("ok"):
            rooms = result.get("total_rooms", "?")
            ok_list.append((rel_raw, json_name, rooms))
            _append_log([
                "OK    {}  ->  {}  ({} rooms, QC {})".format(
                    rel_raw, json_name, rooms, result.get("qc_version", "?")),
            ], flush=FLUSH_LOG_EACH)
        else:
            reason = _format_errors(result or {"stage": "no_result"})
            fail_list.append((rel_raw, reason))
            _append_log(["FAIL  {}  ->  QC/导出: {}".format(rel_raw, reason)], flush=FLUSH_LOG_EACH)

    finished = datetime.now()
    elapsed = finished - started
    summary_lines = [
        "-" * 72,
        "完成 {}  耗时 {}".format(finished.strftime("%Y-%m-%d %H:%M:%S"), elapsed),
        "成功: {}  跳过: {}  失败: {}".format(len(ok_list), len(skip_list), len(fail_list)),
    ]
    if fail_list:
        summary_lines.append("")
        summary_lines.append("失败明细：")
        for rel_raw, reason in fail_list:
            summary_lines.append("  - {} : {}".format(rel_raw, reason))
    summary_lines.append("=" * 72)
    _append_log(summary_lines)

    summary_text = "\n".join([
        "批量导出 V14 完成",
        "",
        "成功: {}  跳过: {}  失败: {}".format(len(ok_list), len(skip_list), len(fail_list)),
        "输出目录: {}".format(OUT_DIR),
        "日志: {}".format(LOG_PATH),
    ])
    if fail_list:
        summary_text += "\n\n失败文件（前 10 个）：\n"
        for rel_raw, reason in fail_list[:10]:
            summary_text += "  • {} — {}\n".format(rel_raw, reason[:120])
        if len(fail_list) > 10:
            summary_text += "  … 另有 {} 个，见日志\n".format(len(fail_list) - 10)

    v14.ui_show_message(summary_text, title="批量导出 V14")

    if original_doc is not None:
        try:
            sc.doc = original_doc
        except Exception:
            pass


if __name__ == "__main__":
    batch_export()
