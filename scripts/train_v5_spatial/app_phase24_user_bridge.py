#!/usr/bin/env python3
"""Gradio UI for the Phase24 user-condition bridge."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr


ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / ".venv-v5-cuda" / "Scripts" / "python.exe"
BRIDGE_SCRIPT = ROOT / "scripts" / "train_v5_spatial" / "generate_phase24_from_user_conditions.py"
VIS_SCRIPT = ROOT / "scripts" / "train_v5_spatial" / "visualize_user_generation.py"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "phase24_user_bridge_ui"
DEFAULT_DECODER = ROOT / "outputs" / "v6_multipart_graph_topology_linked_size_full_phase24" / "decoder.pt"
COARSE_HEAD = ROOT / "outputs" / "v6_coarse_layout_head_full_phase24" / "coarse_layout_head.pt"
GRAPH_COARSE_MODEL = ROOT / "outputs" / "v6_graph_coarse_layout_model_regularized_full_phase24" / "graph_coarse_layout_model.pt"
GRAPH_TOPOLOGY_MODEL = ROOT / "outputs" / "v6_graph_topology_generator_position_full_phase24" / "graph_topology_generator.pt"

ROOM_LABELS = [
    ("entryway", "Entryway"),
    ("living_room", "Living room"),
    ("dining_room", "Dining room"),
    ("kitchen", "Kitchen"),
    ("bedroom", "Bedroom"),
    ("bathroom", "Bathroom"),
    ("corridor", "Corridor"),
    ("stairs", "Stairs"),
    ("utility", "Utility"),
    ("balcony", "Balcony"),
    ("multi_purpose", "Multi-purpose"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counts_from_inputs(values: tuple[Any, ...], infer_missing: bool) -> dict[str, int]:
    counts = {}
    for (room_type, _label), value in zip(ROOM_LABELS, values):
        count = int(value or 0)
        if count > 0:
            counts[room_type] = count
        elif not infer_missing:
            counts[room_type] = 0
    return counts


def _status_text(summary: dict[str, Any], output_dir: Path) -> str:
    request = summary["request"]
    topology = summary["topology"]
    lines = [
        f"Output: {output_dir}",
        f"Site: {request['site_x']:.0f} x {request['site_y']:.0f} mm",
        f"Seed: {request['seed']}",
        f"Coarse layout: {summary.get('coarse_layout_source', 'rule_packer')}",
        f"Topology source: {summary.get('topology_source', 'program_prior_user_bridge')}",
        f"P0 pass: {summary['p0_pass']}",
        f"P1 hard geometry pass: {summary['p1_hard_geometry_pass']}",
        f"P1 spatial organization pass: {summary['p1_spatial_organization_pass']}",
        (
            "Target topology: "
            f"{topology['realized_edge_count']}/{topology['target_edge_count']} "
            f"= {topology['realization_rate']:.2%}"
        ),
        f"Group count: {summary['group_count']}",
        f"Generated blocks: {summary['rectangular_part_count']}",
        "Functional counts: "
        + ", ".join(f"{key}={value}" for key, value in sorted(request["functional_group_counts"].items())),
    ]
    return "\n".join(lines)


def _html_embed(path: Path) -> str:
    if not path.exists():
        return "<p>3D HTML was not generated.</p>"
    return path.read_text(encoding="utf-8")


def generate(
    site_x: float,
    site_y: float,
    seed: int,
    infer_missing: bool,
    coarse_layout_strategy: str,
    use_coarse_head: bool,
    max_topology_move_mm: float,
    max_size_adjustment_mm: float,
    *room_values: Any,
) -> tuple[str, str, str, str, str]:
    if not VENV_PYTHON.exists():
        raise gr.Error(f"CUDA venv python not found: {VENV_PYTHON}")
    if not DEFAULT_DECODER.exists():
        raise gr.Error(f"Phase24 decoder checkpoint not found: {DEFAULT_DECODER}")

    site_x = float(site_x)
    site_y = float(site_y)
    seed = int(seed)
    counts = _counts_from_inputs(tuple(room_values), bool(infer_missing))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_OUTPUT_ROOT / f"site_{int(site_x)}x{int(site_y)}_seed{seed}_{timestamp}"

    command = [
        str(VENV_PYTHON),
        str(BRIDGE_SCRIPT),
        "--site-x",
        str(site_x),
        "--site-y",
        str(site_y),
        "--seed",
        str(seed),
        "--decoder-checkpoint",
        str(DEFAULT_DECODER),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--max-topology-move-mm",
        str(float(max_topology_move_mm)),
        "--max-size-adjustment-mm",
        str(float(max_size_adjustment_mm)),
        "--coarse-layout-strategy",
        str(coarse_layout_strategy),
    ]
    if counts:
        command.extend(["--rooms-json", json.dumps(counts, ensure_ascii=False)])
    if use_coarse_head:
        if not COARSE_HEAD.exists():
            raise gr.Error(f"Coarse layout checkpoint not found: {COARSE_HEAD}")
        command.extend(["--coarse-layout-checkpoint", str(COARSE_HEAD)])
    if coarse_layout_strategy == "learned_graph":
        if not GRAPH_COARSE_MODEL.exists():
            raise gr.Error(f"Graph coarse layout checkpoint not found: {GRAPH_COARSE_MODEL}")
        command.extend(["--graph-coarse-layout-checkpoint", str(GRAPH_COARSE_MODEL)])
    if GRAPH_TOPOLOGY_MODEL.exists():
        command.extend(["--topology-generator-checkpoint", str(GRAPH_TOPOLOGY_MODEL)])

    try:
        subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True, timeout=600)
        subprocess.run([sys.executable, str(VIS_SCRIPT), "--input-dir", str(output_dir)], cwd=ROOT, check=True, text=True, capture_output=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
        raise gr.Error(details[-3500:] or str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise gr.Error(f"Generation timed out: {exc}") from exc

    summary = _read_json(output_dir / "summary.json")
    return (
        _html_embed(output_dir / "layout_3d_interactive.html"),
        str(output_dir / "floor_1.png"),
        str(output_dir / "floor_2.png"),
        str(output_dir / "topology.png"),
        _status_text(summary, output_dir),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="GraphSpace Phase24 User Generator") as demo:
        gr.Markdown("# GraphSpace Phase24 User Generator")
        gr.Markdown("输入场地长宽和可选功能数量，生成两层功能体块。默认使用当前回归更好的 graph-aware Phase24 bridge。")

        with gr.Row():
            site_x = gr.Slider(9000, 30000, value=18000, step=300, label="Site X / 长度 (mm)")
            site_y = gr.Slider(9000, 30000, value=15000, step=300, label="Site Y / 宽度 (mm)")
            seed = gr.Number(value=42, label="Seed", precision=0)

        with gr.Row():
            infer_missing = gr.Checkbox(value=True, label="Auto-fill missing functions / 自动补全未指定功能")
            coarse_layout_strategy = gr.Radio(
                choices=["rule", "graph", "learned_graph"],
                value="learned_graph",
                label="Coarse layout strategy / 粗布局策略",
            )
            use_coarse_head = gr.Checkbox(value=False, label="Use experimental coarse head / 使用实验 coarse head")

        with gr.Row():
            max_topology_move_mm = gr.Slider(300, 3600, value=1800, step=300, label="Max topology move (mm)")
            max_size_adjustment_mm = gr.Slider(0, 1200, value=600, step=300, label="Max size adjustment (mm)")

        gr.Markdown("功能数量。留 0 且开启自动补全时，系统会根据场地补足。")
        room_inputs = []
        for start in range(0, len(ROOM_LABELS), 4):
            with gr.Row():
                for room_type, label in ROOM_LABELS[start : start + 4]:
                    default = {
                        "entryway": 1,
                        "living_room": 1,
                        "dining_room": 1,
                        "kitchen": 1,
                        "bedroom": 3,
                        "bathroom": 2,
                        "corridor": 2,
                        "stairs": 1,
                        "balcony": 1,
                    }.get(room_type, 0)
                    room_inputs.append(gr.Number(value=default, minimum=0, label=label, precision=0))

        run = gr.Button("Generate / 生成", variant="primary")

        html_3d = gr.HTML(label="Interactive 3D")
        with gr.Row():
            floor_1 = gr.Image(label="Floor 1", type="filepath")
            floor_2 = gr.Image(label="Floor 2", type="filepath")
        with gr.Row():
            topology = gr.Image(label="Topology", type="filepath")
            status = gr.Textbox(label="Metrics / 指标", lines=12)

        run.click(
            generate,
            inputs=[
                site_x,
                site_y,
                seed,
                infer_missing,
                coarse_layout_strategy,
                use_coarse_head,
                max_topology_move_mm,
                max_size_adjustment_mm,
                *room_inputs,
            ],
            outputs=[html_3d, floor_1, floor_2, topology, status],
        )
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7862"))
    build_app().launch(server_name="127.0.0.1", server_port=port)
