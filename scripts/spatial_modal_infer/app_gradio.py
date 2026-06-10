#!/usr/bin/env python3
"""Local Gradio UI for SpatialModal V4 inference."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import gradio as gr

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from layout import build_user_request
from pipeline import count_rooms_by_type, generate_user_layout, load_model
from run_meta import build_run_meta
from visualize import (
    plot_3d_layout_static,
    plot_floor_plan,
    plot_topology_graph,
    save_layout_figures,
)

DEFAULT_WEIGHTS = os.environ.get(
    "SPATIAL_MODAL_WEIGHTS",
    str(ROOT / "weights" / "spatial_modal_cvae_v4_88x88x24_probe_best.pth"),
)
DEFAULT_OUT_DIR = os.environ.get(
    "SPATIAL_MODAL_OUT_DIR",
    str(ROOT / "weights" / "outputs_web"),
)

_model = None
_device = None


def _ensure_model(weights_path: str):
    global _model, _device
    if not weights_path:
        raise gr.Error("Please provide a V4 .pth weight path.")
    p = Path(weights_path)
    if not p.exists():
        raise gr.Error(f"Weight file not found: {p}")
    resolved = str(p.resolve())
    if _model is None or resolved != getattr(_ensure_model, "_loaded", ""):
        _model, _device = load_model(p)
        _ensure_model._loaded = resolved
    return _model, _device, p.resolve()


def _format_counts(counts: dict) -> str:
    if not counts:
        return "(none)"
    return ", ".join(f"{k} x {v}" for k, v in sorted(counts.items()))


def generate(
    weights_path,
    site_x,
    site_y,
    seed_in,
    random_each_time,
    sample_k,
    save_files,
    display_style,
    n_entry,
    n_living,
    n_dining,
    n_kitchen,
    n_bed,
    n_bath,
    n_corr,
    n_stairs,
    n_util,
    n_balc,
    n_multi,
):
    model, device, weights = _ensure_model(weights_path)
    counts = {
        k: int(v)
        for k, v in {
            "entryway": n_entry,
            "living_room": n_living,
            "dining_room": n_dining,
            "kitchen": n_kitchen,
            "bedroom": n_bed,
            "bathroom": n_bath,
            "corridor": n_corr,
            "stairs": n_stairs,
            "utility": n_util,
            "balcony": n_balc,
            "multi_purpose": n_multi,
        }.items()
        if int(v) > 0
    }
    if not counts:
        raise gr.Error("Please set at least one room count greater than 0.")

    if random_each_time:
        seed = random.randint(1, 999999)
    elif int(seed_in) > 0:
        seed = int(seed_in)
    else:
        seed = random.randint(1, 999999)

    req = build_user_request(float(site_x), float(site_y), counts)
    result = generate_user_layout(
        req,
        model,
        device,
        seed=seed,
        sample_k=int(sample_k),
        display_style=display_style,
    )

    plot_rooms = result.get("display_rooms", result["rooms"])
    run_meta = build_run_meta(
        weights,
        sample_k=int(sample_k),
        device=str(device),
        source="gradio",
        display_style=display_style,
    )

    saved_lines = []
    save_error = None
    if save_files:
        try:
            paths = save_layout_figures(
                result,
                req,
                DEFAULT_OUT_DIR,
                weights_path=weights,
                run_meta=run_meta,
                extra_meta={"display_style": display_style},
            )
            saved_lines = [f"{k}: {v}" for k, v in paths.items()]
        except Exception as exc:
            save_error = f"{type(exc).__name__}: {exc}"

    title_tail = f" | seed={result['seed']} | {run_meta['weights_file']}"
    fig3d = plot_3d_layout_static(
        plot_rooms,
        site_x,
        site_y,
        title=f"3D layout{title_tail}",
    )
    floor_layers = result.get("floor_layers")
    fig_f1 = plot_floor_plan(
        plot_rooms,
        1,
        site_x,
        site_y,
        title=f"Floor 1{title_tail}",
        floor_layers=floor_layers,
    )
    fig_f2 = plot_floor_plan(
        plot_rooms,
        2,
        site_x,
        site_y,
        title=f"Floor 2{title_tail}",
        floor_layers=floor_layers,
    )
    fig_topo = plot_topology_graph(
        result.get("graph"),
        result.get("pos"),
        result.get("edge_types"),
        title=f"Input topology | seed={result['seed']}",
    )

    shown = count_rooms_by_type(plot_rooms)
    pred_sig = int(result["pred"].astype(int).sum()) if result.get("pred") is not None else 0
    qscore = result.get("quality_score")
    qline = f"{qscore:.3f}" if isinstance(qscore, (int, float)) else "n/a"

    input_signature = (
        f"bedroom={counts.get('bedroom', 0)}, "
        f"bathroom={counts.get('bathroom', 0)}, "
        f"total_rooms={sum(counts.values())}, "
        f"sample_k={int(sample_k)}"
    )
    metrics = (
        f"Generated at: {run_meta['generated_at']}\n"
        f"Device: {device}\n"
        f"Weight: {weights}\n"
        f"Input signature: {input_signature}\n"
        f"Seed: {result['seed']}"
        f"{' (random this run)' if random_each_time else ''}\n"
        f"Display style: {display_style}\n"
        f"Decode mode: {result['decode_mode']}\n"
        f"Non-empty voxels: {result['n_occ']}\n"
        f"Voxel signature: {pred_sig}\n"
        f"Quality score: {qline}\n"
        f"Quality metrics: {result.get('quality_metrics') or {}}\n"
        f"Display source: {result.get('display_source')}\n"
        f"Requested rooms: {_format_counts(counts)}\n"
        f"Shown rooms: {_format_counts(shown)}"
    )
    if save_files:
        if save_error:
            metrics += f"\nSave failed: {save_error}"
        else:
            metrics += f"\nSaved to: {DEFAULT_OUT_DIR}\n" + "\n".join(saved_lines)

    return fig3d, fig_f1, fig_f2, fig_topo, metrics, result["seed"]


def build_app():
    with gr.Blocks(title="SpatialModal V4 Inference") as demo:
        gr.Markdown("# SpatialModal V4 Inference")
        gr.Markdown("Use a V4 88x88x24 weight file, then generate a layout from site size and room counts.")

        weights = gr.Textbox(
            label="V4 weight path (.pth)",
            value=DEFAULT_WEIGHTS,
            placeholder=str(ROOT / "weights" / "spatial_modal_cvae_v4_88x88x24_probe_best.pth"),
        )

        with gr.Row():
            site_x = gr.Slider(6000, 30000, value=18000, step=300, label="Site X (mm)")
            site_y = gr.Slider(6000, 30000, value=15000, step=300, label="Site Y (mm)")
            seed = gr.Number(123, label="Seed", precision=0)
            sample_k = gr.Slider(1, 16, value=1, step=1, label="Samples (K)")

        with gr.Row():
            random_each_time = gr.Checkbox(value=False, label="Use a new random seed each run")
            save_files = gr.Checkbox(value=False, label=f"Save PNG/JSON to {DEFAULT_OUT_DIR}")
            display_style = gr.Radio(
                choices=["regions", "footprint", "boxes"],
                value="regions",
                label="3D display style",
            )

        gr.Markdown("Room counts")
        with gr.Row():
            n_entry = gr.Number(1, label="Entry", precision=0)
            n_living = gr.Number(1, label="Living", precision=0)
            n_dining = gr.Number(1, label="Dining", precision=0)
            n_kitchen = gr.Number(1, label="Kitchen", precision=0)
        with gr.Row():
            n_bed = gr.Number(3, label="Bedroom", precision=0)
            n_bath = gr.Number(2, label="Bathroom", precision=0)
            n_corr = gr.Number(2, label="Corridor", precision=0)
            n_stairs = gr.Number(1, label="Stairs", precision=0)
        with gr.Row():
            n_util = gr.Number(0, label="Utility", precision=0)
            n_balc = gr.Number(1, label="Balcony", precision=0)
            n_multi = gr.Number(0, label="Multi-purpose", precision=0)

        btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            out3d = gr.Plot(label="3D layout")
            out_topo = gr.Plot(label="Input topology")
        with gr.Row():
            out_f1 = gr.Plot(label="Floor 1")
            out_f2 = gr.Plot(label="Floor 2")
        out_txt = gr.Textbox(label="Metrics and saved paths", lines=14)

        btn.click(
            generate,
            inputs=[
                weights,
                site_x,
                site_y,
                seed,
                random_each_time,
                sample_k,
                save_files,
                display_style,
                n_entry,
                n_living,
                n_dining,
                n_kitchen,
                n_bed,
                n_bath,
                n_corr,
                n_stairs,
                n_util,
                n_balc,
                n_multi,
            ],
            outputs=[out3d, out_f1, out_f2, out_topo, out_txt, seed],
        )
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    build_app().launch(server_name="127.0.0.1", server_port=port)
