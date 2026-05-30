"""
SpaceModal — Gradio Demo 占位脚本（仅本地运行）

论文发表并配置本地模型权重前，请勿部署到公网。
作品集模式：localhost 演示交互流程，不暴露权重与数据。

用法:
    pip install gradio
    python website/demo/gradio_app.py
"""

from __future__ import annotations

import gradio as gr

SITE_X_DEFAULT = 18000
SITE_Y_DEFAULT = 15000

ROOM_TYPES = [
    "living", "dining", "kitchen", "bedroom", "bathroom",
    "study", "storage", "balcony", "entrance", "circulation", "stair",
]
ROOM_LABELS = {
    "living": "客厅",
    "dining": "餐厅",
    "kitchen": "厨房",
    "bedroom": "卧室",
    "bathroom": "卫生间",
    "study": "书房",
    "storage": "储藏",
    "balcony": "阳台",
    "entrance": "玄关",
    "circulation": "交通",
    "stair": "楼梯",
}


def generate_layout(
    site_x: int,
    site_y: int,
    living: int,
    bedroom: int,
    kitchen: int,
    orientation: str,
) -> str:
    """占位生成函数；接入 SpatialModalCVAE 权重后替换。"""
    program = {
        "site_mm": [site_x, site_y],
        "rooms": {
            "living": living,
            "bedroom": bedroom,
            "kitchen": kitchen,
        },
        "orientation": orientation,
    }
    return (
        "【Demo 占位输出】\n\n"
        f"用地: {site_x} × {site_y} mm\n"
        f"功能: 客厅×{living}, 卧室×{bedroom}, 厨房×{kitchen}\n"
        f"朝向: {orientation}\n\n"
        "模型权重配置后将在此返回体素预览与指标。\n"
        f"原始参数: {program}"
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="模态户型 ModalPlan Demo") as demo:
        gr.Markdown(
            "# SpaceModal Demo（本地占位）\n"
            "GraphSpace 条件生成 · **请勿公网部署真实权重**"
        )
        with gr.Row():
            site_x = gr.Number(label="用地 X (mm)", value=SITE_X_DEFAULT, precision=0)
            site_y = gr.Number(label="用地 Y (mm)", value=SITE_Y_DEFAULT, precision=0)
        with gr.Row():
            living = gr.Slider(0, 3, value=1, step=1, label=ROOM_LABELS["living"])
            bedroom = gr.Slider(0, 5, value=2, step=1, label=ROOM_LABELS["bedroom"])
            kitchen = gr.Slider(0, 2, value=1, step=1, label=ROOM_LABELS["kitchen"])
        orientation = gr.Dropdown(
            choices=["S", "N", "E", "W"],
            value="S",
            label="主朝向（南向基准）",
        )
        btn = gr.Button("生成体块方案", variant="primary")
        out = gr.Textbox(label="输出", lines=12)
        btn.click(
            generate_layout,
            inputs=[site_x, site_y, living, bedroom, kitchen, orientation],
            outputs=out,
        )
        gr.Markdown(
            "基于 [SpaceModal](https://spacemodal.com) / GraphSpace 研究框架"
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860)
