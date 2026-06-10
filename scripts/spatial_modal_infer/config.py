"""Inference defaults aligned with the V4 88x88x24 training notebook."""
from __future__ import annotations

import json
from pathlib import Path

VOXEL_SIZE = 300.0
RES_X, RES_Y, RES_Z = 88, 88, 24
DECODER_INIT_X = (RES_X + 15) // 16
DECODER_INIT_Y = (RES_Y + 15) // 16
DECODER_INIT_Z = (RES_Z + 15) // 16

CHANNEL_MAP = {
    "empty": 0,
    "entryway": 1,
    "living_room": 2,
    "dining_room": 3,
    "kitchen": 4,
    "bedroom": 5,
    "bathroom": 6,
    "corridor": 7,
    "stairs": 8,
    "utility": 9,
    "balcony": 10,
    "multi_purpose": 11,
}
ROOM_TYPES = [k for k in CHANNEL_MAP if k != "empty"]
NUM_CHANNELS = len(CHANNEL_MAP)
NUM_ROOM_TYPES = len(ROOM_TYPES)
NODE_IN_DIM = 8
COND_DIM = 3 + NUM_ROOM_TYPES + 3

LATENT_DIM = 256
HIDDEN_DIM = 128
COND_COUNT_SCALE = 4.0
COND_EMBED_DIM = 64
AUX_COUNT_WEIGHT = 0.1

DEFAULT_WEIGHT_NAME = "spatial_modal_cvae_v4_88x88x24.pth"
DEFAULT_PROBE_WEIGHT_NAME = "spatial_modal_cvae_v4_88x88x24_probe_best.pth"
MODEL_VERSION = "v4_88x88x24"

LIGHTING_ACCESS_MAP = {"direct": 1.0, "indirect": 0.5, "none": 0.0}

TYPE_COLOR_DICT = {
    "entryway": "#808080",
    "living_room": "#FF8000",
    "dining_room": "#FFFF00",
    "kitchen": "#00FF00",
    "bedroom": "#0000FF",
    "bathroom": "#FF0000",
    "corridor": "#B0B0FF",
    "stairs": "#A000FF",
    "utility": "#3CB371",
    "balcony": "#00FFFF",
    "multi_purpose": "#FFC0CB",
}

CN_NAMES = {
    "entryway": "entryway",
    "living_room": "living room",
    "dining_room": "dining room",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "corridor": "corridor",
    "stairs": "stairs",
    "utility": "utility",
    "balcony": "balcony",
    "multi_purpose": "multi-purpose",
}

DEFAULT_ROOM_SIZE = {
    "entryway": (2400, 2400, 3000),
    "living_room": (6000, 4500, 3000),
    "dining_room": (3600, 3300, 3000),
    "kitchen": (3300, 3000, 3000),
    "bedroom": (3600, 3600, 3000),
    "bathroom": (2400, 2400, 3000),
    "corridor": (1800, 2400, 3000),
    "stairs": (3000, 3000, 6000),
    "utility": (2400, 2400, 3000),
    "balcony": (3000, 1800, 3000),
    "multi_purpose": (3600, 3300, 3000),
}


def load_model_config(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)
