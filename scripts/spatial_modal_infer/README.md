# SpatialModal V4 Inference Package

This folder is the standalone inference/runtime package for the V4 88x88x24
conditional CVAE. Training still happens in the Colab notebook; local generation
only needs this package plus a trained `.pth` weight file.

## Required Files

Copy these from Colab/Drive into the same release folder:

| File | Required | Notes |
| --- | --- | --- |
| `spatial_modal_cvae_v4_88x88x24.pth` | Yes | Best validation V4 weight |
| `spatial_modal_cvae_v4_88x88x24_probe_best.pth` | Optional | Probe-best V4 weight for generation comparison |
| `model_config_v4.json` | Optional | Version record for grid, condition semantics, and architecture |

Do not mix V4 weights with the older V3/V470 `model.py`: V4 uses absolute-count
conditions, a condition embedding, GroupNorm + FiLM decoder injection, count-aware
encoder pooling, and a count prediction head.

## Install

```bash
cd scripts/spatial_modal_infer
pip install -r requirements.txt
```

If `torch-geometric` installation fails on Windows, install the wheel matching
your local PyTorch version first, then rerun the command above.

## CLI

```bash
cd scripts/spatial_modal_infer

python run_cli.py ^
  --weights "D:/release/spatial_modal_cvae_v4_88x88x24.pth" ^
  --site-x 18000 --site-y 15000 --seed 123 ^
  --living-room 1 --bedroom 3 --kitchen 1 ^
  --out-dir ./outputs
```

## Gradio UI

PowerShell:

```powershell
cd e:\Documents\GraphSpace\scripts\spatial_modal_infer
$env:SPATIAL_MODAL_WEIGHTS = "e:\Documents\GraphSpace\weights\spatial_modal_cvae_v4_88x88x24_probe_best.pth"
$env:SPATIAL_MODAL_OUT_DIR  = "e:\Documents\GraphSpace\weights\outputs_web"
python app_gradio.py
```

Then open `http://127.0.0.1:7860`.

## Notes

- V4 defaults are in `config.py`: grid `88x88x24`, voxel size `300mm`, condition
  count scale `4.0`, condition embedding dim `64`.
- `graph_utils.py` builds the condition vector as absolute counts divided by
  `COND_COUNT_SCALE`, not per-room proportions.
- Loading old V3/V470 weights requires restoring the matching old model/config.
