#!/usr/bin/env python3
"""Generate and visualize one user condition with exactly three seeds."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 42, 123)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-x", type=float, required=True)
    parser.add_argument("--site-y", type=float, required=True)
    parser.add_argument("--rooms-json")
    parser.add_argument("--rooms-file", type=Path)
    parser.add_argument("--staged-checkpoint", type=Path, required=True)
    parser.add_argument("--instance-checkpoint", type=Path, required=True)
    parser.add_argument("--program-prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visualization-python", default="python")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in SEEDS:
        seed_dir = args.output_dir / f"seed_{seed}"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_v5_spatial" / "generate_from_user_conditions.py"),
            "--site-x",
            str(args.site_x),
            "--site-y",
            str(args.site_y),
            "--seed",
            str(seed),
            "--staged-checkpoint",
            str(args.staged_checkpoint),
            "--instance-checkpoint",
            str(args.instance_checkpoint),
            "--program-prior",
            str(args.program_prior),
            "--output-dir",
            str(seed_dir),
            "--device",
            args.device,
        ]
        if args.rooms_json:
            command.extend(["--rooms-json", args.rooms_json])
        if args.rooms_file:
            command.extend(["--rooms-file", str(args.rooms_file)])
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(
            [
                args.visualization_python,
                str(ROOT / "scripts" / "train_v5_spatial" / "visualize_user_generation.py"),
                "--input-dir",
                str(seed_dir),
            ],
            cwd=ROOT,
            check=True,
        )
        summaries.append(
            json.loads((seed_dir / "summary.json").read_text(encoding="utf-8"))
        )
    manifest = {
        "schema": "graphspace_v5_three_seed_generation_v1",
        "site_x": args.site_x,
        "site_y": args.site_y,
        "seeds": list(SEEDS),
        "room_counts_source": (
            "user_partial_plus_training_data"
            if args.rooms_json or args.rooms_file
            else "training_data_knn"
        ),
        "runs": summaries,
    }
    (args.output_dir / "three_seed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
