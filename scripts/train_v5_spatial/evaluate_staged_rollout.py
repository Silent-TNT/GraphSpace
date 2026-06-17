#!/usr/bin/env python3
"""Autoregressively replay the seven staged generation actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from staged_dataset import (
    CHANNEL_INDEX,
    DEFAULT_DATA_DIR,
    StagedSpatialDataset,
    collate_staged,
    load_volumes,
    stage_input,
    stage_target,
)
from staged_model import StagedSpatialPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-houses", type=int, default=2)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return float(intersection / union) if union else 1.0


def apply_prediction(
    state: np.ndarray,
    stage_id: int,
    prediction: np.ndarray,
    cut_ratio: float,
) -> None:
    first = prediction[0]
    if stage_id == 0:
        state[CHANNEL_INDEX["protected_stairs"]] = first
    elif stage_id == 1:
        cut_cell = min(max(round(cut_ratio * 20), 1), 19)
        state[CHANNEL_INDEX["floor_boundary"], :, :, cut_cell] = 1
    elif stage_id == 2:
        state[CHANNEL_INDEX["building_envelope"]] = first
        state[CHANNEL_INDEX["explicit_empty"]] = prediction[1]
    elif stage_id == 3:
        state[CHANNEL_INDEX["traffic_reserve"]] = first
    elif stage_id == 4:
        state[CHANNEL_INDEX["rigid_functions"]] = first
    elif stage_id == 5:
        state[CHANNEL_INDEX["service_spaces"]] = first


def floor_slice(volume: np.ndarray, floor: int) -> np.ndarray:
    z0 = floor * 10
    return volume[:, :, z0 : z0 + 10].any(axis=2)


def touches(left: np.ndarray, right: np.ndarray) -> bool:
    expanded = left.copy()
    expanded[1:, :] |= left[:-1, :]
    expanded[:-1, :] |= left[1:, :]
    expanded[:, 1:] |= left[:, :-1]
    expanded[:, :-1] |= left[:, 1:]
    return bool(np.logical_and(expanded, right).any())


def constraint_report(state: np.ndarray) -> dict:
    building = state[CHANNEL_INDEX["building_envelope"]] > 0
    empty = state[CHANNEL_INDEX["explicit_empty"]] > 0
    stairs = state[CHANNEL_INDEX["protected_stairs"]] > 0
    traffic = state[CHANNEL_INDEX["traffic_reserve"]] > 0
    rigid = state[CHANNEL_INDEX["rigid_functions"]] > 0
    service = state[CHANNEL_INDEX["service_spaces"]] > 0
    functions = stairs | traffic | rigid | service
    function_count = max(int(functions.sum()), 1)
    outside_ratio = float(np.logical_and(functions, ~building).sum() / function_count)
    envelope_count = max(int((building | empty).sum()), 1)
    overlap_ratio = float(np.logical_and(building, empty).sum() / envelope_count)
    stair_contacts = []
    rigid_contacts = []
    for floor in range(2):
        stair_contacts.append(
            touches(floor_slice(stairs, floor), floor_slice(traffic, floor))
        )
        rigid_contacts.append(
            touches(floor_slice(rigid, floor), floor_slice(traffic, floor))
        )
    checks = {
        "function_outside_building_ratio": outside_ratio,
        "building_empty_overlap_ratio": overlap_ratio,
        "stairs_contact_traffic_both_floors": all(stair_contacts),
        "rigid_functions_contact_traffic_both_floors": all(rigid_contacts),
    }
    checks["pass"] = (
        outside_ratio <= 0.01
        and overlap_ratio <= 0.01
        and all(stair_contacts)
        and all(rigid_contacts)
    )
    return checks


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    base_channels = int(checkpoint["config"]["base_channels"])
    model = StagedSpatialPolicy(base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = StagedSpatialDataset(args.split, max_houses=args.max_houses)
    house_ids = list(dict.fromkeys(house_id for house_id, _ in dataset.items))
    reports = []
    with torch.no_grad():
        for house_id in house_ids:
            sample_path = DEFAULT_DATA_DIR / f"{house_id}.json"
            volumes = load_volumes(sample_path)
            state = stage_input(volumes, 0)
            stage_reports = []
            for stage_id in range(7):
                dataset_index = dataset.items.index((house_id, stage_id))
                item = dataset[dataset_index]
                item["volume"] = torch.from_numpy(state.copy())
                batch = collate_staged([item])
                output = model(
                    batch["volume"].to(device),
                    batch["nodes"].to(device),
                    batch["node_mask"].to(device),
                    batch["adjacency"].to(device),
                    batch["stage_id"].to(device),
                )
                probability = torch.sigmoid(
                    output["mask_logits"][0]
                ).cpu().numpy()
                prediction = (probability >= args.threshold).astype(np.float32)
                if stage_id == 2:
                    site = state[CHANNEL_INDEX["site"]] > 0
                    stairs = state[CHANNEL_INDEX["protected_stairs"]] > 0
                    building = (probability[0] >= probability[1]) & site
                    empty = (probability[1] > probability[0]) & site
                    building |= stairs
                    empty &= ~stairs
                    prediction[0] = building.astype(np.float32)
                    prediction[1] = empty.astype(np.float32)
                elif stage_id == 3:
                    building = (
                        state[CHANNEL_INDEX["building_envelope"]] > 0
                    )
                    stairs = state[CHANNEL_INDEX["protected_stairs"]] > 0
                    traffic = (prediction[0] > 0) & building
                    prediction[0] = (traffic | stairs).astype(np.float32)
                elif stage_id == 4:
                    building = (
                        state[CHANNEL_INDEX["building_envelope"]] > 0
                    )
                    traffic = state[CHANNEL_INDEX["traffic_reserve"]] > 0
                    prediction[0] = (
                        (prediction[0] > 0) & building & ~traffic
                    ).astype(np.float32)
                elif stage_id == 5:
                    building = (
                        state[CHANNEL_INDEX["building_envelope"]] > 0
                    )
                    traffic = state[CHANNEL_INDEX["traffic_reserve"]] > 0
                    rigid = state[CHANNEL_INDEX["rigid_functions"]] > 0
                    prediction[0] = (
                        (prediction[0] > 0)
                        & building
                        & ~traffic
                        & ~rigid
                    ).astype(np.float32)
                cut_ratio = float(output["cut_ratio"][0].cpu())
                target, valid = stage_target(volumes, stage_id)
                valid_ious = [
                    iou(prediction[channel], target[channel])
                    for channel in range(2)
                    if valid[channel]
                ]
                stage_reports.append(
                    {
                        "stage_id": stage_id,
                        "mask_iou": (
                            sum(valid_ious) / len(valid_ious)
                            if valid_ious
                            else None
                        ),
                        "reachability_probability": (
                            float(
                                torch.sigmoid(
                                    output["reachability_logit"][0]
                                ).cpu()
                            )
                            if stage_id == 6
                            else None
                        ),
                        "cut_ratio": cut_ratio if stage_id == 1 else None,
                        "cut_error": (
                            abs(cut_ratio - 0.5) if stage_id == 1 else None
                        ),
                    }
                )
                apply_prediction(state, stage_id, prediction, cut_ratio)
            mask_values = [
                item["mask_iou"]
                for item in stage_reports
                if item["mask_iou"] is not None
            ]
            reports.append(
                {
                    "house_id": house_id,
                    "mean_mask_iou": sum(mask_values) / len(mask_values),
                    "constraints": constraint_report(state),
                    "stages": stage_reports,
                }
            )
    summary = {
        "checkpoint": str(args.checkpoint),
        "house_count": len(reports),
        "mean_rollout_mask_iou": sum(
            report["mean_mask_iou"] for report in reports
        )
        / max(len(reports), 1),
        "constraint_pass_count": sum(
            report["constraints"]["pass"] for report in reports
        ),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
