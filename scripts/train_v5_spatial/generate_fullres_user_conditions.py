#!/usr/bin/env python3
"""Generate an exact 300 mm instance partition from user conditions."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from fullres_dataset import (
    FLOOR_CELLS,
    GRID_X,
    GRID_Y,
    GRID_Z,
    build_input_volume,
)
from fullres_model import FullResolutionGraphVoxelModel
from generate_from_user_conditions import (
    ROOT,
    TEST_CASES,
    VOXEL_MM,
    graph_item,
    request_graph,
    validate_request,
)
from program_prior import ProgramPrior
from train_fullres import assignment_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(TEST_CASES))
    parser.add_argument("--site-x", type=float)
    parser.add_argument("--site-y", type=float)
    parser.add_argument("--rooms-json")
    parser.add_argument("--rooms-file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--program-prior",
        type=Path,
        default=ROOT / "data" / "phase8_program_prior" / "program_prior.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def centered_site_mask(site_cells: list[int]) -> tuple[np.ndarray, list[int]]:
    x0 = (GRID_X - site_cells[0]) // 2
    y0 = (GRID_Y - site_cells[1]) // 2
    mask = np.zeros((GRID_X, GRID_Y), dtype=np.uint8)
    mask[x0 : x0 + site_cells[0], y0 : y0 + site_cells[1]] = 1
    return mask, [x0, y0, x0 + site_cells[0], y0 + site_cells[1]]


def graph_tensors(model_graph: dict, device: torch.device) -> tuple:
    item = graph_item(model_graph)
    nodes = item["nodes"].unsqueeze(0).to(device)
    node_mask = torch.ones((1, nodes.shape[1]), device=device)
    adjacency = torch.zeros(
        (1, 2, nodes.shape[1], nodes.shape[1]),
        device=device,
    )
    for source, target, relation in model_graph["edges"]:
        adjacency[0, relation, source, target] = 1.0
    return nodes, node_mask, adjacency


def allowed_volume(nodes: list[dict], site_mask: np.ndarray) -> np.ndarray:
    allowed = np.zeros((len(nodes), GRID_X, GRID_Y, GRID_Z), dtype=bool)
    for index, node in enumerate(nodes):
        if node["floor_1"]:
            allowed[index, :, :, :FLOOR_CELLS] = site_mask[:, :, None]
        if node["floor_2"]:
            allowed[index, :, :, FLOOR_CELLS:] = site_mask[:, :, None]
    return allowed


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for start in np.argwhere(mask):
        start_tuple = tuple(int(value) for value in start)
        if visited[start_tuple]:
            continue
        queue = deque([start_tuple])
        visited[start_tuple] = True
        cells = []
        while queue:
            x, y, z = queue.popleft()
            cells.append((x, y, z))
            for dx, dy, dz in (
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ):
                neighbor = (x + dx, y + dy, z + dz)
                if not (
                    0 <= neighbor[0] < mask.shape[0]
                    and 0 <= neighbor[1] < mask.shape[1]
                    and 0 <= neighbor[2] < mask.shape[2]
                ):
                    continue
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(np.asarray(cells, dtype=np.int16))
    return sorted(components, key=len, reverse=True)


def decode_assignments(
    output: dict[str, torch.Tensor],
    node_mask: torch.Tensor,
    nodes: list[dict],
    site_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = output["instance_logits"].clone()
    allowed = torch.from_numpy(allowed_volume(nodes, site_mask)).to(logits.device)
    logits[0] = logits[0].masked_fill(~allowed, -20.0)
    empty_logits = output["empty_logits"].clone()
    site_3d = torch.from_numpy(
        np.repeat(site_mask[:, :, None], GRID_Z, axis=2).astype(bool)
    ).to(logits.device)
    empty_logits[0] = empty_logits[0].masked_fill(~site_3d, 20.0)
    constrained = {
        "instance_logits": logits,
        "empty_logits": empty_logits,
    }
    _, probabilities = assignment_probabilities(constrained, node_mask)
    assignments = probabilities.argmax(dim=1)[0].cpu().numpy().astype(np.uint16)
    assignments[~site_3d.cpu().numpy()] = 0
    return assignments, probabilities[0].cpu().numpy()


def projected_floor(assignments: np.ndarray, floor: int) -> np.ndarray:
    z0 = (floor - 1) * FLOOR_CELLS
    z1 = z0 + FLOOR_CELLS
    layers = assignments[:, :, z0:z1]
    output = np.zeros(layers.shape[:2], dtype=np.uint16)
    for x in range(layers.shape[0]):
        for y in range(layers.shape[1]):
            values = layers[x, y]
            values = values[values > 0]
            if values.size:
                output[x, y] = np.bincount(values).argmax()
    return output


def topology_report(
    assignments: np.ndarray,
    model_graph: dict,
) -> dict:
    masks = assignments[None] == np.arange(
        1,
        len(model_graph["nodes"]) + 1,
    )[:, None, None, None]
    required = {
        tuple(sorted((int(left), int(right))))
        for left, right in model_graph.get("required_edges", [])
    }
    records = []
    for source, target, relation in model_graph["edges"]:
        if source >= target:
            continue
        left = masks[source]
        right = masks[target]
        if relation == 0:
            contact = (
                (left[1:] & right[:-1]).any()
                or (left[:-1] & right[1:]).any()
                or (left[:, 1:] & right[:, :-1]).any()
                or (left[:, :-1] & right[:, 1:]).any()
            )
        else:
            contact = (
                (left[:, :, 1:] & right[:, :, :-1]).any()
                or (left[:, :, :-1] & right[:, :, 1:]).any()
            )
        records.append(
            {
                "source": model_graph["nodes"][source]["instance_token"],
                "target": model_graph["nodes"][target]["instance_token"],
                "relation": "vertical" if relation else "horizontal",
                "required": tuple(sorted((source, target))) in required,
                "realized": bool(contact),
            }
        )
    required_records = [record for record in records if record["required"]]
    return {
        "target_edge_count": len(records),
        "realized_edge_count": sum(record["realized"] for record in records),
        "realization_rate": sum(record["realized"] for record in records)
        / max(len(records), 1),
        "required_edge_count": len(required_records),
        "required_realized_edge_count": sum(
            record["realized"] for record in required_records
        ),
        "required_realization_rate": sum(
            record["realized"] for record in required_records
        )
        / max(len(required_records), 1),
        "edges": records,
    }


def room_records(
    assignments: np.ndarray,
    model_graph: dict,
    placement: list[int],
) -> tuple[list[dict], list[str]]:
    rooms = []
    empty = []
    canvas_x0, canvas_y0 = placement[:2]
    for index, node in enumerate(model_graph["nodes"]):
        mask = assignments == index + 1
        components = connected_components(mask)
        if not components:
            empty.append(node["instance_token"])
            rooms.append(
                {
                    "id": node["instance_token"],
                    "type": node["type"],
                    "requested_floors": [
                        floor
                        for floor, key in ((1, "floor_1"), (2, "floor_2"))
                        if node[key]
                    ],
                    "voxel_count": 0,
                    "component_count": 0,
                }
            )
            continue
        cells = np.argwhere(mask)
        mins = cells.min(axis=0)
        maxs = cells.max(axis=0) + 1
        floors = sorted(set((cells[:, 2] // FLOOR_CELLS + 1).tolist()))
        rooms.append(
            {
                "id": node["instance_token"],
                "type": node["type"],
                "requested_floors": [
                    floor
                    for floor, key in ((1, "floor_1"), (2, "floor_2"))
                    if node[key]
                ],
                "predicted_floors": floors,
                "voxel_count": int(mask.sum()),
                "floor_area_m2": {
                    str(floor): round(
                        float(
                            np.any(
                                mask[
                                    :,
                                    :,
                                    (floor - 1) * FLOOR_CELLS
                                    : floor * FLOOR_CELLS,
                                ],
                                axis=2,
                            ).sum()
                            * 0.09
                        ),
                        2,
                    )
                    for floor in floors
                },
                "component_count": len(components),
                "largest_component_fraction": len(components[0])
                / max(int(mask.sum()), 1),
                "box_min": [
                    int((mins[0] - canvas_x0) * VOXEL_MM),
                    int((mins[1] - canvas_y0) * VOXEL_MM),
                    int(mins[2] * VOXEL_MM),
                ],
                "box_max": [
                    int((maxs[0] - canvas_x0) * VOXEL_MM),
                    int((maxs[1] - canvas_y0) * VOXEL_MM),
                    int(maxs[2] * VOXEL_MM),
                ],
            }
        )
    return rooms, empty


def main() -> None:
    args = parse_args()
    if args.case:
        site_x, site_y = TEST_CASES[args.case]["site"]
        explicit = dict(TEST_CASES[args.case]["rooms"])
    else:
        if args.site_x is None or args.site_y is None:
            raise ValueError("provide --case or --site-x and --site-y")
        site_x, site_y = args.site_x, args.site_y
        payload = (
            json.loads(args.rooms_file.read_text(encoding="utf-8"))
            if args.rooms_file
            else json.loads(args.rooms_json)
            if args.rooms_json
            else {}
        )
        explicit = {str(key): int(value) for key, value in payload.items()}
    prior = ProgramPrior(args.program_prior)
    neighbors = prior.neighbors(site_x, site_y)
    counts, count_evidence = prior.infer_counts(
        neighbors,
        args.seed,
        explicit_counts=explicit,
        infer_missing=not bool(args.case),
    )
    validate_request(site_x, site_y, counts)
    model_graph, topology = request_graph(
        counts,
        site_x,
        site_y,
        args.seed,
        prior,
    )
    topology["count_evidence"] = count_evidence
    site_cells = [
        int(np.floor(site_x / VOXEL_MM)),
        int(np.floor(site_y / VOXEL_MM)),
    ]
    site_mask, placement = centered_site_mask(site_cells)
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    model = FullResolutionGraphVoxelModel(
        spatial_channels=int(config["spatial_channels"]),
        query_channels=int(config["query_channels"]),
        architecture=str(config.get("architecture", "v1")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    nodes, node_mask, adjacency = graph_tensors(model_graph, device)
    volume = torch.from_numpy(build_input_volume(site_mask))[None].to(device)
    with torch.no_grad():
        output = model(volume, nodes, node_mask, adjacency)
        assignments, probabilities = decode_assignments(
            output,
            node_mask,
            model_graph["nodes"],
            site_mask,
        )
    rooms, empty_instances = room_records(
        assignments,
        model_graph,
        placement,
    )
    topology_realization = topology_report(assignments, model_graph)
    floor_grids = np.stack(
        [projected_floor(assignments, floor) for floor in (1, 2)]
    )
    type_ids = np.asarray(
        [0] + [node["type_id"] + 1 for node in model_graph["nodes"]],
        dtype=np.uint8,
    )
    class_grids = type_ids[floor_grids]
    occupancy = [
        float((floor_grids[floor] > 0).sum() / max(site_mask.sum(), 1))
        for floor in range(2)
    ]
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "site_mm": [site_x, site_y, 6000.0],
        "site_cells": site_cells,
        "canvas_placement": placement,
        "seed": args.seed,
        "room_counts": counts,
        "requested_instance_count": len(model_graph["nodes"]),
        "nonempty_instance_count": len(model_graph["nodes"])
        - len(empty_instances),
        "empty_instances": empty_instances,
        "floor_occupancy": occupancy,
        "topology_realization_rate": topology_realization["realization_rate"],
        "required_topology_realization_rate": topology_realization[
            "required_realization_rate"
        ],
        "exact_geometry": "assignment_grid.npz",
        "box_warning": (
            "room box_min/box_max are descriptive bounds only; exact model "
            "geometry is the mutually exclusive 300 mm voxel assignment"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "assignment_grid.npz",
        assignments=assignments,
        probabilities=probabilities.astype(np.float16),
        floor_instance_grid=floor_grids,
        floor_class_grid=class_grids,
        site_mask=site_mask,
    )
    for filename, payload in (
        ("topology.json", topology),
        ("model_graph.json", model_graph),
        ("generated_layout.json", {"metadata": summary, "rooms": rooms}),
        ("topology_realization.json", topology_realization),
        ("summary.json", summary),
    ):
        (args.output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
