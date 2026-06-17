"""Decode noisy V5 dense predictions into floor and building instances."""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np


@dataclass(frozen=True)
class DecodeConfig:
    boundary_weight: float = 1.5
    vote_weight: float = 1.0
    spatial_weight: float = 0.15
    center_nms_radius: float = 3.0
    minimum_instance_cells: int = 2
    cross_floor_threshold: float = 0.5
    cross_floor_overlap_threshold: float = 0.15


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def select_vote_centers(
    cells: np.ndarray,
    offsets: np.ndarray,
    heatmap: np.ndarray,
    count: int,
    radius: float,
) -> list[tuple[float, float]]:
    if count <= 0 or len(cells) == 0:
        return []
    votes = cells.astype(np.float32) + offsets[:, cells[:, 0], cells[:, 1]].T
    weights = 0.2 + heatmap[cells[:, 0], cells[:, 1]]
    candidates = []
    for index, vote in enumerate(votes):
        distance = np.linalg.norm(votes - vote[None], axis=1)
        score = float(weights[distance <= radius].sum())
        candidates.append((score, float(vote[0]), float(vote[1])))
    centers = []
    for _, x, y in sorted(candidates, reverse=True):
        if all((x - cx) ** 2 + (y - cy) ** 2 > radius ** 2 for cx, cy in centers):
            centers.append((x, y))
        if len(centers) == count:
            break
    if not centers:
        centers.append(tuple(cells.mean(axis=0).tolist()))
    while len(centers) < count:
        nearest = np.min(
            [
                (cells[:, 0] - cx) ** 2 + (cells[:, 1] - cy) ** 2
                for cx, cy in centers
            ],
            axis=0,
        )
        x, y = cells[int(nearest.argmax())]
        centers.append((float(x), float(y)))
    return centers


def assign_cells(
    cells: np.ndarray,
    offsets: np.ndarray,
    boundary_probability: np.ndarray,
    centers: list[tuple[float, float]],
    config: DecodeConfig,
) -> np.ndarray:
    """Assign cells by connected multi-source growth from predicted centers."""
    votes = cells.astype(np.float32) + offsets[:, cells[:, 0], cells[:, 1]].T
    center_array = np.asarray(centers, dtype=np.float32)
    cell_lookup = {
        (int(cell[0]), int(cell[1])): index for index, cell in enumerate(cells)
    }
    labels = np.full(len(cells), -1, dtype=np.int32)
    distances = np.full(len(cells), np.inf, dtype=np.float32)
    queue: list[tuple[float, int, int]] = []
    used_seeds = set()
    for label, center in enumerate(center_array):
        seed_order = np.argsort(np.linalg.norm(cells - center[None], axis=1))
        seed_index = next(
            (int(value) for value in seed_order if int(value) not in used_seeds),
            int(seed_order[0]),
        )
        used_seeds.add(seed_index)
        labels[seed_index] = label
        distances[seed_index] = 0.0
        heapq.heappush(queue, (0.0, seed_index, label))

    while queue:
        distance, index, label = heapq.heappop(queue)
        if distance > float(distances[index]) + 1e-6 or labels[index] != label:
            continue
        x, y = (int(value) for value in cells[index])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_index = cell_lookup.get((x + dx, y + dy))
            if neighbor_index is None:
                continue
            vote_cost = float(
                np.linalg.norm(votes[neighbor_index] - center_array[label])
            )
            boundary_cost = float(boundary_probability[x + dx, y + dy])
            new_distance = distance + (
                1.0
                + config.vote_weight * 0.15 * vote_cost
                + config.boundary_weight * boundary_cost
            )
            if new_distance < float(distances[neighbor_index]):
                distances[neighbor_index] = new_distance
                labels[neighbor_index] = label
                heapq.heappush(
                    queue, (new_distance, neighbor_index, label)
                )

    unassigned = np.flatnonzero(labels < 0)
    if len(unassigned):
        spatial_distance = np.linalg.norm(
            cells[unassigned, None, :] - center_array[None, :, :], axis=2
        )
        labels[unassigned] = spatial_distance.argmin(axis=1)
    return labels


def decode_floor_instances(
    class_grid: np.ndarray,
    center_heatmap: np.ndarray,
    center_offset: np.ndarray,
    boundary_probability: np.ndarray,
    class_counts: np.ndarray,
    config: DecodeConfig = DecodeConfig(),
) -> np.ndarray:
    """Decode each floor/class into the requested number of instances."""
    result = np.zeros(class_grid.shape, dtype=np.uint16)
    next_id = 1
    for floor_index in range(2):
        for class_id in range(1, 12):
            cells = np.argwhere(class_grid[floor_index] == class_id)
            requested = int(max(0, round(float(class_counts[floor_index, class_id - 1]))))
            if len(cells) == 0 or requested == 0:
                continue
            requested = min(requested, len(cells))
            centers = select_vote_centers(
                cells,
                center_offset[floor_index],
                center_heatmap[floor_index],
                requested,
                config.center_nms_radius,
            )
            labels = assign_cells(
                cells,
                center_offset[floor_index],
                boundary_probability[floor_index],
                centers,
                config,
            )
            for local_id in range(len(centers)):
                member_cells = cells[labels == local_id]
                if len(member_cells) < config.minimum_instance_cells:
                    continue
                result[
                    floor_index,
                    member_cells[:, 0],
                    member_cells[:, 1],
                ] = next_id
                next_id += 1
    return result


def floor_instance_records(
    instance_grid: np.ndarray,
    class_grid: np.ndarray,
    cross_floor_probability: np.ndarray,
    config: DecodeConfig = DecodeConfig(),
) -> list[dict]:
    records = []
    for floor_index in range(2):
        for instance_id in np.unique(instance_grid[floor_index]):
            instance_id = int(instance_id)
            if instance_id == 0:
                continue
            cells_array = np.argwhere(instance_grid[floor_index] == instance_id)
            cells = {(int(x), int(y)) for x, y in cells_array}
            class_id = int(
                class_grid[floor_index, cells_array[0, 0], cells_array[0, 1]]
            )
            cross_score = float(
                cross_floor_probability[floor_index][
                    instance_grid[floor_index] == instance_id
                ].mean()
            )
            records.append(
                {
                    "floor_index": floor_index,
                    "instance_id": instance_id,
                    "class_id": class_id,
                    "cells": cells,
                    "cross_floor": cross_score >= config.cross_floor_threshold,
                    "cross_floor_score": cross_score,
                }
            )
    return records


def largest_connected_component(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    remaining = set(cells)
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return max(components, key=len) if components else set()


def retain_largest_instance_components(instance_grid: np.ndarray) -> np.ndarray:
    cleaned = np.zeros_like(instance_grid)
    for floor_index in range(instance_grid.shape[0]):
        for instance_id in np.unique(instance_grid[floor_index]):
            instance_id = int(instance_id)
            if instance_id == 0:
                continue
            cells = {
                (int(x), int(y))
                for x, y in np.argwhere(
                    instance_grid[floor_index] == instance_id
                )
            }
            for x, y in largest_connected_component(cells):
                cleaned[floor_index, x, y] = instance_id
    return cleaned


def merge_building_instances(
    floor_instances: np.ndarray,
    class_grid: np.ndarray,
    cross_floor_probability: np.ndarray,
    config: DecodeConfig = DecodeConfig(),
) -> list[dict]:
    floor_records = floor_instance_records(
        floor_instances, class_grid, cross_floor_probability, config
    )
    used: set[tuple[int, int]] = set()
    result = []
    for record in floor_records:
        key = (record["floor_index"], record["instance_id"])
        if key in used:
            continue
        used.add(key)
        floors = [record["floor_index"] + 1]
        cells = set(record["cells"])
        if record["cross_floor"]:
            candidates = []
            for other in floor_records:
                other_key = (other["floor_index"], other["instance_id"])
                if (
                    other_key in used
                    or other["floor_index"] == record["floor_index"]
                    or other["class_id"] != record["class_id"]
                    or not other["cross_floor"]
                ):
                    continue
                intersection = len(record["cells"] & other["cells"])
                denominator = max(1, min(len(record["cells"]), len(other["cells"])))
                overlap = intersection / denominator
                if overlap >= config.cross_floor_overlap_threshold:
                    candidates.append((overlap, other))
            if candidates:
                _, other = max(candidates, key=lambda item: item[0])
                used.add((other["floor_index"], other["instance_id"]))
                floors.append(other["floor_index"] + 1)
                shared_cells = cells & other["cells"]
                if shared_cells:
                    cells = shared_cells
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        result.append(
            {
                "class_id": record["class_id"],
                "floors": sorted(floors),
                "cells": cells,
                "grid_box": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
            }
        )
    return result


def decode_model_output(
    output: dict[str, np.ndarray],
    site_mask: np.ndarray,
    condition_class_counts: np.ndarray | None = None,
    config: DecodeConfig = DecodeConfig(),
) -> dict:
    class_grid = output["class_logits"].argmax(axis=1).astype(np.uint8)
    class_grid[:, site_mask == 0] = 0
    predicted_counts = np.maximum(
        0,
        np.rint(output["count_prediction"][2:].reshape(2, 11)),
    )
    class_counts = (
        condition_class_counts
        if condition_class_counts is not None
        else predicted_counts
    )
    center_heatmap = sigmoid(output["center_logits"])
    boundary_probability = sigmoid(output["boundary_logits"])
    cross_floor_probability = sigmoid(output["cross_floor_logits"])
    floor_instances = decode_floor_instances(
        class_grid,
        center_heatmap,
        output["center_offset"],
        boundary_probability,
        class_counts,
        config,
    )
    floor_instances = retain_largest_instance_components(floor_instances)
    building_instances = merge_building_instances(
        floor_instances,
        class_grid,
        cross_floor_probability,
        config,
    )
    return {
        "class_grid": class_grid,
        "floor_instances": floor_instances,
        "building_instances": building_instances,
        "class_counts": class_counts,
        "center_heatmap": center_heatmap,
        "boundary_probability": boundary_probability,
        "cross_floor_probability": cross_floor_probability,
    }


def matched_instance_metrics(
    predicted_instances: np.ndarray,
    expected_instances: np.ndarray,
    expected_class_grid: np.ndarray,
    predicted_class_grid: np.ndarray | None = None,
) -> dict[str, float]:
    if predicted_class_grid is None:
        predicted_class_grid = expected_class_grid
    matched_ious = []
    expected_count = 0
    predicted_count = 0
    for floor_index in range(2):
        for class_id in range(1, 12):
            expected_ids = [
                int(value)
                for value in np.unique(expected_instances[floor_index])
                if int(value) not in (0, 65535)
                and np.any(
                    (expected_instances[floor_index] == value)
                    & (expected_class_grid[floor_index] == class_id)
                )
            ]
            predicted_ids = [
                int(value)
                for value in np.unique(predicted_instances[floor_index])
                if int(value) != 0
                and np.any(
                    (predicted_instances[floor_index] == value)
                    & (predicted_class_grid[floor_index] == class_id)
                )
            ]
            expected_count += len(expected_ids)
            predicted_count += len(predicted_ids)
            pairs = []
            for expected_id in expected_ids:
                expected_mask = expected_instances[floor_index] == expected_id
                for predicted_id in predicted_ids:
                    predicted_mask = predicted_instances[floor_index] == predicted_id
                    intersection = int((expected_mask & predicted_mask).sum())
                    union = int((expected_mask | predicted_mask).sum())
                    pairs.append(
                        (
                            intersection / max(union, 1),
                            expected_id,
                            predicted_id,
                        )
                    )
            used_expected = set()
            used_predicted = set()
            for iou, expected_id, predicted_id in sorted(pairs, reverse=True):
                if expected_id in used_expected or predicted_id in used_predicted:
                    continue
                used_expected.add(expected_id)
                used_predicted.add(predicted_id)
                matched_ious.append(iou)
            matched_ious.extend([0.0] * (len(expected_ids) - len(used_expected)))
    return {
        "expected_count": float(expected_count),
        "predicted_count": float(predicted_count),
        "count_error": float(abs(expected_count - predicted_count)),
        "mean_matched_iou": float(np.mean(matched_ious) if matched_ious else 0.0),
        "instance_recall_iou50": float(
            np.mean([value >= 0.5 for value in matched_ious])
            if matched_ious
            else 0.0
        ),
    }
