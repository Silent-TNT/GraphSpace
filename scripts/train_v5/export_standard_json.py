"""Convert decoded V5 building instances to the project's standard room JSON."""
from __future__ import annotations

from collections import Counter
from typing import Any


VOXEL_SIZE_MM = 300.0
FLOOR_HEIGHT_MM = 3000.0


def inverse_class_map(metadata: dict) -> dict[int, str]:
    return {
        int(class_id): str(room_type)
        for room_type, class_id in metadata["class_map"].items()
        if str(room_type) != "empty"
    }


def largest_inscribed_rectangle(
    cells: set[tuple[int, int]],
) -> tuple[int, int, int, int]:
    if not cells:
        raise ValueError("Cannot fit a rectangle to an empty instance")
    min_x = min(x for x, _ in cells)
    max_x = max(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    max_y = max(y for _, y in cells)
    width = max_y - min_y + 1
    heights = [0] * width
    best = (0, min_x, min_y, min_x + 1, min_y + 1)
    for x in range(min_x, max_x + 1):
        for local_y in range(width):
            y = min_y + local_y
            heights[local_y] = heights[local_y] + 1 if (x, y) in cells else 0
        stack: list[tuple[int, int]] = []
        for local_y in range(width + 1):
            height = heights[local_y] if local_y < width else 0
            start = local_y
            while stack and stack[-1][1] > height:
                start_index, popped_height = stack.pop()
                area = popped_height * (local_y - start_index)
                if area > best[0]:
                    best = (
                        area,
                        x - popped_height + 1,
                        min_y + start_index,
                        x + 1,
                        min_y + local_y,
                    )
                start = start_index
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    return best[1], best[2], best[3], best[4]


def building_instances_to_rooms(
    building_instances: list[dict],
    canvas_metadata: dict,
    id_prefix: str = "pred",
) -> tuple[list[dict], list[dict]]:
    placement = canvas_metadata["placement"]
    canvas_x0 = int(placement["canvas_x0"])
    canvas_y0 = int(placement["canvas_y0"])
    class_names = inverse_class_map(canvas_metadata)
    rooms = []
    shape_diagnostics = []
    for index, instance in enumerate(building_instances):
        class_id = int(instance["class_id"])
        if class_id not in class_names:
            continue
        floors = sorted(int(value) for value in instance["floors"])
        cells = {
            (int(x), int(y)) for x, y in instance.get("cells", ())
        }
        if cells:
            x0, y0, x1, y1 = largest_inscribed_rectangle(cells)
        else:
            x0, y0, x1, y1 = [
                int(value) for value in instance["grid_box"]
            ]
        room_id = "{}_{:03d}".format(id_prefix, index)
        box_cells = max(1, (x1 - x0) * (y1 - y0))
        instance_cells = len(cells)
        rooms.append(
            {
                "id": room_id,
                "type": class_names[class_id],
                "floor": floors[0],
                "floors": floors,
                "box_min": [
                    float((x0 - canvas_x0) * VOXEL_SIZE_MM),
                    float((y0 - canvas_y0) * VOXEL_SIZE_MM),
                    float((floors[0] - 1) * FLOOR_HEIGHT_MM),
                ],
                "box_max": [
                    float((x1 - canvas_x0) * VOXEL_SIZE_MM),
                    float((y1 - canvas_y0) * VOXEL_SIZE_MM),
                    float(floors[-1] * FLOOR_HEIGHT_MM),
                ],
                "auto_added": False,
            }
        )
        shape_diagnostics.append(
            {
                "room_id": room_id,
                "instance_cells": instance_cells,
                "bounding_box_cells": box_cells,
                "rectangle_cells": box_cells,
                "instance_coverage_by_rectangle": (
                    box_cells / instance_cells if instance_cells else 1.0
                ),
            }
        )
    return rooms, shape_diagnostics


def standard_candidate_payload(
    candidate_id: str,
    rooms: list[dict],
    canvas_metadata: dict,
) -> dict[str, Any]:
    site_x, site_y = canvas_metadata["site_size_mm"]
    return {
        "house_id": candidate_id,
        "metadata": {
            "building_size": {
                "x": float(site_x),
                "y": float(site_y),
                "z": 6000.0,
            },
            "stats": dict(Counter(room["type"] for room in rooms)),
        },
        "rooms": rooms,
    }
