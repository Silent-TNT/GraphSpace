"""Tests for decoded instance conversion to standard room JSON."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_standard_json import building_instances_to_rooms  # noqa: E402


class ExportStandardJsonTest(unittest.TestCase):
    def test_canvas_coordinates_convert_to_local_millimeters(self) -> None:
        metadata = {
            "placement": {"canvas_x0": 10, "canvas_y0": 20},
            "class_map": {"empty": 0, "bedroom": 5, "stairs": 8},
        }
        instances = [
            {
                "class_id": 5,
                "floors": [2],
                "grid_box": [12, 23, 16, 28],
                "cells": {(x, y) for x in range(12, 16) for y in range(23, 28)},
            },
            {
                "class_id": 8,
                "floors": [1, 2],
                "grid_box": [10, 20, 12, 23],
                "cells": {(x, y) for x in range(10, 12) for y in range(20, 23)},
            },
        ]
        rooms, diagnostics = building_instances_to_rooms(instances, metadata)
        self.assertEqual(rooms[0]["box_min"], [600.0, 900.0, 3000.0])
        self.assertEqual(rooms[0]["box_max"], [1800.0, 2400.0, 6000.0])
        self.assertEqual(rooms[1]["floors"], [1, 2])
        self.assertEqual(rooms[1]["box_max"][2], 6000.0)
        self.assertEqual(diagnostics[0]["instance_coverage_by_rectangle"], 1.0)


if __name__ == "__main__":
    unittest.main()
