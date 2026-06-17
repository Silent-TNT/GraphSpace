from __future__ import annotations

import unittest

from scripts.data_phase1.run_phase1 import (
    extract_relations,
    guillotine_upper_bound,
    p1_spatial_organization_report,
)


def room(room_id, room_type, floor, box_min, box_max, floors=None):
    return {
        "id": room_id,
        "type": room_type,
        "floor": floor,
        "floors": floors or [floor],
        "box_min": list(map(float, box_min)),
        "box_max": list(map(float, box_max)),
    }


class Phase1Tests(unittest.TestCase):
    def test_face_adjacency_is_detected(self):
        rooms = [
            room("a", "living_room", 1, (0, 0, 0), (3000, 3000, 3000)),
            room("b", "dining_room", 1, (3000, 0, 0), (6000, 3000, 3000)),
        ]
        relations = extract_relations(rooms, 6000, 3000)
        self.assertEqual(len(relations["adjacency"]), 1)
        self.assertEqual(relations["adjacency"][0]["relation"], "face_adjacent")

    def test_guillotine_separable_layout(self):
        rects = [
            (0.0, 0.0, 3000.0, 3000.0, "a"),
            (3000.0, 0.0, 6000.0, 3000.0, "b"),
            (0.0, 3000.0, 6000.0, 6000.0, "c"),
        ]
        result = guillotine_upper_bound(rects)
        self.assertTrue(result["fully_separable"])
        self.assertEqual(result["resolved_singletons"], 3)

    def test_non_guillotine_cycle_is_reported(self):
        rects = [
            (0.0, 0.0, 4000.0, 1000.0, "top"),
            (3000.0, 0.0, 4000.0, 4000.0, "right"),
            (0.0, 3000.0, 4000.0, 4000.0, "bottom"),
            (0.0, 0.0, 1000.0, 4000.0, "left"),
        ]
        result = guillotine_upper_bound(rects)
        self.assertFalse(result["fully_separable"])
        self.assertTrue(result["blocked_groups"])

    def test_p1_stair_geometry(self):
        rooms = [
            room("entry", "entryway", 1, (0, 0, 0), (3000, 3000, 3000)),
            room("stair", "stairs", 1, (3000, 0, 0), (6000, 3000, 6000), [1, 2]),
            room("corridor2", "corridor", 2, (6000, 0, 3000), (9000, 3000, 6000)),
        ]
        relations = extract_relations(rooms, 9000, 3000)
        report = p1_spatial_organization_report(rooms, relations)
        self.assertTrue(report["checks"]["all_stairs_span_both_floors"])
        self.assertTrue(report["checks"]["stairs_contact_both_floors"])


if __name__ == "__main__":
    unittest.main()
