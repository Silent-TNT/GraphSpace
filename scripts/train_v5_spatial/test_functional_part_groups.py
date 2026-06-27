import unittest

from scripts.train_v5_spatial.build_functional_part_groups import (
    infer_functional_groups,
    normalize_source_rooms,
)


def room(room_id, room_type, x0, y0, x1, y1, z0=0, z1=3000, **extra):
    return {
        "id": room_id,
        "type": room_type,
        "box_min": [x0, y0, z0],
        "box_max": [x1, y1, z1],
        **extra,
    }


class FunctionalPartGroupTests(unittest.TestCase):
    def test_same_type_adjacent_corridor_parts_become_one_group(self):
        source = {
            "rooms": [
                room("corridor_a", "corridor", 0, 0, 900, 1200),
                room("corridor_b", "corridor", 900, 0, 1800, 1200),
                room("bedroom_0", "bedroom", 1800, 0, 3000, 1200),
            ]
        }

        grouped_rooms, groups = infer_functional_groups(normalize_source_rooms(source))

        corridor_groups = [group for group in groups if group["type"] == "corridor"]
        self.assertEqual(len(corridor_groups), 1)
        self.assertEqual(corridor_groups[0]["part_count"], 2)
        self.assertEqual(
            corridor_groups[0]["inference"],
            "same_type_adjacent_component",
        )
        self.assertEqual(
            {
                room["functional_id"]
                for room in grouped_rooms
                if room["type"] == "corridor"
            },
            {corridor_groups[0]["functional_id"]},
        )

    def test_adjacent_bedrooms_remain_separate_functional_groups(self):
        source = {
            "rooms": [
                room("bedroom_0", "bedroom", 0, 0, 1200, 1200),
                room("bedroom_1", "bedroom", 1200, 0, 2400, 1200),
            ]
        }

        grouped_rooms, groups = infer_functional_groups(normalize_source_rooms(source))

        self.assertEqual(len(groups), 2)
        self.assertEqual({group["part_count"] for group in groups}, {1})
        self.assertEqual(
            {group["inference"] for group in groups},
            {"non_groupable_singleton"},
        )
        self.assertEqual(
            {room["functional_id"] for room in grouped_rooms},
            {"bedroom_0", "bedroom_1"},
        )

    def test_explicit_functional_id_is_preserved(self):
        source = {
            "rooms": [
                room(
                    "living_0_part_0",
                    "living_room",
                    0,
                    0,
                    1200,
                    1200,
                    functional_id="living_room_0",
                ),
                room(
                    "living_0_part_1",
                    "living_room",
                    1200,
                    0,
                    2400,
                    1200,
                    functional_id="living_room_0",
                ),
            ]
        }

        grouped_rooms, groups = infer_functional_groups(normalize_source_rooms(source))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["functional_id"], "living_room_0")
        self.assertEqual(groups[0]["part_count"], 2)
        self.assertEqual(
            {room["functional_id"] for room in grouped_rooms},
            {"living_room_0"},
        )


if __name__ == "__main__":
    unittest.main()
