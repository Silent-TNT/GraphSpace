import unittest

import numpy as np

from build_v5_representation import (
    CLASS_MAP,
    OUTSIDE_CLASS,
    decode_instances,
    encode_house,
    evaluate_round_trip,
)


class V5RepresentationTest(unittest.TestCase):
    def test_site_empty_and_outside_are_distinct(self):
        data = {
            "house_id": "synthetic_l",
            "metadata": {"building_size": {"x": 1200, "y": 1200, "z": 6000}},
            "rooms": [
                {
                    "id": "room_0",
                    "type": "living_room",
                    "floor": 1,
                    "floors": [1],
                    "box_min": [0, 0, 0],
                    "box_max": [600, 1200, 3000],
                },
                {
                    "id": "room_1",
                    "type": "bedroom",
                    "floor": 1,
                    "floors": [1],
                    "box_min": [600, 0, 0],
                    "box_max": [1200, 600, 3000],
                },
                {
                    "id": "room_2",
                    "type": "stairs",
                    "floor": 1,
                    "floors": [1, 2],
                    "box_min": [0, 0, 0],
                    "box_max": [300, 300, 6000],
                },
            ],
        }
        # Avoid overlap in the synthetic example by moving the stair into the
        # unoccupied corner on both floors.
        data["rooms"][2]["box_min"] = [900, 900, 0]
        data["rooms"][2]["box_max"] = [1200, 1200, 6000]
        arrays, metadata = encode_house(data)
        placement = metadata["placement"]
        x0, y0 = placement["canvas_x0"], placement["canvas_y0"]

        self.assertEqual(int(arrays["class_grid"][0, 0, 0]), OUTSIDE_CLASS)
        self.assertEqual(int(arrays["class_grid"][0, x0 + 2, y0 + 3]), 0)
        self.assertEqual(
            int(arrays["class_grid"][0, x0, y0]), CLASS_MAP["living_room"]
        )
        self.assertEqual(int(arrays["cross_floor_mask"][:, x0 + 3, y0 + 3].sum()), 2)
        self.assertEqual(
            int(arrays["double_height_void_mask"][:, x0 + 3, y0 + 3].sum()), 0
        )
        self.assertEqual(len(decode_instances(arrays, metadata)), 3)
        self.assertTrue(evaluate_round_trip(data, arrays, metadata)["exact"])

    def test_masks_partition_site(self):
        data = {
            "house_id": "partition",
            "metadata": {"building_size": {"x": 600, "y": 600, "z": 6000}},
            "rooms": [{
                "id": "room_0",
                "type": "living_room",
                "floor": 1,
                "floors": [1],
                "box_min": [0, 0, 0],
                "box_max": [300, 600, 3000],
            }],
        }
        arrays, _ = encode_house(data)
        for floor_index in range(2):
            partition = (
                arrays["building_mask"][floor_index]
                + arrays["empty_inside_mask"][floor_index]
            )
            np.testing.assert_array_equal(partition, arrays["site_mask"])


if __name__ == "__main__":
    unittest.main()
