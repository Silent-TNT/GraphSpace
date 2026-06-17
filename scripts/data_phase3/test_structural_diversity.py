import unittest

import numpy as np

from evaluate_structural_diversity import compare_layouts


def sample():
    site = np.ones((6, 6), dtype=np.uint8)
    class_grid = np.zeros((2, 6, 6), dtype=np.uint8)
    instance_grid = np.zeros((2, 6, 6), dtype=np.uint16)
    class_grid[0, :4, :4] = 2
    class_grid[1, :4, :4] = 5
    instance_grid[0, :2, :4] = 1
    instance_grid[0, 2:4, :4] = 2
    instance_grid[1, :4, :2] = 3
    instance_grid[1, :4, 2:4] = 4
    building = (class_grid > 0).astype(np.uint8)
    return {
        "site_mask": site,
        "class_grid": class_grid,
        "instance_grid": instance_grid,
        "building_mask": building,
        "floor_overlap_mask": (building[0] & building[1]).astype(np.uint8),
        "cross_floor_mask": np.zeros_like(building),
        "double_height_void_mask": np.zeros_like(building),
    }


class StructuralDiversityTest(unittest.TestCase):
    def test_identical_layout_has_zero_distance(self):
        a = sample()
        result = compare_layouts(a, {key: value.copy() for key, value in a.items()})
        self.assertAlmostEqual(result["structural_distance"], 0.0)
        self.assertAlmostEqual(result["semantic_class_distance"], 0.0)

    def test_label_swap_is_not_structural_diversity(self):
        a = sample()
        b = {key: value.copy() for key, value in a.items()}
        b["class_grid"][b["class_grid"] == 2] = 7
        result = compare_layouts(a, b)
        self.assertAlmostEqual(result["geometry_distance"], 0.0)
        self.assertGreater(result["organization_distance"], 0.0)
        self.assertGreater(result["semantic_class_distance"], 0.0)
        self.assertEqual(result["category"], "label_only")

    def test_new_partition_counts_as_structural_change(self):
        a = sample()
        b = {key: value.copy() for key, value in a.items()}
        b["instance_grid"][0] = 0
        b["instance_grid"][0, :4, :2] = 1
        b["instance_grid"][0, :4, 2:4] = 2
        result = compare_layouts(a, b)
        self.assertGreater(result["components"]["partition"], 0.0)
        self.assertGreater(result["structural_distance"], 0.0)

    def test_new_footprint_counts_as_structural_change(self):
        a = sample()
        b = {key: value.copy() for key, value in a.items()}
        b["class_grid"][0, 3, :] = 0
        b["instance_grid"][0, 3, :] = 0
        b["building_mask"] = (b["class_grid"] > 0).astype(np.uint8)
        b["floor_overlap_mask"] = (
            b["building_mask"][0] & b["building_mask"][1]
        ).astype(np.uint8)
        result = compare_layouts(a, b)
        self.assertGreater(result["components"]["footprint"], 0.0)
        self.assertGreater(result["structural_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
