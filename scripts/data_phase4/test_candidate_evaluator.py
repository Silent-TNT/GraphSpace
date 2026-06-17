import unittest

from evaluate_candidates import evaluate_candidate, summarize_candidate_set


REQUEST = {
    "entryway": 1,
    "living_room": 1,
    "dining_room": 1,
    "bedroom": 1,
    "bathroom": 1,
    "corridor": 2,
    "stairs": 1,
}
SITE = (9000, 6000)


def room(room_id, room_type, floor, box_min, box_max, floors=None):
    return {
        "id": room_id,
        "type": room_type,
        "floor": floor,
        "floors": floors or [floor],
        "box_min": list(box_min),
        "box_max": list(box_max),
    }


def valid_rooms():
    return [
        room("entry", "entryway", 1, (0, 0, 0), (1800, 3000, 3000)),
        room("living", "living_room", 1, (1800, 0, 0), (6000, 6000, 3000)),
        room("dining", "dining_room", 1, (6000, 0, 0), (9000, 3000, 3000)),
        room("corridor1", "corridor", 1, (0, 3000, 0), (1800, 6000, 3000)),
        room("stair", "stairs", 1, (6000, 3000, 0), (9000, 6000, 6000), [1, 2]),
        room("corridor2", "corridor", 2, (3000, 3000, 3000), (6000, 6000, 6000)),
        room("bedroom", "bedroom", 2, (0, 0, 3000), (3000, 6000, 6000)),
        room("bathroom", "bathroom", 2, (3000, 0, 3000), (6000, 3000, 6000)),
    ]


class CandidateEvaluatorTest(unittest.TestCase):
    def test_valid_candidate_passes_hard_checks_and_round_trip(self):
        report, arrays = evaluate_candidate("valid", valid_rooms(), REQUEST, SITE)
        self.assertTrue(report["p0"]["pass"])
        self.assertTrue(
            report["p1_spatial_organization"]["hard_geometry_pass"]
        )
        self.assertTrue(report["instance_recovery"]["pass"])
        self.assertTrue(report["eligible_for_diversity"])
        self.assertIsNotNone(arrays)

    def test_wrong_count_is_not_eligible(self):
        rooms = valid_rooms()[:-1]
        report, _ = evaluate_candidate("missing", rooms, REQUEST, SITE)
        self.assertFalse(report["p0"]["checks"]["requested_counts_match"])
        self.assertFalse(report["eligible_for_diversity"])

    def test_overlap_fails_p0_and_instance_recovery(self):
        rooms = valid_rooms()
        rooms[-1] = room(
            "bathroom", "bathroom", 2, (0, 0, 3000), (3000, 3000, 6000)
        )
        report, _ = evaluate_candidate("overlap", rooms, REQUEST, SITE)
        self.assertFalse(report["p0"]["checks"]["no_volume_overlap"])
        self.assertFalse(report["instance_recovery"]["pass"])

    def test_label_only_set_does_not_pass_diversity(self):
        candidates = []
        for index in range(4):
            rooms = valid_rooms()
            if index:
                rooms = [dict(value) for value in rooms]
                rooms[1], rooms[2] = (
                    {**rooms[1], "type": "dining_room", "id": "dining"},
                    {**rooms[2], "type": "living_room", "id": "living"},
                )
            candidates.append({"candidate_id": "c{}".format(index), "rooms": rooms})
        summary = summarize_candidate_set(candidates, REQUEST, SITE)
        self.assertFalse(summary["diversity"]["pass"])


if __name__ == "__main__":
    unittest.main()
