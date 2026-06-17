"""Site-conditioned room counts, floor assignment and contact topology."""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx


def _floor_set(signature: str) -> set[int]:
    return {1, 2} if signature == "1&2" else {int(signature)}


def _node_key(node: dict) -> tuple[str, str]:
    return str(node["type"]), str(node["floor"])


class ProgramPrior:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        if self.payload.get("source_split") != "train":
            raise ValueError("program prior must be built from the train split")
        self.houses = list(self.payload["houses"])

    @staticmethod
    def _distance(house: dict, site_x: float, site_y: float) -> float:
        x_ratio = math.log(max(float(house["site_x"]), 1.0) / site_x)
        y_ratio = math.log(max(float(house["site_y"]), 1.0) / site_y)
        area_ratio = math.log(
            max(float(house["site_x"]) * float(house["site_y"]), 1.0)
            / max(site_x * site_y, 1.0)
        )
        return math.sqrt(
            x_ratio * x_ratio + y_ratio * y_ratio + 0.35 * area_ratio * area_ratio
        )

    def neighbors(
        self,
        site_x: float,
        site_y: float,
        count: int = 32,
        exclude_house_id: str | None = None,
    ) -> list[dict]:
        ranked = sorted(
            (
                {
                    "distance": self._distance(house, site_x, site_y),
                    "weight": 0.0,
                    "house": house,
                }
                for house in self.houses
                if str(house["house_id"]) != str(exclude_house_id)
            ),
            key=lambda item: (item["distance"], item["house"]["house_id"]),
        )[: min(count, len(self.houses))]
        for item in ranked:
            item["weight"] = 1.0 / max(item["distance"], 0.035)
        total = sum(item["weight"] for item in ranked)
        for item in ranked:
            item["weight"] /= total
        return ranked

    @staticmethod
    def _weighted_choice(rng: random.Random, values: list, weights: list[float]):
        return rng.choices(values, weights=weights, k=1)[0]

    def infer_counts(
        self,
        neighbors: list[dict],
        seed: int,
        explicit_counts: dict[str, int] | None = None,
        infer_missing: bool = True,
    ) -> tuple[dict[str, int], dict]:
        rng = random.Random(seed)
        explicit = {
            str(key): int(value)
            for key, value in (explicit_counts or {}).items()
            if int(value) >= 0
        }
        room_types = sorted(
            {
                room_type
                for item in neighbors
                for room_type in item["house"]["room_counts"]
            }
        )
        weighted_total = sum(
            sum(int(value) for value in item["house"]["room_counts"].values())
            * float(item["weight"])
            for item in neighbors
        )
        capacity_limit = max(
            sum(explicit.values()),
            int(math.ceil(weighted_total + 2.0)),
        )
        feasible_neighbors = []
        for item in neighbors:
            proposed = dict(item["house"]["room_counts"])
            proposed.update(explicit)
            proposed_total = sum(int(value) for value in proposed.values())
            if proposed_total <= capacity_limit:
                feasible_neighbors.append(item)
        donor_candidates = feasible_neighbors or neighbors
        donor_weights = []
        for item in donor_candidates:
            donor_counts = item["house"]["room_counts"]
            mismatch = sum(
                abs(int(donor_counts.get(room_type, 0)) - count)
                for room_type, count in explicit.items()
            )
            donor_weights.append(float(item["weight"]) * math.exp(-0.35 * mismatch))
        donor_item = self._weighted_choice(
            rng,
            donor_candidates,
            donor_weights,
        )
        donor = donor_item["house"]
        inferred = {
            room_type: int(donor["room_counts"].get(room_type, 0))
            for room_type in room_types
        }
        distributions = {}
        for room_type in room_types:
            values = [
                int(item["house"]["room_counts"].get(room_type, 0))
                for item in neighbors
            ]
            weights = [float(item["weight"]) for item in neighbors]
            distributions[room_type] = {
                "weighted_mean": sum(v * w for v, w in zip(values, weights)),
                "sampled": inferred[room_type],
            }
        counts = dict(inferred) if infer_missing else {}
        counts.update(explicit)
        if infer_missing:
            counts["corridor"] = max(int(counts.get("corridor", 0)), 2)
        counts = {key: value for key, value in counts.items() if value > 0}
        if not counts:
            raise ValueError("the learned program prior produced no rooms")
        return counts, {
            "explicit_counts": explicit,
            "inferred_counts": inferred,
            "count_donor_house": donor["house_id"],
            "count_donor_distance": donor_item["distance"],
            "neighbor_weighted_total_rooms": weighted_total,
            "capacity_limit": capacity_limit,
            "infer_missing": infer_missing,
            "distributions": distributions,
        }

    def _assign_floors(
        self,
        counts: dict[str, int],
        neighbors: list[dict],
        rng: random.Random,
    ) -> list[dict]:
        node_samples = defaultdict(list)
        for item in neighbors:
            for node in item["house"]["nodes"]:
                node_samples[str(node["type"])].append(
                    (
                        {
                            "floor": str(node["floor"]),
                            "area_ratio": float(node.get("area_ratio", 0.0)),
                            "lighting_access": str(
                                node.get("lighting_access", "none")
                            ),
                            "lighting_priority": int(
                                node.get("lighting_priority", 0)
                            ),
                        },
                        float(item["weight"]),
                    )
                )
        nodes = []
        for room_type in sorted(counts):
            samples = node_samples[room_type] or [
                (
                    {
                        "floor": "1",
                        "area_ratio": 0.04,
                        "lighting_access": "none",
                        "lighting_priority": 0,
                    },
                    1.0,
                )
            ]
            values = [sample[0] for sample in samples]
            weights = [sample[1] for sample in samples]
            for offset in range(int(counts[room_type])):
                sampled = dict(self._weighted_choice(rng, values, weights))
                nodes.append(
                    {
                        "id": f"{room_type}_{offset}",
                        "type": room_type,
                        **sampled,
                    }
                )
        by_type = defaultdict(list)
        for node in nodes:
            by_type[node["type"]].append(node)
        if by_type["living_room"] and by_type["dining_room"]:
            living_floor = by_type["living_room"][0]["floor"]
            if not (
                _floor_set(living_floor)
                & _floor_set(by_type["dining_room"][0]["floor"])
            ):
                by_type["dining_room"][0]["floor"] = living_floor
        if by_type["dining_room"] and by_type["kitchen"]:
            dining_floor = by_type["dining_room"][0]["floor"]
            if not (
                _floor_set(dining_floor)
                & _floor_set(by_type["kitchen"][0]["floor"])
            ):
                by_type["kitchen"][0]["floor"] = dining_floor
        return nodes

    @staticmethod
    def _pair_probabilities(
        nodes: list[dict],
        neighbors: list[dict],
    ) -> dict:
        requested_keys = {_node_key(node) for node in nodes}
        possible = defaultdict(float)
        contacts = defaultdict(lambda: defaultdict(float))
        for item in neighbors:
            weight = float(item["weight"])
            house_nodes = item["house"]["nodes"]
            counts = Counter(_node_key(node) for node in house_nodes)
            for left in requested_keys:
                for right in requested_keys:
                    if left > right:
                        continue
                    combinations = counts[left] * counts[right]
                    if left == right:
                        combinations = counts[left] * (counts[left] - 1) / 2
                    possible[(left, right)] += weight * combinations
            for left, right, relation in item["house"]["edges"]:
                pair = tuple(
                    sorted(
                        (
                            _node_key(house_nodes[int(left)]),
                            _node_key(house_nodes[int(right)]),
                        )
                    )
                )
                contacts[pair][int(relation)] += weight
        return {
            pair: {
                relation: min(1.0, value / max(denominator, 1e-9))
                for relation, value in contacts[pair].items()
            }
            for pair, denominator in possible.items()
        }

    def build_topology(
        self,
        counts: dict[str, int],
        site_x: float,
        site_y: float,
        seed: int,
        neighbor_count: int = 32,
        exclude_house_id: str | None = None,
    ) -> tuple[nx.Graph, dict, list[tuple[str, str, str]], dict, dict]:
        rng = random.Random(seed)
        neighbors = self.neighbors(
            site_x,
            site_y,
            neighbor_count,
            exclude_house_id=exclude_house_id,
        )
        node_records = self._assign_floors(counts, neighbors, rng)
        probabilities = self._pair_probabilities(node_records, neighbors)
        graph = nx.Graph()
        for node in node_records:
            graph.add_node(node["id"], type=node["type"], floor=node["floor"])

        candidates = []
        for left_index, left in enumerate(node_records):
            for right in node_records[left_index + 1 :]:
                pair = tuple(sorted((_node_key(left), _node_key(right))))
                relation_probs = probabilities.get(pair, {})
                relation = max(relation_probs, key=relation_probs.get, default=0)
                probability = float(relation_probs.get(relation, 0.0))
                if relation == 0 and not (
                    _floor_set(left["floor"]) & _floor_set(right["floor"])
                ):
                    probability *= 0.15
                candidates.append(
                    (
                        probability * rng.uniform(0.92, 1.08),
                        probability,
                        relation,
                        left["id"],
                        right["id"],
                    )
                )

        edge_types = {}
        fallback_edges = []
        required_edges = set()
        components = {node_id: node_id for node_id in graph.nodes}

        def find(node_id: str) -> str:
            while components[node_id] != node_id:
                components[node_id] = components[components[node_id]]
                node_id = components[node_id]
            return node_id

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                components[right_root] = left_root

        sorted_candidates = sorted(candidates, reverse=True)
        for _score, probability, relation, left, right in sorted_candidates:
            if (
                find(left) != find(right)
                and graph.degree(left) < 4
                and graph.degree(right) < 4
            ):
                graph.add_edge(left, right)
                union(left, right)
                edge_types[(left, right)] = "vertical" if relation == 1 else "horizontal"
                if probability <= 0:
                    fallback_edges.append([left, right])

        def ensure_edge(
            left: str,
            right: str,
            relation: int = 0,
            required: bool = False,
        ) -> None:
            pair = tuple(sorted((left, right)))
            if required:
                required_edges.add(pair)
            if graph.has_edge(left, right):
                edge_types[(left, right)] = (
                    "vertical" if relation == 1 else "horizontal"
                )
                return
            graph.add_edge(left, right)
            edge_types[(left, right)] = "vertical" if relation == 1 else "horizontal"

        def ensure_type_pair(left_type: str, right_type: str) -> None:
            matching = [
                item
                for item in sorted_candidates
                if {
                    graph.nodes[item[3]]["type"],
                    graph.nodes[item[4]]["type"],
                }
                == {left_type, right_type}
            ]
            if not matching:
                return
            _score, probability, _relation, left, right = matching[0]
            ensure_edge(left, right, 0, required=True)
            if probability <= 0:
                fallback_edges.append([left, right])

        # These are acceptance constraints, while the selected instances and
        # edge preference still come from the learned contact probabilities.
        ensure_type_pair("living_room", "dining_room")
        ensure_type_pair("kitchen", "dining_room")

        circulation_types = {"corridor", "entryway", "living_room", "stairs"}
        circulation_nodes = [
            node for node in node_records if node["type"] in circulation_types
        ]
        bedroom_target_usage = Counter()
        for bedroom in (
            node for node in node_records if node["type"] == "bedroom"
        ):
            compatible = [
                node
                for node in circulation_nodes
                if node["id"] != bedroom["id"]
                and _floor_set(node["floor"]) & _floor_set(bedroom["floor"])
            ]
            if compatible:
                target = min(
                    compatible,
                    key=lambda node: (
                        bedroom_target_usage[node["id"]],
                        graph.degree(node["id"]),
                    ),
                )
                bedroom_target_usage[target["id"]] += 1
                ensure_edge(
                    bedroom["id"],
                    target["id"],
                    0,
                    required=True,
                )

        for stairs in (
            node for node in node_records if node["type"] == "stairs"
        ):
            for floor in (1, 2):
                compatible = [
                    node
                    for node in circulation_nodes
                    if node["id"] != stairs["id"]
                    and floor in _floor_set(node["floor"])
                ]
                if compatible:
                    target = min(
                        compatible,
                        key=lambda node: graph.degree(node["id"]),
                    )
                    ensure_edge(
                        stairs["id"],
                        target["id"],
                        0,
                        required=True,
                    )
        for _score, probability, relation, left, right in sorted_candidates:
            if find(left) == find(right):
                continue
            graph.add_edge(left, right)
            union(left, right)
            edge_types[(left, right)] = "vertical" if relation == 1 else "horizontal"
            if probability <= 0:
                fallback_edges.append([left, right])

        existing = {tuple(sorted(edge)) for edge in graph.edges}
        maximum_edges = max(len(node_records) - 1, round(len(node_records) * 1.45))
        for _score, probability, relation, left, right in sorted(
            candidates,
            reverse=True,
        ):
            pair = tuple(sorted((left, right)))
            if (
                pair in existing
                or probability < 0.18
                or graph.number_of_edges() >= maximum_edges
                or graph.degree(left) >= 4
                or graph.degree(right) >= 4
            ):
                continue
            if rng.random() < min(0.72, probability):
                graph.add_edge(left, right)
                edge_types[(left, right)] = (
                    "vertical" if relation == 1 else "horizontal"
                )
                existing.add(pair)

        positions = nx.spring_layout(graph, seed=seed, k=1.2)
        topology_nodes = [
            (node["id"], node["type"], node["floor"]) for node in node_records
        ]
        evidence = {
            "source": "training_data_knn",
            "prior_path": str(self.path),
            "source_split": self.payload["source_split"],
            "neighbor_count": len(neighbors),
            "nearest_houses": [
                {
                    "house_id": item["house"]["house_id"],
                    "site_x": item["house"]["site_x"],
                    "site_y": item["house"]["site_y"],
                    "distance": item["distance"],
                    "weight": item["weight"],
                }
                for item in neighbors[:10]
            ],
            "connectivity_fallback_edges": fallback_edges,
            "required_edges": [list(pair) for pair in sorted(required_edges)],
            "node_conditions": {
                node["id"]: {
                    "area_ratio": float(node["area_ratio"]),
                    "lighting_access": str(node["lighting_access"]),
                    "lighting_priority": int(node["lighting_priority"]),
                }
                for node in node_records
            },
            "topology_semantics": self.payload["topology_semantics"],
        }
        return graph, positions, topology_nodes, edge_types, evidence
