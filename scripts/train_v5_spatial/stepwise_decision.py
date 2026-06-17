"""Stepwise decision environment for learned spatial layout decoding.

This module defines the action contract used after a graph/semantic/voxel
model proposes the next layout operation.  It is intentionally model-agnostic:
the neural policy can make a proposal, while this environment validates it,
records rejected attempts, and keeps enough snapshots for rollback.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum


Bounds = tuple[int, int, int, int, int, int]
GRID_BOUNDS: Bounds = (0, 0, 0, 88, 88, 20)


class ActionKind(str, Enum):
    CUT = "cut"
    PLACE = "place"
    RESERVE_EMPTY = "reserve_empty"
    MERGE = "merge"
    ACCEPT = "accept"
    DEFER = "defer"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class StepAction:
    kind: ActionKind
    region_id: str | None = None
    axis: int | None = None
    cut: int | None = None
    left_node_ids: tuple[int, ...] = ()
    right_node_ids: tuple[int, ...] = ()
    node_ids: tuple[int, ...] = ()
    bounds: Bounds | None = None
    source_region_ids: tuple[str, ...] = ()
    target_action_index: int | None = None
    reason: str = ""


@dataclass
class DecisionRegion:
    id: str
    bounds: Bounds
    node_ids: tuple[int, ...]
    status: str = "open"


@dataclass
class DecisionState:
    regions: dict[str, DecisionRegion] = field(default_factory=dict)
    assignments: dict[int, list[Bounds]] = field(default_factory=dict)
    empty_regions: list[Bounds] = field(default_factory=list)
    history: list[StepAction] = field(default_factory=list)


@dataclass
class StepResult:
    accepted: bool
    issues: list[str]
    action_index: int | None = None
    rollback_available: bool = False


@dataclass
class ActionAttempt:
    action: StepAction
    accepted: bool
    issues: list[str]


def volume(bounds: Bounds) -> int:
    x0, y0, z0, x1, y1, z1 = bounds
    return max(0, x1 - x0) * max(0, y1 - y0) * max(0, z1 - z0)


def contains(outer: Bounds, inner: Bounds) -> bool:
    return all(outer[index] <= inner[index] for index in range(3)) and all(
        inner[index + 3] <= outer[index + 3] for index in range(3)
    )


def intersects(a: Bounds, b: Bounds) -> bool:
    return all(
        min(a[index + 3], b[index + 3]) > max(a[index], b[index])
        for index in range(3)
    )


def bounding_box(boxes: list[Bounds]) -> Bounds:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
        max(box[4] for box in boxes),
        max(box[5] for box in boxes),
    )


def face_adjacent(a: Bounds, b: Bounds) -> bool:
    for axis in range(3):
        touches = a[axis + 3] == b[axis] or b[axis + 3] == a[axis]
        if not touches:
            continue
        other_axes = [value for value in range(3) if value != axis]
        if all(
            min(a[item + 3], b[item + 3]) > max(a[item], b[item])
            for item in other_axes
        ):
            return True
    return False


class StepwiseDecisionEnvironment:
    """Validate and apply stepwise spatial layout actions."""

    def __init__(
        self,
        site_bounds: Bounds = GRID_BOUNDS,
        node_ids: tuple[int, ...] = (),
        root_region_id: str = "root",
    ) -> None:
        self.site_bounds = site_bounds
        root = DecisionRegion(root_region_id, site_bounds, tuple(node_ids))
        self.state = DecisionState(regions={root_region_id: root})
        self._snapshots: list[DecisionState] = [deepcopy(self.state)]
        self.attempt_log: list[ActionAttempt] = []
        self._next_region_index = 1

    def apply(self, action: StepAction) -> StepResult:
        if action.kind == ActionKind.ROLLBACK:
            return self._rollback(action)

        issues = self._validate(action)
        if issues:
            self.attempt_log.append(ActionAttempt(action, False, issues))
            return StepResult(
                accepted=False,
                issues=issues,
                rollback_available=len(self._snapshots) > 1,
            )

        self._mutate(action)
        self.state.history.append(action)
        self._snapshots.append(deepcopy(self.state))
        action_index = len(self.state.history) - 1
        self.attempt_log.append(ActionAttempt(action, True, []))
        return StepResult(
            accepted=True,
            issues=[],
            action_index=action_index,
            rollback_available=len(self._snapshots) > 1,
        )

    def _validate(self, action: StepAction) -> list[str]:
        if action.kind == ActionKind.CUT:
            return self._validate_cut(action)
        if action.kind == ActionKind.PLACE:
            return self._validate_place(action)
        if action.kind == ActionKind.RESERVE_EMPTY:
            return self._validate_reserve_empty(action)
        if action.kind == ActionKind.MERGE:
            return self._validate_merge(action)
        if action.kind in {ActionKind.ACCEPT, ActionKind.DEFER}:
            return self._validate_status_action(action)
        return [f"unsupported action kind: {action.kind}"]

    def _validate_region_action(self, action: StepAction) -> list[str]:
        if action.region_id is None:
            return ["region_id is required"]
        if action.region_id not in self.state.regions:
            return [f"unknown region_id: {action.region_id}"]
        return []

    def _validate_cut(self, action: StepAction) -> list[str]:
        issues = self._validate_region_action(action)
        if issues:
            return issues
        region = self.state.regions[action.region_id or ""]
        if action.axis not in (0, 1, 2):
            issues.append("cut axis must be 0, 1, or 2")
        elif action.cut is None:
            issues.append("cut coordinate is required")
        elif not (region.bounds[action.axis] < action.cut < region.bounds[action.axis + 3]):
            issues.append("cut coordinate must be strictly inside the region")

        left = set(action.left_node_ids)
        right = set(action.right_node_ids)
        expected = set(region.node_ids)
        if not left or not right:
            issues.append("cut must put at least one node on each side")
        if left & right:
            issues.append("left_node_ids and right_node_ids must not overlap")
        if left | right != expected:
            issues.append("cut node partition must match the region node_ids")
        return issues

    def _validate_place(self, action: StepAction) -> list[str]:
        issues = self._validate_region_action(action)
        if issues:
            return issues
        region = self.state.regions[action.region_id or ""]
        if action.bounds is None:
            issues.append("bounds are required")
            return issues
        issues.extend(self._validate_bounds(action.bounds, container=region.bounds))
        if not action.node_ids:
            issues.append("place action needs at least one node_id")
        unknown = set(action.node_ids) - set(region.node_ids)
        if unknown:
            issues.append(f"node_ids are not active in region: {sorted(unknown)}")
        if self._overlaps_taken_space(action.bounds):
            issues.append("bounds overlap existing assigned or empty space")
        return issues

    def _validate_reserve_empty(self, action: StepAction) -> list[str]:
        issues = self._validate_region_action(action)
        if issues:
            return issues
        region = self.state.regions[action.region_id or ""]
        if action.bounds is None:
            issues.append("bounds are required")
            return issues
        issues.extend(self._validate_bounds(action.bounds, container=region.bounds))
        if self._overlaps_taken_space(action.bounds):
            issues.append("empty bounds overlap existing assigned or empty space")
        return issues

    def _validate_merge(self, action: StepAction) -> list[str]:
        if len(action.source_region_ids) < 2:
            return ["merge needs at least two source regions"]
        regions = []
        for region_id in action.source_region_ids:
            if region_id not in self.state.regions:
                return [f"unknown source region_id: {region_id}"]
            regions.append(self.state.regions[region_id])
        if not any(
            face_adjacent(left.bounds, right.bounds)
            for index, left in enumerate(regions)
            for right in regions[index + 1 :]
        ):
            return ["merge source regions must have at least one face contact"]
        merged = bounding_box([region.bounds for region in regions])
        total_volume = sum(volume(region.bounds) for region in regions)
        if volume(merged) != total_volume:
            return ["merge would create a bounding box with hidden voids"]
        return []

    def _validate_status_action(self, action: StepAction) -> list[str]:
        issues = self._validate_region_action(action)
        if issues:
            return issues
        if action.kind == ActionKind.ACCEPT:
            region = self.state.regions[action.region_id or ""]
            if region.node_ids:
                issues.append("accept requires the region to have no active node_ids")
        return issues

    def _validate_bounds(self, bounds: Bounds, container: Bounds) -> list[str]:
        issues = []
        if volume(bounds) <= 0:
            issues.append("bounds must have positive volume")
        if not contains(self.site_bounds, bounds):
            issues.append("bounds must stay inside site_bounds")
        if not contains(container, bounds):
            issues.append("bounds must stay inside the active region")
        return issues

    def _overlaps_taken_space(self, bounds: Bounds) -> bool:
        assigned = [
            assigned_bounds
            for boxes in self.state.assignments.values()
            for assigned_bounds in boxes
        ]
        taken = assigned + list(self.state.empty_regions)
        return any(intersects(bounds, existing) for existing in taken)

    def _mutate(self, action: StepAction) -> None:
        if action.kind == ActionKind.CUT:
            self._mutate_cut(action)
        elif action.kind == ActionKind.PLACE:
            self._mutate_place(action)
        elif action.kind == ActionKind.RESERVE_EMPTY:
            self.state.empty_regions.append(action.bounds or GRID_BOUNDS)
        elif action.kind == ActionKind.MERGE:
            self._mutate_merge(action)
        elif action.kind in {ActionKind.ACCEPT, ActionKind.DEFER}:
            region = self.state.regions[action.region_id or ""]
            region.status = "accepted" if action.kind == ActionKind.ACCEPT else "deferred"

    def _mutate_cut(self, action: StepAction) -> None:
        region_id = action.region_id or ""
        region = self.state.regions.pop(region_id)
        axis = int(action.axis or 0)
        cut = int(action.cut or 0)
        left_bounds = list(region.bounds)
        right_bounds = list(region.bounds)
        left_bounds[axis + 3] = cut
        right_bounds[axis] = cut
        left_id = self._new_region_id(region_id)
        right_id = self._new_region_id(region_id)
        self.state.regions[left_id] = DecisionRegion(
            left_id, tuple(left_bounds), tuple(action.left_node_ids)
        )
        self.state.regions[right_id] = DecisionRegion(
            right_id, tuple(right_bounds), tuple(action.right_node_ids)
        )

    def _mutate_place(self, action: StepAction) -> None:
        region = self.state.regions[action.region_id or ""]
        for node_id in action.node_ids:
            self.state.assignments.setdefault(node_id, []).append(action.bounds or GRID_BOUNDS)
        remaining = tuple(node for node in region.node_ids if node not in set(action.node_ids))
        region.node_ids = remaining
        if not remaining:
            region.status = "accepted"

    def _mutate_merge(self, action: StepAction) -> None:
        regions = [self.state.regions.pop(region_id) for region_id in action.source_region_ids]
        merged_id = self._new_region_id("merged")
        merged_nodes = tuple(
            node for region in regions for node in region.node_ids
        )
        self.state.regions[merged_id] = DecisionRegion(
            merged_id,
            bounding_box([region.bounds for region in regions]),
            merged_nodes,
        )

    def _rollback(self, action: StepAction) -> StepResult:
        if len(self._snapshots) <= 1:
            issues = ["no accepted action is available for rollback"]
            self.attempt_log.append(ActionAttempt(action, False, issues))
            return StepResult(False, issues, rollback_available=False)

        target = action.target_action_index
        if target is None:
            snapshot_index = len(self._snapshots) - 2
        else:
            snapshot_index = target + 1
        if snapshot_index < 0 or snapshot_index >= len(self._snapshots):
            issues = ["target_action_index is outside the available history"]
            self.attempt_log.append(ActionAttempt(action, False, issues))
            return StepResult(False, issues, rollback_available=True)

        self.state = deepcopy(self._snapshots[snapshot_index])
        self.state.history.append(action)
        self._snapshots = self._snapshots[: snapshot_index + 1]
        self._snapshots.append(deepcopy(self.state))
        action_index = len(self.state.history) - 1
        self.attempt_log.append(ActionAttempt(action, True, []))
        return StepResult(True, [], action_index, rollback_available=len(self._snapshots) > 1)

    def _new_region_id(self, prefix: str) -> str:
        value = f"{prefix}_{self._next_region_index}"
        self._next_region_index += 1
        return value
