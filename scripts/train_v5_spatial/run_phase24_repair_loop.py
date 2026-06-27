#!/usr/bin/env python3
"""Run a small Phase24 fix-and-verify loop on fixed user-generation cases.

The loop is deliberately conservative: it does not edit source code. Each
iteration changes one generation strategy knob, reruns the same case/seeds,
then records whether P0 and target-topology realization improved.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "phase24_repair_loop"
DEFAULT_GRAPH_CHECKPOINT = (
    ROOT / "outputs" / "v6_graph_coarse_layout_model_regularized_full_phase24" / "graph_coarse_layout_model.pt"
)
DEFAULT_CANDIDATE_SCORER = ROOT / "outputs" / "v6_candidate_scorer_full_phase24" / "candidate_scorer.pt"
DEFAULT_GENERATOR = ROOT / "scripts" / "train_v5_spatial" / "generate_phase24_from_user_conditions.py"


@dataclass(frozen=True)
class RepairAction:
    name: str
    candidate_scorer_weight: float | None
    max_topology_move_mm: float
    max_size_adjustment_mm: float


DEFAULT_ACTIONS = [
    RepairAction("baseline_learned_graph", None, 1800.0, 600.0),
    RepairAction("candidate_scorer_150", 150.0, 1800.0, 600.0),
    RepairAction("candidate_scorer_300", 300.0, 1800.0, 600.0),
    RepairAction("candidate_scorer_600", 600.0, 1800.0, 600.0),
    RepairAction("wider_topology_move", 300.0, 2400.0, 600.0),
    RepairAction("wider_size_adjustment", 300.0, 2400.0, 900.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="medium")
    parser.add_argument("--seeds", default="11,42,123", help="Comma-separated integer seeds.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--graph-coarse-layout-checkpoint", type=Path, default=DEFAULT_GRAPH_CHECKPOINT)
    parser.add_argument("--candidate-scorer-checkpoint", type=Path, default=DEFAULT_CANDIDATE_SCORER)
    parser.add_argument("--max-iterations", type=int, default=len(DEFAULT_ACTIONS))
    parser.add_argument("--stop-on-full-topology", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    topology = summary.get("topology", {})
    target = int(topology.get("target_edge_count", 0))
    realized = int(topology.get("realized_edge_count", 0))
    return {
        "p0_pass": bool(summary.get("p0_pass", False)),
        "p1_hard_geometry_pass": bool(summary.get("p1_hard_geometry_pass", False)),
        "p1_spatial_organization_pass": bool(summary.get("p1_spatial_organization_pass", False)),
        "target_edge_count": target,
        "realized_edge_count": realized,
        "missing_edge_count": max(target - realized, 0),
        "realization_rate": float(topology.get("realization_rate", 0.0)),
    }


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(seed_results)
    total_target = sum(result["metrics"]["target_edge_count"] for result in seed_results)
    total_realized = sum(result["metrics"]["realized_edge_count"] for result in seed_results)
    return {
        "seed_count": count,
        "p0_pass_count": sum(1 for result in seed_results if result["metrics"]["p0_pass"]),
        "p1_spatial_pass_count": sum(
            1 for result in seed_results if result["metrics"]["p1_spatial_organization_pass"]
        ),
        "total_target_edges": total_target,
        "total_realized_edges": total_realized,
        "total_missing_edges": max(total_target - total_realized, 0),
        "aggregate_realization_rate": float(total_realized / total_target) if total_target else 0.0,
    }


def primary_issue(aggregate_metrics: dict[str, Any]) -> str:
    if aggregate_metrics["p0_pass_count"] < aggregate_metrics["seed_count"]:
        return "p0_regression"
    if aggregate_metrics["total_missing_edges"] > 0:
        return "missing_target_contacts"
    if aggregate_metrics["p1_spatial_pass_count"] < aggregate_metrics["seed_count"]:
        return "p1_non_topology_failure"
    return "passed_current_loop_targets"


def improvement_score(aggregate_metrics: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(aggregate_metrics["p0_pass_count"]),
        int(aggregate_metrics["total_realized_edges"]),
        float(aggregate_metrics["aggregate_realization_rate"]),
    )


def run_generation(args: argparse.Namespace, action: RepairAction, seed: int, iteration_dir: Path) -> dict[str, Any]:
    run_dir = iteration_dir / f"seed_{seed}"
    command = [
        str(args.python),
        str(args.generator),
        "--case",
        str(args.case),
        "--seed",
        str(seed),
        "--output-dir",
        str(run_dir),
        "--coarse-layout-strategy",
        "learned_graph",
        "--graph-coarse-layout-checkpoint",
        str(args.graph_coarse_layout_checkpoint),
        "--max-topology-move-mm",
        str(action.max_topology_move_mm),
        "--max-size-adjustment-mm",
        str(action.max_size_adjustment_mm),
        "--device",
        str(args.device),
    ]
    if action.candidate_scorer_weight is not None:
        command.extend(
            [
                "--candidate-scorer-checkpoint",
                str(args.candidate_scorer_checkpoint),
                "--candidate-scorer-weight",
                str(action.candidate_scorer_weight),
            ]
        )
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    write_json(
        run_dir / "command_result.json",
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        },
    )
    if completed.returncode != 0:
        return {
            "seed": seed,
            "run_dir": str(run_dir),
            "failed": True,
            "returncode": completed.returncode,
            "metrics": {
                "p0_pass": False,
                "p1_hard_geometry_pass": False,
                "p1_spatial_organization_pass": False,
                "target_edge_count": 0,
                "realized_edge_count": 0,
                "missing_edge_count": 0,
                "realization_rate": 0.0,
            },
        }
    summary = read_json(run_dir / "summary.json")
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "failed": False,
        "metrics": summary_metrics(summary),
    }


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [int(value.strip()) for value in str(args.seeds).split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions = DEFAULT_ACTIONS[: max(1, int(args.max_iterations))]
    iterations = []
    best: dict[str, Any] | None = None
    best_score: tuple[int, int, float] | None = None
    for index, action in enumerate(actions, start=1):
        iteration_dir = args.output_dir / f"{index:02d}_{action.name}"
        seed_results = [run_generation(args, action, seed, iteration_dir) for seed in seeds]
        aggregate_metrics = aggregate(seed_results)
        issue = primary_issue(aggregate_metrics)
        score = improvement_score(aggregate_metrics)
        record = {
            "iteration": index,
            "action": asdict(action),
            "issue_after_iteration": issue,
            "aggregate": aggregate_metrics,
            "seed_results": seed_results,
        }
        write_json(iteration_dir / "iteration_summary.json", record)
        iterations.append(record)
        if best_score is None or score > best_score:
            best_score = score
            best = record
        print(
            json.dumps(
                {
                    "iteration": index,
                    "action": action.name,
                    "issue": issue,
                    "p0": f"{aggregate_metrics['p0_pass_count']}/{aggregate_metrics['seed_count']}",
                    "topology": f"{aggregate_metrics['total_realized_edges']}/{aggregate_metrics['total_target_edges']}",
                    "rate": aggregate_metrics["aggregate_realization_rate"],
                },
                ensure_ascii=False,
            )
        )
        if args.stop_on_full_topology and issue == "passed_current_loop_targets":
            break
    summary = {
        "schema": "graphspace_phase24_repair_loop_v1",
        "case": args.case,
        "seeds": seeds,
        "iterations": iterations,
        "best_iteration": best,
    }
    write_json(args.output_dir / "repair_loop_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(run_loop(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
