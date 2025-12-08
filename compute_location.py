"""Compute-location optimization module.

Implements data structures and scoring utilities for selecting the best
compute tier for a space/ground system based on user-supplied metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import matplotlib.pyplot as plt


@dataclass
class CriterionConfig:
    """Configuration for a single evaluation criterion."""

    name: str
    weight: float
    higher_is_better: bool
    min_val: float
    max_val: float


@dataclass
class Task:
    """Represents the requirements for a computation task."""

    name: str
    deadline_ms: Optional[float] = None
    min_success_prob: Optional[float] = None
    min_quality: Optional[float] = None
    max_cost_per_task: Optional[float] = None


@dataclass
class TierMetrics:
    """Holds raw metrics for a candidate compute tier."""

    tier: str
    metrics: Dict[str, float]
    feasible: bool = True
    infeasibility_reason: Optional[str] = None


def normalize(value: float, cfg: CriterionConfig) -> float:
    """Normalize a raw metric value to [0, 1]."""

    if cfg.max_val == cfg.min_val:
        return 0.0

    if cfg.higher_is_better:
        score = (value - cfg.min_val) / (cfg.max_val - cfg.min_val)
    else:
        score = (cfg.max_val - value) / (cfg.max_val - cfg.min_val)

    return max(0.0, min(1.0, score))


def apply_feasibility_constraints(task: Task, tm: TierMetrics) -> None:
    """Mutate TierMetrics to reflect hard feasibility constraints."""

    m = tm.metrics

    if "reg_ok" in m and m["reg_ok"] < 0.5:
        tm.feasible = False
        tm.infeasibility_reason = "regulatory policy forbids this tier"
        return

    if task.deadline_ms is not None and "latency_p99_ms" in m:
        if m["latency_p99_ms"] > task.deadline_ms:
            tm.feasible = False
            tm.infeasibility_reason = "p99 latency exceeds deadline"
            return

    if tm.feasible and task.min_success_prob is not None and "success_prob" in m:
        if m["success_prob"] < task.min_success_prob:
            tm.feasible = False
            tm.infeasibility_reason = "success probability below requirement"
            return

    if tm.feasible and task.min_quality is not None and "quality" in m:
        if m["quality"] < task.min_quality:
            tm.feasible = False
            tm.infeasibility_reason = "quality below requirement"
            return

    if tm.feasible and task.max_cost_per_task is not None and "cost_per_task" in m:
        if m["cost_per_task"] > task.max_cost_per_task:
            tm.feasible = False
            tm.infeasibility_reason = "cost per task above maximum"
            return

    if tm.feasible and "power_margin_W" in m and m["power_margin_W"] < 0:
        tm.feasible = False
        tm.infeasibility_reason = "negative power margin"


def compute_utility(
    task: Task,
    tier_metrics: TierMetrics,
    criteria: Dict[str, CriterionConfig],
    lambda_penalty: float = 0.2,
) -> float:
    """Compute the penalized utility score for a tier."""

    apply_feasibility_constraints(task, tier_metrics)
    if not tier_metrics.feasible:
        return -math.inf

    if not criteria:
        return 0.0

    w_total = sum(cfg.weight for cfg in criteria.values())
    w_known = 0.0
    numerator = 0.0

    for cfg in criteria.values():
        metric_name = cfg.name
        if metric_name not in tier_metrics.metrics:
            continue

        value = tier_metrics.metrics[metric_name]
        score = normalize(value, cfg)
        numerator += cfg.weight * score
        w_known += cfg.weight

    if w_known == 0.0:
        return -math.inf

    u_base = numerator / w_known
    phi = (w_known / w_total) if w_total > 0 else 1.0
    u_eff = u_base - lambda_penalty * (1.0 - phi)
    return u_eff


def choose_best_tier(
    task: Task,
    candidates: List[TierMetrics],
    criteria: Dict[str, CriterionConfig],
    lambda_penalty: float = 0.2,
) -> Tuple[Optional[TierMetrics], Dict[str, float]]:
    """Evaluate all tiers and return the best one with their scores."""

    scores: Dict[str, float] = {}
    best_tier: Optional[TierMetrics] = None
    best_score = -math.inf

    for tm in candidates:
        score = compute_utility(task, tm, criteria, lambda_penalty)
        scores[tm.tier] = score
        if score > best_score:
            best_score = score
            best_tier = tm

    if best_score == -math.inf:
        best_tier = None

    return best_tier, scores


def plot_tier_scores(
    scores: Dict[str, float],
    title: str = "Tier utility scores",
    output_path: Optional[str] = None,
) -> None:
    """Plot a bar chart of the utility scores for each tier.

    If output_path is provided, the figure is saved there instead of shown.
    """

    filtered = {tier: score for tier, score in scores.items() if score != -math.inf}
    if not filtered:
        print("No feasible tiers to plot.")
        return

    tiers = list(filtered.keys())
    vals = [filtered[tier] for tier in tiers]

    plt.figure(figsize=(8, 4))
    plt.bar(tiers, vals, color="steelblue")
    plt.ylabel("Utility score")
    plt.xlabel("Tier")
    plt.title(title)
    plt.ylim(bottom=min(vals) - 0.1, top=1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved tier score plot to {output_path}")
        plt.close()
    else:
        plt.show()


def _build_example_criteria() -> Dict[str, CriterionConfig]:
    """Helper to construct example criteria for the demo."""

    return {
        "latency_p99_ms": CriterionConfig(
            name="latency_p99_ms",
            weight=0.2,
            higher_is_better=False,
            min_val=50,
            max_val=1000,
        ),
        "success_prob": CriterionConfig(
            name="success_prob",
            weight=0.2,
            higher_is_better=True,
            min_val=0.8,
            max_val=0.999,
        ),
        "quality": CriterionConfig(
            name="quality",
            weight=0.15,
            higher_is_better=True,
            min_val=0.7,
            max_val=1.0,
        ),
        "energy_J": CriterionConfig(
            name="energy_J",
            weight=0.1,
            higher_is_better=False,
            min_val=10,
            max_val=1000,
        ),
        "cost_per_task": CriterionConfig(
            name="cost_per_task",
            weight=0.1,
            higher_is_better=False,
            min_val=0.01,
            max_val=10.0,
        ),
        "link_availability": CriterionConfig(
            name="link_availability",
            weight=0.15,
            higher_is_better=True,
            min_val=0.7,
            max_val=0.999,
        ),
        "ops_minutes_per_1000_tasks": CriterionConfig(
            name="ops_minutes_per_1000_tasks",
            weight=0.05,
            higher_is_better=False,
            min_val=5,
            max_val=300,
        ),
        "reduction_ratio": CriterionConfig(
            name="reduction_ratio",
            weight=0.05,
            higher_is_better=True,
            min_val=1.0,
            max_val=20.0,
        ),
    }


if __name__ == "__main__":
    task = Task(
        name="Intrusion detection pipeline",
        deadline_ms=250.0,
        min_success_prob=0.92,
        min_quality=0.9,
        max_cost_per_task=3.0,
    )

    criteria = _build_example_criteria()

    candidates = [
        TierMetrics(
            tier="FC",
            metrics={
                "latency_p99_ms": 90.0,
                "success_prob": 0.96,
                "quality": 0.93,
                "energy_J": 150.0,
                "cost_per_task": 2.4,
                "link_availability": 0.78,
                "ops_minutes_per_1000_tasks": 40.0,
                "reduction_ratio": 12.0,
                "reg_ok": 1.0,
            },
        ),
        TierMetrics(
            tier="ODC",
            metrics={
                "latency_p99_ms": 180.0,
                "success_prob": 0.93,
                "quality": 0.91,
                "energy_J": 90.0,
                "cost_per_task": 1.6,
                "link_availability": 0.88,
                "ops_minutes_per_1000_tasks": 25.0,
                "reduction_ratio": 10.0,
                "reg_ok": 1.0,
            },
        ),
        TierMetrics(
            tier="GSE",
            metrics={
                "latency_p99_ms": 600.0,
                "success_prob": 0.98,
                "quality": 0.95,
                "energy_J": 40.0,
                "cost_per_task": 0.9,
                "link_availability": 0.97,
                "ops_minutes_per_1000_tasks": 15.0,
                "reduction_ratio": 6.0,
                "reg_ok": 1.0,
            },
        ),
        TierMetrics(
            tier="TDC",
            metrics={
                "latency_p99_ms": 320.0,
                "success_prob": 0.9,
                "quality": 0.88,
                "energy_J": 25.0,
                "cost_per_task": 0.4,
                "link_availability": 0.99,
                "ops_minutes_per_1000_tasks": 12.0,
                "reduction_ratio": 14.0,
                "reg_ok": 1.0,
            },
        ),
    ]

    best_tier, scores = choose_best_tier(task, candidates, criteria)
    for tier, score in scores.items():
        tm = next(tm for tm in candidates if tm.tier == tier)
        reason = tm.infeasibility_reason
        print(f"Tier {tier}: score={score:.3f}, feasible={tm.feasible}, reason={reason}")

    if best_tier is not None:
        print(f"Best tier: {best_tier.tier} with score {scores[best_tier.tier]:.3f}")
    else:
        print("No feasible tier meets the requirements.")

    plot_tier_scores(
        scores, title=f"Compute-location choice for {task.name}", output_path="tier_scores.png"
    )
