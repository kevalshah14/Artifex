"""
Fitness scoring functions for the Evolution Engine.

All fitness functions return a normalized score in [0.0, 1.0].
Follows VLMgineer's reward design: smooth, distance-based, normalized.
"""

import json
from typing import Optional


def score_code_result(result: dict) -> float:
    """
    Score the result of a code tool execution.

    Heuristics:
      - success=True with no error: base score 0.5
      - Bonus for meaningful result data (objects found, positions returned)
      - Full score 1.0 for confirmed physical action (pick_up success, place success)
      - 0.0 for errors or success=False
    """
    if not isinstance(result, dict):
        return 0.0

    # Check for errors at any nesting level
    if "error" in result:
        return 0.0

    inner = result.get("result", result)
    if isinstance(inner, dict) and "error" in inner:
        return 0.0

    success = result.get("success", False)
    if not success:
        inner_success = inner.get("success", False) if isinstance(inner, dict) else False
        if not inner_success:
            return 0.0

    # Base score for not crashing
    score = 0.5

    # Bonus: did it produce meaningful output?
    if isinstance(inner, dict):
        if inner.get("holding"):
            score = 1.0  # successfully picked up
        elif inner.get("placed_at"):
            score = 1.0  # successfully placed
        elif inner.get("objects"):
            # Found objects — partial credit (info gathering)
            score = 0.3
        elif inner.get("position"):
            score = 0.4
        elif inner.get("success") is True:
            score = 0.8  # generic success from sim

    return min(score, 1.0)


def score_geometry_result(result: dict, task_hint: str = "") -> float:
    """
    Score the result of a geometry tool evaluation from the sim.

    The sim returns a fitness score directly (computed browser-side).
    This function normalizes and validates it.
    """
    if not isinstance(result, dict):
        return 0.0

    if "error" in result:
        return 0.0

    # The sim computes fitness directly
    fitness = result.get("fitness", 0.0)
    if isinstance(fitness, (int, float)):
        return max(0.0, min(float(fitness), 1.0))

    return 0.0


def compute_distance_reward(
    current_pos: list[float],
    target_pos: list[float],
    initial_pos: list[float],
) -> float:
    """
    VLMgineer-style distance reward: 1 - (current_dist / initial_dist).

    Returns 0.0 if no progress, 1.0 if at target.
    """
    def dist(a, b):
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

    initial_distance = dist(initial_pos, target_pos)
    if initial_distance < 1e-6:
        return 1.0  # already at target

    current_distance = dist(current_pos, target_pos)
    reward = max(0.0, 1.0 - current_distance / initial_distance)
    return min(reward, 1.0)
