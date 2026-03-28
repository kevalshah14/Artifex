"""
Evolution Engine — VLMgineer Algorithm 1 for ForgeBot.

Population-based evolutionary search that generates, evaluates, and refines
tool candidates through LLM-guided mutation and crossover.

Two modes:
  - "code": evolve async Python tool functions
  - "geometry": evolve MJCF tool shapes + action waypoints
"""

import json
import asyncio
import os
import time
from typing import Optional, Callable

from agent.fitness import score_code_result, score_geometry_result
from agent.prompts import evolution_code, evolution_geometry
from agent.tool_registry import registry
from agent.skill_registry import skill_registry
from agent.llm_client import generate as llm_generate

GEOMETRY_MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory", "geometry")


class EvolutionEngine:
    """VLMgineer Algorithm 1 — population-based evolutionary search."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        n_iterations: int = 3,
        n_candidates: int = 5,
        top_k: int = 2,
    ):
        self.model = model
        self.n_iterations = n_iterations
        self.n_candidates = n_candidates
        self.top_k = top_k
        self._event_callback: Optional[Callable] = None

    def on_event(self, callback: Callable):
        """Register event callback for streaming progress to frontend."""
        self._event_callback = callback

    async def _emit(self, event_type: str, data: dict):
        if self._event_callback:
            await self._event_callback(event_type, data)

    # ──────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────

    async def evolve(
        self,
        task: str,
        mode: str,
        compile_fn: Optional[Callable] = None,
        sim_eval_fn: Optional[Callable] = None,
        scene_info: str = "",
    ) -> dict:
        """
        Run the full evolutionary loop.

        Args:
            task: Natural language task description
            mode: "code" or "geometry"
            compile_fn: For code mode — callable(name, source) -> fn
            sim_eval_fn: For geometry mode — async callable(mjcf, waypoints) -> result dict
            scene_info: For geometry mode — scene description for LLM context

        Returns:
            dict with best candidate, score, and metadata
        """
        await self._emit("evolution_start", {
            "mode": mode,
            "task": task,
            "n_iterations": self.n_iterations,
            "n_candidates": self.n_candidates,
            "message": f"Starting {mode} evolution: {self.n_iterations} generations x {self.n_candidates} candidates",
        })

        winners: list[tuple[dict, float]] = []
        all_best_score = 0.0
        all_best_candidate = None

        for iteration in range(1, self.n_iterations + 1):
            await self._emit("evolution_generation", {
                "iteration": iteration,
                "total": self.n_iterations,
                "message": f"Generation {iteration}/{self.n_iterations}",
            })

            # 1. SAMPLE — LLM generates candidates
            candidates = await self._sample_candidates(task, mode, winners, scene_info)
            if not candidates:
                await self._emit("evolution_error", {
                    "message": f"Generation {iteration}: LLM returned no valid candidates",
                })
                continue

            await self._emit("evolution_sampled", {
                "iteration": iteration,
                "count": len(candidates),
                "names": [c.get("tool_name", "?") for c in candidates],
                "message": f"Generated {len(candidates)} candidates",
            })

            # 2. EVALUATE — score each candidate
            scored: list[tuple[dict, float]] = []
            for i, candidate in enumerate(candidates):
                name = candidate.get("tool_name", f"candidate_{i}")
                await self._emit("evolution_evaluating", {
                    "candidate": name,
                    "index": i + 1,
                    "total": len(candidates),
                    "message": f"Evaluating {name} ({i+1}/{len(candidates)})",
                })

                score = await self._evaluate_candidate(
                    candidate, mode, compile_fn, sim_eval_fn
                )
                scored.append((candidate, score))

                await self._emit("evolution_scored", {
                    "candidate": name,
                    "score": score,
                    "message": f"{name}: score {score:.2f}",
                })

            # 3. SELECT — keep top-k
            scored.sort(key=lambda x: x[1], reverse=True)
            winners = scored[:self.top_k]

            gen_best = winners[0] if winners else (None, 0.0)
            if gen_best[1] > all_best_score:
                all_best_score = gen_best[1]
                all_best_candidate = gen_best[0]

            await self._emit("evolution_selected", {
                "iteration": iteration,
                "winners": [
                    {"name": c.get("tool_name", "?"), "score": s}
                    for c, s in winners
                ],
                "best_score": all_best_score,
                "message": f"Generation {iteration} best: {gen_best[0].get('tool_name', '?') if gen_best[0] else 'none'} ({gen_best[1]:.2f})",
            })

            # Early exit if perfect score
            if all_best_score >= 0.99:
                await self._emit("evolution_converged", {
                    "iteration": iteration,
                    "score": all_best_score,
                    "message": "Perfect score — stopping early",
                })
                break

        if not all_best_candidate:
            return {"success": False, "error": "Evolution produced no viable candidates"}

        # Persist the winner
        self._persist_winner(all_best_candidate, all_best_score, mode)

        await self._emit("evolution_complete", {
            "mode": mode,
            "winner": all_best_candidate.get("tool_name", "?"),
            "score": all_best_score,
            "iterations": iteration,
            "message": f"Evolution complete — {all_best_candidate.get('tool_name', '?')} (score: {all_best_score:.2f})",
        })

        return {
            "success": True,
            "candidate": all_best_candidate,
            "score": all_best_score,
            "mode": mode,
        }

    # ──────────────────────────────────────────
    # Candidate sampling
    # ──────────────────────────────────────────

    async def _sample_candidates(
        self,
        task: str,
        mode: str,
        winners: list[tuple[dict, float]],
        scene_info: str = "",
    ) -> list[dict]:
        """Generate candidates via LLM — initial sampling or evolution."""
        if mode == "code":
            prompt_mod = evolution_code
        else:
            prompt_mod = evolution_geometry

        if not winners:
            # Initial sampling
            template = prompt_mod.INITIAL_SAMPLING_PROMPT
            prompt = template.format(
                task=task,
                n_candidates=self.n_candidates,
                previous_context="",
                scene_info=scene_info if mode == "geometry" else "",
            )
        else:
            # Evolution — mutation + crossover
            template = prompt_mod.EVOLUTION_PROMPT
            winners_context = prompt_mod.build_winners_context(winners)
            prompt = template.format(
                task=task,
                n_candidates=self.n_candidates,
                winners_context=winners_context,
                scene_info=scene_info if mode == "geometry" else "",
            )

        try:
            raw = await llm_generate(
                model=self.model,
                contents=prompt,
                temperature=0.7,  # higher than normal for diversity
                response_json=True,
            )
            candidates = json.loads(raw)
            if isinstance(candidates, list):
                return candidates
            return []
        except Exception as e:
            print(f"[evolution] Sampling failed: {e}")
            return []

    # ──────────────────────────────────────────
    # Candidate evaluation
    # ──────────────────────────────────────────

    async def _evaluate_candidate(
        self,
        candidate: dict,
        mode: str,
        compile_fn: Optional[Callable],
        sim_eval_fn: Optional[Callable],
    ) -> float:
        """Evaluate a single candidate and return its fitness score."""
        if mode == "code":
            return await self._eval_code_candidate(candidate, compile_fn)
        else:
            return await self._eval_geometry_candidate(candidate, sim_eval_fn)

    async def _eval_code_candidate(
        self,
        candidate: dict,
        compile_fn: Optional[Callable],
    ) -> float:
        """Compile and execute a code tool candidate, return fitness."""
        name = candidate.get("tool_name", "unnamed")
        source = candidate.get("source_code", "")

        if not source:
            return 0.0

        # Compile
        try:
            fn = await compile_fn(name, source)
        except Exception as e:
            print(f"[evolution] Compile failed for {name}: {e}")
            return 0.0

        # Execute
        try:
            result = await fn()
            return score_code_result(result)
        except Exception as e:
            print(f"[evolution] Execution failed for {name}: {e}")
            return 0.0

    async def _eval_geometry_candidate(
        self,
        candidate: dict,
        sim_eval_fn: Optional[Callable],
    ) -> float:
        """Send geometry + waypoints to sim, return fitness."""
        if not sim_eval_fn:
            return 0.0

        mjcf = candidate.get("tool_mjcf", "")
        waypoints = candidate.get("action_waypoints", [])

        if not mjcf or not waypoints:
            return 0.0

        try:
            result = await sim_eval_fn(mjcf, waypoints)
            return score_geometry_result(result)
        except Exception as e:
            print(f"[evolution] Geometry eval failed: {e}")
            return 0.0

    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def _persist_winner(self, candidate: dict, score: float, mode: str):
        """Save the winning candidate to disk."""
        name = candidate.get("tool_name", "unnamed")

        if mode == "code":
            # Register as an invented tool (the forgebot caller handles this)
            pass  # handled by forgebot after evolve() returns

        elif mode == "geometry":
            os.makedirs(GEOMETRY_MEMORY_DIR, exist_ok=True)
            path = os.path.join(GEOMETRY_MEMORY_DIR, f"{name}.json")
            data = {
                "name": name,
                "description": candidate.get("description", ""),
                "tool_mjcf": candidate.get("tool_mjcf", ""),
                "action_waypoints": candidate.get("action_waypoints", []),
                "score": score,
                "created_at": time.time(),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[evolution] Persisted geometry tool: {name} (score: {score:.2f})")
