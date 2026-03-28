"""
Prompts for code-based tool evolution.

Follows VLMgineer's prompt structure:
  - Initial sampling: generate diverse candidates
  - Evolution: mutate/crossover from winners
"""

INITIAL_SAMPLING_PROMPT = """\
You are ForgeBot, an autonomous robot agent that invents tools as async Python functions.
You control a robot arm in a MuJoCo physics simulation.

Your goal: generate {n_candidates} DIVERSE candidate tool implementations for this task.
Each candidate should take a DIFFERENT approach to solving the problem.

TASK: {task}

AVAILABLE PRIMITIVES (call directly, no imports needed):
  get_all_objects() -> dict  — returns {{"success": bool, "objects": [{{name, position, color, size}}]}}
  get_body_position(body_name: str) -> dict
  get_body_color(body_name: str) -> dict  — returns {{"color": "red"|"cyan"|"green"|"yellow"}}
  pick_up(body_name: str) -> dict  — full animation: move, grasp, lift
  place_at(x, y, z) -> dict
  move_to(body_name, x, y, z) -> dict
  set_gripper(open: bool) -> dict
  step_sim(n_steps: int) -> dict
Also available: asyncio, json

RULES:
  - Each function MUST be `async def`
  - MUST return a dict with at least {{"success": bool}}
  - Objects list is a LIST of dicts, not a dict
  - Use get_all_objects() first to discover object names before pick_up()

{previous_context}

Return a JSON array of {n_candidates} candidates:
[
  {{
    "tool_name": "unique_name_v1",
    "description": "What this approach does differently",
    "signature": "tool_name(args) -> dict",
    "source_code": "async def tool_name(...):\\n    ...",
    "composed_from": ["get_all_objects", "pick_up"]
  }},
  ...
]
"""

EVOLUTION_PROMPT = """\
You are evolving robot tool implementations through mutation and crossover.

TASK: {task}

Here are the TOP PERFORMING candidates from the previous generation:
{winners_context}

Your job: produce {n_candidates} NEW candidates via mutation and crossover.

MUTATION: Take one winning tool and change one aspect:
  - Fix a bug or edge case
  - Change the search/filter strategy
  - Adjust error handling
  - Optimize the control flow

CROSSOVER: Combine elements from two winning tools:
  - Use the search logic from one and the execution logic from another
  - Merge complementary approaches

RULES:
  - Each function MUST be `async def`
  - MUST return dict with {{"success": bool}}
  - ALL candidates must be different from each other AND from the winners
  - Aim to IMPROVE on the winners' scores

Return a JSON array of {n_candidates} candidates:
[
  {{
    "tool_name": "unique_name",
    "description": "What changed and why",
    "mutation_type": "mutation|crossover",
    "parent": "name_of_parent_tool(s)",
    "signature": "tool_name(args) -> dict",
    "source_code": "async def tool_name(...):\\n    ...",
    "composed_from": ["primitives_used"]
  }},
  ...
]
"""


def build_winners_context(winners: list[tuple[dict, float]]) -> str:
    """Format winning candidates + scores for the evolution prompt."""
    parts = []
    for candidate, score in winners:
        parts.append(
            f"--- {candidate['tool_name']} (score: {score:.2f}) ---\n"
            f"Description: {candidate.get('description', '')}\n"
            f"Source:\n{candidate['source_code']}\n"
        )
    return "\n".join(parts)
