"""
Prompts for geometry-based tool evolution (VLMgineer-faithful).

The LLM generates:
  1. MJCF XML snippets defining tool geometry (boxes, capsules, etc.)
  2. Action waypoints — [x, y, z] positions for the end-effector to follow

Tools are attached to the robot's end-effector in the MuJoCo sim.
"""

INITIAL_SAMPLING_PROMPT = """\
You are a robotics hardware and controls expert designing physical tools for a Franka Panda
robot arm in a MuJoCo simulation.

TASK: {task}

SCENE INFO:
{scene_info}

Your goal: design {n_candidates} DIVERSE physical tool geometries + action plans.
Each design should take a fundamentally different approach (different shapes, strategies).

TOOL DESIGN RULES:
  - Tools are defined as MuJoCo MJCF XML body elements
  - Use only simple geometries: box, sphere, capsule, cylinder
  - Tool is attached to the robot end-effector at position [0, 0, 0.1] relative to the TCP
  - Keep tools lightweight (mass < 0.1 kg per geom)
  - Tools should be physically connected (no floating parts)
  - Think about how the tool shape enables the task — smarter geometry = simpler motion

ACTION WAYPOINT RULES:
  - Waypoints are [x, y, z] positions in world frame for the end-effector
  - The robot will move through these positions sequentially using IK
  - Include enough waypoints for: approach, contact, manipulation, retreat
  - Typical workspace: x=[0.2, 0.8], y=[-0.5, 0.5], z=[0.0, 0.6]

{previous_context}

Return a JSON array of {n_candidates} candidates:
[
  {{
    "tool_name": "descriptive_name",
    "description": "Design rationale — what shape and why",
    "tool_mjcf": "<body name='tool' pos='0 0 0.1'>\\n  <geom type='box' size='0.05 0.02 0.1' rgba='0.3 0.6 1 1'/>\\n</body>",
    "action_waypoints": [
      [0.5, 0.0, 0.4],
      [0.5, 0.0, 0.15],
      [0.3, 0.0, 0.15],
      [0.3, 0.0, 0.4]
    ]
  }},
  ...
]
"""

EVOLUTION_PROMPT = """\
You are evolving physical tool designs through mutation and crossover, inspired by
genetic algorithms. Your goal is to produce better tools than the previous generation.

TASK: {task}

SCENE INFO:
{scene_info}

TOP PERFORMING DESIGNS from previous generation:
{winners_context}

Produce {n_candidates} NEW tool designs via mutation and crossover:

MUTATION (change exactly one aspect of a winning tool):
  - Adjust a component's dimensions (longer, wider, curved)
  - Add a new component (side wall, hook, scoop lip)
  - Remove a component that isn't helping
  - Change orientation of a component

CROSSOVER (combine elements from two winning tools):
  - Use the base shape from one and the tip from another
  - Merge the action strategy of one with the geometry of another

All mutations/crossovers should plausibly enhance task success while preserving diversity.

Return a JSON array of {n_candidates} candidates:
[
  {{
    "tool_name": "descriptive_name",
    "description": "What changed from parent(s) and why",
    "mutation_type": "mutation|crossover",
    "parent": "name_of_parent(s)",
    "tool_mjcf": "<body name='tool' pos='0 0 0.1'>...</body>",
    "action_waypoints": [[x, y, z], ...]
  }},
  ...
]
"""


def build_winners_context(winners: list[tuple[dict, float]]) -> str:
    """Format winning geometry candidates + scores for the evolution prompt."""
    parts = []
    for candidate, score in winners:
        parts.append(
            f"--- {candidate['tool_name']} (score: {score:.2f}) ---\n"
            f"Description: {candidate.get('description', '')}\n"
            f"MJCF:\n{candidate['tool_mjcf']}\n"
            f"Waypoints: {candidate['action_waypoints']}\n"
        )
    return "\n".join(parts)
