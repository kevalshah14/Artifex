"""
ForgeBot Agent — the self-evolving brain that invents tools and skills.

Four-layer capability hierarchy
─────────────────────────────────
  Primitives   Built-in robot actions (move_to, pick_up, …).  Never invented.
  Tools        Atomic async functions invented by the agent, composed from
               primitives (and other tools).  Reused across tasks.
  Skills       High-level multi-step strategies invented by the agent.
  Evolution    VLMgineer-style population-based search that evolves code tools
               or physical tool geometry (MJCF) through LLM-guided mutation
               and crossover.  Auto-escalates from failed single-shot invention.

Self-evolving lifecycle
───────────────────────
  1. Plan    — LLM sees scene image + text, decides: reuse, invent, or evolve
  2. Execute — run the tool/skill
  3. Retry   — if fail, capture new image, feed error back to LLM (max 3)
  4. Evolve  — if retries exhausted, escalate to population search
  5. Persist — working tools/skills/geometry saved to disk
  6. Reload  — on next startup, everything is loaded back
"""

import base64
import json
import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from google.genai import types

from agent.primitives import PRIMITIVES, capture_scene_image
from agent.tool_registry import registry, Tool
from agent.skill_registry import skill_registry, Skill
from agent.evolution import EvolutionEngine
from agent.llm_client import generate as llm_generate

# Load .env from agent/, sim/, or project root
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'sim', '.env.local'))
load_dotenv()

MAX_RETRIES = 3


# ──────────────────────────────────────────
# Initialize registry with primitives
# ──────────────────────────────────────────

def init_primitives():
    """Load all primitives into the tool registry."""
    for name, info in PRIMITIVES.items():
        registry.register_primitive(
            name=name,
            fn=info["fn"],
            signature=info["signature"],
            description=info["description"],
        )


# ──────────────────────────────────────────
# System prompt — tuned for Gemini Robotics ER
# ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are ForgeBot, an autonomous robot agent with vision. You control a Franka Panda
robot arm in a MuJoCo physics simulation. You receive a live image of the scene with
every message so you can SEE the current state of the world.

You work step-by-step: each response is ONE action. After each action you will
receive an updated scene image and the action result, then decide the NEXT action.

{tool_descriptions}
{skill_descriptions}

════════════════════════════════════════
SCENE & COORDINATE SYSTEM
════════════════════════════════════════
  • The image shows a top-down-ish view of the table and robot.
  • Coordinate frame: X forward/back, Y left/right, Z up/down.
  • The table surface is at roughly Z ≈ 0.02.
  • Cubes are ~0.04 m on each side (size [0.02, 0.02, 0.02] half-extents).
  • ALWAYS call get_all_objects() first to get exact positions — do NOT guess
    coordinates from the image alone.

════════════════════════════════════════
MANIPULATION STRATEGY
════════════════════════════════════════
  For pick-and-place:
    1. get_all_objects() — get names, positions, colors
    2. grasp(body_name)  — grasp the target object (verifies lift)
    3. place_at(x, y, z) — place at target WITH approach-from-above
       • For stacking: z = target_top_z + half_cube_height + small_gap (~0.03)
       • place_at handles the approach, release, and retract automatically

  For moving without releasing:
    1. move_to(body_name, x, y, z) — pure IK move, does NOT open gripper
    2. set_gripper(open) — explicit gripper control

  IMPORTANT:
    • grasp() is preferred over pick_up() — it verifies the object lifted.
    • place_at() approaches from above, releases, and retracts. Use it for all
      placement tasks.
    • ALWAYS verify success after each action before proceeding.
    • If grasp fails, try again — the object may have shifted.

════════════════════════════════════════
STRIKING / HITTING STRATEGY  (hockey, pushing, sweeping)
════════════════════════════════════════
  USE strike_toward() — it handles all direction math automatically:

    strike_toward(object_name="ball", target_name="goal_post")

  This primitive:
    1. Reads the ball position and goal position from the sim
    2. Computes the correct XY direction vector
    3. Moves to a wind-up position behind the ball (opposite to goal)
    4. Executes a fast swing through the ball toward the goal
    5. Returns pre_strike_pos, post_strike_pos, and distance_moved

  WORKFLOW:
    1. get_all_objects() — find the ball name and the goal landmark name
    2. strike_toward(object_name="ball", target_name="goal_post")
    3. CHECK the result: if distance_moved < 0.01, the strike missed.
       Retry with adjusted strike_z or approach_dist.
    4. get_all_objects() again to verify the ball is near the goal.
    5. NEVER say "done" unless the ball actually moved toward the goal.

  Optional params:
    • approach_dist=0.15  — wind-up distance (increase if missing)
    • strike_duration=400 — ms for the swing (lower = faster = harder hit)
    • strike_z=0.03       — override hit height (default = ball's current Z)
    • target_pos=[x,y,z]  — explicit target instead of target_name

  If you need manual control, move_to() still works — its `duration`
  parameter controls speed (300-500 ms for fast strikes).

════════════════════════════════════════
SCENE MODIFICATION
════════════════════════════════════════
  Simple shapes (one call):
    • add_object(name, x, y, z, shape, color, size)
        Shapes : box, sphere, cylinder, capsule, ellipsoid
        Colors : red, green, blue, yellow, cyan, purple, orange, white, black, pink
        size   : half-extent / radius (default 0.02). Place on table: z ≈ 0.04

  Custom / composite objects — use when the user asks for any real-world
  object (plate, bowl, mug, table, chair, ramp, wall, etc.):
    • add_custom_object(name, x, y, z, body_xml)
      body_xml = raw MuJoCo MJCF geom elements (no <body> wrapper, <freejoint/>
      is added automatically).

      MJCF geom reference:
        type     | size attribute             | notes
        ---------|----------------------------|---------------------------
        box      | "sx sy sz"  (half-extents) | rectangular solid
        sphere   | "r"                        | ball
        cylinder | "r h"  (radius, half-h)    | round tube
        capsule  | "r h"  (radius, half-h)    | rounded-end cylinder
        ellipsoid| "rx ry rz" (semi-axes)     | squished sphere

      Each <geom> can have:
        pos="x y z"        — offset from body origin
        rgba="r g b a"     — color (0-1 floats)
        mass="m"           — mass in kg
        condim="4"         — contact dimensions
        friction="1 0.5 0.01"

      ── Example: plate (flat cylinder) ──
        add_custom_object("plate", 0, 0, 0.025, body_xml=
          '<geom type="cylinder" size="0.06 0.003" rgba="0.95 0.95 0.9 1" mass="0.1" condim="4" friction="1 0.5 0.01"/>')

      ── Example: bowl (base + angled ring of capsules) ──
        add_custom_object("bowl", 0.1, 0, 0.03, body_xml=
          '<geom type="cylinder" size="0.04 0.003" rgba="0.8 0.5 0.2 1" mass="0.05" condim="4" friction="1 0.5 0.01"/>'
          '<geom type="capsule" size="0.004 0.025" pos="0.035 0 0.015" euler="0 20 0" rgba="0.8 0.5 0.2 1" mass="0.01"/>'
          '<geom type="capsule" size="0.004 0.025" pos="-0.035 0 0.015" euler="0 -20 0" rgba="0.8 0.5 0.2 1" mass="0.01"/>'
          '<geom type="capsule" size="0.004 0.025" pos="0 0.035 0.015" euler="-20 0 0" rgba="0.8 0.5 0.2 1" mass="0.01"/>'
          '<geom type="capsule" size="0.004 0.025" pos="0 -0.035 0.015" euler="20 0 0" rgba="0.8 0.5 0.2 1" mass="0.01"/>')

      ── Example: small table (top + 4 legs) ──
        add_custom_object("mini_table", 0, 0.2, 0.06, body_xml=
          '<geom type="box" size="0.06 0.06 0.004" pos="0 0 0.04" rgba="0.6 0.35 0.15 1" mass="0.1"/>'
          '<geom type="cylinder" size="0.005 0.04" pos="0.045 0.045 0" rgba="0.6 0.35 0.15 1" mass="0.02"/>'
          '<geom type="cylinder" size="0.005 0.04" pos="-0.045 0.045 0" rgba="0.6 0.35 0.15 1" mass="0.02"/>'
          '<geom type="cylinder" size="0.005 0.04" pos="0.045 -0.045 0" rgba="0.6 0.35 0.15 1" mass="0.02"/>'
          '<geom type="cylinder" size="0.005 0.04" pos="-0.045 -0.045 0" rgba="0.6 0.35 0.15 1" mass="0.02"/>')

      ── Example: ramp ──
        add_custom_object("ramp", -0.1, 0, 0.02, body_xml=
          '<geom type="box" size="0.06 0.04 0.015" euler="0 15 0" rgba="0.4 0.4 0.4 1" mass="0.2" condim="4" friction="1 0.5 0.01"/>')

      TIPS:
        - Think about what real-world shape looks like, then approximate with
          MuJoCo primitives (combinations of boxes, cylinders, capsules, spheres).
        - Use 'pos' on child geoms for offsets from the body center.
        - Use 'euler' (degrees) for rotations.
        - Keep mass realistic (~0.01-0.5 kg for tabletop objects).
        - Test z so the object sits correctly on the table surface.

  Other scene commands:
    • remove_body(body_name)       — remove a single body
    • clear_objects(name_prefix)   — bulk remove: '' = all objects, 'cube' = only cubes
    • set_body_color(name, color)  — change color instantly
    • move_body(name, x, y, z)     — teleport a body
    • reset_scene()                — reset to initial state (brings cubes back)

════════════════════════════════════════
CUSTOM GRIPPER ATTACHMENT
════════════════════════════════════════
  You can attach custom tools/grippers to the robot's hand alongside the
  existing Panda fingers.  The geometry is injected as a rigid child of
  the 'hand' body in the MJCF model.

    • attach_gripper(gripper_xml, tcp_offset="0 0 0.1")
      gripper_xml : MJCF body/geom elements (rigidly attached to hand)
      tcp_offset  : IK target point relative to hand origin ("x y z").
                    Default "0 0 0.1" (10 cm below flange = fingertips).
                    If your tool extends further, increase z (e.g. "0 0 0.18").

    • detach_gripper()  — remove any custom gripper, restore default

  The hand frame: Z points downward (toward the table when arm is upright).
  So pos="0 0 0.12" means 12 cm below the flange = past the fingertips.

  ── Example: spatula ──
    attach_gripper(
      gripper_xml='<body name="spatula" pos="0 0 0.11">'
        '<geom type="box" size="0.04 0.002 0.06" rgba="0.7 0.7 0.7 1" mass="0.05"/>'
        '</body>',
      tcp_offset="0 0 0.17")

  ── Example: suction cup ──
    attach_gripper(
      gripper_xml='<body name="suction" pos="0 0 0.11">'
        '<geom type="cylinder" size="0.008 0.03" rgba="0.2 0.2 0.2 1" mass="0.02"/>'
        '<geom type="sphere" size="0.015" pos="0 0 0.03" rgba="0.8 0.2 0.2 1" mass="0.01"/>'
        '</body>',
      tcp_offset="0 0 0.15")

  ── Example: wide paddle ──
    attach_gripper(
      gripper_xml='<body name="paddle" pos="0 0 0.105">'
        '<geom type="box" size="0.06 0.06 0.003" rgba="0.9 0.8 0.3 1" mass="0.08"/>'
        '</body>',
      tcp_offset="0 0 0.11")

  ── Example: hockey stick (for striking / pushing balls) ──
    attach_gripper(
      gripper_xml='<body name="hockey_stick" pos="0 0 0.11">'
        '<geom type="cylinder" size="0.025 0.15" rgba="0.4 0.2 0.1 1" mass="0.08"/>'
        '</body>',
      tcp_offset="0 0 0.26")

  TIPS:
    - pos="0 0 0.1" on a child body starts at the fingertip level.
    - Keep tools lightweight (mass 0.01–0.1 kg) to avoid IK instability.
    - Update tcp_offset so the IK target lands at the tool's working tip.
    - The default fingers remain — this is an ADD-ON, not a replacement.
    - CRITICAL FOR STRIKING: use type="cylinder" (NOT box) for any tool
      that hits/pushes objects.  A cylinder is rotationally symmetric so the
      ball always deflects along the swing direction regardless of hand yaw.
      A box has flat faces that cause the ball to glance sideways.

════════════════════════════════════════
CAPABILITY HIERARCHY
════════════════════════════════════════
  Primitives  →  Tools  →  Skills
  • Primitives  Built-in robot actions. Use them directly or inside tools.
  • Tools       Atomic async functions you invent. One tool = one focused job.
  • Skills      High-level multi-step strategies you invent.

════════════════════════════════════════
YOUR DECISION PROCESS (use the image!)
════════════════════════════════════════
  1. LOOK at the image — what do you see? Where are objects?
  2. If the task is fully accomplished            → action: done
  3. If an existing TOOL covers the next step     → action: execute
  4. If an existing SKILL covers the next step    → action: use_skill
  5. If you need a new single atomic capability   → action: invent
  6. If the step is multi-step / complex          → action: create_skill
  7. If the task needs a PHYSICAL tool design     → action: evolve (mode: geometry)
  8. If previous attempts keep failing            → action: evolve (mode: code)

════════════════════════════════════════
DATA FORMAT NOTES
════════════════════════════════════════
  • get_all_objects() → {{"success": bool, "objects": [list], "landmarks": [list]}}
    objects:   movable bodies {{"name", "shape", "position": [x,y,z], "color", "movable": true}}
    landmarks: fixed bodies   {{"name", "shape", "position": [x,y,z], "movable": false}}
    ALWAYS check landmarks for targets like "goal_post".
  • move_to(x, y, z) → {{"success": bool, "requested": [x,y,z], "actual_tcp": [x,y,z]}}
    Compare requested vs actual_tcp — if they differ significantly,
    the target was unreachable (out of workspace or collision).
  • grasp(body_name) → {{"success": bool, "holding": str|null}}
    Pass the object NAME (e.g. "cube0"), NOT a description.
  • place_at(x, y, z) → {{"success": bool, "placed_at": [x,y,z]}}
  • Always check "success" before using results.

════════════════════════════════════════
WRITING CODE RULES
════════════════════════════════════════
  • All functions MUST be `async def`.
  • Available in namespace (no imports needed):
      Primitives : move_to, set_gripper, get_body_position, get_body_color,
                   get_all_objects, pick_up, grasp, place_at, step_sim,
                   add_object, add_custom_object, remove_body,
                   clear_objects, set_body_color, move_body, reset_scene,
                   attach_gripper, detach_gripper, strike_toward
      Invented   : all registered tools/skills by name
      Stdlib     : asyncio, json
  • Always return {{"success": bool, ...}}.
  • Keep tools focused (one job). Skills may be longer.

════════════════════════════════════════
RESPONSE FORMAT  (strict JSON, no markdown)
════════════════════════════════════════

── task is fully done ─────────────────
{{
    "thought": "Looking at the scene, the task is complete because ...",
    "action": "done",
    "summary": "Brief summary of what was accomplished"
}}

── execute a single existing tool ─────
{{
    "thought": "I can see [object] at [position]. I need to ...",
    "action": "execute",
    "tool_name": "existing_tool_name",
    "tool_args": {{}}
}}

── use an existing skill ──────────────
{{
    "thought": "...",
    "action": "use_skill",
    "skill_name": "existing_skill_name",
    "skill_args": {{}}
}}

── invent one new atomic tool ─────────
{{
    "thought": "I need a tool that ...",
    "action": "invent",
    "tool_name": "new_tool_name",
    "tool_description": "What it does",
    "tool_signature": "new_tool_name(arg: type) -> dict",
    "composed_from": ["primitive_or_tool"],
    "source_code": "async def new_tool_name(arg):\\n    ...",
    "then_execute_with": {{}}
}}

── create a new skill (complex task) ──
{{
    "thought": "...",
    "action": "create_skill",
    "skill_name": "new_skill_name",
    "skill_description": "What the skill accomplishes",
    "new_tools": [
        {{
            "tool_name": "helper_tool",
            "tool_description": "...",
            "tool_signature": "helper_tool(arg: type) -> dict",
            "composed_from": ["primitive"],
            "source_code": "async def helper_tool(arg):\\n    ..."
        }}
    ],
    "skill_source_code": "async def new_skill_name():\\n    ...",
    "tools_used": ["helper_tool", "pick_up", "place_at"],
    "then_execute_with": {{}}
}}

── evolve a tool via population search ─
{{
    "thought": "I need to design a physical tool / previous code attempts failed...",
    "action": "evolve",
    "evolve_mode": "code|geometry",
    "task_description": "What the evolved tool should accomplish"
}}
"""


def build_system_prompt() -> str:
    """Insert registry text without str.format — avoids KeyError if prompt or descriptions contain `{`."""
    text = SYSTEM_PROMPT.replace("{tool_descriptions}", registry.get_tool_descriptions()).replace(
        "{skill_descriptions}", skill_registry.get_skill_descriptions()
    )
    return text.replace("{{", "{").replace("}}", "}")


# ──────────────────────────────────────────
# Agent Core
# ──────────────────────────────────────────

class ForgeBotAgent:
    """The main agent that plans, invents, and executes — with vision."""

    def __init__(self, model: str = "gemini-3.1-flash-lite-preview"):
        self.model = model
        self._event_callback = None
        init_primitives()
        self._load_persisted()

    def _load_persisted(self):
        """Load all previously invented tools and skills from disk."""
        def sync_compiler(name, source_code):
            namespace = self._build_execution_namespace(exclude_name=name)
            exec(source_code, namespace)
            if name not in namespace:
                raise ValueError(f"Source code did not define '{name}'")
            return namespace[name]

        loaded_tools = registry.load_all_from_disk(sync_compiler)
        if loaded_tools:
            print(f"[forgebot] ♻ Loaded {len(loaded_tools)} persisted tools: {', '.join(loaded_tools)}")

        loaded_skills = skill_registry.load_all_from_disk(sync_compiler)
        if loaded_skills:
            print(f"[forgebot] ♻ Loaded {len(loaded_skills)} persisted skills: {', '.join(loaded_skills)}")

    def on_event(self, callback):
        """Register callback for agent events (for streaming to frontend)."""
        self._event_callback = callback

    async def _emit(self, event_type: str, data: dict):
        """Emit an event to the frontend."""
        if self._event_callback:
            await self._event_callback(event_type, data)

    # ──────────────────────────────────────────
    # Scene image capture
    # ──────────────────────────────────────────

    async def _capture_image(self) -> Optional[bytes]:
        """Capture scene image from sim; returns raw JPEG bytes or None."""
        try:
            result = await capture_scene_image()
            if result.get("success") and result.get("image_base64"):
                return base64.b64decode(result["image_base64"])
        except Exception as e:
            print(f"[forgebot] Image capture failed: {e}")
        return None

    # ──────────────────────────────────────────
    # Main entry point — multi-step agentic loop
    # ──────────────────────────────────────────

    MAX_STEPS = 10
    MAX_RETRIES = 3

    async def handle_task(self, user_message: str) -> dict:
        """
        Multi-step agentic loop with vision:
          1. Capture scene image
          2. LLM sees image + text, picks an action
          3. Execute it
          4. If fail → capture new image, retry with error context
          5. If succeed → loop back with updated image
          6. LLM says "done" → persist artifacts, return
        """
        await self._emit("thinking", {"message": f"Analyzing task: {user_message}"})

        history: list[dict] = []
        retries_this_step = 0

        for step in range(1, self.MAX_STEPS + 1):
            system = build_system_prompt()

            # Capture current scene image
            await self._emit("thinking", {"message": "Capturing scene image..."})
            image_bytes = await self._capture_image()

            # Build multimodal contents: [image, text]
            contents = self._build_contents(user_message, history, image_bytes)

            label = "Planning approach..." if step == 1 and retries_this_step == 0 else f"Step {step}..."
            if retries_this_step > 0:
                label = f"Retrying (attempt {retries_this_step + 1}/{self.MAX_RETRIES})..."
            await self._emit("llm_call", {"message": label})

            try:
                raw = await llm_generate(
                    model=self.model,
                    contents=contents,
                    system_instruction=system,
                    temperature=0.2,
                    response_json=True,
                )
            except Exception as e:
                return {"error": f"LLM call failed: {e}", "success": False}

            try:
                plan = json.loads(raw)
            except json.JSONDecodeError:
                history.append({"role": "error", "content": f"Invalid JSON: {raw[:300]}"})
                retries_this_step += 1
                if retries_this_step >= self.MAX_RETRIES:
                    return {"error": "LLM returned invalid JSON after retries", "success": False}
                continue

            action = plan.get("action", "")
            await self._emit("plan", {
                "thought": plan.get("thought", ""),
                "action": action,
                "tool_name": plan.get("tool_name") or plan.get("skill_name", ""),
                "step": step,
            })

            # ── "done" ──
            if action == "done":
                await self._emit("done", {
                    "summary": plan.get("summary", ""),
                    "message": "Task complete",
                })
                self._persist_all_invented()
                return {"success": True, "summary": plan.get("summary", "")}

            # ── Execute ──
            result = await self._route_action(action, plan)
            error_msg = self._extract_error(result)

            if error_msg:
                retries_this_step += 1
                await self._emit("retry", {
                    "attempt": retries_this_step,
                    "max_retries": self.MAX_RETRIES,
                    "error": error_msg,
                    "message": f"Step failed — {'retrying' if retries_this_step < self.MAX_RETRIES else 'giving up'}",
                })

                source_code = plan.get("source_code", "") or plan.get("skill_source_code", "")
                history.append({
                    "role": "step_error",
                    "action": action,
                    "name": plan.get("tool_name") or plan.get("skill_name", ""),
                    "error": error_msg,
                    "source_code": source_code,
                })

                if retries_this_step >= self.MAX_RETRIES:
                    await self._emit("auto_escalate", {
                        "message": "Retries exhausted — escalating to evolution mode",
                        "error": error_msg,
                    })
                    evolve_result = await self._handle_evolve({
                        "evolve_mode": "code",
                        "task_description": f"{user_message} (previous error: {error_msg})",
                    })
                    if evolve_result.get("success"):
                        history.append({
                            "role": "step_result",
                            "action": "evolve",
                            "name": evolve_result.get("candidate", {}).get("tool_name", "evolved"),
                            "result": json.dumps(evolve_result, default=str)[:500],
                        })
                        retries_this_step = 0
                        continue
                    return {"success": False, "error": f"Evolution also failed: {evolve_result.get('error', 'unknown')}"}
            else:
                retries_this_step = 0
                self._persist_new_artifacts(action, plan)

                result_summary = json.dumps(result, default=str)[:500]
                history.append({
                    "role": "step_result",
                    "action": action,
                    "name": plan.get("tool_name") or plan.get("skill_name", ""),
                    "result": result_summary,
                })

        return {"success": False, "error": f"Reached max steps ({self.MAX_STEPS}) without completing task"}

    def _build_contents(self, original_task: str, history: list[dict], image_bytes: Optional[bytes] = None) -> list:
        """Build multimodal LLM contents: [scene_image, text_prompt]."""
        parts = []

        # Scene image (if available)
        if image_bytes:
            parts.append(
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            )

        # Text prompt
        text_parts = [f"TASK: {original_task}"]
        for entry in history:
            role = entry["role"]
            if role == "step_result":
                text_parts.append(f"\n--- Step completed: {entry['action']} ({entry['name']}) ---")
                text_parts.append(f"Result: {entry['result']}")
            elif role == "step_error":
                text_parts.append(f"\n--- Step FAILED: {entry['action']} ({entry['name']}) ---")
                text_parts.append(f"Error: {entry['error']}")
                if entry.get("source_code"):
                    text_parts.append(f"Failed source code:\n{entry['source_code']}")
                text_parts.append("Look at the updated image. Fix the issue and try a different approach.")
            elif role == "error":
                text_parts.append(f"\nError: {entry['content']}")

        if history:
            text_parts.append(
                "\nLook at the scene image above. What is the current state? "
                "What is the next action? If the task is fully done, use action: done."
            )

        parts.append("\n".join(text_parts))
        return parts

    @staticmethod
    def _extract_error(result: dict) -> Optional[str]:
        """Extract error message from a result dict, checking nested levels."""
        if "error" in result:
            return result["error"]
        inner = result.get("result", {})
        if isinstance(inner, dict):
            if "error" in inner:
                return inner["error"]
            if inner.get("success") is False:
                return inner.get("error", f"Tool returned success=false: {json.dumps(inner)[:200]}")
        if result.get("success") is False:
            return "Action returned success=false"
        return None

    async def _route_action(self, action: str, plan: dict) -> dict:
        """Route a plan to the correct handler."""
        if action == "invent":
            return await self._handle_invent(plan)
        elif action == "execute":
            return await self._handle_execute(plan)
        elif action == "use_skill":
            return await self._handle_use_skill(plan)
        elif action == "create_skill":
            return await self._handle_create_skill(plan)
        elif action == "evolve":
            return await self._handle_evolve(plan)
        else:
            return {"error": f"Unknown action: {action}", "plan": plan}

    def _persist_new_artifacts(self, action: str, plan: dict):
        """Persist any tools/skills created during this specific action."""
        if action == "invent":
            name = plan.get("tool_name", "")
            if name and registry.has_tool(name):
                registry.save_tool(name)
                print(f"[forgebot] 💾 Persisted tool: {name}")
        elif action == "create_skill":
            for tool_spec in plan.get("new_tools", []):
                tool_name = tool_spec.get("tool_name", "")
                if tool_name and registry.has_tool(tool_name):
                    registry.save_tool(tool_name)
                    print(f"[forgebot] 💾 Persisted supporting tool: {tool_name}")
            skill_name = plan.get("skill_name", "")
            if skill_name and skill_registry.has_skill(skill_name):
                skill_registry.save_skill(skill_name)
                print(f"[forgebot] 💾 Persisted skill: {skill_name}")

    def _persist_all_invented(self):
        """Persist ALL invented tools and skills (called on task completion)."""
        for tool in registry.get_invented_tools():
            registry.save_tool(tool.name)
        for skill in skill_registry.get_all_skills():
            skill_registry.save_skill(skill.name)
        count = len(registry.get_invented_tools()) + len(skill_registry.get_all_skills())
        if count:
            print(f"[forgebot] 💾 Persisted {count} artifacts to memory/")

    # ──────────────────────────────────────────
    # Action handlers
    # ──────────────────────────────────────────

    async def _handle_invent(self, plan: dict) -> dict:
        """Handle tool invention: compile, register, then execute."""
        tool_name = plan["tool_name"]
        source_code = plan["source_code"]

        await self._emit("inventing", {
            "tool_name": tool_name,
            "description": plan.get("tool_description", ""),
            "source_code": source_code,
            "composed_from": plan.get("composed_from", []),
        })

        try:
            fn = await self._compile_tool(tool_name, source_code)
        except Exception as e:
            await self._emit("error", {"message": f"Failed to compile tool: {e}"})
            return {"error": f"Compilation failed: {e}", "source_code": source_code}

        tool = registry.register_invented_tool(
            name=tool_name,
            description=plan.get("tool_description", ""),
            signature=plan.get("tool_signature", f"{tool_name}()"),
            source_code=source_code,
            composed_from=plan.get("composed_from", []),
            fn=fn,
        )

        await self._emit("tool_registered", {
            "tool": tool.to_dict(),
            "message": f"Invented new tool: {tool_name}",
        })

        exec_args = plan.get("then_execute_with", {})
        return await self._execute_tool(tool_name, exec_args)

    async def _handle_evolve(self, plan: dict) -> dict:
        """Handle evolution: population-based search for code or geometry tools."""
        mode = plan.get("evolve_mode", "code")
        task_desc = plan.get("task_description", "")

        engine = EvolutionEngine(model=self.model)
        engine.on_event(self._event_callback)

        if mode == "code":
            result = await engine.evolve(
                task=task_desc,
                mode="code",
                compile_fn=self._compile_tool,
            )
            if result.get("success"):
                candidate = result["candidate"]
                name = candidate.get("tool_name", "evolved_tool")
                source = candidate.get("source_code", "")
                try:
                    fn = await self._compile_tool(name, source)
                    registry.register_invented_tool(
                        name=name,
                        description=candidate.get("description", ""),
                        signature=candidate.get("signature", f"{name}()"),
                        source_code=source,
                        composed_from=candidate.get("composed_from", []),
                        fn=fn,
                    )
                    registry.save_tool(name)
                    await self._emit("tool_registered", {
                        "tool": {"name": name, "score": result["score"]},
                        "message": f"Evolution winner registered: {name} (score: {result['score']:.2f})",
                    })
                except Exception as e:
                    return {"error": f"Failed to register evolved tool: {e}"}
            return result

        elif mode == "geometry":
            from agent.primitives import send_command

            async def sim_eval_fn(mjcf: str, waypoints: list) -> dict:
                return await send_command({
                    "action": "eval_tool",
                    "tool_mjcf": mjcf,
                    "waypoints": waypoints,
                }, timeout=60.0)

            result = await engine.evolve(
                task=task_desc,
                mode="geometry",
                sim_eval_fn=sim_eval_fn,
            )
            return result

        return {"error": f"Unknown evolve mode: {mode}"}

    async def _handle_execute(self, plan: dict) -> dict:
        """Execute an existing tool."""
        tool_name = plan["tool_name"]
        tool_args = plan.get("tool_args", {})
        steps = plan.get("steps", [])

        for i, step in enumerate(steps):
            await self._emit("step", {"index": i, "description": step})

        return await self._execute_tool(tool_name, tool_args)

    # ──────────────────────────────────────────
    # Skill handlers
    # ──────────────────────────────────────────

    async def _handle_use_skill(self, plan: dict) -> dict:
        """Execute an existing skill."""
        skill_name = plan["skill_name"]
        skill_args = plan.get("skill_args", {})

        skill = skill_registry.get_skill(skill_name)
        if not skill:
            return {"error": f"Skill '{skill_name}' not found in registry"}

        return await self._execute_skill(skill_name, skill_args)

    async def _handle_create_skill(self, plan: dict) -> dict:
        """Create a new skill with optional supporting tools, then execute it."""
        skill_name = plan["skill_name"]
        skill_source = plan["skill_source_code"]
        tools_created: list[str] = []

        for tool_spec in plan.get("new_tools", []):
            tool_name = tool_spec["tool_name"]
            await self._emit("inventing", {
                "tool_name": tool_name,
                "description": tool_spec.get("tool_description", ""),
                "source_code": tool_spec.get("source_code", ""),
                "composed_from": tool_spec.get("composed_from", []),
                "context": f"Supporting tool for skill '{skill_name}'",
            })
            try:
                fn = await self._compile_tool(tool_name, tool_spec["source_code"])
            except Exception as e:
                await self._emit("error", {"message": f"Failed to compile tool '{tool_name}': {e}"})
                return {"error": f"Tool compilation failed: {e}", "tool_name": tool_name}

            tool = registry.register_invented_tool(
                name=tool_name,
                description=tool_spec.get("tool_description", ""),
                signature=tool_spec.get("tool_signature", f"{tool_name}()"),
                source_code=tool_spec["source_code"],
                composed_from=tool_spec.get("composed_from", []),
                fn=fn,
            )
            tools_created.append(tool_name)
            await self._emit("tool_registered", {
                "tool": tool.to_dict(),
                "message": f"Created supporting tool: {tool_name}",
            })

        await self._emit("creating_skill", {
            "skill_name": skill_name,
            "description": plan.get("skill_description", ""),
            "source_code": skill_source,
            "tools_used": plan.get("tools_used", []),
            "tools_created": tools_created,
        })
        try:
            skill_fn = await self._compile_skill(skill_name, skill_source)
        except Exception as e:
            await self._emit("error", {"message": f"Failed to compile skill '{skill_name}': {e}"})
            return {"error": f"Skill compilation failed: {e}", "skill_name": skill_name}

        skill = skill_registry.register_skill(
            name=skill_name,
            description=plan.get("skill_description", ""),
            source_code=skill_source,
            tools_used=plan.get("tools_used", []),
            tools_created=tools_created,
            fn=skill_fn,
        )
        await self._emit("skill_registered", {
            "skill": skill.to_dict(),
            "message": f"Created new skill: {skill_name}",
        })

        exec_args = plan.get("then_execute_with", {})
        return await self._execute_skill(skill_name, exec_args)

    async def _execute_skill(self, skill_name: str, args: dict) -> dict:
        """Execute a registered skill."""
        skill = skill_registry.get_skill(skill_name)
        if not skill:
            return {"error": f"Skill '{skill_name}' not found in registry"}
        if not skill.fn:
            return {"error": f"Skill '{skill_name}' has no executable function"}

        clean_args = {k: v for k, v in args.items() if k not in self._META_KEYS}

        await self._emit("executing_skill", {
            "skill_name": skill_name,
            "args": clean_args,
            "message": f"Executing skill {skill_name}...",
        })
        try:
            result = await skill.fn(**clean_args)
            skill.invocation_count += 1
            skill_success = result.get("success", True) if isinstance(result, dict) else True
            await self._emit("skill_result", {
                "skill_name": skill_name,
                "result": result,
                "message": f"Skill {skill_name} complete",
            })
            return {"success": skill_success, "skill_name": skill_name, "result": result}
        except Exception as e:
            await self._emit("error", {
                "skill_name": skill_name,
                "message": f"Skill execution failed: {e}",
            })
            return {"error": str(e), "skill_name": skill_name}

    # ──────────────────────────────────────────
    # Compilation helpers
    # ──────────────────────────────────────────

    def _build_execution_namespace(self, exclude_name: str = "") -> dict:
        """Build the shared namespace for compiled tools/skills."""
        from agent import primitives

        namespace: dict = {
            "asyncio": asyncio,
            "json": json,
            "move_to": primitives.move_to,
            "set_gripper": primitives.set_gripper,
            "get_body_position": primitives.get_body_position,
            "get_body_color": primitives.get_body_color,
            "get_all_objects": primitives.get_all_objects,
            "pick_up": primitives.pick_up,
            "grasp": primitives.grasp,
            "place_at": primitives.place_at,
            "step_sim": primitives.step_sim,
            "add_object": primitives.add_object,
            "add_custom_object": primitives.add_custom_object,
            "remove_body": primitives.remove_body,
            "set_body_color": primitives.set_body_color,
            "move_body": primitives.move_body,
            "clear_objects": primitives.clear_objects,
            "reset_scene": primitives.reset_scene,
            "attach_gripper": primitives.attach_gripper,
            "detach_gripper": primitives.detach_gripper,
            "strike_toward": primitives.strike_toward,
        }
        for tool in registry.get_invented_tools():
            if tool.fn and tool.name != exclude_name:
                namespace[tool.name] = tool.fn
        for skill in skill_registry.get_all_skills():
            if skill.fn and skill.name != exclude_name:
                namespace[skill.name] = skill.fn
        return namespace

    async def _compile_tool(self, name: str, source_code: str) -> callable:
        """Safely compile an invented tool from source code."""
        namespace = self._build_execution_namespace(exclude_name=name)
        exec(source_code, namespace)
        if name not in namespace:
            raise ValueError(f"Source code did not define function '{name}'")
        return namespace[name]

    async def _compile_skill(self, name: str, source_code: str) -> callable:
        """Safely compile an invented skill from source code."""
        namespace = self._build_execution_namespace(exclude_name=name)
        exec(source_code, namespace)
        if name not in namespace:
            raise ValueError(f"Source code did not define function '{name}'")
        return namespace[name]

    _META_KEYS = frozenset({
        "tool_name", "tool_args", "skill_name", "skill_args",
        "action", "thought", "steps", "then_execute_with",
        "tool_description", "tool_signature", "composed_from",
        "source_code", "skill_description", "skill_source_code",
        "new_tools", "tools_used",
    })

    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute a registered tool."""
        tool = registry.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found in registry"}

        if not tool.fn:
            return {"error": f"Tool '{tool_name}' has no executable function"}

        clean_args = {k: v for k, v in args.items() if k not in self._META_KEYS}

        await self._emit("executing", {
            "tool_name": tool_name,
            "args": clean_args,
            "message": f"Executing {tool_name}...",
        })

        try:
            result = await tool.fn(**clean_args)
            tool.invocation_count += 1
            tool_success = result.get("success", True) if isinstance(result, dict) else True
            await self._emit("result", {
                "tool_name": tool_name,
                "result": result,
                "message": f"Completed {tool_name}",
            })
            return {"success": tool_success, "tool_name": tool_name, "result": result}
        except Exception as e:
            await self._emit("error", {
                "tool_name": tool_name,
                "message": f"Execution failed: {e}",
            })
            return {"error": str(e), "tool_name": tool_name}


# Singleton agent
agent = ForgeBotAgent()
