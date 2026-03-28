"""
ForgeBot Agent — the self-evolving brain that invents tools and skills.

Three-layer capability hierarchy
─────────────────────────────────
  Primitives   Built-in robot actions (move_to, pick_up, …).  Never invented.
  Tools        Atomic async functions invented by the agent, composed from
               primitives (and other tools).  Reused across tasks.
  Skills       High-level multi-step strategies invented by the agent.
               A skill orchestrates tools and primitives, and may trigger
               the invention of new tools at creation time.  Also reused.

Self-evolving lifecycle
───────────────────────
  1. Plan   — LLM decides: reuse existing tool/skill, or invent new one
  2. Execute — run the tool/skill
  3. Retry  — if it fails, feed the error back to LLM and re-invent (max 3)
  4. Persist — once a tool/skill works, save to disk (agent/memory/)
  5. Reload — on next startup, all persisted tools/skills are loaded back
"""

import json
import asyncio
import os
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from agent.primitives import PRIMITIVES
from agent.tool_registry import registry, Tool
from agent.skill_registry import skill_registry, Skill

# Load .env from sim folder or project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'sim', '.env'))
load_dotenv()  # also check project root

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
# LLM Client (Google Gemini)
# ──────────────────────────────────────────

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Add it to .env or export it.")
        _client = genai.Client(api_key=api_key)
    return _client


# ──────────────────────────────────────────
# The Tool Invention Prompt
# ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are ForgeBot, an autonomous robot agent that invents tools and skills to accomplish tasks.

You control a robot arm in a MuJoCo physics simulation.
You work step-by-step: each response is ONE action. After each action completes,
you see the result and decide the NEXT action. When the task is fully done, use "done".

{tool_descriptions}
{skill_descriptions}

════════════════════════════════════════
CAPABILITY HIERARCHY
════════════════════════════════════════
  Primitives  →  Tools  →  Skills
  • Primitives  Built-in robot actions. Use them inside tools.
  • Tools       Atomic async functions you invent. One tool = one focused job.
                They compose primitives (and other tools).
  • Skills      High-level multi-step strategies you invent. A skill
                orchestrates several tools/primitives to accomplish a complex
                task. Skills can create new tools as part of their setup.

════════════════════════════════════════
YOUR DECISION PROCESS
════════════════════════════════════════
  1. If the task is fully accomplished            → action: done
  2. If an existing SKILL covers the next step    → action: use_skill
  3. If a single existing TOOL covers the step    → action: execute
  4. If you need a new single atomic capability   → action: invent
  5. If the step is multi-step / complex          → action: create_skill
     (create_skill may also invent supporting tools first)

════════════════════════════════════════
IMPORTANT DATA FORMAT NOTES
════════════════════════════════════════
  • get_all_objects() returns {{"success": bool, "objects": [list of dicts]}}
    Each object dict has: {{"name": str, "position": [x,y,z], "color": str, "size": [x,y,z]}}
    The "color" field is a human-readable string like "red", "cyan", "green", "yellow".
    Objects is a LIST, not a dict. Iterate with: for obj in objects_data["objects"]:
  • get_body_color() returns {{"success": bool, "color": str}} where color is a string.
  • pick_up(body_name) takes the object's NAME (e.g. "cube0"), NOT a description.
    Use get_all_objects() first to find the correct name.
  • Always check the "success" field before using results.

════════════════════════════════════════
WRITING CODE RULES
════════════════════════════════════════
  • All functions MUST be `async def`.
  • Available in every execution namespace (no imports needed):
      Primitives : move_to, set_gripper, get_body_position, get_body_color,
                   get_all_objects, pick_up, place_at, step_sim
      Invented tools  : all previously registered tools by name
      Invented skills : all previously registered skills by name
      Stdlib : asyncio, json
  • Always return a dict with at least {{"success": bool}}.
  • Keep tools focused (one job). Skills may be longer.
  • Do NOT put extra keys like "tool_name" in then_execute_with —
    only pass the actual function arguments.

════════════════════════════════════════
RESPONSE FORMAT  (strict JSON, no markdown)
════════════════════════════════════════

── task is fully done ─────────────────
{{
    "thought": "The task is complete because ...",
    "action": "done",
    "summary": "Brief summary of what was accomplished"
}}

── use an existing skill ──────────────
{{
    "thought": "...",
    "action": "use_skill",
    "skill_name": "existing_skill_name",
    "skill_args": {{}}
}}

── execute a single existing tool ─────
{{
    "thought": "...",
    "action": "execute",
    "tool_name": "existing_tool_name",
    "tool_args": {{}}
}}

── invent one new atomic tool ─────────
{{
    "thought": "...",
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
"""


# ──────────────────────────────────────────
# Agent Core
# ──────────────────────────────────────────

class ForgeBotAgent:
    """The main agent that plans, invents, and executes — with self-healing retry."""

    def __init__(self, model: str = "gemini-2.5-flash"):
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
    # Main entry point — with retry loop
    # ──────────────────────────────────────────

    async def handle_task(self, user_message: str) -> dict:
        """Main entry point: plan → execute → retry on failure → persist on success."""
        await self._emit("thinking", {"message": f"Analyzing task: {user_message}"})

        error_context = None

        for attempt in range(1, MAX_RETRIES + 1):
            # Build prompt with current tool + skill state
            system = SYSTEM_PROMPT.format(
                tool_descriptions=registry.get_tool_descriptions(),
                skill_descriptions=skill_registry.get_skill_descriptions(),
            )

            # Build the user message — include error feedback on retries
            if error_context:
                await self._emit("evolving", {
                    "attempt": attempt,
                    "max_retries": MAX_RETRIES,
                    "message": f"Attempt {attempt}/{MAX_RETRIES} — fixing previous error...",
                    "error": error_context["error"],
                })
                contents = RETRY_PROMPT.format(**error_context)
            else:
                contents = user_message

            # Call LLM
            await self._emit("llm_call", {
                "message": f"Planning approach..." if attempt == 1 else f"Re-planning (attempt {attempt}/{MAX_RETRIES})...",
            })

            try:
                response = await get_client().aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
            except Exception as e:
                return {"error": f"LLM call failed: {e}", "success": False}

            raw = response.text
            try:
                plan = json.loads(raw)
            except json.JSONDecodeError:
                error_context = {
                    "action": "parse",
                    "name": "LLM response",
                    "error": f"Invalid JSON from LLM: {raw[:200]}",
                    "source_context": "",
                    "original_task": user_message,
                }
                continue

            action = plan.get("action", "")
            await self._emit("plan", {
                "thought": plan.get("thought", ""),
                "action": action,
                "tool_name": plan.get("tool_name") or plan.get("skill_name", ""),
                "attempt": attempt,
            })

            # Route based on action
            result = await self._route_action(action, plan)

            # Check if it succeeded
            succeeded = result.get("success", False)
            has_error = "error" in result

            if succeeded and not has_error:
                # Persist any newly invented tools/skills
                self._persist_new_artifacts(action, plan)
                return result

            # Failed — build error context for retry
            error_msg = result.get("error", "Unknown error")
            source_code = plan.get("source_code", "")
            if not source_code and action == "create_skill":
                source_code = plan.get("skill_source_code", "")

            error_context = {
                "action": action,
                "name": plan.get("tool_name") or plan.get("skill_name", "unknown"),
                "error": error_msg,
                "source_context": f"Source code that failed:\n{source_code}" if source_code else "",
                "original_task": user_message,
            }

            await self._emit("retry", {
                "attempt": attempt,
                "max_retries": MAX_RETRIES,
                "error": error_msg,
                "message": f"Failed on attempt {attempt} — will retry" if attempt < MAX_RETRIES else f"Failed after {MAX_RETRIES} attempts",
            })

        # All retries exhausted
        return {
            "success": False,
            "error": f"Failed after {MAX_RETRIES} attempts. Last error: {error_context['error'] if error_context else 'unknown'}",
        }

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
        else:
            return {"error": f"Unknown action: {action}", "plan": plan}

    def _persist_new_artifacts(self, action: str, plan: dict):
        """Persist any tools/skills created during this action."""
        if action == "invent":
            name = plan.get("tool_name", "")
            if name and registry.has_tool(name):
                registry.save_tool(name)
                print(f"[forgebot] 💾 Persisted tool: {name}")
        elif action == "create_skill":
            # Persist supporting tools
            for tool_spec in plan.get("new_tools", []):
                tname = tool_spec.get("tool_name", "")
                if tname and registry.has_tool(tname):
                    registry.save_tool(tname)
                    print(f"[forgebot] 💾 Persisted supporting tool: {tname}")
            # Persist the skill
            sname = plan.get("skill_name", "")
            if sname and skill_registry.has_skill(sname):
                skill_registry.save_skill(sname)
                print(f"[forgebot] 💾 Persisted skill: {sname}")

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

        # Compile the invented tool
        try:
            fn = await self._compile_tool(tool_name, source_code)
        except Exception as e:
            await self._emit("error", {"message": f"Failed to compile tool: {e}"})
            return {"error": f"Compilation failed: {e}", "source_code": source_code}

        # Register it
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

        # Now execute it
        exec_args = plan.get("then_execute_with", {})
        return await self._execute_tool(tool_name, exec_args)

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
        """
        Create a new skill:
          1. Optionally invent supporting tools listed in plan["new_tools"].
          2. Compile the skill function.
          3. Register the skill.
          4. Execute it.
        """
        skill_name = plan["skill_name"]
        skill_source = plan["skill_source_code"]
        tools_created: list[str] = []

        # Step 1 — invent any supporting tools
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

        # Step 2 — compile the skill itself
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

        # Step 3 — register the skill
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

        # Step 4 — execute
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
        """
        Build the shared execution namespace that every compiled function can use:
          - stdlib helpers (asyncio, json)
          - all primitives
          - all registered tools (invented)
          - all registered skills
        """
        from agent import primitives

        namespace: dict = {
            "asyncio": asyncio,
            "json": json,
            # primitives
            "move_to": primitives.move_to,
            "set_gripper": primitives.set_gripper,
            "get_body_position": primitives.get_body_position,
            "get_body_color": primitives.get_body_color,
            "get_all_objects": primitives.get_all_objects,
            "pick_up": primitives.pick_up,
            "place_at": primitives.place_at,
            "step_sim": primitives.step_sim,
        }
        # inject invented tools
        for tool in registry.get_invented_tools():
            if tool.fn and tool.name != exclude_name:
                namespace[tool.name] = tool.fn
        # inject skills
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

    # Keys that are plan metadata, not function arguments
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

        # Strip plan metadata keys that the LLM may leak into execution args
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
