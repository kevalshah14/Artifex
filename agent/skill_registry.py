"""
ForgeBot Skill Registry — stores high-level reusable skills.

Skills sit above tools in the hierarchy:
  Primitives  →  Tools (atomic invented functions)  →  Skills (multi-step strategies)

Unlike tools (single atomic async functions), a skill is a complete behavioural
strategy that can orchestrate multiple tools and primitives, and may itself have
triggered the invention of new tools at creation time.  Skills are remembered and
reused across tasks exactly like tools are.

Skills are persisted to agent/memory/skills/ so they survive restarts.
"""

import json
import os
import time
from typing import Callable, Optional
from dataclasses import dataclass, field

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory", "skills")


@dataclass
class Skill:
    name: str
    description: str
    source_code: str
    tools_used: list[str] = field(default_factory=list)
    tools_created: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    invocation_count: int = 0
    fn: Optional[Callable] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize for frontend / WebSocket broadcast."""
        return {
            "name": self.name,
            "description": self.description,
            "source_code": self.source_code,
            "tools_used": self.tools_used,
            "tools_created": self.tools_created,
            "created_at": self.created_at,
            "invocation_count": self.invocation_count,
        }


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._listeners: list[Callable] = []

    def register_skill(
        self,
        name: str,
        description: str,
        source_code: str,
        tools_used: list[str],
        tools_created: list[str],
        fn: Callable,
    ) -> Skill:
        """Register a skill invented by the agent."""
        skill = Skill(
            name=name,
            description=description,
            source_code=source_code,
            tools_used=tools_used,
            tools_created=tools_created,
            fn=fn,
        )
        self._skills[name] = skill
        for listener in self._listeners:
            listener("skill_created", skill.to_dict())
        return skill

    # ── Persistence ──────────────────────────────

    def save_skill(self, name: str):
        """Persist a skill to disk as JSON."""
        skill = self._skills.get(name)
        if not skill:
            return
        os.makedirs(MEMORY_DIR, exist_ok=True)
        path = os.path.join(MEMORY_DIR, f"{name}.json")
        data = {
            "name": skill.name,
            "description": skill.description,
            "source_code": skill.source_code,
            "tools_used": skill.tools_used,
            "tools_created": skill.tools_created,
            "created_at": skill.created_at,
            "invocation_count": skill.invocation_count,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_all_from_disk(self, compiler) -> list[str]:
        """
        Load all persisted skills from disk and compile them.

        Args:
            compiler: callable(name, source_code) -> fn

        Returns:
            list of skill names that were loaded.
        """
        loaded: list[str] = []
        if not os.path.isdir(MEMORY_DIR):
            return loaded
        for fname in sorted(os.listdir(MEMORY_DIR)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(MEMORY_DIR, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                name = data["name"]
                if name in self._skills:
                    continue
                fn = compiler(name, data["source_code"])
                skill = Skill(
                    name=name,
                    description=data.get("description", ""),
                    source_code=data["source_code"],
                    tools_used=data.get("tools_used", []),
                    tools_created=data.get("tools_created", []),
                    created_at=data.get("created_at", time.time()),
                    invocation_count=data.get("invocation_count", 0),
                    fn=fn,
                )
                self._skills[name] = skill
                loaded.append(name)
            except Exception as e:
                print(f"[skill_registry] Failed to load {fname}: {e}")
        return loaded

    # ── Query helpers ────────────────────────────

    def get_skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def get_all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def has_skill(self, name: str) -> bool:
        return name in self._skills

    def get_skill_descriptions(self) -> str:
        """Format all skills as a string for the LLM prompt."""
        if not self._skills:
            return ""
        lines = ["\n## Skills (high-level reusable strategies):"]
        for s in self._skills.values():
            lines.append(f"  - {s.name}()")
            lines.append(f"    {s.description}")
            if s.tools_used:
                lines.append(f"    Orchestrates: {', '.join(s.tools_used)}")
        return "\n".join(lines)

    def on_change(self, callback: Callable):
        """Register a listener for registry changes (e.g. WebSocket broadcast)."""
        self._listeners.append(callback)

    def snapshot(self) -> list[dict]:
        """Get full registry state for frontend sync."""
        return [s.to_dict() for s in self._skills.values()]


# Singleton
skill_registry = SkillRegistry()
