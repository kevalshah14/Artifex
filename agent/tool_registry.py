"""
ForgeBot Tool Registry — stores primitives and agent-invented tools.

The registry tracks:
- Built-in primitives (from primitives.py)
- Invented tools (composed by the agent at runtime)
- Metadata for the UI (description, source code, lineage)

Invented tools are persisted to agent/memory/tools/ so they survive restarts.
"""

import json
import os
import time
from typing import Callable, Optional
from dataclasses import dataclass, field

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory", "tools")


@dataclass
class Tool:
    name: str
    description: str
    signature: str
    source_code: str = ""
    is_primitive: bool = True
    composed_from: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    invocation_count: int = 0
    fn: Optional[Callable] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize for frontend / WebSocket broadcast."""
        return {
            "name": self.name,
            "description": self.description,
            "signature": self.signature,
            "source_code": self.source_code,
            "is_primitive": self.is_primitive,
            "composed_from": self.composed_from,
            "created_at": self.created_at,
            "invocation_count": self.invocation_count,
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._listeners: list[Callable] = []

    def register_primitive(self, name: str, fn: Callable, signature: str, description: str):
        """Register a built-in primitive."""
        self._tools[name] = Tool(
            name=name,
            description=description,
            signature=signature,
            source_code="[built-in primitive]",
            is_primitive=True,
            fn=fn,
        )

    def register_invented_tool(
        self,
        name: str,
        description: str,
        signature: str,
        source_code: str,
        composed_from: list[str],
        fn: Callable,
    ) -> Tool:
        """Register a tool invented by the agent."""
        tool = Tool(
            name=name,
            description=description,
            signature=signature,
            source_code=source_code,
            is_primitive=False,
            composed_from=composed_from,
            fn=fn,
        )
        self._tools[name] = tool
        for listener in self._listeners:
            listener("tool_invented", tool.to_dict())
        return tool

    # ── Persistence ──────────────────────────────

    def save_tool(self, name: str):
        """Persist an invented tool to disk as JSON."""
        tool = self._tools.get(name)
        if not tool or tool.is_primitive:
            return
        os.makedirs(MEMORY_DIR, exist_ok=True)
        path = os.path.join(MEMORY_DIR, f"{name}.json")
        data = {
            "name": tool.name,
            "description": tool.description,
            "signature": tool.signature,
            "source_code": tool.source_code,
            "composed_from": tool.composed_from,
            "created_at": tool.created_at,
            "invocation_count": tool.invocation_count,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_all_from_disk(self, compiler) -> list[str]:
        """
        Load all persisted tools from disk and compile them.

        Args:
            compiler: async callable(name, source_code) -> fn

        Returns:
            list of tool names that were loaded.
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
                if name in self._tools:
                    continue
                fn = compiler(name, data["source_code"])
                tool = Tool(
                    name=name,
                    description=data.get("description", ""),
                    signature=data.get("signature", f"{name}()"),
                    source_code=data["source_code"],
                    is_primitive=False,
                    composed_from=data.get("composed_from", []),
                    created_at=data.get("created_at", time.time()),
                    invocation_count=data.get("invocation_count", 0),
                    fn=fn,
                )
                self._tools[name] = tool
                loaded.append(name)
            except Exception as e:
                print(f"[tool_registry] Failed to load {fname}: {e}")
        return loaded

    # ── Query helpers ────────────────────────────

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_primitives(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.is_primitive]

    def get_invented_tools(self) -> list[Tool]:
        return [t for t in self._tools.values() if not t.is_primitive]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool_descriptions(self) -> str:
        """Format all tools as a string for the LLM prompt."""
        lines = []
        lines.append("=== AVAILABLE TOOLS ===\n")

        lines.append("## Primitives (built-in):")
        for t in self.get_primitives():
            lines.append(f"  - {t.signature}")
            lines.append(f"    {t.description}")

        invented = self.get_invented_tools()
        if invented:
            lines.append("\n## Invented Tools (previously created):")
            for t in invented:
                lines.append(f"  - {t.signature}")
                lines.append(f"    {t.description}")
                lines.append(f"    Composed from: {', '.join(t.composed_from)}")

        return "\n".join(lines)

    def on_change(self, callback: Callable):
        """Register a listener for registry changes."""
        self._listeners.append(callback)

    def snapshot(self) -> list[dict]:
        """Get full registry state for frontend sync."""
        return [t.to_dict() for t in self._tools.values()]


# Singleton
registry = ToolRegistry()
