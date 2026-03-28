"""
ForgeBot Server — FastAPI + WebSocket bridge between chat UI and MuJoCo sim.

Two WebSocket endpoints:
  /ws/chat    — Chat UI sends user messages, receives agent events
  /ws/sim     — MuJoCo WASM frontend sends sim state, receives commands

Plus REST endpoints for tool registry state.
"""

import sys
import os

# Ensure the parent directory (Artifex/) is on sys.path so `agent` resolves as a package
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import json
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent.forgebot import agent
from agent.tool_registry import registry
from agent.skill_registry import skill_registry
from agent.primitives import set_sim_connection, update_sim_state, resolve_command_result


# ──────────────────────────────────────────
# Connection Manager
# ──────────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections for chat clients and sim frontend."""

    def __init__(self):
        self.chat_clients: list[WebSocket] = []
        self.sim_ws: Optional[WebSocket] = None

    async def connect_chat(self, ws: WebSocket):
        await ws.accept()
        self.chat_clients.append(ws)

    async def connect_sim(self, ws: WebSocket):
        await ws.accept()
        self.sim_ws = ws
        set_sim_connection(ws)

    def disconnect_chat(self, ws: WebSocket):
        self.chat_clients.remove(ws)

    def disconnect_sim(self):
        self.sim_ws = None
        set_sim_connection(None)

    async def broadcast_to_chat(self, event_type: str, data: dict):
        """Send an event to all connected chat clients."""
        message = json.dumps({"type": event_type, **data})
        disconnected = []
        for client in self.chat_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.append(client)
        for client in disconnected:
            self.chat_clients.remove(client)

    async def send_to_sim(self, command: dict):
        """Send a command to the sim frontend."""
        if self.sim_ws:
            await self.sim_ws.send_text(json.dumps(command))


manager = ConnectionManager()


# ──────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wire up agent events to broadcast to chat clients
    async def on_agent_event(event_type: str, data: dict):
        await manager.broadcast_to_chat(event_type, data)

    agent.on_event(on_agent_event)

    # Wire up tool registry changes to broadcast
    def on_registry_change(event_type: str, data: dict):
        asyncio.create_task(
            manager.broadcast_to_chat(event_type, data)
        )

    registry.on_change(on_registry_change)

    # Wire up skill registry changes to broadcast
    def on_skill_change(event_type: str, data: dict):
        asyncio.create_task(
            manager.broadcast_to_chat(event_type, data)
        )

    skill_registry.on_change(on_skill_change)

    yield


app = FastAPI(title="ForgeBot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────
# WebSocket: Chat UI
# ──────────────────────────────────────────

@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await manager.connect_chat(ws)
    
    # Send initial state (tools + skills)
    await ws.send_text(json.dumps({
        "type": "init",
        "tools": registry.snapshot(),
        "skills": skill_registry.snapshot(),
        "sim_connected": manager.sim_ws is not None,
    }))
    
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            if data.get("type") == "task":
                user_message = data.get("message", "")
                # Broadcast that we received the task
                await manager.broadcast_to_chat("task_received", {
                    "message": user_message
                })
                # Run the agent
                result = await agent.handle_task(user_message)
                # Send final result
                await manager.broadcast_to_chat("task_complete", {
                    "result": result
                })

            elif data.get("type") == "get_tools":
                await ws.send_text(json.dumps({
                    "type": "tools",
                    "tools": registry.snapshot(),
                }))

    except WebSocketDisconnect:
        manager.disconnect_chat(ws)


# ──────────────────────────────────────────
# WebSocket: Sim Frontend
# ──────────────────────────────────────────

@app.websocket("/ws/sim")
async def sim_ws(ws: WebSocket):
    await manager.connect_sim(ws)
    
    # Notify chat clients that sim connected
    await manager.broadcast_to_chat("sim_status", {"connected": True})
    
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            if data.get("type") == "state_update":
                # Frontend sends periodic sim state updates
                update_sim_state(data.get("state", {}))

            elif data.get("type") == "command_result":
                resolve_command_result(data.get("result", {}))

    except WebSocketDisconnect:
        manager.disconnect_sim()
        await manager.broadcast_to_chat("sim_status", {"connected": False})


# ──────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "ForgeBot",
        "description": "A robot that invents its own tools and skills",
        "sim_connected": manager.sim_ws is not None,
        "tools_count": len(registry.get_all_tools()),
        "invented_tools_count": len(registry.get_invented_tools()),
        "skills_count": len(skill_registry.get_all_skills()),
    }


@app.get("/tools")
async def get_tools():
    return {
        "tools": registry.snapshot(),
        "primitives": len(registry.get_primitives()),
        "invented": len(registry.get_invented_tools()),
    }


@app.get("/tools/{tool_name}")
async def get_tool(tool_name: str):
    tool = registry.get_tool(tool_name)
    if not tool:
        return {"error": f"Tool '{tool_name}' not found"}
    return tool.to_dict()


@app.get("/skills")
async def get_skills():
    return {
        "skills": skill_registry.snapshot(),
        "count": len(skill_registry.get_all_skills()),
    }


@app.get("/skills/{skill_name}")
async def get_skill(skill_name: str):
    skill = skill_registry.get_skill(skill_name)
    if not skill:
        return {"error": f"Skill '{skill_name}' not found"}
    return skill.to_dict()


# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
