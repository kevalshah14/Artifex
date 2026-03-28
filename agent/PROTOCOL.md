# ForgeBot WebSocket Protocol

## Overview

Two WebSocket connections:
- **`/ws/chat`** — Chat UI ↔ Server (user messages + agent events)
- **`/ws/sim`** — MuJoCo WASM Frontend ↔ Server (commands + state)

---

## `/ws/sim` — Sim Frontend Protocol

Your coworker implements this side.

### Server → Sim (Commands)

The server sends JSON commands. The sim must execute them and respond.

```json
// Move end-effector
{"action": "move_to", "body_name": "end_effector", "target": [0.2, 0.1, 0.15]}

// Gripper control
{"action": "set_gripper", "open": true}

// Query position
{"action": "get_body_position", "body_name": "red_block_1"}

// Query color
{"action": "get_body_color", "body_name": "block_1"}

// Get all objects in scene
{"action": "get_all_objects"}

// Pick up object (move to it + grasp + lift)
{"action": "pick_up", "body_name": "red_block_1"}

// Place held object at position
{"action": "place_at", "target": [0.3, 0.0, 0.02]}

// Step simulation forward
{"action": "step", "n_steps": 100}
```

### Sim → Server (Responses)

After each command, respond with a result:

```json
// Success
{"success": true, "position": [0.2, 0.1, 0.15]}

// Object list
{"success": true, "objects": [
  {"name": "red_block_1", "position": [0.2, 0.1, 0.02], "color": "red", "size": 0.04},
  {"name": "blue_block_1", "position": [-0.1, 0.2, 0.02], "color": "blue", "size": 0.05}
]}

// Error
{"success": false, "error": "Body 'xyz' not found"}
```

### Sim → Server (State Updates)

Periodically send sim state (e.g., every 500ms):

```json
{
  "type": "state_update",
  "state": {
    "objects": [...],
    "end_effector": [x, y, z],
    "sim_time": 1.234
  }
}
```

---

## `/ws/chat` — Chat UI Protocol

### Client → Server

```json
// Send a task
{"type": "task", "message": "Sort the blocks by color"}

// Request current tool list
{"type": "get_tools"}
```

### Server → Client (Events stream)

Events are streamed in order as the agent works:

```json
// Initial state on connect
{"type": "init", "tools": [...], "sim_connected": true}

// Agent is analyzing
{"type": "thinking", "message": "Analyzing task: sort blocks by color"}

// Agent is calling LLM
{"type": "llm_call", "message": "Planning approach..."}

// Agent has a plan
{"type": "plan", "thought": "I need to...", "action": "invent", "tool_name": "sort_by_color"}

// Agent is inventing a tool (BIG UI MOMENT)
{"type": "inventing", "tool_name": "sort_by_color", "description": "...", "source_code": "async def...", "composed_from": ["get_all_objects", "pick_up", "place_at"]}

// Tool was registered
{"type": "tool_registered", "tool": {"name": "sort_by_color", ...}, "message": "Invented new tool"}

// Executing a tool
{"type": "executing", "tool_name": "sort_by_color", "args": {}}

// Execution step
{"type": "step", "index": 0, "description": "Getting all objects..."}

// Result
{"type": "result", "tool_name": "sort_by_color", "result": {...}}

// Task complete
{"type": "task_complete", "result": {"success": true, ...}}

// Error
{"type": "error", "message": "Failed to compile tool"}

// Sim connection status
{"type": "sim_status", "connected": true}
```

---

## REST Endpoints

```
GET /           → Server status + tool counts
GET /tools      → All tools (primitives + invented)
GET /tools/:name → Single tool details
```

---

## Quick Start

```bash
# Terminal 1: Start server
cd forgebot
cp .env.example .env  # Add your OPENAI_API_KEY
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start mock sim (for testing without MuJoCo)
python mock_sim.py

# Terminal 3: Test with a task
python test_chat.py "Sort the blocks by color into groups"
```
