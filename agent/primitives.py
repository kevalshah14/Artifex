"""
ForgeBot Primitives — base robot actions the agent can compose into tools.

These primitives send commands over WebSocket to the MuJoCo WASM frontend.
Each returns a result dict that the agent can use for planning.
"""

import json
import asyncio
from typing import Optional

# Global reference to the WebSocket connection to the sim frontend
_sim_ws = None
_sim_state = {}
_pending_command: Optional[asyncio.Future] = None


def set_sim_connection(ws):
    """Set the active WebSocket connection to the sim."""
    global _sim_ws
    _sim_ws = ws


def update_sim_state(state: dict):
    """Update cached sim state from frontend heartbeat."""
    global _sim_state
    _sim_state.update(state)


def get_sim_state() -> dict:
    """Get the latest cached sim state."""
    return _sim_state.copy()


def resolve_command_result(result: dict):
    """Called by the server when sim sends back a command_result."""
    global _pending_command
    print(f"[primitives] resolve_command_result called, pending={_pending_command is not None}, result_keys={list(result.keys()) if result else 'None'}")
    if _pending_command and not _pending_command.done():
        _pending_command.set_result(result)
    else:
        print(f"[primitives] WARNING: no pending command to resolve (pending={_pending_command}, done={_pending_command.done() if _pending_command else 'N/A'})")


async def send_command(cmd: dict, timeout: float = 30.0) -> dict:
    """Send a command to the sim frontend and wait for result via Future."""
    global _pending_command
    if _sim_ws is None:
        return {"error": "No sim connection", "success": False}
    try:
        loop = asyncio.get_running_loop()
        _pending_command = loop.create_future()
        print(f"[primitives] Sending command: {cmd.get('action', 'unknown')}")
        await _sim_ws.send_text(json.dumps(cmd))
        print(f"[primitives] Awaiting response for: {cmd.get('action', 'unknown')}")
        result = await asyncio.wait_for(_pending_command, timeout=timeout)
        print(f"[primitives] Got response for: {cmd.get('action', 'unknown')}")
        return result
    except asyncio.TimeoutError:
        print(f"[primitives] TIMEOUT for: {cmd.get('action', 'unknown')}")
        return {"error": "Sim command timed out", "success": False}
    except Exception as e:
        print(f"[primitives] ERROR for {cmd.get('action', 'unknown')}: {e}")
        return {"error": str(e), "success": False}
    finally:
        _pending_command = None


# ─────────────────────────────────────────────
# PRIMITIVES — these are what the agent composes
# ─────────────────────────────────────────────

async def move_to(body_name: str, x: float, y: float, z: float) -> dict:
    """Move the robot end-effector to a target position (x, y, z).
    
    Args:
        body_name: Name of the target body or 'end_effector' for the robot arm tip.
        x: Target x coordinate in world frame.
        y: Target y coordinate in world frame.  
        z: Target z coordinate in world frame.
    
    Returns:
        dict with 'success' bool and 'position' after move.
    """
    return await send_command({
        "action": "move_to",
        "body_name": body_name,
        "target": [x, y, z]
    })


async def set_gripper(open: bool) -> dict:
    """Open or close the robot gripper.
    
    Args:
        open: True to open gripper, False to close/grasp.
    
    Returns:
        dict with 'success' bool and 'gripper_state'.
    """
    return await send_command({
        "action": "set_gripper",
        "open": open
    })


async def get_body_position(body_name: str) -> dict:
    """Get the current 3D position of a named body in the sim.
    
    Args:
        body_name: Name of the body (e.g., 'red_block_1', 'end_effector').
    
    Returns:
        dict with 'position' [x, y, z] and 'success' bool.
    """
    return await send_command({
        "action": "get_body_position",
        "body_name": body_name
    })


async def get_body_color(body_name: str) -> dict:
    """Get the color of a named body in the sim.
    
    Args:
        body_name: Name of the body (e.g., 'block_1').
    
    Returns:
        dict with 'color' string (e.g., 'red', 'blue', 'green') and 'success' bool.
    """
    return await send_command({
        "action": "get_body_color",
        "body_name": body_name
    })


async def get_all_objects() -> dict:
    """Get a list of all manipulable objects in the sim scene.
    
    Returns:
        dict with 'objects' list of {name, position, color, size} dicts.
    """
    return await send_command({
        "action": "get_all_objects"
    })


async def pick_up(body_name: str) -> dict:
    """Pick up an object: move to it, close gripper, lift.
    
    Args:
        body_name: Name of the object to pick up.
    
    Returns:
        dict with 'success' bool and 'holding' body name.
    """
    return await send_command({
        "action": "pick_up",
        "body_name": body_name
    })


async def grasp(body_name: str) -> dict:
    """Grasp an object and hold it (no tray placement).
    Uses the full animated sequence for proper IK + gripper control,
    but stops after lifting instead of moving to a tray.
    
    Args:
        body_name: Name of the object to grasp.
    
    Returns:
        dict with 'success' bool, 'holding' body name, and verification data.
    """
    return await send_command({
        "action": "grasp",
        "body_name": body_name
    })


async def place_at(x: float, y: float, z: float) -> dict:
    """Place the currently held object at the target position.
    
    Args:
        x: Target x coordinate.
        y: Target y coordinate.
        z: Target z coordinate.
    
    Returns:
        dict with 'success' bool and 'placed_at' position.
    """
    return await send_command({
        "action": "place_at",
        "target": [x, y, z]
    })


async def step_sim(n_steps: int = 100) -> dict:
    """Advance the simulation by n physics steps.
    
    Args:
        n_steps: Number of physics steps to advance.
    
    Returns:
        dict with 'success' bool and 'sim_time' after stepping.
    """
    return await send_command({
        "action": "step",
        "n_steps": n_steps
    })


async def capture_scene_image() -> dict:
    """Capture a JPEG screenshot of the current sim scene.

    Returns:
        dict with 'image_base64' (base64-encoded JPEG), 'mime_type', and 'success'.
    """
    return await send_command({"action": "capture_image"})


# Registry of all primitives for the agent to reference
PRIMITIVES = {
    "move_to": {
        "fn": move_to,
        "signature": "move_to(body_name: str, x: float, y: float, z: float) -> dict",
        "description": "Move the robot end-effector to target position (x, y, z)."
    },
    "set_gripper": {
        "fn": set_gripper,
        "signature": "set_gripper(open: bool) -> dict",
        "description": "Open or close the robot gripper."
    },
    "get_body_position": {
        "fn": get_body_position,
        "signature": "get_body_position(body_name: str) -> dict",
        "description": "Get current 3D position of a named body."
    },
    "get_body_color": {
        "fn": get_body_color,
        "signature": "get_body_color(body_name: str) -> dict",
        "description": "Get the color of a named body."
    },
    "get_all_objects": {
        "fn": get_all_objects,
        "signature": "get_all_objects() -> dict",
        "description": "Get list of all manipulable objects with their positions, colors, sizes."
    },
    "pick_up": {
        "fn": pick_up,
        "signature": "pick_up(body_name: str) -> dict",
        "description": "Pick up an object by name (moves to it, grasps, lifts)."
    },
    "grasp": {
        "fn": grasp,
        "signature": "grasp(body_name: str) -> dict",
        "description": "Grasp an object and hold it — uses full animated sequence with proper IK + gripper control, but stops after lifting (no tray placement). Verifies the object actually lifted."
    },
    "place_at": {
        "fn": place_at,
        "signature": "place_at(x: float, y: float, z: float) -> dict",
        "description": "Place the currently held object at the target position."
    },
    "step_sim": {
        "fn": step_sim,
        "signature": "step_sim(n_steps: int = 100) -> dict",
        "description": "Advance simulation by n physics steps."
    },
}
