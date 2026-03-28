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
        action = cmd.get('action', 'unknown')
        print(f"[primitives] Sending command: {action}")
        print(f"[primitives]   full cmd: {json.dumps(cmd, default=str)[:500]}")
        await _sim_ws.send_text(json.dumps(cmd))
        print(f"[primitives] Awaiting response for: {action}")
        result = await asyncio.wait_for(_pending_command, timeout=timeout)
        print(f"[primitives] Got response for: {action}")
        print(f"[primitives]   result keys: {list(result.keys()) if result else 'None'}")
        # Log key values for grasp/place_at
        if action == 'grasp':
            print(f"[primitives]   grasp result: success={result.get('success')}, original_z={result.get('original_z')}, new_z={result.get('new_z')}, holding={result.get('holding')}")
        elif action == 'place_at':
            print(f"[primitives]   place_at result: success={result.get('success')}, placed_at={result.get('placed_at')}")
        elif action == 'get_all_objects':
            objects = result.get('objects', [])
            landmarks = result.get('landmarks', [])
            print(f"[primitives]   objects count: {len(objects)}, landmarks count: {len(landmarks)}")
            for obj in objects:
                pos = obj.get('position', [])
                print(f"[primitives]     obj {obj.get('name')}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), color={obj.get('color')}")
            for lm in landmarks:
                pos = lm.get('position', [])
                print(f"[primitives]     lmk {lm.get('name')}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
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

async def move_to(body_name: str, x: float, y: float, z: float, duration: int = 1500) -> dict:
    """Move the robot end-effector (TCP) to a target position (x, y, z).
    
    Args:
        body_name: Name of the target body or 'end_effector' for the robot arm tip.
        x: Target x coordinate in world frame.
        y: Target y coordinate in world frame.  
        z: Target z coordinate in world frame.
        duration: Motion duration in ms (default 1500). Use 300-500 for fast strikes.
    
    Returns:
        dict with 'success', 'requested' [x,y,z], and 'actual_tcp' [x,y,z].
    """
    return await send_command({
        "action": "move_to",
        "body_name": body_name,
        "target": [x, y, z],
        "duration": duration,
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


async def add_object(name: str, x: float, y: float, z: float,
                     shape: str = "box", color: str = "red",
                     size: float = 0.02) -> dict:
    """Add a new object to the scene.

    Args:
        name: Unique name (e.g. 'sphere_1', 'my_box').
        x, y, z: Position in world frame.
        shape: One of 'box', 'sphere', 'cylinder', 'capsule', 'ellipsoid'.
        color: Color name (red, green, blue, yellow, cyan, purple, orange, white, black, pink).
        size: Characteristic half-extent / radius (default 0.02).

    Returns:
        dict with 'success', 'added', 'shape', 'position', 'color'.
    """
    return await send_command({
        "action": "add_object",
        "name": name,
        "position": [x, y, z],
        "shape": shape,
        "color": color,
        "size": size,
    })


async def add_custom_object(name: str, x: float, y: float, z: float, body_xml: str) -> dict:
    """Add a custom-shaped object by providing raw MuJoCo MJCF XML for the geom(s).

    The body_xml should contain one or more <geom .../> elements that together
    form the desired shape.  A <freejoint/> is added automatically.

    Args:
        name: Unique body name.
        x, y, z: Position in world frame.
        body_xml: Inner MJCF XML string (geom elements only, no <body> wrapper).

    Returns:
        dict with 'success' and 'added' name.
    """
    return await send_command({
        "action": "add_custom_object",
        "name": name,
        "position": [x, y, z],
        "body_xml": body_xml,
    })


async def remove_body(body_name: str) -> dict:
    """Remove a body from the scene entirely.

    Args:
        body_name: Name of the body to remove.

    Returns:
        dict with 'success' and 'removed' name.
    """
    return await send_command({
        "action": "remove_body",
        "body_name": body_name,
    })


async def set_body_color(body_name: str, color: str) -> dict:
    """Change the color of an existing body.

    Args:
        body_name: Name of the body.
        color: Color name (red, green, blue, yellow, cyan, purple, orange, white, black, pink).

    Returns:
        dict with 'success', 'body_name', 'new_color'.
    """
    return await send_command({
        "action": "set_body_color",
        "body_name": body_name,
        "color": color,
    })


async def move_body(body_name: str, x: float, y: float, z: float) -> dict:
    """Teleport a body to a new position (instant, no physics).

    Args:
        body_name: Name of the body to move.
        x, y, z: New position in world frame.

    Returns:
        dict with 'success', 'body_name', 'new_position'.
    """
    return await send_command({
        "action": "move_body",
        "body_name": body_name,
        "position": [x, y, z],
    })


async def clear_objects(name_prefix: str = "") -> dict:
    """Remove all manipulable objects from the scene in one reload.

    Args:
        name_prefix: If provided, only remove objects whose name starts with
                     this prefix (e.g. 'cube' removes cube0-cubeN). If empty,
                     removes ALL free-joint objects.

    Returns:
        dict with 'success', 'removed' list, and 'count'.
    """
    cmd: dict = {"action": "clear_objects"}
    if name_prefix:
        cmd["name_prefix"] = name_prefix
    return await send_command(cmd)


async def reset_scene() -> dict:
    """Reset the simulation to its initial state (re-randomizes cubes).

    Returns:
        dict with 'success' and 'message'.
    """
    return await send_command({"action": "reset_scene"})


async def attach_gripper(gripper_xml: str, tcp_offset: str = "0 0 0.1") -> dict:
    """Attach a custom gripper/tool to the robot's hand body.

    The gripper_xml is injected as a child of the Panda's 'hand' body,
    alongside the existing default fingers.  Use MuJoCo MJCF body/geom
    elements.  A <freejoint/> is NOT added (the tool is rigidly attached).

    Args:
        gripper_xml: MJCF XML for the gripper (body/geom elements).
        tcp_offset: New TCP site position relative to the hand body
                    as "x y z" string (default "0 0 0.1").  Set further
                    out if the tool extends beyond the default fingers.

    Returns:
        dict with 'success' and 'message'.
    """
    return await send_command({
        "action": "attach_gripper",
        "gripper_xml": gripper_xml,
        "tcp_offset": tcp_offset,
    })


async def detach_gripper() -> dict:
    """Remove any previously attached custom gripper from the robot.

    Returns:
        dict with 'success' and 'message'.
    """
    return await send_command({"action": "detach_gripper"})


# Registry of all primitives for the agent to reference
PRIMITIVES = {
    "move_to": {
        "fn": move_to,
        "signature": "move_to(body_name: str, x: float, y: float, z: float, duration: int = 1500) -> dict",
        "description": "Move the robot TCP (tool tip) to target position. duration in ms (default 1500; use 300-500 for fast strikes). Returns requested + actual_tcp positions."
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
        "description": "Get all scene bodies: 'objects' (movable, with freejoint) + 'landmarks' (fixed bodies like goal_post). Check both!"
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
    "add_object": {
        "fn": add_object,
        "signature": "add_object(name: str, x: float, y: float, z: float, shape: str = 'box', color: str = 'red', size: float = 0.02) -> dict",
        "description": "Add a new object to the scene. Shapes: box, sphere, cylinder, capsule, ellipsoid. Colors: red, green, blue, yellow, cyan, purple, orange, white, black, pink."
    },
    "add_custom_object": {
        "fn": add_custom_object,
        "signature": "add_custom_object(name: str, x: float, y: float, z: float, body_xml: str) -> dict",
        "description": "Add a custom-shaped object by composing raw MuJoCo MJCF geom elements. Use for complex objects like plates, bowls, tables, L-shapes, etc."
    },
    "remove_body": {
        "fn": remove_body,
        "signature": "remove_body(body_name: str) -> dict",
        "description": "Remove a body from the scene entirely (modifies XML and reloads)."
    },
    "set_body_color": {
        "fn": set_body_color,
        "signature": "set_body_color(body_name: str, color: str) -> dict",
        "description": "Change the color of an existing body. Colors: red, green, blue, yellow, cyan, purple, orange, white, black, pink."
    },
    "move_body": {
        "fn": move_body,
        "signature": "move_body(body_name: str, x: float, y: float, z: float) -> dict",
        "description": "Teleport a body instantly to a new position (no physics, just sets qpos)."
    },
    "clear_objects": {
        "fn": clear_objects,
        "signature": "clear_objects(name_prefix: str = '') -> dict",
        "description": "Remove all manipulable objects in one reload. Pass name_prefix (e.g. 'cube') to only remove matching objects, or empty string for all."
    },
    "reset_scene": {
        "fn": reset_scene,
        "signature": "reset_scene() -> dict",
        "description": "Reset the simulation to its initial state with randomized cube positions."
    },
    "attach_gripper": {
        "fn": attach_gripper,
        "signature": "attach_gripper(gripper_xml: str, tcp_offset: str = '0 0 0.1') -> dict",
        "description": "Attach a custom gripper/tool to the robot hand. Provide MJCF body/geom XML. Set tcp_offset to adjust IK target point."
    },
    "detach_gripper": {
        "fn": detach_gripper,
        "signature": "detach_gripper() -> dict",
        "description": "Remove any previously attached custom gripper from the robot."
    },
}
