"""
Mock Sim Client — pretends to be the MuJoCo WASM frontend.

Run this alongside the server to test the agent loop without real sim.
Usage: python mock_sim.py
"""

import json
import asyncio
import random
import websockets


# Fake scene state
SCENE = {
    "objects": [
        {"name": "red_block_1", "position": [0.2, 0.1, 0.02], "color": "red", "size": 0.04},
        {"name": "red_block_2", "position": [0.3, -0.1, 0.02], "color": "red", "size": 0.03},
        {"name": "blue_block_1", "position": [-0.1, 0.2, 0.02], "color": "blue", "size": 0.05},
        {"name": "blue_block_2", "position": [0.0, -0.2, 0.02], "color": "blue", "size": 0.035},
        {"name": "green_block_1", "position": [0.15, 0.0, 0.02], "color": "green", "size": 0.045},
    ],
    "end_effector": [0.0, 0.0, 0.3],
    "gripper_open": True,
    "holding": None,
    "sim_time": 0.0,
}


def handle_command(cmd: dict) -> dict:
    """Process a command and return a result."""
    action = cmd.get("action")

    if action == "get_all_objects":
        return {"success": True, "objects": SCENE["objects"]}

    elif action == "get_body_position":
        name = cmd.get("body_name")
        if name == "end_effector":
            return {"success": True, "position": SCENE["end_effector"]}
        for obj in SCENE["objects"]:
            if obj["name"] == name:
                return {"success": True, "position": obj["position"]}
        return {"success": False, "error": f"Body '{name}' not found"}

    elif action == "get_body_color":
        name = cmd.get("body_name")
        for obj in SCENE["objects"]:
            if obj["name"] == name:
                return {"success": True, "color": obj["color"]}
        return {"success": False, "error": f"Body '{name}' not found"}

    elif action == "move_to":
        target = cmd.get("target", [0, 0, 0])
        SCENE["end_effector"] = target
        print(f"  🤖 Moving to {target}")
        return {"success": True, "position": target}

    elif action == "set_gripper":
        SCENE["gripper_open"] = cmd.get("open", True)
        state = "open" if SCENE["gripper_open"] else "closed"
        print(f"  🤖 Gripper {state}")
        return {"success": True, "gripper_state": state}

    elif action == "pick_up":
        name = cmd.get("body_name")
        for obj in SCENE["objects"]:
            if obj["name"] == name:
                SCENE["holding"] = name
                SCENE["end_effector"] = obj["position"]
                print(f"  🤖 Picked up {name}")
                return {"success": True, "holding": name}
        return {"success": False, "error": f"Object '{name}' not found"}

    elif action == "grasp":
        name = cmd.get("body_name")
        for obj in SCENE["objects"]:
            if obj["name"] == name:
                original_z = obj["position"][2]
                SCENE["holding"] = name
                # Simulate the object being lifted
                new_pos = [obj["position"][0], obj["position"][1], obj["position"][2] + 0.2]
                obj["position"] = new_pos
                SCENE["end_effector"] = new_pos
                print(f"  🤖 Grasped {name}")
                return {
                    "success": True,
                    "holding": name,
                    "original_z": original_z,
                    "new_z": new_pos[2],
                    "message": f"Successfully grasped {name}"
                }
        return {"success": False, "error": f"Object '{name}' not found"}

    elif action == "place_at":
        target = cmd.get("target", [0, 0, 0])
        if SCENE["holding"]:
            held = SCENE["holding"]
            for obj in SCENE["objects"]:
                if obj["name"] == held:
                    obj["position"] = target
            print(f"  🤖 Placed {held} at {target}")
            SCENE["holding"] = None
            SCENE["gripper_open"] = True
            return {"success": True, "placed_at": target}
        return {"success": False, "error": "Not holding anything"}

    elif action == "step":
        n = cmd.get("n_steps", 100)
        SCENE["sim_time"] += n * 0.002
        return {"success": True, "sim_time": SCENE["sim_time"]}

    return {"success": False, "error": f"Unknown action: {action}"}


async def main():
    uri = "ws://localhost:8000/ws/sim"
    print("🔌 Mock sim connecting to ForgeBot server...")
    
    async with websockets.connect(uri) as ws:
        print("✅ Connected! Listening for commands...\n")

        # Send initial state
        await ws.send(json.dumps({
            "type": "state_update",
            "state": {
                "objects": SCENE["objects"],
                "end_effector": SCENE["end_effector"],
            }
        }))

        while True:
            try:
                raw = await ws.recv()
                cmd = json.loads(raw)
                print(f"📥 Command: {cmd.get('action', 'unknown')}")
                
                # Simulate some latency
                await asyncio.sleep(0.3)
                
                result = handle_command(cmd)
                await ws.send(json.dumps(result))
                print(f"📤 Result: {json.dumps(result, indent=2)[:200]}\n")

            except websockets.ConnectionClosed:
                print("❌ Connection closed")
                break


if __name__ == "__main__":
    asyncio.run(main())
