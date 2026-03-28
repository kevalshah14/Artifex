"""
Test Chat Client — sends a task and prints agent events.

Usage: python test_chat.py "sort the blocks by color"
"""

import json
import sys
import asyncio
import websockets


async def main(task: str):
    uri = "ws://localhost:8000/ws/chat"
    print(f"🔌 Connecting to ForgeBot...")

    async with websockets.connect(uri) as ws:
        # Receive init message
        raw = await ws.recv()
        init = json.loads(raw)
        tools = init.get("tools", [])
        skills = init.get("skills", [])
        primitives = [t for t in tools if t.get("is_primitive")]
        invented = [t for t in tools if not t.get("is_primitive")]
        print(f"✅ Connected! {len(primitives)} primitives, {len(invented)} invented tools, {len(skills)} skills")
        print(f"   Sim connected: {init.get('sim_connected', False)}")
        if invented:
            print(f"   Invented tools: {', '.join(t['name'] for t in invented)}")
        if skills:
            print(f"   Skills: {', '.join(s['name'] for s in skills)}")
        print()

        # Send the task
        print(f"📤 Sending task: {task}\n")
        await ws.send(json.dumps({
            "type": "task",
            "message": task,
        }))

        # Listen for events
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
                event = json.loads(raw)
                event_type = event.get("type", "unknown")

                if event_type == "thinking":
                    print(f"🧠 {event.get('message', '')}")

                elif event_type == "llm_call":
                    print(f"🤖 {event.get('message', '')}")

                elif event_type == "plan":
                    attempt = event.get("attempt", 1)
                    prefix = f" (attempt {attempt})" if attempt > 1 else ""
                    print(f"\n📋 Plan{prefix}:")
                    print(f"   Thought: {event.get('thought', '')}")
                    print(f"   Action: {event.get('action', '')}")
                    print(f"   Tool: {event.get('tool_name', '')}")

                elif event_type == "evolving":
                    print(f"\n🔄 EVOLVING — {event.get('message', '')}")
                    print(f"   Previous error: {event.get('error', '')[:200]}")

                elif event_type == "retry":
                    attempt = event.get("attempt", 0)
                    max_r = event.get("max_retries", 3)
                    print(f"\n⚠️  RETRY {attempt}/{max_r}: {event.get('error', '')[:200]}")

                elif event_type == "inventing":
                    print(f"\n🔧 INVENTING: {event.get('tool_name', '')}")
                    print(f"   Description: {event.get('description', '')}")
                    print(f"   Composed from: {event.get('composed_from', [])}")
                    src = event.get("source_code", "")
                    if src:
                        print(f"   Source code:\n{src}")

                elif event_type == "tool_registered":
                    print(f"\n✅ Tool registered: {event.get('tool', {}).get('name', '')}")

                elif event_type == "creating_skill":
                    print(f"\n🎯 CREATING SKILL: {event.get('skill_name', '')}")
                    print(f"   Description: {event.get('description', '')}")
                    print(f"   Tools used: {event.get('tools_used', [])}")

                elif event_type == "skill_registered":
                    print(f"\n✅ Skill registered: {event.get('skill', {}).get('name', '')}")

                elif event_type == "executing":
                    print(f"\n⚡ Executing: {event.get('tool_name', '')}")

                elif event_type == "executing_skill":
                    print(f"\n⚡ Executing skill: {event.get('skill_name', '')}")

                elif event_type == "step":
                    print(f"   Step {event.get('index', 0)}: {event.get('description', '')}")

                elif event_type == "result":
                    print(f"\n🎯 Result from {event.get('tool_name', '')}:")
                    print(f"   {json.dumps(event.get('result', {}), indent=2)[:500]}")

                elif event_type == "skill_result":
                    print(f"\n🎯 Skill result from {event.get('skill_name', '')}:")
                    print(f"   {json.dumps(event.get('result', {}), indent=2)[:500]}")

                elif event_type == "task_complete":
                    print(f"\n{'='*50}")
                    result = event.get("result", {})
                    success = result.get("success", False)
                    icon = "✅" if success else "❌"
                    print(f"{icon} TASK {'COMPLETE' if success else 'FAILED'}")
                    if result.get("error"):
                        print(f"   Error: {result['error'][:300]}")
                    if result.get("tool_name"):
                        print(f"   Tool: {result['tool_name']}")
                    if result.get("skill_name"):
                        print(f"   Skill: {result['skill_name']}")
                    print(f"{'='*50}")
                    break

                elif event_type == "error":
                    print(f"\n❌ Error: {event.get('message', '')}")

                elif event_type in ("sim_status", "task_received"):
                    pass  # quiet

                else:
                    print(f"📨 {event_type}: {json.dumps(event)[:200]}")

            except asyncio.TimeoutError:
                print("⏰ Timeout waiting for response")
                break


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Sort the blocks by color into groups"
    asyncio.run(main(task))
