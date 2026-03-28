import { GoogleGenAI } from "@google/genai";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  useAui,
  Suggestions,
  type ChatModelAdapter,
} from "@assistant-ui/react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Thread } from "@/components/assistant-ui/thread";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

const SYSTEM_PROMPT = `You are an assistant for Artifex, an interactive robotics simulation environment. The user is working with:

**Robot**: Franka Emika Panda — a 7-DOF robotic arm with a parallel gripper, simulated in real-time.

**Simulation**: MuJoCo physics engine running in the browser via WebAssembly, rendered with Three.js. The robot operates on a tabletop scene with manipulable objects (cubes, blocks, etc.).

**Detection Modes**: The user can analyze the scene using Google Gemini vision models with three detection types:
- **2D Bounding Boxes**: Detects objects and returns labeled bounding boxes
- **Segmentation Masks**: Returns per-object segmentation masks with bounding boxes
- **Points**: Detects object center points for targeting

**Pick-and-Place Workflow**:
1. The user types a prompt describing objects to detect (e.g., "red cubes")
2. The system captures a top-down snapshot of the scene
3. Gemini analyzes the image and returns detected object locations
4. The detected positions are projected from 2D image coordinates into 3D world coordinates
5. The robot arm uses inverse kinematics to move to each target, pick it up with the gripper, and place it in a tray or stacking position

**Controls**: The user can enable freeform IK mode to manually control the robot's end-effector position and orientation using a 3D gizmo, adjust simulation speed, pause/resume.

Help the user understand and use this simulation effectively. Answer questions about the robot, the detection pipeline, the pick-and-place workflow, and troubleshooting.`;

const GeminiAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const contents = messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: m.content
          .filter((p) => p.type === "text")
          .map((p) => ({ text: p.type === "text" ? p.text : "" })),
      }));

    const response = await ai.models.generateContentStream({
      model: "gemini-2.5-flash",
      contents,
      config: {
        abortSignal,
        systemInstruction: SYSTEM_PROMPT,
      },
    });

    let text = "";
    for await (const chunk of response) {
      text += chunk.text ?? "";
      yield { content: [{ type: "text" as const, text }] };
    }
  },
};

export function ChatPanel() {
  const runtime = useLocalRuntime(GeminiAdapter);

  const aui = useAui({
    suggestions: Suggestions([
      {
        title: "Sort Objects",
        label: "Sort the red blocks to the left side",
        prompt: "Sort the red blocks to the left side",
      },
      {
        title: "Stack Objects",
        label: "Stack all objects by color",
        prompt: "Stack all objects by color",
      },
      {
        title: "Move Gripper",
        label: "Move the gripper to position (0.3, 0, 0.4)",
        prompt: "Move the gripper to position (0.3, 0, 0.4)",
      },
      {
        title: "Scene Understanding",
        label: "What objects are on the table?",
        prompt: "What objects are on the table?",
      },
    ]),
  });

  return (
    <AssistantRuntimeProvider aui={aui} runtime={runtime}>
      <TooltipProvider>
        <div className="flex flex-col h-full w-full bg-zinc-900 text-zinc-100">
          {/* Chat header */}
          <div className="px-4 py-2.5 border-b border-zinc-800 flex items-center gap-2 shrink-0 bg-zinc-900/80">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-mono font-bold tracking-widest text-zinc-400">
              CHAT
            </span>
            <span className="ml-auto text-[10px] font-mono text-zinc-600">
              gemini-2.5-flash
            </span>
          </div>
          {/* Thread */}
          <div className="flex-1 min-h-0 dark">
            <Thread />
          </div>
        </div>
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}
