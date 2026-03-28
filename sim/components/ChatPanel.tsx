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
import { MessageCircle, X } from "lucide-react";

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

**Controls**: The user can enable freeform IK mode to manually control the robot's end-effector position and orientation using a 3D gizmo, adjust simulation speed, pause/resume, and toggle dark mode.

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

interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  isDarkMode: boolean;
}

export function ChatPanel({ isOpen, onClose, isDarkMode }: ChatPanelProps) {
  const runtime = useLocalRuntime(GeminiAdapter);

  const aui = useAui({
    suggestions: Suggestions([
      {
        title: "Scene Understanding",
        label: "What objects are on the table?",
        prompt: "What objects are on the table?",
      },
      {
        title: "Robot Info",
        label: "How does the Franka Panda robot work?",
        prompt: "How does the Franka Panda robot work?",
      },
      {
        title: "Pick & Place",
        label: "Explain pick and place in robotics",
        prompt: "Explain pick and place in robotics",
      },
      {
        title: "Detection Modes",
        label: "What detection modes are available?",
        prompt: "What detection modes are available?",
      },
    ]),
  });

  if (!isOpen) return null;

  return (
    <AssistantRuntimeProvider aui={aui} runtime={runtime}>
      <TooltipProvider>
        <div
          className={`absolute top-4 bottom-4 left-4 right-4 min-[660px]:right-auto min-[660px]:top-10 min-[660px]:left-10 min-[660px]:bottom-10 min-[660px]:w-[420px] rounded-[2.5rem] flex flex-col z-40 overflow-hidden shadow-2xl transition-all border ${
            isDarkMode
              ? "bg-slate-900/90 border-white/10 text-slate-100 shadow-slate-950/40"
              : "bg-white/90 border-white/60 text-slate-800 shadow-slate-200/40"
          }`}
          style={{ backdropFilter: "blur(20px) saturate(180%)" }}
        >
          {/* Header */}
          <div
            className={`px-7 py-5 border-b flex justify-between items-center shrink-0 ${
              isDarkMode
                ? "border-white/5 bg-white/5"
                : "border-slate-100 bg-white/40"
            }`}
          >
            <div className="flex items-center gap-3">
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center ${
                  isDarkMode ? "bg-indigo-500/20" : "bg-indigo-50"
                }`}
              >
                <MessageCircle
                  className={`w-4 h-4 ${isDarkMode ? "text-indigo-400" : "text-indigo-600"}`}
                />
              </div>
              <div>
                <h2 className="text-sm font-bold leading-none">Chat</h2>
                <p
                  className={`text-[10px] mt-1 ${isDarkMode ? "text-slate-500" : "text-slate-400"}`}
                >
                  Gemini 2.5 Flash
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-200/20 rounded-full transition-colors text-slate-400"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Thread */}
          <div className={`flex-1 min-h-0 ${isDarkMode ? "dark" : ""}`}>
            <Thread />
          </div>
        </div>
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}
