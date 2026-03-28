import { GoogleGenAI } from "@google/genai";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Thread } from "@/components/assistant-ui/thread";
import { useMemo, type RefObject } from "react";
import { MujocoSim } from "../MujocoSim";
import {
  robotFunctionDeclarations,
  executeRobotTool,
} from "../robotTools";
import { useToolCallStore } from "../toolCallStore";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

const SYSTEM_PROMPT = `You are the AI controller for Artifex, an interactive robotics simulation. You can directly control a Franka Emika Panda 7-DOF robotic arm via the tools provided.

**Environment**: A tabletop scene with coloured cubes simulated in MuJoCo (physics engine running in-browser). The robot has a parallel gripper.

**How to work**:
1. When asked to manipulate objects, ALWAYS call get_all_objects first to discover what's on the table, their names, positions, and colours.
2. Use pick_up with the body name to grab an object. This runs the full approach-grasp-lift sequence automatically.
3. Use place_at to move the arm somewhere and release the held object.
4. For fine-grained control, use move_to and set_gripper individually.
5. For complex tasks requiring tool invention, call run_vlmgineer(task, max_iterations, candidates_per_iteration, solve_threshold). This performs iterative candidate generation, simulation evaluation, mutation/crossover, and early stopping when solved.
6. The system learns across runs: previous elite designs for similar tasks are reused as memory seeds.
7. Call get_vlmgineer_status after a run to inspect best designs, fitness progression, and memory coverage.
8. After completing actions, briefly confirm what you did and report if solved.

**Coordinate system**: X and Y are the horizontal plane (table surface), Z is up. The table center is roughly (0, 0, 0). Typical cube Z is ~0.02. Safe hover height is Z ≈ 0.15–0.25.

**Object naming**: Objects are named like "cube0", "cube1", etc. Use get_all_objects to learn the mapping between names and colours.

Be concise. When executing multi-step tasks (sorting, stacking), call tools sequentially — one pick_up, then one place_at, then the next object.`;

function createAdapter(
  simRef: RefObject<MujocoSim | null>,
): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      // Build Gemini-compatible message history
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const contents: any[] = messages
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({
          role: m.role === "assistant" ? "model" : "user",
          parts: m.content
            .filter((p) => p.type === "text")
            .map((p) => ({ text: p.type === "text" ? p.text : "" })),
        }));

      const toolsConfig = {
        tools: [{ functionDeclarations: robotFunctionDeclarations }],
        systemInstruction: SYSTEM_PROMPT,
        abortSignal,
      };

      // Multi-turn tool-use loop: keep calling until the model produces text
      const MAX_TOOL_ROUNDS = 15;
      for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
        const response = await ai.models.generateContent({
          model: "gemini-2.5-flash",
          contents,
          config: toolsConfig,
        });

        const functionCalls = response.functionCalls;
        if (!functionCalls || functionCalls.length === 0) {
          // No tool calls — stream the final text back
          const text = response.text ?? "";
          yield { content: [{ type: "text" as const, text }] };
          return;
        }

        // Execute each function call and collect responses
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const functionResponseParts: any[] = [];

        for (const fc of functionCalls) {
          const callId = `${fc.id ?? "call"}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
          const callArgs = ((fc.args as Record<string, unknown>) ?? {});
          useToolCallStore.getState().addCall({
            id: callId,
            name: fc.name,
            args: callArgs,
            status: "running",
            startedAt: new Date().toISOString(),
          });

          // Yield a progress update so the user sees activity
          yield {
            content: [
              {
                type: "text" as const,
                text: `\u200B\n\n⏳ *Executing ${fc.name}*...\n\n`,
              },
            ],
          };

          const sim = simRef.current;
          let result: Record<string, unknown>;
          if (sim) {
            result = await executeRobotTool(
              sim,
              fc.name,
              callArgs,
            );
          } else {
            result = { success: false, error: "Simulation not available" };
          }

          useToolCallStore.getState().updateCall(callId, {
            result,
            status: result.success === false ? "error" : "success",
            finishedAt: new Date().toISOString(),
          });

          functionResponseParts.push({
            functionResponse: {
              name: fc.name,
              response: { result },
              id: fc.id,
            },
          });
        }

        // Append the model's response (with function calls) to history
        contents.push(response.candidates![0].content);
        // Append function results so the model can see them
        contents.push({ role: "user", parts: functionResponseParts });
      }

      // Fell through MAX_TOOL_ROUNDS — generate a final text response
      const finalResponse = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents,
        config: toolsConfig,
      });
      yield {
        content: [
          { type: "text" as const, text: finalResponse.text ?? "" },
        ],
      };
    },
  };
}

interface ChatPanelProps {
  simRef: RefObject<MujocoSim | null>;
}

export function ChatPanel({ simRef }: ChatPanelProps) {
  const adapter = useMemo(() => createAdapter(simRef), [simRef]);
  const runtime = useLocalRuntime(adapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <div className="flex flex-col h-full w-full bg-zinc-900 text-zinc-100">
          {/* Chat header */}
          <div className="px-3 py-2 border-b border-zinc-800 shrink-0 bg-zinc-900/80">
            <span className="text-[11px] font-mono font-semibold tracking-wide text-zinc-400">
              CHAT
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
