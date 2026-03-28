import { useState } from "react";
import { Wrench, Sparkles, ChevronDown, History } from "lucide-react";
import { useInventedToolsStore } from "../inventedToolsStore";
import { useToolCallStore } from "../toolCallStore";

interface Tool {
  name: string;
  description: string;
  code: string;
  badge?: string;
}

const PRIMITIVE_TOOLS: Tool[] = [
  {
    name: "move_to",
    description: "Move end-effector to target XYZ position",
    code: `def move_to(x: float, y: float, z: float) -> bool:\n    """Move robot end-effector to target position."""\n    return robot.ik_solve(target=[x, y, z])`,
  },
  {
    name: "set_gripper",
    description: "Open or close the parallel gripper",
    code: `def set_gripper(state: str) -> bool:\n    """Set gripper state: 'open' or 'closed'."""\n    return robot.gripper.set(state)`,
  },
  {
    name: "get_body_position",
    description: "Get the 3D position of a named body",
    code: `def get_body_position(name: str) -> tuple:\n    """Returns (x, y, z) world position of body."""\n    return sim.get_body_pos(name)`,
  },
  {
    name: "detect_objects",
    description: "Run vision model to detect objects in scene",
    code: `def detect_objects(prompt: str) -> list:\n    """Detect objects matching prompt via Gemini."""\n    return vision.detect(prompt, mode="2d_bbox")`,
  },
  {
    name: "pick_and_place",
    description: "Pick object at source and place at target",
    code: `def pick_and_place(src: tuple, dst: tuple) -> bool:\n    """Pick from src (x,y,z) and place at dst."""\n    return robot.pick(src).place(dst)`,
  },
];

type TabType = "primitives" | "invented";
type TabTypeExtended = TabType | "calls";

function ToolCard({ tool }: { tool: Tool }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={`bg-zinc-950 border rounded-lg px-3 py-2.5 transition-all cursor-pointer group ${
      isExpanded ? "border-violet-500/30 shadow-[0_0_12px_rgba(167,139,250,0.06)]" : "border-zinc-800 hover:border-zinc-700"
    }`} onClick={() => setIsExpanded(!isExpanded)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-1.5 h-1.5 rounded-full bg-violet-400/60 shrink-0" />
          <span className="font-mono font-semibold text-sm text-violet-300 truncate">
            {tool.name}
          </span>
          <span className="text-[10px] bg-violet-500/10 text-violet-400/70 px-1.5 py-px rounded font-mono shrink-0 border border-violet-500/10">
            {tool.badge ?? "PRIMITIVE"}
          </span>
        </div>
        <ChevronDown
          className={`w-3.5 h-3.5 text-zinc-600 group-hover:text-zinc-400 transition-all shrink-0 ${isExpanded ? "rotate-180 text-violet-400" : ""}`}
        />
      </div>
      <p className="text-[11px] text-zinc-500 mt-1 ml-3.5">{tool.description}</p>
      {isExpanded && (
        <pre className="bg-black/60 border border-zinc-800 rounded-md p-2.5 text-[11px] font-mono text-emerald-400/90 mt-2 overflow-x-auto whitespace-pre leading-relaxed">
          {tool.code}
        </pre>
      )}
    </div>
  );
}

interface ToolRegistryProps {
  className?: string;
}

export function ToolRegistry({ className = "" }: ToolRegistryProps) {
  const [activeTab, setActiveTab] = useState<TabTypeExtended>("calls");
  const inventedTools = useInventedToolsStore((s) => s.inventedTools);
  const calls = useToolCallStore((s) => s.calls);

  const tools = activeTab === "primitives"
    ? PRIMITIVE_TOOLS
    : activeTab === "invented"
      ? inventedTools.map((tool) => ({
          name: tool.name,
          description: `${tool.description} (fitness ${tool.bestFitness.toFixed(3)})`,
          code: `${tool.toolMjcf}\n\n# waypoints\n${JSON.stringify(tool.waypoints, null, 2)}`,
          badge: "INVENTED",
        }))
      : calls.map((call) => ({
          name: call.name,
          description: `${call.status.toUpperCase()} • ${new Date(call.startedAt).toLocaleTimeString()}`,
          code: `args:\n${JSON.stringify(call.args, null, 2)}\n\nresult:\n${JSON.stringify(call.result ?? {}, null, 2)}`,
          badge: "CALL",
        }));

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Wrench className="w-3.5 h-3.5 text-violet-400/60" />
          <span className="text-[11px] font-mono font-bold tracking-widest text-zinc-400">
            TOOL REGISTRY
          </span>
        </div>
        <span className="text-[10px] bg-violet-500/10 text-violet-400/70 px-1.5 py-0.5 rounded font-mono border border-violet-500/10">
          {tools.length}
        </span>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-zinc-800 px-3">
        <button
          onClick={() => setActiveTab("primitives")}
          className={`flex items-center gap-1.5 px-2 py-1.5 text-xs font-mono transition-colors border-b-2 -mb-px ${
            activeTab === "primitives"
              ? "border-violet-400 text-violet-300"
              : "border-transparent text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <Wrench className="w-3 h-3" />
          Primitives
        </button>
        <button
          onClick={() => setActiveTab("invented")}
          className={`flex items-center gap-1.5 px-2 py-1.5 text-xs font-mono transition-colors border-b-2 -mb-px ${
            activeTab === "invented"
              ? "border-violet-400 text-violet-300"
              : "border-transparent text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <Sparkles className="w-3 h-3" />
          Invented
        </button>
        <button
          onClick={() => setActiveTab("calls")}
          className={`flex items-center gap-1.5 px-2 py-1.5 text-xs font-mono transition-colors border-b-2 -mb-px ${
            activeTab === "calls"
              ? "border-violet-400 text-violet-300"
              : "border-transparent text-zinc-500 hover:text-zinc-300"
          }`}
        >
          <History className="w-3 h-3" />
          Live Calls
        </button>
      </div>

      {/* Tool cards */}
      <div className="flex-1 overflow-y-auto p-2.5 space-y-1.5">
        {tools.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-zinc-600">
            <div className="w-10 h-10 rounded-xl bg-zinc-800/50 border border-zinc-800 flex items-center justify-center mb-3">
              <Sparkles className="w-4 h-4 text-zinc-600" />
            </div>
            <p className="text-xs font-mono text-zinc-600">
              {activeTab === "calls" ? "No tool calls yet" : "No invented tools yet"}
            </p>
            <p className="text-[10px] text-zinc-700 mt-1">
              {activeTab === "calls" ? "Run an action in chat to see live calls" : "Ask Artifex to create one"}
            </p>
          </div>
        ) : (
          tools.map((tool) => <ToolCard key={tool.name} tool={tool} />)
        )}
      </div>
    </div>
  );
}
