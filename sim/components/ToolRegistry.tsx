import { useState } from "react";
import { Wrench, Sparkles, ChevronDown } from "lucide-react";

interface Tool {
  name: string;
  description: string;
  code: string;
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

function ToolCard({ tool }: { tool: Tool }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono font-semibold text-sm text-violet-300 truncate">
            {tool.name}
          </span>
          <span className="text-[10px] bg-zinc-800 text-zinc-400 px-1.5 rounded font-mono shrink-0">
            PRIMITIVE
          </span>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors shrink-0"
        >
          <ChevronDown
            className={`w-3.5 h-3.5 transition-transform ${isExpanded ? "rotate-180" : ""}`}
          />
        </button>
      </div>
      <p className="text-xs text-zinc-500 mt-1">{tool.description}</p>
      {isExpanded && (
        <pre className="bg-black rounded p-2 text-xs font-mono text-emerald-300 mt-2 overflow-x-auto whitespace-pre">
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
  const [activeTab, setActiveTab] = useState<TabType>("primitives");

  const tools = activeTab === "primitives" ? PRIMITIVE_TOOLS : [];

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Wrench className="w-3.5 h-3.5 text-zinc-400" />
          <span className="text-xs font-mono font-bold tracking-widest text-zinc-400">
            TOOL REGISTRY
          </span>
        </div>
        <span className="text-[10px] bg-zinc-800 text-zinc-500 px-1.5 py-0.5 rounded font-mono">
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
      </div>

      {/* Tool cards */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {tools.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-zinc-600">
            <Sparkles className="w-6 h-6 mb-2" />
            <p className="text-xs font-mono">No invented tools yet</p>
          </div>
        ) : (
          tools.map((tool) => <ToolCard key={tool.name} tool={tool} />)
        )}
      </div>
    </div>
  );
}
