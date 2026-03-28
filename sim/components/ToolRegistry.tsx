import { useState, useEffect } from "react";
import { Wrench, Sparkles, ChevronDown, History, Zap } from "lucide-react";
import { useToolCallStore } from "../toolCallStore";
import { agentBridge, type AgentEventHandler } from "../agentBridge";

interface ToolItem {
  name: string;
  description: string;
  code: string;
  badge?: string;
}

interface BackendTool {
  name: string;
  type: string;
  description: string;
  signature: string;
  source_code?: string;
  composed_from?: string[];
  invocation_count?: number;
}

function ToolCard({ tool }: { tool: ToolItem }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={`bg-zinc-950 border rounded-lg px-3 py-2.5 transition-all cursor-pointer group ${
        isExpanded ? "border-violet-500/30 shadow-[0_0_12px_rgba(167,139,250,0.06)]" : "border-zinc-800 hover:border-zinc-700"
      }`}
      onClick={() => setIsExpanded(!isExpanded)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-1.5 h-1.5 rounded-full bg-violet-400/60 shrink-0" />
          <span className="font-mono font-semibold text-sm text-violet-300 truncate">{tool.name}</span>
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

type TabId = "tools" | "skills" | "calls";

export function ToolRegistry({ className = "" }: { className?: string }) {
  const [activeTab, setActiveTab] = useState<TabId>("calls");
  const [backendTools, setBackendTools] = useState<BackendTool[]>([]);
  const [backendSkills, setBackendSkills] = useState<BackendTool[]>([]);
  const calls = useToolCallStore((s) => s.calls);

  // Fetch initial tools/skills from backend on mount and subscribe to updates.
  useEffect(() => {
    const fetchTools = async () => {
      try {
        const resp = await fetch("http://localhost:8000/tools");
        if (resp.ok) {
          const data = await resp.json();
          setBackendTools(data.tools ?? []);
        }
      } catch { /* backend may be offline */ }
    };

    const fetchSkills = async () => {
      try {
        const resp = await fetch("http://localhost:8000/skills");
        if (resp.ok) {
          const data = await resp.json();
          setBackendSkills(data.skills ?? []);
        }
      } catch { /* backend may be offline */ }
    };

    fetchTools();
    fetchSkills();

    const onToolChange: AgentEventHandler = () => { fetchTools(); };
    const onSkillChange: AgentEventHandler = () => { fetchSkills(); };

    agentBridge.on("tool_registered", onToolChange);
    agentBridge.on("tool_change", onToolChange);
    agentBridge.on("skill_registered", onSkillChange);
    agentBridge.on("skill_change", onSkillChange);
    agentBridge.on("init", () => { fetchTools(); fetchSkills(); });

    return () => {
      agentBridge.off("tool_registered", onToolChange);
      agentBridge.off("tool_change", onToolChange);
      agentBridge.off("skill_registered", onSkillChange);
      agentBridge.off("skill_change", onSkillChange);
    };
  }, []);

  const toolItems: ToolItem[] =
    activeTab === "tools"
      ? backendTools.map((t) => ({
          name: t.name,
          description: `${t.description}${t.invocation_count ? ` (${t.invocation_count} calls)` : ""}`,
          code: t.source_code ?? t.signature,
          badge: t.type === "primitive" ? "PRIMITIVE" : "INVENTED",
        }))
      : activeTab === "skills"
        ? backendSkills.map((s) => ({
            name: s.name,
            description: s.description,
            code: s.source_code ?? s.signature,
            badge: "SKILL",
          }))
        : calls.map((c) => ({
            name: c.name,
            description: `${c.status.toUpperCase()} • ${new Date(c.startedAt).toLocaleTimeString()}`,
            code: `args:\n${JSON.stringify(c.args, null, 2)}\n\nresult:\n${JSON.stringify(c.result ?? {}, null, 2)}`,
            badge: c.status === "running" ? "RUNNING" : c.status === "error" ? "ERROR" : "DONE",
          }));

  const count = activeTab === "tools" ? backendTools.length : activeTab === "skills" ? backendSkills.length : calls.length;

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Wrench className="w-3.5 h-3.5 text-violet-400/60" />
          <span className="text-[11px] font-mono font-bold tracking-widest text-zinc-400">TOOL REGISTRY</span>
        </div>
        <span className="text-[10px] bg-violet-500/10 text-violet-400/70 px-1.5 py-0.5 rounded font-mono border border-violet-500/10">
          {count}
        </span>
      </div>

      <div className="flex border-b border-zinc-800 px-3">
        {([
          { id: "calls" as TabId, icon: History, label: "Live" },
          { id: "tools" as TabId, icon: Wrench, label: "Tools" },
          { id: "skills" as TabId, icon: Sparkles, label: "Skills" },
        ]).map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-1.5 px-2 py-1.5 text-xs font-mono transition-colors border-b-2 -mb-px ${
              activeTab === id ? "border-violet-400 text-violet-300" : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            <Icon className="w-3 h-3" />
            {label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-2.5 space-y-1.5">
        {toolItems.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 text-zinc-600">
            <div className="w-10 h-10 rounded-xl bg-zinc-800/50 border border-zinc-800 flex items-center justify-center mb-3">
              <Zap className="w-4 h-4 text-zinc-600" />
            </div>
            <p className="text-xs font-mono text-zinc-600">
              {activeTab === "calls" ? "No tool calls yet" : activeTab === "tools" ? "No tools loaded" : "No skills loaded"}
            </p>
            <p className="text-[10px] text-zinc-700 mt-1">
              {activeTab === "calls" ? "Send a task in chat to see live calls" : "Start the backend to load tools"}
            </p>
          </div>
        ) : (
          toolItems.map((tool, i) => <ToolCard key={`${tool.name}-${i}`} tool={tool} />)
        )}
      </div>
    </div>
  );
}
