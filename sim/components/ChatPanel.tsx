import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Thread } from "@/components/assistant-ui/thread";
import { useEffect, useState } from "react";
import { agentBridge, type AgentEventHandler } from "../agentBridge";
import { useToolCallStore } from "../toolCallStore";

// ─── Format backend events into readable markdown lines ────────────

function formatEvent(data: Record<string, unknown>): string | null {
  const t = data.type as string;
  switch (t) {
    case "thinking":
      return `**Thinking** — ${data.message}`;
    case "llm_call":
      return `**LLM** — ${data.message}`;
    case "plan":
      return `**Plan** (step ${data.step}) — ${data.thought}\n> action: \`${data.action}\`${data.tool_name ? ` → \`${data.tool_name}\`` : ""}`;
    case "executing":
      return `**Executing** \`${data.tool_name}\`…`;
    case "executing_skill":
      return `**Executing skill** \`${data.skill_name}\`…`;
    case "result":
    case "skill_result": {
      const r = data.result as Record<string, unknown> | undefined;
      const name = data.tool_name ?? data.skill_name ?? "?";
      const preview = r ? JSON.stringify(r).slice(0, 200) : "";
      return `**Result** \`${name}\` — \`${preview}\``;

    }
    case "retry":
      return `**Retry** ${data.attempt}/${data.max_retries} — ${(data.error as string ?? "").slice(0, 150)}`;
    case "inventing":
      return `**Inventing tool** \`${data.tool_name}\``;
    case "creating_skill":
      return `**Creating skill** \`${data.skill_name}\``;
    case "tool_registered":
      return `**Registered** \`${(data.tool as Record<string, unknown>)?.name ?? "tool"}\``;
    case "skill_registered":
      return `**Registered skill** \`${(data.skill as Record<string, unknown>)?.name ?? "skill"}\``;
    case "evolving":
      return `**Evolving** — ${data.message}`;
    case "auto_escalate":
      return `**Auto-escalate** — ${data.message}`;
    case "error":
      return `**Error** — ${data.message}`;
    case "done":
      return `**Done** — ${data.summary ?? data.message ?? ""}`;
    default:
      return null;
  }
}

// ─── Adapter: streams agent backend events into assistant-ui ────────

function createBackendAdapter(): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const lastUserMsg = messages.filter((m) => m.role === "user").pop();
      const task = lastUserMsg?.content
        ?.filter((p) => p.type === "text")
        .map((p) => (p as { type: "text"; text: string }).text)
        .join("\n") ?? "";

      if (!task.trim()) {
        yield { content: [{ type: "text" as const, text: "Please enter a task." }] };
        return;
      }

      const ok = await agentBridge.connect();
      if (!ok) {
        yield {
          content: [{
            type: "text" as const,
            text: "Could not connect to the agent backend.\n\nStart it with:\n```bash\ncd agent && python server.py\n```",
          }],
        };
        return;
      }

      // Async queue: events pushed from WS callbacks, consumed by the generator.
      const queue: Array<{ kind: "event"; text: string } | { kind: "done"; result: Record<string, unknown> }> = [];
      let wakeUp: (() => void) | null = null;
      const push = (item: (typeof queue)[number]) => {
        queue.push(item);
        wakeUp?.();
        wakeUp = null;
      };
      const waitForItem = () => new Promise<void>((r) => { wakeUp = r; });

      const { addCall, updateCall } = useToolCallStore.getState();

      const onEvent: AgentEventHandler = (data) => {
        const t = data.type as string;

        // Push executing/result events to the toolCallStore for the Live Calls tab.
        if (t === "executing" || t === "executing_skill") {
          const name = (data.tool_name ?? data.skill_name) as string;
          addCall({
            id: `${name}-${Date.now()}`,
            name,
            args: (data.args as Record<string, unknown>) ?? {},
            status: "running",
            startedAt: new Date().toISOString(),
          });
        } else if (t === "result" || t === "skill_result") {
          const name = (data.tool_name ?? data.skill_name) as string;
          const calls = useToolCallStore.getState().calls;
          const pending = [...calls].reverse().find((c) => c.name === name && c.status === "running");
          if (pending) {
            const r = data.result as Record<string, unknown> | undefined;
            updateCall(pending.id, {
              result: r,
              status: r?.success === false ? "error" : "success",
              finishedAt: new Date().toISOString(),
            });
          }
        }

        if (t === "task_complete") {
          push({ kind: "done", result: (data.result as Record<string, unknown>) ?? {} });
          return;
        }

        const line = formatEvent(data);
        if (line) push({ kind: "event", text: line });
      };

      agentBridge.on("*", onEvent);

      // Handle abort
      let aborted = false;
      const onAbort = () => {
        aborted = true;
        push({ kind: "done", result: { success: false, error: "Aborted" } });
      };
      abortSignal?.addEventListener("abort", onAbort, { once: true });

      // Send the task
      agentBridge.send({ type: "task", message: task });

      const lines: string[] = [];
      let finished = false;
      let finalResult: Record<string, unknown> = {};

      // Yield initial state
      yield { content: [{ type: "text" as const, text: "Sending task to agent…\n" }] };

      while (!finished && !aborted) {
        if (queue.length === 0) await waitForItem();
        while (queue.length > 0) {
          const item = queue.shift()!;
          if (item.kind === "done") {
            finished = true;
            finalResult = item.result;
          } else {
            lines.push(item.text);
          }
        }
        const progress = finished ? "" : "\n\n_Processing…_";
        yield { content: [{ type: "text" as const, text: lines.join("\n\n") + progress }] };
      }

      // Cleanup
      agentBridge.off("*", onEvent);
      abortSignal?.removeEventListener("abort", onAbort);

      // Final yield
      const success = finalResult?.success as boolean;
      const summary = (finalResult?.summary as string) ?? "";
      const error = (finalResult?.error as string) ?? "";
      const suffix = success
        ? `\n\n**Task complete**${summary ? ` — ${summary}` : ""}`
        : `\n\n**Task failed**${error ? ` — ${error}` : ""}`;

      yield { content: [{ type: "text" as const, text: lines.join("\n\n") + suffix }] };
    },
  };
}

// ─── ChatPanel component ────────────────────────────────────────────

export function ChatPanel() {
  const [status, setStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");

  const adapter = createBackendAdapter();
  const runtime = useLocalRuntime(adapter);

  useEffect(() => {
    agentBridge.connect().then((ok) => {
      setStatus(ok ? "connected" : "disconnected");
    });

    const onSim: AgentEventHandler = (data) => {
      setStatus(data.connected ? "connected" : "disconnected");
    };
    const onConn: AgentEventHandler = (data) => {
      if (!data.connected) setStatus("disconnected");
    };
    agentBridge.on("sim_status", onSim);
    agentBridge.on("connection", onConn);

    return () => {
      agentBridge.off("sim_status", onSim);
      agentBridge.off("connection", onConn);
    };
  }, []);

  const dot =
    status === "connected" ? "bg-emerald-400"
      : status === "connecting" ? "bg-amber-400 animate-pulse"
        : "bg-red-400";

  const label =
    status === "connected" ? "agent connected"
      : status === "connecting" ? "connecting…"
        : "backend offline";

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <div className="flex flex-col h-full w-full bg-zinc-900 text-zinc-100">
          <div className="px-4 py-2.5 border-b border-zinc-800 flex items-center gap-2 shrink-0 bg-zinc-900/80">
            <div className={`w-2 h-2 rounded-full ${dot}`} />
            <span className="text-xs font-mono font-bold tracking-widest text-zinc-400">CHAT</span>
            <span className="ml-auto text-[10px] font-mono text-zinc-600">{label}</span>
          </div>
          <div className="flex-1 min-h-0 dark">
            <Thread />
          </div>
        </div>
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}
