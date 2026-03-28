import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Thread } from "@/components/assistant-ui/thread";
import { useMemo, useState, useEffect, type RefObject } from "react";
import { MujocoSim } from "../MujocoSim";

// ─── WebSocket singleton for the agent backend ──────────────────────

const WS_URL = "ws://localhost:8000/ws/chat";

class AgentConnection {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Set<(data: Record<string, unknown>) => void>> = new Map();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private disposed = false;
  private _ready = false;
  private _simConnected = false;
  private initResolve: ((value: unknown) => void) | null = null;
  private initPromise: Promise<unknown> | null = null;

  get ready() { return this._ready; }
  get simConnected() { return this._simConnected; }

  connect(): Promise<unknown> {
    if (this.ws?.readyState === WebSocket.OPEN) return Promise.resolve(true);
    if (this.initPromise) return this.initPromise;

    this.initPromise = new Promise((resolve) => {
      this.initResolve = resolve;
      this.doConnect();
    });

    return this.initPromise;
  }

  private doConnect() {
    if (this.disposed) return;
    try {
      this.ws = new WebSocket(WS_URL);
    } catch {
      if (this.initResolve) { this.initResolve(false); this.initResolve = null; this.initPromise = null; }
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      console.log("[ChatPanel] WS connected to agent backend");
    };

    this.ws.onmessage = (ev) => {
      let msg: Record<string, unknown>;
      try { msg = JSON.parse(ev.data as string); } catch { return; }

      const type = msg.type as string;

      if (type === "init") {
        this._ready = true;
        this._simConnected = msg.sim_connected as boolean;
        if (this.initResolve) { this.initResolve(true); this.initResolve = null; }
        this.initPromise = null;
      }

      // Dispatch to type-specific listeners
      const handlers = this.listeners.get(type);
      if (handlers) {
        for (const fn of handlers) fn(msg);
      }
      // Dispatch to wildcard listeners
      const wildcard = this.listeners.get("*");
      if (wildcard) {
        for (const fn of wildcard) fn(msg);
      }
    };

    this.ws.onclose = () => {
      this._ready = false;
      console.log("[ChatPanel] WS disconnected");
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  private scheduleReconnect() {
    if (this.disposed) return;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      if (this.initResolve) { this.initResolve(false); this.initResolve = null; this.initPromise = null; }
      this.doConnect();
    }, 3000);
  }

  on(event: string, fn: (data: Record<string, unknown>) => void) {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(fn);
  }

  off(event: string, fn: (data: Record<string, unknown>) => void) {
    this.listeners.get(event)?.delete(fn);
  }

  send(data: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  dispose() {
    this.disposed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }
}

// Singleton — one WS connection shared across re-renders
const agent = new AgentConnection();

// ─── Adapter: routes chat messages through the agent backend ────────

function createBackendAdapter(): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      // Get the last user message
      const lastUserMsg = messages.filter((m) => m.role === "user").pop();
      const task = lastUserMsg?.content
        ?.filter((p) => p.type === "text")
        .map((p) => (p as { type: "text"; text: string }).text)
        .join("\n") ?? "";

      if (!task.trim()) {
        yield { content: [{ type: "text" as const, text: "Please enter a task." }] };
        return;
      }

      // Make sure we're connected
      const ok = await agent.connect();
      if (!ok) {
        yield { content: [{ type: "text" as const, text: "Could not connect to the agent backend. Is the server running?" }] };
        return;
      }

      // Set up listeners for agent events
      let parts: string[] = [];

      const onEvent = (data: Record<string, unknown>) => {
        const type = data.type as string;

        if (type === "thinking") {
          parts.push(`🧠 ${data.message}`);
        } else if (type === "plan") {
          parts.push(`\n📋 **Plan**: ${data.thought}`);
        } else if (type === "executing") {
          parts.push(`⚡ Executing: ${data.tool_name}`);
        } else if (type === "result") {
          const result = data.result as Record<string, unknown> | undefined;
          if (result) {
            const r = result.result ?? result;
            parts.push(`✅ ${data.tool_name}: \`${JSON.stringify(r).slice(0, 200)}\``);
          }
        } else if (type === "retry") {
          parts.push(`⚠️ Retry ${data.attempt}/${data.max_retries}: ${(data.error as string ?? "").slice(0, 150)}`);
        } else if (type === "inventing") {
          parts.push(`🔧 Inventing tool: ${data.tool_name}`);
        } else if (type === "tool_registered") {
          parts.push(`✅ Registered: ${(data.tool as Record<string, unknown>)?.name ?? "tool"}`);
        } else if (type === "evolving") {
          parts.push(`🧬 Evolving: ${data.message}`);
        } else if (type === "auto_escalate") {
          parts.push(`🚀 Auto-escalating: ${data.message}`);
        } else if (type === "error") {
          parts.push(`❌ ${data.message}`);
        }
      };

      agent.on("*", onEvent);

      // Send the task
      agent.send({ type: "task", message: task });

      // Wait for completion
      yield { content: [{ type: "text" as const, text: "⏳ *Sending task to agent...*\n" }] };

      // Wait for task_complete event
      const result = await new Promise<Record<string, unknown>>((resolve) => {
        const onComplete = (data: Record<string, unknown>) => {
          resolve(data.result as Record<string, unknown>);
        };
        agent.on("task_complete", onComplete);

        // Also handle abort
        if (abortSignal) {
          const onAbort = () => {
            agent.off("task_complete", onComplete);
            resolve({ success: false, error: "Aborted" });
          };
          abortSignal.addEventListener("abort", onAbort, { once: true });
        }
      });

      agent.off("*", onEvent);
      agent.off("task_complete", () => {});

      // Build the final response
      const success = result?.success as boolean;
      const summary = result?.summary as string ?? "";
      const error = result?.error as string ?? "";

      let text = "";
      if (parts.length > 0) {
        text += parts.join("\n") + "\n\n";
      }
      if (success) {
        text += `✅ **Done**${summary ? `: ${summary}` : ""}`;
      } else {
        text += `❌ **Failed**${error ? `: ${error}` : ""}`;
      }

      yield { content: [{ type: "text" as const, text }] };
    },
  };
}

// ─── ChatPanel component ────────────────────────────────────────────

interface ChatPanelProps {
  simRef: RefObject<MujocoSim | null>;
}

export function ChatPanel({ simRef }: ChatPanelProps) {
  const [backendStatus, setBackendStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");

  const adapter = useMemo(() => createBackendAdapter(), []);
  const runtime = useLocalRuntime(adapter);

  // Connect on mount
  useEffect(() => {
    agent.connect().then((ok) => {
      setBackendStatus(ok ? "connected" : "disconnected");
    });

    const onStatus = (data: Record<string, unknown>) => {
      setBackendStatus(data.connected ? "connected" : "disconnected");
    };
    agent.on("sim_status", onStatus);

    return () => {
      agent.off("sim_status", onStatus);
    };
  }, []);

  const statusColor =
    backendStatus === "connected"
      ? "bg-emerald-400"
      : backendStatus === "connecting"
        ? "bg-amber-400"
        : "bg-red-400";

  const statusLabel =
    backendStatus === "connected"
      ? "agent backend"
      : backendStatus === "connecting"
        ? "connecting..."
        : "backend offline";

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <div className="flex flex-col h-full w-full bg-zinc-900 text-zinc-100">
          {/* Chat header */}
          <div className="px-4 py-2.5 border-b border-zinc-800 flex items-center gap-2 shrink-0 bg-zinc-900/80">
            <div className={`w-2 h-2 rounded-full ${statusColor} ${backendStatus === "connecting" ? "animate-pulse" : ""}`} />
            <span className="text-xs font-mono font-bold tracking-widest text-zinc-400">
              CHAT
            </span>
            <span className="ml-auto text-[10px] font-mono text-zinc-600">
              {statusLabel}
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
