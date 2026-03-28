/**
 * AgentBridge — singleton WebSocket connection to the ForgeBot backend.
 *
 * Connects to ws://localhost:8000/ws/chat and exposes an event-emitter style API
 * so multiple UI components (ChatPanel, ToolRegistry, etc.) can subscribe.
 */

const WS_URL = import.meta.env.VITE_WS_URL ? `${import.meta.env.VITE_WS_URL}/ws/chat` : "ws://localhost:8000/ws/chat";

export type AgentEventHandler = (data: Record<string, unknown>) => void;

export class AgentBridge {
  private ws: WebSocket | null = null;
  private listeners = new Map<string, Set<AgentEventHandler>>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private disposed = false;

  private _ready = false;
  private _simConnected = false;
  private connectResolve: ((ok: boolean) => void) | null = null;
  private connectPromise: Promise<boolean> | null = null;

  get ready() { return this._ready; }
  get simConnected() { return this._simConnected; }

  /** Connect to backend; resolves true when the init handshake completes. */
  connect(): Promise<boolean> {
    if (this.ws?.readyState === WebSocket.OPEN && this._ready) return Promise.resolve(true);
    if (this.connectPromise) return this.connectPromise;

    this.connectPromise = new Promise<boolean>((resolve) => {
      this.connectResolve = resolve;
      this._doConnect();
    });
    return this.connectPromise;
  }

  private _doConnect() {
    if (this.disposed) return;
    try {
      this.ws = new WebSocket(WS_URL);
    } catch {
      this._resolveConnect(false);
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      console.log("[AgentBridge] connected to backend");
    };

    this.ws.onmessage = (ev) => {
      let msg: Record<string, unknown>;
      try { msg = JSON.parse(ev.data as string); } catch { return; }
      const type = msg.type as string;

      if (type === "init") {
        this._ready = true;
        this._simConnected = msg.sim_connected as boolean;
        this._resolveConnect(true);
      }

      this._dispatch(type, msg);
      this._dispatch("*", msg);
    };

    this.ws.onclose = () => {
      this._ready = false;
      console.log("[AgentBridge] disconnected");
      this._dispatch("connection", { type: "connection", connected: false });
      this._scheduleReconnect();
    };

    this.ws.onerror = () => { this.ws?.close(); };
  }

  private _resolveConnect(ok: boolean) {
    if (this.connectResolve) {
      this.connectResolve(ok);
      this.connectResolve = null;
    }
    this.connectPromise = null;
  }

  private _scheduleReconnect() {
    if (this.disposed) return;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      this._resolveConnect(false);
      this._doConnect();
    }, 3000);
  }

  private _dispatch(event: string, data: Record<string, unknown>) {
    const handlers = this.listeners.get(event);
    if (handlers) for (const fn of handlers) fn(data);
  }

  on(event: string, fn: AgentEventHandler) {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(fn);
  }

  off(event: string, fn: AgentEventHandler) {
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

/** Singleton — one connection shared app-wide. */
export const agentBridge = new AgentBridge();
