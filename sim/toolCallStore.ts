import { create } from 'zustand';

export type ToolCallStatus = 'running' | 'success' | 'error';

export interface ToolCallEntry {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: Record<string, unknown>;
  status: ToolCallStatus;
  startedAt: string;
  finishedAt?: string;
}

interface ToolCallState {
  calls: ToolCallEntry[];
  addCall: (call: ToolCallEntry) => void;
  updateCall: (
    id: string,
    patch: Partial<Pick<ToolCallEntry, 'result' | 'status' | 'finishedAt'>>,
  ) => void;
  clearCalls: () => void;
}

export const useToolCallStore = create<ToolCallState>((set) => ({
  calls: [],
  addCall: (call) =>
    set((state) => ({
      calls: [call, ...state.calls].slice(0, 100),
    })),
  updateCall: (id, patch) =>
    set((state) => ({
      calls: state.calls.map((c) => (c.id === id ? { ...c, ...patch } : c)),
    })),
  clearCalls: () => set({ calls: [] }),
}));
