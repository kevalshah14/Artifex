import { create } from 'zustand';

export interface InventedTool {
  id: string;
  name: string;
  description: string;
  task: string;
  toolMjcf: string;
  waypoints: number[][];
  bestFitness: number;
  iteration: number;
  createdAt: string;
}

interface InventedToolsState {
  inventedTools: InventedTool[];
  addInventedTool: (tool: InventedTool) => void;
  clearInventedTools: () => void;
}

export const useInventedToolsStore = create<InventedToolsState>((set) => ({
  inventedTools: [],
  addInventedTool: (tool) =>
    set((state) => ({
      inventedTools: [tool, ...state.inventedTools].slice(0, 30),
    })),
  clearInventedTools: () => set({ inventedTools: [] }),
}));
