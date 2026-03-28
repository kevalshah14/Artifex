/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { PanelRight, Pause, Play, RotateCcw } from 'lucide-react';

interface ToolbarProps {
  isPaused: boolean;
  togglePause: () => void;
  onReset: () => void;
  showSidebar: boolean;
  toggleSidebar: () => void;
}

/**
 * Toolbar
 * Floating control bar for simulation actions. Dark-only styling.
 */
export function Toolbar({
  isPaused,
  togglePause,
  onReset,
  showSidebar,
  toggleSidebar,
}: ToolbarProps) {
  const btnBase =
    "w-14 h-14 rounded-2xl glass-panel flex items-center justify-center transition-all hover:scale-105 active:scale-95 shadow-xl bg-zinc-900/80 border border-zinc-800 text-zinc-100";

  return (
    <div className="absolute bottom-10 left-1/2 -translate-x-1/2 min-[660px]:left-10 min-[660px]:translate-x-0 flex items-center gap-4 z-30">
      {/* Play/Pause */}
      <button onClick={togglePause} className={btnBase} title={isPaused ? "Resume" : "Pause"}>
        {isPaused ? <Play className="w-6 h-6 fill-zinc-100" /> : <Pause className="w-6 h-6 fill-zinc-100" />}
      </button>

      {/* Reset */}
      <button onClick={onReset} className={btnBase} title="Reset Simulation">
        <RotateCcw className="w-6 h-6" />
      </button>

      {/* Sidebar Toggle */}
      <button
        onClick={toggleSidebar}
        className={`${btnBase} ${showSidebar ? 'text-violet-400 bg-zinc-800' : ''}`}
        title="Toggle Analysis Panel"
      >
        <PanelRight className="w-6 h-6" />
      </button>
    </div>
  );
}
