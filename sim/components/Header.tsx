import { Hexagon, Wrench } from "lucide-react";

interface HeaderProps {
  modelName: string;
  showToolRegistry?: boolean;
  onToggleToolRegistry?: () => void;
}

export function Header({ modelName, showToolRegistry, onToggleToolRegistry }: HeaderProps) {
  return (
    <header className="h-10 px-4 border-b border-zinc-800 bg-zinc-950 flex items-center justify-between shrink-0">
      {/* Left: Logo */}
      <div className="flex items-center gap-2">
        <Hexagon className="w-5 h-5 text-violet-400" strokeWidth={2} />
        <span className="font-mono font-bold text-sm tracking-widest text-zinc-100">
          ARTIFEX
        </span>
        <span className="text-xs text-zinc-500 ml-2">by Artifex</span>
      </div>

      {/* Right: Tools toggle + Model badge */}
      <div className="flex items-center gap-2">
        {onToggleToolRegistry && (
          <button
            onClick={onToggleToolRegistry}
            className={`p-1.5 rounded transition-colors ${
              showToolRegistry
                ? "bg-violet-500/20 text-violet-400"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
            title="Toggle Tool Registry"
          >
            <Wrench className="w-3.5 h-3.5" />
          </button>
        )}
        <span className="bg-zinc-800 text-violet-400 text-xs font-mono px-2 py-0.5 rounded">
          {modelName}
        </span>
      </div>
    </header>
  );
}
