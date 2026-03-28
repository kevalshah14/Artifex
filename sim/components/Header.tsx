import { Hexagon } from "lucide-react";

interface HeaderProps {
  modelName: string;
}

export function Header({ modelName }: HeaderProps) {
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

      {/* Right: Model badge */}
      <div className="flex items-center gap-2">
        <span className="bg-zinc-800 text-violet-400 text-xs font-mono px-2 py-0.5 rounded">
          {modelName}
        </span>
      </div>
    </header>
  );
}
