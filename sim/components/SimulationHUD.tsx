import { Hexagon } from "lucide-react";

interface SimulationHUDProps {
  fps: number;
  coordinates: { x: string; y: string; z: string } | null;
  isLoaded: boolean;
}

export function SimulationHUD({ fps, coordinates, isLoaded }: SimulationHUDProps) {
  if (!isLoaded) {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10">
        <Hexagon
          className="w-12 h-12 text-violet-400 animate-pulse-soft"
          strokeWidth={1.5}
        />
        <p className="mt-4 text-sm text-zinc-500 font-mono">
          Initializing simulation...
        </p>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 pointer-events-none z-10">
      {/* TOP LEFT: SIM badge + FPS */}
      <div className="absolute top-3 left-3 flex items-center gap-2">
        <span className="text-xs text-green-400 font-mono font-semibold bg-zinc-950/70 px-1.5 py-0.5 rounded">
          SIM
        </span>
        <span className="text-xs text-green-400 font-mono bg-zinc-950/70 px-1.5 py-0.5 rounded">
          {fps} FPS
        </span>
      </div>

      {/* BOTTOM LEFT: XYZ coordinates */}
      {coordinates && (
        <div className="absolute bottom-3 left-3 flex items-center gap-2 text-xs text-violet-300 font-mono bg-zinc-950/70 px-2 py-1 rounded">
          <span>X: {coordinates.x}</span>
          <span>Y: {coordinates.y}</span>
          <span>Z: {coordinates.z}</span>
        </div>
      )}

      {/* BOTTOM RIGHT: Watermark */}
      <div className="absolute bottom-3 right-3">
        <span className="text-[10px] text-zinc-700 font-mono tracking-widest">
          ARTIFEX
        </span>
      </div>
    </div>
  );
}
