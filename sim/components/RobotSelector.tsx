/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

interface RobotSelectorProps {
  gizmoStats: { pos: string; rot: string } | null;
}

/**
 * RobotSelector
 * Overlay displaying current robot info. Dark-only styling.
 */
export function RobotSelector({ gizmoStats }: RobotSelectorProps) {
  return (
    <div className="absolute top-10 left-1/2 -translate-x-1/2 min-[660px]:left-10 min-[660px]:translate-x-0 z-20 flex flex-col gap-4">
      <div className="glass-panel px-8 py-5 rounded-[2rem] min-w-[240px] shadow-2xl bg-zinc-900/80 border border-zinc-800 text-zinc-100">
        <h1 className="text-xl font-bold tracking-tight leading-none text-center">Franka Panda</h1>
      </div>

      {gizmoStats && (
        <div className="glass-card px-5 py-3 rounded-2xl flex flex-col gap-2 shadow-sm bg-zinc-800/60 border border-zinc-800">
          <div className="font-mono text-[9px] space-y-0.5">
            <p className="flex justify-between gap-4">
              <span className="text-zinc-400">POSITION:</span>
              <span className="text-zinc-300 font-semibold">{gizmoStats.pos}</span>
            </p>
            <p className="flex justify-between gap-4">
              <span className="text-zinc-400">ROTATION:</span>
              <span className="text-zinc-300 font-semibold">{gizmoStats.rot}</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
