/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { AlertCircle, Loader2 } from 'lucide-react';
import loadMujoco from 'mujoco-js';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { MujocoSim } from './MujocoSim';
import { SimBridge } from './SimBridge';
import { ChatPanel } from './components/ChatPanel';
import { Header } from './components/Header';
import { RobotSelector } from './components/RobotSelector';
import { SimulationHUD } from './components/SimulationHUD';
import { ToolRegistry } from './components/ToolRegistry';
import { MujocoModule } from './types';

export function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const simRef = useRef<MujocoSim | null>(null);
  const bridgeRef = useRef<SimBridge | null>(null);
  const isMounted = useRef(true);
  const mujocoModuleRef = useRef<MujocoModule | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [loadingStatus, setLoadingStatus] = useState("Initializing Spatial Engine...");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [mujocoReady, setMujocoReady] = useState(false);
  const [gizmoStats, setGizmoStats] = useState<{ pos: string; rot: string } | null>(null);
  const [fps, setFps] = useState(0);

  // Load MuJoCo WASM
  useEffect(() => {
    isMounted.current = true;
    loadMujoco({
      locateFile: (path: string) =>
        path.endsWith('.wasm') ? "https://unpkg.com/mujoco-js@0.0.7/dist/mujoco_wasm.wasm" : path,
      printErr: (text: string) => {
        if (text.includes("Aborted") && isMounted.current) {
          setLoadError(prev => prev ? prev : "Simulation crashed. Reload page.");
        }
      },
    }).then((inst: unknown) => {
      if (isMounted.current) {
        mujocoModuleRef.current = inst as MujocoModule;
        setMujocoReady(true);
      }
    }).catch((err: Error) => {
      if (isMounted.current) {
        setLoadError(err.message || "Failed to init spatial simulation");
        setIsLoading(false);
      }
    });
    return () => {
      isMounted.current = false;
      bridgeRef.current?.dispose();
      simRef.current?.dispose();
    };
  }, []);

  // Init sim + attach bridge to backend
  useEffect(() => {
    if (!mujocoReady || !containerRef.current || !mujocoModuleRef.current) return;
    setIsLoading(true);
    setLoadError(null);

    simRef.current?.dispose();

    try {
      simRef.current = new MujocoSim(containerRef.current, mujocoModuleRef.current);
      simRef.current.renderSys.setDarkMode(true);

      simRef.current
        .init("franka_panda_stack", "scene.xml", (msg) => {
          if (isMounted.current) setLoadingStatus(msg);
        })
        .then(() => {
          if (isMounted.current) {
            simRef.current?.setIkEnabled(false);
            setIsLoading(false);
            bridgeRef.current?.dispose();
            const bridge = new SimBridge();
            bridge.attach(simRef.current!);
            bridgeRef.current = bridge;
          }
        })
        .catch((err) => {
          if (isMounted.current) {
            setLoadError(err.message);
            setIsLoading(false);
          }
        });
    } catch (err: unknown) {
      if (isMounted.current) {
        setLoadError((err as Error).message);
        setIsLoading(false);
      }
    }
  }, [mujocoReady]);

  // FPS + gizmo stats loop
  useEffect(() => {
    if (isLoading) return;
    let animId: number;
    let lastTime = performance.now();
    let frameCount = 0;
    const uiLoop = () => {
      frameCount++;
      const now = performance.now();
      if (now - lastTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastTime = now;
      }
      if (simRef.current) {
        const s = simRef.current.getGizmoStats();
        setGizmoStats(
          s
            ? {
                pos: `X: ${s.pos.x.toFixed(2)}, Y: ${s.pos.y.toFixed(2)}, Z: ${s.pos.z.toFixed(2)}`,
                rot: `X: ${s.rot.x.toFixed(2)}, Y: ${s.rot.y.toFixed(2)}, Z: ${s.rot.z.toFixed(2)}`,
              }
            : null,
        );
      }
      animId = requestAnimationFrame(uiLoop);
    };
    uiLoop();
    return () => cancelAnimationFrame(animId);
  }, [isLoading]);

  // Click on 3D markers to move robot
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (simRef.current && !isLoading) {
        const markerPos = simRef.current.renderSys.checkMarkerClick(e.clientX, e.clientY);
        if (markerPos) {
          simRef.current.moveIkTargetTo(markerPos, 2000);
          simRef.current.setIkEnabled(true);
        }
      }
    };
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [isLoading]);

  const hudCoordinates = gizmoStats
    ? {
        x: gizmoStats.pos.split(',')[0]?.replace('X: ', '').trim() ?? '0.00',
        y: gizmoStats.pos.split(',')[1]?.replace('Y: ', '').trim() ?? '0.00',
        z: gizmoStats.pos.split(',')[2]?.replace('Z: ', '').trim() ?? '0.00',
      }
    : null;

  return (
    <div className="h-screen flex flex-col bg-zinc-950 text-zinc-50 font-sans overflow-hidden">
      <Header modelName="gemini-robotics-er" />

      <div className="flex flex-1 min-h-0">
        {/* Sim viewport */}
        <div className="w-[58%] relative bg-black">
          <div ref={containerRef} className="w-full h-full absolute inset-0" />
          <SimulationHUD fps={fps} coordinates={hudCoordinates} isLoaded={!isLoading && !loadError} />
          {!loadError && <RobotSelector gizmoStats={gizmoStats} />}

          {isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center z-50 backdrop-blur-md px-6 bg-zinc-950/40">
              <div className="flex flex-col min-[660px]:flex-row gap-8 max-w-4xl w-full items-stretch">
                <div className="glass-panel p-12 rounded-[3rem] flex-1 flex flex-col justify-center shadow-2xl transition-colors bg-zinc-900/70 border border-zinc-800">
                  <h3 className="text-sm font-bold uppercase tracking-widest mb-4 text-violet-400">System Overview</h3>
                  <p className="text-sm leading-relaxed mb-6 text-zinc-300">
                    This demo showcases spatial reasoning for robotics. Using <strong>Gemini</strong>, the agent analyzes the scene, plans actions, invents tools, and controls the robot arm.
                  </p>
                  <ul className="text-[13px] space-y-3 list-disc list-inside text-zinc-400">
                    <li>Real-time MuJoCo physics simulation</li>
                    <li>Analytical Inverse Kinematics for Franka Panda</li>
                    <li>ForgeBot agent with tool invention and evolution</li>
                  </ul>
                </div>
                <div className="glass-panel p-10 rounded-[3rem] flex flex-col items-center justify-center shrink-0 min-[660px]:w-[260px] shadow-2xl transition-colors bg-zinc-900/70 border border-zinc-800">
                  <div className="w-16 h-16 rounded-2xl bg-violet-600 flex items-center justify-center shadow-lg shadow-violet-500/20 animate-pulse-soft mb-6">
                    <Loader2 className="w-8 h-8 text-white animate-spin" />
                  </div>
                  <h2 className="text-base font-bold text-center px-2 text-zinc-100">{loadingStatus}</h2>
                </div>
              </div>
            </div>
          )}

          {loadError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/60 backdrop-blur-xl z-50">
              <div className="glass-panel p-10 rounded-[2.5rem] border border-red-900/50 max-w-md text-center bg-zinc-900">
                <div className="w-16 h-16 bg-red-900/30 text-red-400 rounded-full flex items-center justify-center mx-auto mb-6">
                  <AlertCircle className="w-8 h-8" />
                </div>
                <h3 className="text-2xl text-zinc-100 font-bold mb-2">Simulation Halted</h3>
                <p className="text-zinc-400 mb-8 leading-relaxed">{loadError}</p>
                <button
                  onClick={() => window.location.reload()}
                  className="w-full py-4 bg-zinc-100 text-zinc-900 rounded-2xl font-bold hover:bg-white transition-all shadow-xl active:scale-95"
                >
                  Restart System
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right panel: Chat + Tool Registry */}
        <div className="w-[42%] flex flex-col border-l border-zinc-800">
          <div className="flex-[6] min-h-0 border-b border-zinc-800">
            <ChatPanel />
          </div>
          <div className="flex-[4] min-h-0 bg-zinc-900 overflow-hidden">
            <ToolRegistry />
          </div>
        </div>
      </div>
    </div>
  );
}
