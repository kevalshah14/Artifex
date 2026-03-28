/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { GoogleGenAI } from "@google/genai";
import { AlertCircle, Loader2, X } from 'lucide-react';
import loadMujoco from 'mujoco-js';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { v4 as uuidv4 } from 'uuid';
import { MujocoSim } from './MujocoSim';
import { SimBridge } from './SimBridge';
import { ChatPanel } from './components/ChatPanel';
import { Header } from './components/Header';
import { RobotSelector } from './components/RobotSelector';
import { SimulationHUD } from './components/SimulationHUD';
import { Toolbar } from './components/Toolbar';
import { ToolRegistry } from './components/ToolRegistry';
import { UnifiedSidebar } from './components/UnifiedSidebar';
import { DetectedItem, DetectType, LogEntry, MujocoModule } from './types';

/**
 * Default prompt parts for different detection types.
 */
export const defaultPromptParts = {
  '2D bounding boxes': [
    'Detect',
    'items',
    ', with no more than 25 items. DO NOT detect items that only match the description partially. Output a json list where each entry contains the 2D bounding box in "box_2d" and a text label in "label".',
  ],
  'Segmentation masks': [
    `Give the segmentation masks for`,
    'all objects',
    `. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label". Use descriptive labels."`,
  ],
  'Points': [
    'Identify ',
    'items',
    ' in the scene and mark them with points. DO NOT mark items that only match the description partially. Follow the JSON format: [{"point": [y, x], "label": "label"}, ...]. The points are in [y, x] format normalized to 0-1000.',
  ],
};

interface LogOverlayProps {
  log: LogEntry;
}

/**
 * LogOverlay
 * Draws Gemini detection results (boxes/points) over an image.
 * Uses a normalized 1000x1000 coordinate system.
 */
export function LogOverlay({ log }: LogOverlayProps) {
  if (!log.result || !Array.isArray(log.result)) return null;

  const results = log.result as DetectedItem[];
  const shapes = results.map((item, idx) => {
    if (item.box_2d) {
      const [ymin, xmin, ymax, xmax] = item.box_2d;
      return (
        <rect
          key={idx} x={xmin} y={ymin} width={xmax - xmin} height={ymax - ymin}
          fill="rgba(79, 70, 229, 0.15)" stroke="#4f46e5" strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      );
    } else if (item.point) {
      const [y, x] = item.point;
      // Using vector-effect="non-scaling-stroke" ensures the circle border is visible even in small miniatures.
      // cx/cy are normalized 0-1000.
      return <circle key={idx} cx={x} cy={y} r="10" fill="#4f46e5" stroke="white" strokeWidth="2" vectorEffect="non-scaling-stroke" />;
    }
    return null;
  });

  return (
    <svg
      viewBox="0 0 1000 1000"
      preserveAspectRatio="none"
      className="absolute inset-0 pointer-events-none w-full h-full z-10"
    >
      {shapes}
    </svg>
  );
}

/**
 * Main Application Component
 */
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

  const [isPaused, setIsPaused] = useState(false);
  // Initialize sidebar based on screen width (hidden on mobile by default)
  const [showSidebar, setShowSidebar] = useState(() => window.innerWidth >= 660);

  const [erLoading, setErLoading] = useState(false);
  const [logs, setLogs] = useState<Array<LogEntry>>([]);
  const [expandedLogId, setExpandedLogId] = useState<string | null>(null);
  const [flash, setFlash] = useState(false);
  const detectedTargets = useRef<Array<{pos: THREE.Vector3, markerId: number}>>([]);
  const [detectedCount, setDetectedCount] = useState(0);

  const [isPickingUp, setIsPickingUp] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  const [gizmoStats, setGizmoStats] = useState<{pos: string, rot: string} | null>(null);
  const [fps, setFps] = useState(0);
  const [showToolRegistry, setShowToolRegistry] = useState(false);

  // Deriving activeLog directly from the latest logs state ensures UI reactivity
  const activeLog = expandedLogId ? logs.find(l => l.id === expandedLogId) : null;

  useEffect(() => {
    isMounted.current = true;
    loadMujoco({
      locateFile: (path: string) => path.endsWith('.wasm') ? "https://unpkg.com/mujoco-js@0.0.7/dist/mujoco_wasm.wasm" : path,
      printErr: (text: string) => {
        if (text.includes("Aborted") && isMounted.current) {
            setLoadError(prev => prev ? prev : "Simulation crashed. Reload page.");
        }
      }
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
    return () => { isMounted.current = false; bridgeRef.current?.dispose(); simRef.current?.dispose(); };
  }, []);

  useEffect(() => {
      if (!mujocoReady || !containerRef.current || !mujocoModuleRef.current) return;
      setIsLoading(true);
      setLoadError(null);
      setIsPaused(false);

      simRef.current?.dispose();

      try {
          simRef.current = new MujocoSim(containerRef.current, mujocoModuleRef.current);
          simRef.current.renderSys.setDarkMode(true);

          simRef.current.init("franka_panda_stack", "scene.xml", (msg) => {
             if (isMounted.current) setLoadingStatus(msg);
          })
             .then(() => {
                 if (isMounted.current) {
                     simRef.current?.setIkEnabled(false);
                     setIsLoading(false);
                     // Connect sim to the ForgeBot server
                     bridgeRef.current?.dispose();
                     const bridge = new SimBridge();
                     bridge.attach(simRef.current!);
                     bridgeRef.current = bridge;
                 }
             })
             .catch(err => {
                 if (isMounted.current) {
                     setLoadError(err.message);
                     setIsLoading(false);
                 }
             });

      } catch (err: unknown) {
          if (isMounted.current) { setLoadError((err as Error).message); setIsLoading(false); }
      }
  }, [mujocoReady]);

  // Effect to move camera when sidebar toggles
  useEffect(() => {
    if (isLoading || !simRef.current || erLoading) return;

    // Standard view when sidebar is closed
    const standardPos = new THREE.Vector3(2.2, -1.2, 2.2);
    const standardTarget = new THREE.Vector3(0, 0, 0);

    // Offset view to shift robot left when sidebar is open
    const offsetPos = new THREE.Vector3(2.35, -0.7, 2.2);
    const offsetTarget = new THREE.Vector3(0.15, 0.4, 0.05);

    // Only offset camera on desktop/tablet (width >= 660px). On mobile, keep centered.
    if (showSidebar && window.innerWidth >= 660) {
      simRef.current.renderSys.moveCameraTo(offsetPos, offsetTarget, 1000);
    } else {
      simRef.current.renderSys.moveCameraTo(standardPos, standardTarget, 1000);
    }
  }, [showSidebar, isLoading, erLoading]);

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
              setGizmoStats(s ? {
                  pos: `X: ${s.pos.x.toFixed(2)}, Y: ${s.pos.y.toFixed(2)}, Z: ${s.pos.z.toFixed(2)}`,
                  rot: `X: ${s.rot.x.toFixed(2)}, Y: ${s.rot.y.toFixed(2)}, Z: ${s.rot.z.toFixed(2)}`
              } : null);
          }
          animId = requestAnimationFrame(uiLoop);
      };
      uiLoop();
      return () => cancelAnimationFrame(animId);
  }, [isLoading]);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
        if (simRef.current && !isLoading && !erLoading) {
            const markerPos = simRef.current.renderSys.checkMarkerClick(e.clientX, e.clientY);
            if (markerPos) {
                simRef.current.moveIkTargetTo(markerPos, 2000);
                simRef.current.setIkEnabled(true);
            }
        }
    };
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [isLoading, erLoading]);

  const handleErSend = async (prompt: string, type: DetectType, temperature: number, enableThinking: boolean, modelId: string) => {
      if (!simRef.current || erLoading) return;
      setErLoading(true);
      simRef.current.renderSys.clearErMarkers();
      detectedTargets.current = [];
      setDetectedCount(0);
      setIsPickingUp(false);
      setPlaybackSpeed(1);

      const savedState = simRef.current.renderSys.getCameraState();
      const topPos = new THREE.Vector3(0, -0.01, 2.0);
      const target = new THREE.Vector3(0, 0, 0);
      await simRef.current.renderSys.moveCameraTo(topPos, target, 1500);
      await new Promise(r => setTimeout(r, 100));

      setFlash(true);
      setTimeout(() => setFlash(false), 100);

      // Dynamic Resizing: Limit max dimension to 640px while preserving aspect ratio.
      const canvas = simRef.current.renderSys.renderer.domElement;
      const width = canvas.width;
      const height = canvas.height;
      const scaleFactor = Math.min(640 / width, 640 / height);
      const snapshotWidth = Math.floor(width * scaleFactor);
      const snapshotHeight = Math.floor(height * scaleFactor);

      // Serialization: Convert to PNG.
      const imageBase64 = simRef.current.renderSys.getCanvasSnapshot(snapshotWidth, snapshotHeight, 'image/png');
      // Payload Preparation: Strip data URI prefix.
      const base64Data = imageBase64.replace('data:image/png;base64,', '');

      const parts = defaultPromptParts[type];
      const subject = prompt.trim() || parts[1];
      const textPrompt = `${parts[0]} ${subject}${parts[2]}`;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const config: any = {
          temperature,
          responseMimeType: "application/json",
      };

      if (!enableThinking) {
          config.thinkingConfig = { thinkingBudget: 0 };
      }

      const requestLogData = {
          model: modelId,
          contents: {
              parts: [
                  { inlineData: { data: "<IMAGE>", mimeType: "image/png" } },
                  { text: textPrompt }
              ]
          },
          config: config
      };

      const logId = uuidv4();
      const newLog: LogEntry = {
          id: logId,
          timestamp: new Date(),
          imageSrc: imageBase64,
          prompt,
          fullPrompt: textPrompt,
          type,
          result: null,
          requestData: requestLogData
      };
      setLogs(prev => [newLog, ...prev]);

      await simRef.current.renderSys.moveCameraTo(savedState.position, savedState.target, 1500);

      try {
          const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
          const response = await ai.models.generateContent({
              model: modelId,
              contents: {
                  parts: [
                      { inlineData: { mimeType: 'image/png', data: base64Data } },
                      { text: textPrompt }
                  ]
              },
              // tslint:disable-next-line:no-any
              config: config
          });

          const text = response.text;
          if (!text) throw new Error("No response text returned.");

          let jsonText = text.replace(/```json|```/g, '').trim();
          const firstBracket = jsonText.indexOf('[');
          const lastBracket = jsonText.lastIndexOf(']');
          if (firstBracket !== -1 && lastBracket !== -1) {
              jsonText = jsonText.substring(firstBracket, lastBracket + 1);
          }

          let result;
          try { result = JSON.parse(jsonText); } catch (e) { result = []; }

          // Remove absolute duplicates
          if (Array.isArray(result)) {
              const seen = new Set();
              result = result.filter((item: unknown) => {
                  const serialized = JSON.stringify(item);
                  if (seen.has(serialized)) return false;
                  seen.add(serialized);
                  return true;
              });
          }

          setLogs(prev => prev.map(l => l.id === logId ? { ...l, result } : l));

          if (Array.isArray(result)) {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              result.forEach((item: any) => {
                  let center2d: {x: number, y: number} | null = null;
                  if (item.box_2d) {
                      const [ymin, xmin, ymax, xmax] = item.box_2d;
                      center2d = { x: (xmin + xmax) / 2, y: (ymin + ymax) / 2 };
                  } else if (item.point) {
                      const [y, x] = item.point;
                      center2d = { x, y };
                  }

                  if (center2d) {
                      const projection = simRef.current?.renderSys.project2DTo3D(center2d.x, center2d.y, topPos, target);
                      if (projection) {
                          const markerId = Date.now() + Math.random();
                          simRef.current?.renderSys.addErMarker(projection.point, item.label, markerId);
                          detectedTargets.current.push({ pos: projection.point, markerId });
                      }
                  }
              });
              setDetectedCount(detectedTargets.current.length);
          }
      } catch (error: unknown) {
          console.error("Gemini API Error", error);
          const errorMsg = (error as Error).message || "Unknown error";
          setLogs(prev => prev.map(l => l.id === logId && l.result === null ? { ...l, result: { error: errorMsg } } : l));
      } finally {
          setErLoading(false);
      }
  };

  const handlePickup = () => {
    if (simRef.current) {
        // If already picking up, this button acts as a speed toggle
        if (isPickingUp) {
            let nextSpeed = 1;
            if (playbackSpeed === 1) nextSpeed = 2;
            else if (playbackSpeed === 2) nextSpeed = 5;
            else if (playbackSpeed === 5) nextSpeed = 10;
            else if (playbackSpeed === 10) nextSpeed = 20;
            else if (playbackSpeed === 20) nextSpeed = 2; // Cycle back to 2x for continuous fast forward feeling

            setPlaybackSpeed(nextSpeed);
            simRef.current.setSpeedMultiplier(nextSpeed);
            return;
        }

        // Otherwise start the pickup sequence
        if (detectedTargets.current.length > 0) {
            setIsPickingUp(true);
            setPlaybackSpeed(1);
            const positions = detectedTargets.current.map(t => t.pos);
            const markerIds = detectedTargets.current.map(t => t.markerId);

            simRef.current.pickupItems(positions, markerIds, () => {
                // On Finished
                setIsPickingUp(false);
                setPlaybackSpeed(1);
                setDetectedCount(0); // Deactivates the button
                detectedTargets.current = [];
                simRef.current?.setSpeedMultiplier(1);
            });
        }
    }
  };

  const handleReset = () => {
    simRef.current?.reset();
    setLogs([]);
    setDetectedCount(0);
    setIsPickingUp(false);
    setPlaybackSpeed(1);
    detectedTargets.current = [];
  };

  // Compute HUD coordinates from gizmoStats
  const hudCoordinates = gizmoStats ? {
    x: gizmoStats.pos.split(',')[0]?.replace('X: ', '').trim() ?? '0.00',
    y: gizmoStats.pos.split(',')[1]?.replace('Y: ', '').trim() ?? '0.00',
    z: gizmoStats.pos.split(',')[2]?.replace('Z: ', '').trim() ?? '0.00',
  } : null;

  return (
    <div className="h-screen flex flex-col bg-zinc-950 text-zinc-50 font-sans overflow-hidden">
      {/* Header */}
      <Header
        modelName="gemini-2.5-flash"
        showToolRegistry={showToolRegistry}
        onToggleToolRegistry={() => setShowToolRegistry(!showToolRegistry)}
      />

      {/* Main content area */}
      <div className="flex flex-1 min-h-0">
        {/* Left panel: Sim viewport */}
        <div className="w-[58%] relative bg-black">
          {/* 3D Container */}
          <div ref={containerRef} className="w-full h-full absolute inset-0" />

          {/* Simulation HUD */}
          <SimulationHUD fps={fps} coordinates={hudCoordinates} isLoaded={!isLoading && !loadError} />

          {/* Robot Info Overlay */}
          {!loadError && <RobotSelector gizmoStats={gizmoStats} />}

          {/* UnifiedSidebar as overlay inside viewport */}
          {!isLoading && !loadError && (
            <UnifiedSidebar
              isOpen={showSidebar}
              onClose={() => setShowSidebar(false)}
              onSend={handleErSend}
              onPickup={handlePickup}
              isLoading={erLoading}
              hasDetectedItems={detectedCount > 0}
              logs={logs}
              onOpenLog={(log) => setExpandedLogId(log.id)}
              isPickingUp={isPickingUp}
              playbackSpeed={playbackSpeed}
            />
          )}

          {/* Toolbar */}
          {!isLoading && !loadError && (
            <Toolbar
              isPaused={isPaused}
              togglePause={() => setIsPaused(simRef.current?.togglePause() ?? false)}
              onReset={handleReset}
              showSidebar={showSidebar}
              toggleSidebar={() => setShowSidebar(!showSidebar)}
            />
          )}

          {/* Loading Screen */}
          {isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center z-50 backdrop-blur-md px-6 bg-zinc-950/40">
              <div className="flex flex-col min-[660px]:flex-row gap-8 max-w-4xl w-full items-stretch">
                <div className="glass-panel p-12 rounded-[3rem] flex-1 flex flex-col justify-center shadow-2xl transition-colors bg-zinc-900/70 border border-zinc-800">
                  <h3 className="text-sm font-bold uppercase tracking-widest mb-4 text-violet-400">System Overview</h3>
                  <p className="text-sm leading-relaxed mb-6 text-zinc-300">
                    This demo showcases spatial reasoning for robotics. Using <strong>Gemini Robotics Embodied Reasoning 1.5</strong>, the system analyzes a 2D image to identify objects and calculate manipulation coordinates.
                  </p>
                  <ul className="text-[13px] space-y-3 list-disc list-inside text-zinc-400">
                    <li>Real-time MuJoCo physics simulation</li>
                    <li>Analytical Inverse Kinematics for Franka Panda</li>
                    <li>Call Gemini Robotics Embodied Reasoning 1.5 for detection</li>
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

          {/* Flash Effect */}
          {flash && <div className="absolute inset-0 bg-white z-[60] pointer-events-none opacity-50" />}

          {/* Error State */}
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
          {/* Chat panel — top 60% */}
          <div className="flex-[6] min-h-0 border-b border-zinc-800">
            <ChatPanel />
          </div>
          {/* Tool Registry — bottom 40% */}
          <div className="flex-[4] min-h-0 bg-zinc-900 overflow-hidden">
            <ToolRegistry />
          </div>
        </div>
      </div>

      {/* Expanded View Modal - Overlay everything */}
      {activeLog && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center min-[660px]:p-10 bg-zinc-950/60 backdrop-blur-xl animate-in fade-in" onClick={() => setExpandedLogId(null)}>
          <div className="glass-panel overflow-hidden flex flex-col shadow-2xl transition-colors fixed top-4 bottom-4 left-4 right-4 rounded-[2.5rem] min-[660px]:relative min-[660px]:inset-auto min-[660px]:w-full min-[660px]:max-w-4xl min-[660px]:max-h-[85vh] bg-zinc-900 border border-zinc-800 text-zinc-100" onClick={e => e.stopPropagation()}>
            <div className="p-6 border-b flex justify-between items-center shrink-0 border-zinc-800 bg-zinc-900/50">
              <div>
                <h3 className="text-xl font-bold">API Call</h3>
                <p className="text-xs font-medium text-zinc-400">{activeLog.timestamp.toLocaleString()}</p>
              </div>
              <button onClick={() => setExpandedLogId(null)} className="w-10 h-10 flex items-center justify-center rounded-full shadow-sm border transition-colors bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 flex max-[659px]:flex-col max-[659px]:overflow-y-auto custom-scrollbar min-[660px]:flex-row min-[660px]:overflow-hidden">
              <div className="flex items-center justify-center border-b min-[660px]:border-b-0 min-[660px]:border-r min-[660px]:flex-1 min-[660px]:p-6 min-[660px]:overflow-hidden max-[659px]:shrink-0 max-[659px]:p-6 bg-zinc-950/50 border-zinc-800">
                <div className="relative rounded-2xl overflow-hidden shadow-lg border-2 flex items-center justify-center min-[660px]:w-auto min-[660px]:h-auto min-[660px]:max-w-full min-[660px]:max-h-full max-[659px]:w-full max-[659px]:h-auto border-zinc-800 bg-black/20">
                  <img src={activeLog.imageSrc} className="block w-full h-auto min-[660px]:max-w-full min-[660px]:max-h-full" alt="Detailed log" />
                  <LogOverlay log={activeLog} />
                </div>
              </div>
              <div className="min-[660px]:w-[320px] p-6 flex flex-col gap-5 min-[660px]:overflow-y-auto min-[660px]:custom-scrollbar bg-zinc-900/50">
                <div className="space-y-1">
                  <h4 className="text-[9px] font-bold text-zinc-400 uppercase tracking-widest">User Prompt</h4>
                  <p className="text-sm font-bold leading-tight">{activeLog.prompt}</p>
                </div>
                <div className="space-y-1">
                  <h4 className="text-[9px] font-bold text-zinc-400 uppercase tracking-widest">Full Prompt</h4>
                  <p className="text-[10px] font-mono p-3 r
                                ounded-xl leading-relaxed border whitespace-pre-wrap bg-zinc-950 border-zinc-800 text-zinc-400">{activeLog.fullPrompt}</p>
                </div>
                <div className="space-y-3 flex flex-col min-[660px]:flex-1 min-[660px]:min-h-0">
                  <h4 className="text-[9px] font-bold text-zinc-400 uppercase tracking-widest">API Call Results</h4>
                  <div className="p-3 rounded-xl font-mono text-[10px] border overflow-y-auto shadow-inner min-[660px]:flex-1 max-[659px]:h-96 bg-zinc-950 border-zinc-800 text-violet-400">
                    {activeLog.result === null ? (
                      <div className="h-full flex flex-col items-center justify-center gap-3 text-violet-400 animate-pulse">
                        <Loader2 className="w-6 h-6 animate-spin" />
                        <span className="font-sans font-bold text-[8px] uppercase tracking-widest">Processing...</span>
                      </div>
                    ) : (
                      <pre className="whitespace-pre-wrap break-all leading-relaxed">{JSON.stringify(activeLog.result, null, 2)}</pre>
                    )}
                  </div>
                </div>
                <div className="min-[660px]:hidden h-8 shrink-0" />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}