/**
 * ToolLoader — Loads MJCF tool geometry into the live MuJoCo sim,
 * executes action waypoints, and computes fitness scores.
 *
 * This is the browser-side evaluator for VLMgineer geometry evolution.
 *
 * Flow:
 *   1. Receive MJCF tool XML + waypoints from server
 *   2. Inject tool body into the scene XML
 *   3. Reload the MuJoCo model
 *   4. Execute waypoints via IK
 *   5. Measure object positions → compute fitness
 *   6. Reset sim to clean state
 *   7. Return fitness score
 */

import * as THREE from 'three';
import { MujocoSim } from './MujocoSim';
import { getName } from './utils/StringUtils';

/** Snapshot of object positions for fitness comparison. */
interface SceneSnapshot {
    objects: Array<{ name: string; position: [number, number, number]; color: string }>;
}

export class ToolLoader {
    private sim: MujocoSim;
    private originalSceneXml: string | null = null;
    private originalPandaXml: string | null = null;
    private pandaXmlPath: string | null = null;

    constructor(sim: MujocoSim) {
        this.sim = sim;
    }

    /**
     * Evaluate a tool design: inject geometry, run waypoints, measure fitness.
     *
     * @param toolMjcf - MJCF XML body element to inject as an end-effector tool
     * @param waypoints - Array of [x, y, z] world positions to move through
     * @param taskHint - Optional task description for fitness heuristics
     * @returns Result with fitness score (0-1) and final state
     */
    async evaluate(
        toolMjcf: string,
        waypoints: number[][],
        taskHint: string = ''
    ): Promise<{ success: boolean; fitness: number; final_state: SceneSnapshot; error?: string }> {
        const sim = this.sim;
        if (!sim.mjModel || !sim.mjData) {
            return { success: false, fitness: 0, final_state: { objects: [] }, error: 'Sim not ready' };
        }

        // 1) Inject tool geometry into model files and reload.
        try {
            this.ensureOriginalXmlCached();
            this.injectToolIntoPandaXml(toolMjcf);
            sim.reloadFromWorkingScene();
        } catch (e) {
            return {
                success: false,
                fitness: 0,
                final_state: { objects: [] },
                error: `Tool injection failed: ${String(e)}`,
            };
        }

        // 2) Snapshot post-injection initial state.
        const initialState = this.captureSnapshot();

        // 3) Execute waypoints sequentially.
        try {
            for (const wp of waypoints) {
                if (wp.length < 3) continue;
                const pos = new THREE.Vector3(wp[0], wp[1], wp[2]);
                sim.moveIkTargetTo(pos, 800);
                sim.setIkEnabled(true);
                await this.waitFrames(60); // ~1 second per waypoint
            }

            // Let physics settle
            await this.waitFrames(30);

        } catch (e) {
            this.restoreOriginalXml();
            sim.reloadFromWorkingScene();
            return {
                success: false,
                fitness: 0,
                final_state: this.captureSnapshot(),
                error: `Waypoint execution failed: ${e}`,
            };
        }

        // 4) Capture final state and compute fitness.
        const finalState = this.captureSnapshot();
        const fitness = this.computeFitness(initialState, finalState, taskHint);

        // 5) Restore original no-tool XML and reload for next candidate.
        this.restoreOriginalXml();
        sim.reloadFromWorkingScene();
        await this.waitFrames(15);

        return {
            success: true,
            fitness,
            final_state: finalState,
        };
    }

    private ensureOriginalXmlCached() {
        if (!this.originalSceneXml) {
            this.originalSceneXml = this.readWorkingText('/working/scene.xml');
        }
        if (!this.originalPandaXml) {
            const pandaCandidates = ['/working/assets/panda.xml', '/working/panda.xml'];
            for (const path of pandaCandidates) {
                try {
                    const txt = this.readWorkingText(path);
                    this.originalPandaXml = txt;
                    this.pandaXmlPath = path;
                    break;
                } catch {
                    // Try next candidate path.
                }
            }
        }
        if (!this.originalPandaXml || !this.pandaXmlPath) {
            throw new Error('Unable to locate panda.xml in /working');
        }
    }

    private injectToolIntoPandaXml(toolMjcf: string) {
        if (!this.originalPandaXml || !this.pandaXmlPath) {
            throw new Error('Tool XML cache not initialized');
        }
        const cleanTool = toolMjcf.trim();
        if (!cleanTool) {
            throw new Error('Empty tool_mjcf');
        }

        // Normalize candidate payload: allow body snippet or geom snippet.
        const toolSnippet = cleanTool.includes('<body')
            ? cleanTool
            : `<body name="vlmgineer_tool"><geom type="box" size="0.01 0.01 0.08" pos="0 0 0.08" mass="0.01" rgba="0.6 0.2 0.9 1"/>${cleanTool}</body>`;

        const markerStart = '<!-- VLMGINEER_TOOL_START -->';
        const markerEnd = '<!-- VLMGINEER_TOOL_END -->';
        const wrappedTool = `${markerStart}\n${toolSnippet}\n${markerEnd}`;

        const injected = this.originalPandaXml.replace(
            /(<body[^>]*name=["']hand["'][^>]*>)/,
            `$1\n${wrappedTool}\n`,
        );

        this.writeWorkingText(this.pandaXmlPath, injected);
    }

    private restoreOriginalXml() {
        if (this.originalSceneXml) {
            this.writeWorkingText('/working/scene.xml', this.originalSceneXml);
        }
        if (this.originalPandaXml && this.pandaXmlPath) {
            this.writeWorkingText(this.pandaXmlPath, this.originalPandaXml);
        }
    }

    private readWorkingText(path: string): string {
        // MuJoCo Emscripten FS exposes readFile at runtime, though not typed in our slim TS interface.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const fsAny = this.sim.mujoco.FS as any;
        const raw = fsAny.readFile(path, { encoding: 'utf8' });
        return typeof raw === 'string' ? raw : new TextDecoder().decode(raw);
    }

    private writeWorkingText(path: string, text: string) {
        this.sim.mujoco.FS.writeFile(path, text);
    }

    /** Capture positions of all manipulable objects. */
    private captureSnapshot(): SceneSnapshot {
        const sim = this.sim;
        if (!sim.mjModel || !sim.mjData) return { objects: [] };

        const objects: SceneSnapshot['objects'] = [];
        for (let i = 0; i < sim.mjModel.nbody; i++) {
            const name = getName(sim.mjModel, sim.mjModel.name_bodyadr[i]);
            if (!name.startsWith('cube')) continue;

            const pos: [number, number, number] = [
                sim.mjData.xpos[i * 3],
                sim.mjData.xpos[i * 3 + 1],
                sim.mjData.xpos[i * 3 + 2],
            ];

            // Get color
            let color = 'unknown';
            for (let g = 0; g < sim.mjModel.ngeom; g++) {
                if (sim.mjModel.geom_bodyid[g] === i) {
                    const r = sim.mjModel.geom_rgba[g * 4];
                    const gr = sim.mjModel.geom_rgba[g * 4 + 1];
                    const b = sim.mjModel.geom_rgba[g * 4 + 2];
                    if (r > 0.5 && gr < 0.3 && b < 0.3) color = 'red';
                    else if (r < 0.3 && gr > 0.5 && b > 0.5) color = 'cyan';
                    else if (r < 0.3 && gr > 0.5 && b < 0.3) color = 'green';
                    else if (r > 0.5 && gr > 0.5 && b < 0.3) color = 'yellow';
                    break;
                }
            }

            objects.push({ name, position: pos, color });
        }
        return { objects };
    }

    /**
     * Compute fitness score (0-1) by comparing initial and final states.
     *
     * Heuristics:
     * - Did any object move significantly? (general manipulation success)
     * - Did objects move upward? (lifting/picking tasks)
     * - Did objects move toward a target zone? (sorting/placement tasks)
     */
    private computeFitness(
        initial: SceneSnapshot,
        final_: SceneSnapshot,
        taskHint: string
    ): number {
        if (initial.objects.length === 0 || final_.objects.length === 0) return 0;

        const hint = taskHint.toLowerCase();
        const wantsLeft = /\bleft\b/.test(hint);
        const wantsRight = /\bright\b/.test(hint);
        const wantsLift = /(lift|elevate|raise|up|stack|pick)/.test(hint);
        const wantsGather = /(gather|group|cluster|together|collect)/.test(hint);
        const wantsSpread = /(clean|separate|spread|disperse|away)/.test(hint);

        let totalScore = 0;
        let count = 0;
        const deltas: Array<{ dx: number; dy: number; dz: number; displacement: number }> = [];
        const finalPos: Array<[number, number, number]> = [];

        for (const initObj of initial.objects) {
            const finalObj = final_.objects.find(o => o.name === initObj.name);
            if (!finalObj) continue;

            const dx = finalObj.position[0] - initObj.position[0];
            const dy = finalObj.position[1] - initObj.position[1];
            const dz = finalObj.position[2] - initObj.position[2];
            const displacement = Math.sqrt(dx * dx + dy * dy + dz * dz);
            deltas.push({ dx, dy, dz, displacement });
            finalPos.push(finalObj.position);

            // Base score: did the object move at all? (0-0.5)
            const moveScore = Math.min(displacement / 0.3, 1.0) * 0.5;

            // Bonus: did it move upward? (picking/lifting) (0-0.3)
            const liftScore = dz > 0.02 ? Math.min(dz / 0.2, 1.0) * 0.3 : 0;

            // Bonus: did it move toward center/robot? (fetching) (0-0.2)
            const initDist = Math.sqrt(
                initObj.position[0] ** 2 + initObj.position[1] ** 2
            );
            const finalDist = Math.sqrt(
                finalObj.position[0] ** 2 + finalObj.position[1] ** 2
            );
            const fetchScore = finalDist < initDist
                ? Math.min((initDist - finalDist) / 0.3, 1.0) * 0.2
                : 0;

            totalScore += moveScore + liftScore + fetchScore;
            count++;
        }

        if (count === 0) return 0;

        // Directional shaping from task language.
        let taskBonus = 0;
        if (wantsLeft) {
            const meanLeft = deltas.reduce((s, d) => s + d.dy, 0) / count;
            taskBonus += Math.max(0, Math.min(meanLeft / 0.25, 1)) * 0.25;
        }
        if (wantsRight) {
            const meanRight = deltas.reduce((s, d) => s - d.dy, 0) / count;
            taskBonus += Math.max(0, Math.min(meanRight / 0.25, 1)) * 0.25;
        }
        if (wantsLift) {
            const meanLift = deltas.reduce((s, d) => s + Math.max(0, d.dz), 0) / count;
            taskBonus += Math.max(0, Math.min(meanLift / 0.2, 1)) * 0.25;
        }

        // Cluster/spread shaping.
        if (finalPos.length > 1) {
            let pairwise = 0;
            let pairs = 0;
            for (let i = 0; i < finalPos.length; i++) {
                for (let j = i + 1; j < finalPos.length; j++) {
                    const dx = finalPos[i][0] - finalPos[j][0];
                    const dy = finalPos[i][1] - finalPos[j][1];
                    pairwise += Math.sqrt(dx * dx + dy * dy);
                    pairs++;
                }
            }
            const meanPairwise = pairs > 0 ? pairwise / pairs : 0;
            if (wantsGather) {
                // Smaller pairwise distance is better for grouping.
                taskBonus += Math.max(0, Math.min((0.25 - meanPairwise) / 0.25, 1)) * 0.2;
            } else if (wantsSpread) {
                // Larger pairwise distance is better for spreading/clearing.
                taskBonus += Math.max(0, Math.min(meanPairwise / 0.35, 1)) * 0.2;
            }
        }

        const base = totalScore / count;
        return Math.min(base * 0.75 + taskBonus, 1.0);
    }

    /** Wait for N animation frames. */
    private waitFrames(n: number): Promise<void> {
        return new Promise((resolve) => {
            let count = 0;
            const tick = () => {
                if (++count >= n) { resolve(); return; }
                requestAnimationFrame(tick);
            };
            requestAnimationFrame(tick);
        });
    }
}
