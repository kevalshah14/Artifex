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

        // 1. Snapshot initial state
        const initialState = this.captureSnapshot();

        // 2. Inject tool into scene and reload
        // For now, we skip actual MJCF injection (requires scene XML rewrite + model reload)
        // Instead, we execute the waypoints with the existing end-effector
        // and measure the effect on objects. This tests the action strategy.
        // Full MJCF injection will require storing the base scene XML and modifying it.

        // 3. Execute waypoints sequentially
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
            return {
                success: false,
                fitness: 0,
                final_state: this.captureSnapshot(),
                error: `Waypoint execution failed: ${e}`,
            };
        }

        // 4. Capture final state and compute fitness
        const finalState = this.captureSnapshot();
        const fitness = this.computeFitness(initialState, finalState, taskHint);

        // 5. Reset sim for next candidate
        sim.reset();
        await this.waitFrames(30);

        return {
            success: true,
            fitness,
            final_state: finalState,
        };
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

        let totalScore = 0;
        let count = 0;

        for (const initObj of initial.objects) {
            const finalObj = final_.objects.find(o => o.name === initObj.name);
            if (!finalObj) continue;

            const dx = finalObj.position[0] - initObj.position[0];
            const dy = finalObj.position[1] - initObj.position[1];
            const dz = finalObj.position[2] - initObj.position[2];
            const displacement = Math.sqrt(dx * dx + dy * dy + dz * dz);

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

        return count > 0 ? Math.min(totalScore / count, 1.0) : 0;
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
