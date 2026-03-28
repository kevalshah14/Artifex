/**
 * robotTools — Gemini function calling declarations and executor for
 * direct in-browser robot control via the MujocoSim instance.
 *
 * Mirrors the command set exposed by SimBridge over WebSocket, but runs
 * entirely client-side so no external server is required.
 */

import { Type } from '@google/genai';
import * as THREE from 'three';
import { MujocoSim } from './MujocoSim';
import { getName } from './utils/StringUtils';

// ─── Helpers ──────────────────────────────────────────────────────────

function waitFrames(n: number): Promise<void> {
    return new Promise((resolve) => {
        let count = 0;
        const tick = () => {
            if (++count >= n) { resolve(); return; }
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    });
}

function findBodyId(sim: MujocoSim, bodyName: string): number | null {
    if (!sim.mjModel) return null;
    for (let i = 0; i < sim.mjModel.nbody; i++) {
        if (getName(sim.mjModel, sim.mjModel.name_bodyadr[i]) === bodyName) return i;
    }
    return null;
}

function rgbaToColorName(r: number, g: number, b: number): string {
    if (r > 0.5 && g < 0.3 && b < 0.3) return 'red';
    if (r < 0.3 && g > 0.5 && b > 0.5) return 'cyan';
    if (r < 0.3 && g > 0.5 && b < 0.3) return 'green';
    if (r > 0.5 && g > 0.5 && b < 0.3) return 'yellow';
    if (r > 0.5 && g > 0.5 && b > 0.5) return 'white';
    if (r < 0.2 && g < 0.2 && b < 0.2) return 'black';
    if (r > 0.5 && g < 0.3 && b > 0.5) return 'purple';
    if (r < 0.3 && g < 0.3 && b > 0.5) return 'blue';
    return `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
}

function getBodyGeomColor(sim: MujocoSim, bodyId: number): string {
    const m = sim.mjModel;
    if (!m) return 'unknown';
    for (let g = 0; g < m.ngeom; g++) {
        if (m.geom_bodyid[g] === bodyId) {
            return rgbaToColorName(
                m.geom_rgba[g * 4],
                m.geom_rgba[g * 4 + 1],
                m.geom_rgba[g * 4 + 2],
            );
        }
    }
    return 'unknown';
}

function getBodyPosition(sim: MujocoSim, bodyId: number): [number, number, number] {
    const d = sim.mjData;
    if (!d) return [0, 0, 0];
    return [d.xpos[bodyId * 3], d.xpos[bodyId * 3 + 1], d.xpos[bodyId * 3 + 2]];
}

// ─── Gemini function declarations ────────────────────────────────────

export const robotFunctionDeclarations = [
    {
        name: 'get_all_objects',
        description:
            'Lists every manipulable object currently on the table. Returns each object\'s body name, world position [x,y,z], and color label. Use this first to discover what is in the scene before acting.',
        parameters: {
            type: Type.OBJECT,
            properties: {},
        },
    },
    {
        name: 'get_body_position',
        description:
            'Returns the current world position [x,y,z] of a specific body by name (e.g. "cube0", "cube1").',
        parameters: {
            type: Type.OBJECT,
            properties: {
                body_name: {
                    type: Type.STRING,
                    description: 'The MuJoCo body name, e.g. "cube0".',
                },
            },
            required: ['body_name'],
        },
    },
    {
        name: 'get_body_color',
        description: 'Returns the color label of a specific body by name.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                body_name: {
                    type: Type.STRING,
                    description: 'The MuJoCo body name.',
                },
            },
            required: ['body_name'],
        },
    },
    {
        name: 'move_to',
        description:
            'Moves the robot arm end-effector to a target position in world coordinates. The arm uses inverse kinematics to reach the target. Takes ~2 seconds.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                x: { type: Type.NUMBER, description: 'X world coordinate.' },
                y: { type: Type.NUMBER, description: 'Y world coordinate.' },
                z: { type: Type.NUMBER, description: 'Z world coordinate. Use ~0.05 for table level, higher to hover above.' },
            },
            required: ['x', 'y', 'z'],
        },
    },
    {
        name: 'set_gripper',
        description:
            'Opens or closes the robot parallel gripper. Open to release objects, close to grasp them.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                open: {
                    type: Type.BOOLEAN,
                    description: 'true to open the gripper, false to close it.',
                },
            },
            required: ['open'],
        },
    },
    {
        name: 'pick_up',
        description:
            'Runs the full automated pick-up sequence for a named object: approaches, grasps, and lifts it. This is a high-level command that handles the entire motion. Takes several seconds.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                body_name: {
                    type: Type.STRING,
                    description: 'Name of the body to pick up, e.g. "cube0".',
                },
            },
            required: ['body_name'],
        },
    },
    {
        name: 'place_at',
        description:
            'Moves the robot arm to the given position and opens the gripper to release whatever it is holding. Use after pick_up to place an object somewhere.',
        parameters: {
            type: Type.OBJECT,
            properties: {
                x: { type: Type.NUMBER, description: 'X world coordinate to place at.' },
                y: { type: Type.NUMBER, description: 'Y world coordinate to place at.' },
                z: { type: Type.NUMBER, description: 'Z world coordinate to place at.' },
            },
            required: ['x', 'y', 'z'],
        },
    },
    {
        name: 'reset_simulation',
        description:
            'Resets the entire simulation: returns the robot arm to its home position and re-randomizes object positions on the table.',
        parameters: {
            type: Type.OBJECT,
            properties: {},
        },
    },
];

// ─── Executor ────────────────────────────────────────────────────────

export type ToolResult = Record<string, unknown>;

/**
 * Execute a robot tool call against the live MujocoSim instance.
 * Returns a JSON-serialisable result object that gets sent back to Gemini
 * as a functionResponse.
 */
export async function executeRobotTool(
    sim: MujocoSim,
    name: string,
    args: Record<string, unknown>,
): Promise<ToolResult> {
    if (!sim.mjModel || !sim.mjData) {
        return { success: false, error: 'Simulation not ready' };
    }

    switch (name) {
        case 'get_all_objects': {
            const objects: Array<Record<string, unknown>> = [];
            for (let i = 0; i < sim.mjModel.nbody; i++) {
                const bodyName = getName(sim.mjModel, sim.mjModel.name_bodyadr[i]);
                if (!bodyName.startsWith('cube')) continue;
                objects.push({
                    name: bodyName,
                    position: getBodyPosition(sim, i),
                    color: getBodyGeomColor(sim, i),
                    size: [0.02, 0.02, 0.02],
                });
            }
            return { success: true, objects };
        }

        case 'get_body_position': {
            const bodyId = findBodyId(sim, args.body_name as string);
            if (bodyId === null) return { success: false, error: `Body '${args.body_name}' not found` };
            return { success: true, position: getBodyPosition(sim, bodyId) };
        }

        case 'get_body_color': {
            const bodyId = findBodyId(sim, args.body_name as string);
            if (bodyId === null) return { success: false, error: `Body '${args.body_name}' not found` };
            return { success: true, color: getBodyGeomColor(sim, bodyId) };
        }

        case 'move_to': {
            const pos = new THREE.Vector3(
                args.x as number,
                args.y as number,
                args.z as number,
            );
            sim.moveIkTargetTo(pos, 1500);
            sim.setIkEnabled(true);
            await waitFrames(120);
            return { success: true, position: [args.x, args.y, args.z] };
        }

        case 'set_gripper': {
            if (sim.gripperActuatorId < 0) {
                return { success: false, error: 'Gripper not available' };
            }
            const open = args.open as boolean;
            sim.mjData.ctrl[sim.gripperActuatorId] = open ? 0.08 : 0.0;
            await waitFrames(60);
            return { success: true, gripper_state: open ? 'open' : 'closed' };
        }

        case 'pick_up': {
            const bodyId = findBodyId(sim, args.body_name as string);
            if (bodyId === null) return { success: false, error: `Body '${args.body_name}' not found` };

            const pos = new THREE.Vector3(
                sim.mjData.xpos[bodyId * 3],
                sim.mjData.xpos[bodyId * 3 + 1],
                sim.mjData.xpos[bodyId * 3 + 2],
            );

            return new Promise<ToolResult>((resolve) => {
                sim.pickupItems([pos], [0], () => {
                    resolve({ success: true, holding: args.body_name });
                });
            });
        }

        case 'place_at': {
            const pos = new THREE.Vector3(
                args.x as number,
                args.y as number,
                args.z as number,
            );
            sim.moveIkTargetTo(pos, 1500);
            sim.setIkEnabled(true);
            await waitFrames(120);

            if (sim.gripperActuatorId >= 0) {
                sim.mjData.ctrl[sim.gripperActuatorId] = 0.08;
            }
            await waitFrames(60);
            return { success: true, placed_at: [args.x, args.y, args.z] };
        }

        case 'reset_simulation': {
            sim.reset();
            await waitFrames(30);
            return { success: true, message: 'Simulation reset' };
        }

        default:
            return { success: false, error: `Unknown tool: ${name}` };
    }
}
