/**
 * SimBridge — WebSocket client connecting the MuJoCo sim to the ForgeBot server.
 *
 * Connects to ws://localhost:8000/ws/sim, receives commands from the agent,
 * executes them against the live MujocoSim instance, and sends results back.
 */

import * as THREE from 'three';
import { MujocoSim } from './MujocoSim';
import { ToolLoader } from './ToolLoader';
import { getName } from './utils/StringUtils';

const WS_URL = 'ws://localhost:8000/ws/sim';
const RECONNECT_DELAY_MS = 3000;

type CommandHandler = (cmd: Record<string, unknown>) => Promise<Record<string, unknown>>;

export class SimBridge {
    private ws: WebSocket | null = null;
    private sim: MujocoSim | null = null;
    private toolLoader: ToolLoader | null = null;
    private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    private disposed = false;
    private handlers: Record<string, CommandHandler> = {};

    constructor() {
        this.handlers = {
            get_all_objects: (cmd) => this.handleGetAllObjects(cmd),
            get_body_position: (cmd) => this.handleGetBodyPosition(cmd),
            get_body_color: (cmd) => this.handleGetBodyColor(cmd),
            move_to: (cmd) => this.handleMoveTo(cmd),
            set_gripper: (cmd) => this.handleSetGripper(cmd),
            pick_up: (cmd) => this.handlePickUp(cmd),
            place_at: (cmd) => this.handlePlaceAt(cmd),
            step: (cmd) => this.handleStep(cmd),
            eval_tool: (cmd) => this.handleEvalTool(cmd),
        };
    }

    /** Attach a live sim instance (call after sim.init resolves). */
    attach(sim: MujocoSim) {
        this.sim = sim;
        this.toolLoader = new ToolLoader(sim);
        this.connect();
    }

    private connect() {
        if (this.disposed) return;
        try {
            this.ws = new WebSocket(WS_URL);
        } catch {
            this.scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            console.log('[SimBridge] connected to server');
        };

        this.ws.onmessage = async (ev) => {
            let cmd: Record<string, unknown>;
            try {
                cmd = JSON.parse(ev.data as string);
            } catch {
                return;
            }
            const action = cmd.action as string | undefined;
            if (!action) return;

            const handler = this.handlers[action];
            let result: Record<string, unknown>;
            if (handler) {
                try {
                    result = await handler(cmd);
                } catch (e) {
                    result = { success: false, error: String(e) };
                }
            } else {
                result = { success: false, error: `Unknown action: ${action}` };
            }

            this.send({ type: 'command_result', result });
        };

        this.ws.onclose = () => {
            console.log('[SimBridge] disconnected');
            this.scheduleReconnect();
        };

        this.ws.onerror = () => {
            this.ws?.close();
        };
    }

    private send(data: Record<string, unknown>) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    private scheduleReconnect() {
        if (this.disposed) return;
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        this.reconnectTimer = setTimeout(() => this.connect(), RECONNECT_DELAY_MS);
    }

    dispose() {
        this.disposed = true;
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        this.ws?.close();
        this.ws = null;
    }

    // ─── Helpers ──────────────────────────────────────────

    private findBodyId(bodyName: string): number | null {
        const sim = this.sim;
        if (!sim?.mjModel) return null;
        for (let i = 0; i < sim.mjModel.nbody; i++) {
            if (getName(sim.mjModel, sim.mjModel.name_bodyadr[i]) === bodyName) return i;
        }
        return null;
    }

    private rgbaToColorName(r: number, g: number, b: number): string {
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

    private getBodyGeomColor(bodyId: number): string {
        const m = this.sim?.mjModel;
        if (!m) return 'unknown';
        for (let g = 0; g < m.ngeom; g++) {
            if (m.geom_bodyid[g] === bodyId) {
                const r = m.geom_rgba[g * 4];
                const gr = m.geom_rgba[g * 4 + 1];
                const b = m.geom_rgba[g * 4 + 2];
                return this.rgbaToColorName(r, gr, b);
            }
        }
        return 'unknown';
    }

    private getBodyPosition(bodyId: number): [number, number, number] {
        const d = this.sim?.mjData;
        if (!d) return [0, 0, 0];
        return [d.xpos[bodyId * 3], d.xpos[bodyId * 3 + 1], d.xpos[bodyId * 3 + 2]];
    }

    /** Wait for N animation frames (lets the sim step and IK converge). */
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

    // ─── Command handlers ─────────────────────────────────

    private async handleGetAllObjects(_cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjModel || !sim.mjData) return { success: false, error: 'Sim not ready' };

        const objects: Array<Record<string, unknown>> = [];
        for (let i = 0; i < sim.mjModel.nbody; i++) {
            const name = getName(sim.mjModel, sim.mjModel.name_bodyadr[i]);
            if (!name.startsWith('cube')) continue;
            const pos = this.getBodyPosition(i);
            const color = this.getBodyGeomColor(i);
            objects.push({ name, position: pos, color, size: [0.02, 0.02, 0.02] });
        }
        return { success: true, objects };
    }

    private async handleGetBodyPosition(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const bodyName = cmd.body_name as string;
        const bodyId = this.findBodyId(bodyName);
        if (bodyId === null) return { success: false, error: `Body '${bodyName}' not found` };
        return { success: true, position: this.getBodyPosition(bodyId) };
    }

    private async handleGetBodyColor(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const bodyName = cmd.body_name as string;
        const bodyId = this.findBodyId(bodyName);
        if (bodyId === null) return { success: false, error: `Body '${bodyName}' not found` };
        return { success: true, color: this.getBodyGeomColor(bodyId) };
    }

    private async handleMoveTo(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjData) return { success: false, error: 'Sim not ready' };

        const target = cmd.target as number[];
        const pos = new THREE.Vector3(target[0], target[1], target[2]);
        sim.moveIkTargetTo(pos, 1500);
        sim.setIkEnabled(true);
        await this.waitFrames(120); // ~2 seconds at 60fps
        return { success: true, position: target };
    }

    private async handleSetGripper(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjData || sim.gripperActuatorId < 0) return { success: false, error: 'Gripper not available' };

        const open = cmd.open as boolean;
        sim.mjData.ctrl[sim.gripperActuatorId] = open ? 0.08 : 0.0;
        await this.waitFrames(60); // ~1 second to settle
        return { success: true, gripper_state: open ? 'open' : 'closed' };
    }

    private async handlePickUp(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjModel || !sim.mjData) return { success: false, error: 'Sim not ready' };

        const bodyName = cmd.body_name as string;
        const bodyId = this.findBodyId(bodyName);
        if (bodyId === null) return { success: false, error: `Body '${bodyName}' not found` };

        const pos = new THREE.Vector3(
            sim.mjData.xpos[bodyId * 3],
            sim.mjData.xpos[bodyId * 3 + 1],
            sim.mjData.xpos[bodyId * 3 + 2]
        );

        // Use the existing pickup sequence
        return new Promise((resolve) => {
            sim.pickupItems([pos], [0], () => {
                resolve({ success: true, holding: bodyName });
            });
        });
    }

    private async handlePlaceAt(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjData) return { success: false, error: 'Sim not ready' };

        const target = cmd.target as number[];
        const pos = new THREE.Vector3(target[0], target[1], target[2]);
        sim.moveIkTargetTo(pos, 1500);
        sim.setIkEnabled(true);
        await this.waitFrames(120);

        // Open gripper to release
        if (sim.gripperActuatorId >= 0) {
            sim.mjData.ctrl[sim.gripperActuatorId] = 0.08;
        }
        await this.waitFrames(60);

        return { success: true, placed_at: target };
    }

    private async handleStep(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjModel || !sim.mjData) return { success: false, error: 'Sim not ready' };

        const nSteps = (cmd.n_steps as number) ?? 100;
        for (let i = 0; i < nSteps; i++) {
            sim.mujoco.mj_step(sim.mjModel, sim.mjData);
        }
        return { success: true, sim_time: sim.mjData.time };
    }

    private async handleEvalTool(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        if (!this.toolLoader) return { success: false, error: 'ToolLoader not initialized' };

        const toolMjcf = cmd.tool_mjcf as string ?? '';
        const waypoints = cmd.waypoints as number[][] ?? [];
        const taskHint = cmd.task_hint as string ?? '';

        return await this.toolLoader.evaluate(toolMjcf, waypoints, taskHint);
    }
}
