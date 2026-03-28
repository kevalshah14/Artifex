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

const WS_URL = import.meta.env.VITE_WS_URL ? `${import.meta.env.VITE_WS_URL}/ws/sim` : 'ws://localhost:8000/ws/sim';
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
            grasp: (cmd) => this.handleGrasp(cmd),
            place_at: (cmd) => this.handlePlaceAt(cmd),
            step: (cmd) => this.handleStep(cmd),
            eval_tool: (cmd) => this.handleEvalTool(cmd),
            capture_image: (cmd) => this.handleCaptureImage(cmd),
            add_object: (cmd) => this.handleAddObject(cmd),
            add_custom_object: (cmd) => this.handleAddCustomObject(cmd),
            remove_body: (cmd) => this.handleRemoveBody(cmd),
            set_body_color: (cmd) => this.handleSetBodyColor(cmd),
            move_body: (cmd) => this.handleMoveBody(cmd),
            reset_scene: () => this.handleResetScene(),
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

            console.log(`[SimBridge] >>> Received command: ${action}`, JSON.stringify(cmd).slice(0, 300));

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

        // Collect body IDs that have a free joint (= manipulable objects)
        const freeJointBodies = new Set<number>();
        for (let j = 0; j < sim.mjModel.njnt; j++) {
            if (sim.mjModel.jnt_type[j] === 0) { // mjJNT_FREE
                freeJointBodies.add(sim.mjModel.jnt_bodyid[j]);
            }
        }

        const GEOM_TYPES: Record<number, string> = { 0: 'plane', 2: 'sphere', 3: 'capsule', 4: 'ellipsoid', 5: 'cylinder', 6: 'box' };

        const objects: Array<Record<string, unknown>> = [];
        for (const bodyId of freeJointBodies) {
            const name = getName(sim.mjModel, sim.mjModel.name_bodyadr[bodyId]);
            if (!name || name === 'world') continue;
            const pos = this.getBodyPosition(bodyId);
            const color = this.getBodyGeomColor(bodyId);

            let shape = 'box';
            for (let g = 0; g < sim.mjModel.ngeom; g++) {
                if (sim.mjModel.geom_bodyid[g] === bodyId) {
                    shape = GEOM_TYPES[sim.mjModel.geom_type[g]] ?? 'unknown';
                    break;
                }
            }

            objects.push({ name, shape, position: pos, color });
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
        console.log(`[SimBridge] handleSetGripper: open=${open}, gripperActuatorId=${sim.gripperActuatorId}`);
        sim.mjData.ctrl[sim.gripperActuatorId] = open ? 0.08 : 0.0;
        console.log(`[SimBridge] handleSetGripper: ctrl[${sim.gripperActuatorId}] = ${open ? 0.08 : 0.0}`);
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

    private async handleGrasp(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
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

        console.log(`[SimBridge] handleGrasp: bodyName=${bodyName}, bodyId=${bodyId}`);
        console.log(`[SimBridge] handleGrasp: body position from xpos = (${pos.x.toFixed(4)}, ${pos.y.toFixed(4)}, ${pos.z.toFixed(4)})`);
        console.log(`[SimBridge] handleGrasp: using graspObjectById (livePosition mode) for accurate depth tracking`);

        // Grasp and hold (no tray placement) — use body ID for live position tracking
        return new Promise((resolve) => {
            sim.graspObjectById(bodyId, () => {
                // Verify grasp by checking if object lifted
                const newZ = sim.mjData!.xpos[bodyId * 3 + 2];
                const lifted = newZ > pos.z + 0.05;
                console.log(`[SimBridge] handleGrasp COMPLETE: bodyName=${bodyName}, original_z=${pos.z.toFixed(4)}, new_z=${newZ.toFixed(4)}, lifted=${lifted}`);
                console.log(`[SimBridge] handleGrasp: z_delta=${(newZ - pos.z).toFixed(4)}, threshold=0.05`);
                resolve({
                    success: lifted,
                    holding: lifted ? bodyName : null,
                    original_z: pos.z,
                    new_z: newZ,
                    message: lifted ? `Successfully grasped ${bodyName}` : `Grasp failed — ${bodyName} did not lift (z_delta=${(newZ - pos.z).toFixed(4)})`
                });
            });
        });
    }

    private async handlePlaceAt(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjData) return { success: false, error: 'Sim not ready' };

        const target = cmd.target as number[];
        const APPROACH_HEIGHT = 0.12;

        // 1) Move above the target (approach from above to avoid collisions)
        const above = new THREE.Vector3(target[0], target[1], target[2] + APPROACH_HEIGHT);
        sim.moveIkTargetTo(above, 1500);
        sim.setIkEnabled(true);
        await this.waitFrames(120);

        // 2) Descend to the actual placement height
        const place = new THREE.Vector3(target[0], target[1], target[2]);
        sim.moveIkTargetTo(place, 1000);
        await this.waitFrames(90);

        // 3) Open gripper to release
        if (sim.gripperActuatorId >= 0) {
            sim.mjData.ctrl[sim.gripperActuatorId] = 0.08;
            console.log(`[SimBridge] handlePlaceAt: gripper opened (ctrl[${sim.gripperActuatorId}] = 0.08)`);
        }
        await this.waitFrames(45);

        // 4) Retract upward so the arm doesn't knock the placed object
        sim.moveIkTargetTo(above, 800);
        await this.waitFrames(60);

        console.log(`[SimBridge] handlePlaceAt: COMPLETE at (${target[0].toFixed(4)}, ${target[1].toFixed(4)}, ${target[2].toFixed(4)})`);
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

    private async handleCaptureImage(_cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.renderSys) return { success: false, error: 'Sim not ready' };

        try {
            const dataUrl = sim.renderSys.getCanvasSnapshot(640, 480, 'image/jpeg');
            const base64 = dataUrl.replace(/^data:image\/\w+;base64,/, '');
            return { success: true, image_base64: base64, mime_type: 'image/jpeg' };
        } catch (e) {
            return { success: false, error: `Screenshot failed: ${e}` };
        }
    }

    // ─── Virtual FS helpers ──────────────────────────────

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private get FS(): any { return (this.sim!.mujoco as any).FS; }

    private readWorkingText(path: string): string {
        const raw = this.FS.readFile(path, { encoding: 'utf8' });
        return typeof raw === 'string' ? raw : new TextDecoder().decode(raw);
    }

    private writeWorkingText(path: string, text: string) {
        this.FS.writeFile(path, text);
    }

    // ─── Scene modification commands ─────────────────────

    private colorNameToRgba(color: string): [number, number, number, number] {
        const map: Record<string, [number, number, number, number]> = {
            red:     [0.9, 0.15, 0.15, 1],
            green:   [0.15, 0.85, 0.25, 1],
            blue:    [0.15, 0.25, 0.9, 1],
            yellow:  [0.95, 0.85, 0.1, 1],
            cyan:    [0.1, 0.85, 0.85, 1],
            purple:  [0.7, 0.15, 0.85, 1],
            orange:  [1.0, 0.55, 0.1, 1],
            white:   [0.95, 0.95, 0.95, 1],
            black:   [0.12, 0.12, 0.12, 1],
            pink:    [1.0, 0.4, 0.7, 1],
        };
        return map[color.toLowerCase()] ?? [0.5, 0.5, 0.5, 1];
    }

    /**
     * Build the MuJoCo `size` attribute for a geom.
     *
     *  shape      | size meaning
     *  -----------|---------------------------------------------------
     *  box        | half-extents: "sx sy sz"
     *  sphere     | radius: "r"
     *  cylinder   | radius + half-height: "r h"
     *  capsule    | radius + half-height: "r h"
     *  ellipsoid  | semi-axes: "rx ry rz"
     */
    private geomSizeAttr(shape: string, size: number | number[]): string {
        const s = typeof size === 'number' ? size : size[0];
        switch (shape) {
            case 'sphere':    return `${s}`;
            case 'cylinder':  return `${s} ${s}`;
            case 'capsule':   return `${s} ${s * 1.5}`;
            case 'ellipsoid': return `${s} ${s} ${s * 1.5}`;
            case 'box':
            default:          return `${s} ${s} ${s}`;
        }
    }

    private async handleAddObject(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjModel) return { success: false, error: 'Sim not ready' };

        const name  = cmd.name as string ?? `obj_${Date.now()}`;
        const pos   = cmd.position as number[] ?? [0, 0, 0.05];
        const color = cmd.color as string ?? 'red';
        const size  = (cmd.size as number) ?? 0.02;
        const shape = (cmd.shape as string ?? 'box').toLowerCase();
        const rgba  = this.colorNameToRgba(color);

        const VALID_SHAPES = ['box', 'sphere', 'cylinder', 'capsule', 'ellipsoid'];
        if (!VALID_SHAPES.includes(shape)) {
            return { success: false, error: `Unsupported shape '${shape}'. Use: ${VALID_SHAPES.join(', ')}` };
        }

        try {
            const sceneXml = this.readWorkingText('/working/scene.xml');

            const sizeAttr = this.geomSizeAttr(shape, size);
            const bodyXml =
                `    <body name="${name}" pos="${pos[0]} ${pos[1]} ${pos[2]}">\n` +
                `      <freejoint/>\n` +
                `      <geom type="${shape}" size="${sizeAttr}" rgba="${rgba.join(' ')}" mass="0.05" condim="4" friction="1 0.5 0.01"/>\n` +
                `    </body>\n`;

            const modified = sceneXml.replace('</worldbody>', bodyXml + '  </worldbody>');
            this.writeWorkingText('/working/scene.xml', modified);
            sim.reloadFromWorkingScene();
            await this.waitFrames(20);

            return { success: true, added: name, shape, position: pos, color };
        } catch (e) {
            return { success: false, error: `Failed to add object: ${e}` };
        }
    }

    private async handleAddCustomObject(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjModel) return { success: false, error: 'Sim not ready' };

        const name = cmd.name as string;
        const pos  = cmd.position as number[] ?? [0, 0, 0.05];
        const innerXml = cmd.body_xml as string;

        if (!name)     return { success: false, error: 'Missing name' };
        if (!innerXml) return { success: false, error: 'Missing body_xml' };

        try {
            const sceneXml = this.readWorkingText('/working/scene.xml');

            const bodyXml =
                `    <body name="${name}" pos="${pos[0]} ${pos[1]} ${pos[2]}">\n` +
                `      <freejoint/>\n` +
                `      ${innerXml}\n` +
                `    </body>\n`;

            const modified = sceneXml.replace('</worldbody>', bodyXml + '  </worldbody>');
            this.writeWorkingText('/working/scene.xml', modified);
            sim.reloadFromWorkingScene();
            await this.waitFrames(20);

            return { success: true, added: name, position: pos };
        } catch (e) {
            return { success: false, error: `Failed to add custom object: ${e}` };
        }
    }

    private async handleRemoveBody(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjModel) return { success: false, error: 'Sim not ready' };

        const name = cmd.body_name as string;
        if (!name) return { success: false, error: 'Missing body_name' };

        try {
            const sceneXml = this.readWorkingText('/working/scene.xml');

            // Match the full <body name="NAME" ...> ... </body> block
            const pattern = new RegExp(
                `\\s*<body\\s+name="${name}"[^>]*>[\\s\\S]*?</body>`,
                'm'
            );
            const modified = sceneXml.replace(pattern, '');

            if (modified === sceneXml) {
                return { success: false, error: `Body '${name}' not found in scene XML` };
            }

            this.writeWorkingText('/working/scene.xml', modified);
            sim.reloadFromWorkingScene();
            await this.waitFrames(20);

            return { success: true, removed: name };
        } catch (e) {
            return { success: false, error: `Failed to remove body: ${e}` };
        }
    }

    private async handleSetBodyColor(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjModel) return { success: false, error: 'Sim not ready' };

        const name = cmd.body_name as string;
        const color = cmd.color as string;
        if (!name || !color) return { success: false, error: 'Missing body_name or color' };

        const bodyId = this.findBodyId(name);
        if (bodyId === null) return { success: false, error: `Body '${name}' not found` };

        const rgba = this.colorNameToRgba(color);

        for (let g = 0; g < sim.mjModel.ngeom; g++) {
            if (sim.mjModel.geom_bodyid[g] === bodyId) {
                sim.mjModel.geom_rgba[g * 4]     = rgba[0];
                sim.mjModel.geom_rgba[g * 4 + 1] = rgba[1];
                sim.mjModel.geom_rgba[g * 4 + 2] = rgba[2];
                sim.mjModel.geom_rgba[g * 4 + 3] = rgba[3];
            }
        }

        return { success: true, body_name: name, new_color: color };
    }

    private async handleMoveBody(cmd: Record<string, unknown>): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjModel || !sim.mjData) return { success: false, error: 'Sim not ready' };

        const name = cmd.body_name as string;
        const pos = cmd.position as number[];
        if (!name || !pos) return { success: false, error: 'Missing body_name or position' };

        const bodyId = this.findBodyId(name);
        if (bodyId === null) return { success: false, error: `Body '${name}' not found` };

        // Find the free joint for this body and set its qpos
        for (let j = 0; j < sim.mjModel.njnt; j++) {
            if (sim.mjModel.jnt_bodyid[j] === bodyId && sim.mjModel.jnt_type[j] === 0) {
                // type 0 = mjJNT_FREE: qpos = [x, y, z, qw, qx, qy, qz]
                const addr = sim.mjModel.jnt_qposadr[j];
                sim.mjData.qpos[addr]     = pos[0];
                sim.mjData.qpos[addr + 1] = pos[1];
                sim.mjData.qpos[addr + 2] = pos[2];
                sim.mujoco.mj_forward(sim.mjModel, sim.mjData);
                await this.waitFrames(5);
                return { success: true, body_name: name, new_position: pos };
            }
        }

        return { success: false, error: `Body '${name}' has no free joint — cannot teleport` };
    }

    private async handleResetScene(): Promise<Record<string, unknown>> {
        const sim = this.sim;
        if (!sim?.mjModel) return { success: false, error: 'Sim not ready' };

        sim.reset();
        await this.waitFrames(30);

        return { success: true, message: 'Scene reset to initial state' };
    }
}
