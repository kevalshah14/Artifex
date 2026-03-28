import { GoogleGenAI } from '@google/genai';
import { v4 as uuidv4 } from 'uuid';
import { MujocoSim } from './MujocoSim';
import { ToolLoader } from './ToolLoader';
import { useInventedToolsStore } from './inventedToolsStore';
import { getName } from './utils/StringUtils';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

interface Candidate {
  name: string;
  description: string;
  tool_mjcf: string;
  waypoints: number[][];
}

interface CandidateEval {
  candidate: Candidate;
  fitness: number;
  success: boolean;
  error?: string;
}

interface VLMgineerRunOptions {
  maxIterations?: number;
  candidatesPerIteration?: number;
  model?: string;
  solveThreshold?: number;
}

interface SceneObject {
  name: string;
  position: [number, number, number];
  color: string;
}

interface VLMgineerRunSummary {
  success: boolean;
  task: string;
  iterations: number;
  bestFitness: number;
  solved: boolean;
  bestToolName?: string;
  bestToolDescription?: string;
  bestToolMjcf?: string;
  bestWaypoints?: number[][];
  iterationResults: Array<{
    iteration: number;
    bestFitness: number;
    avgFitness: number;
    candidateCount: number;
  }>;
}

const runHistory: VLMgineerRunSummary[] = [];
const taskMemory = new Map<string, CandidateEval[]>();

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function captureSceneObjects(sim: MujocoSim): SceneObject[] {
  if (!sim.mjModel || !sim.mjData) return [];
  const objects: SceneObject[] = [];
  for (let i = 0; i < sim.mjModel.nbody; i++) {
    const name = getName(sim.mjModel, sim.mjModel.name_bodyadr[i]);
    if (!name.startsWith('cube')) continue;
    const pos: [number, number, number] = [
      sim.mjData.xpos[i * 3],
      sim.mjData.xpos[i * 3 + 1],
      sim.mjData.xpos[i * 3 + 2],
    ];
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
  return objects;
}

function sanitizeWaypoints(input: unknown): number[][] {
  if (!Array.isArray(input)) return [];
  return input
    .filter(
      (w): w is number[] =>
        Array.isArray(w) &&
        w.length >= 3 &&
        typeof w[0] === 'number' &&
        typeof w[1] === 'number' &&
        typeof w[2] === 'number',
    )
    .map((w) => [w[0], w[1], w[2]]);
}

function taskKey(task: string): string {
  return task
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 12)
    .join(' ');
}

function parseCandidates(text: string, fallbackCount: number): Candidate[] {
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = null;
  }
  const rawList = Array.isArray((parsed as { candidates?: unknown[] } | null)?.candidates)
    ? ((parsed as { candidates: unknown[] }).candidates)
    : [];
  const candidates: Candidate[] = rawList
    .map((raw, idx) => {
      const c = raw as Record<string, unknown>;
      return {
        name: (c.name as string) || `candidate_${idx + 1}`,
        description: (c.description as string) || 'Generated tool candidate',
        tool_mjcf: typeof c.tool_mjcf === 'string' ? c.tool_mjcf : '',
        waypoints: sanitizeWaypoints(c.waypoints),
      };
    })
    .filter((c) => c.waypoints.length > 0);

  if (candidates.length > 0) return candidates.slice(0, fallbackCount);

  // Safety fallback to keep the evolutionary loop moving.
  return Array.from({ length: fallbackCount }, (_, i) => ({
    name: `fallback_${i + 1}`,
    description: 'Fallback direct-push strategy',
    tool_mjcf:
      '<body name="tool_fallback"><geom type="box" size="0.02 0.02 0.08" pos="0 0 0.08" rgba="0.7 0.2 0.9 1"/></body>',
    waypoints: [
      [0.25, 0.0, 0.22],
      [0.35, 0.0, 0.22],
      [0.25, 0.0, 0.22],
    ],
  }));
}

async function generateCandidates(
  task: string,
  sceneObjects: SceneObject[],
  elites: CandidateEval[],
  priorMemory: CandidateEval[],
  count: number,
  model: string,
): Promise<Candidate[]> {
  const eliteSummary = elites
    .slice(0, 3)
    .map(
      (e, i) =>
        `${i + 1}. ${e.candidate.name} | fitness=${e.fitness.toFixed(3)} | ${e.candidate.description}${e.error ? ` | error=${e.error}` : ''}`,
    )
    .join('\n');

  const memorySummary = priorMemory
    .slice(0, 3)
    .map(
      (e, i) =>
        `${i + 1}. ${e.candidate.name} | fitness=${e.fitness.toFixed(3)} | ${e.candidate.description}`,
    )
    .join('\n');

  const prompt = `You are VLMGINEER for a tabletop Franka Panda simulation.
Design custom tool geometry + action waypoints to solve the task.
You MUST improve over previous attempts using mutation and crossover logic.

Task:
${task}

Scene objects:
${JSON.stringify(sceneObjects, null, 2)}

Elite previous designs (if any):
${eliteSummary || 'None'}

Cross-run memory from similar tasks:
${memorySummary || 'None'}

Return STRICT JSON with this exact schema:
{
  "candidates": [
    {
      "name": "short_name",
      "description": "what this tool is trying to do",
      "tool_mjcf": "<body ...>...</body>",
      "waypoints": [[x,y,z], [x,y,z], ...]
    }
  ]
}

Requirements:
- return exactly ${count} candidates.
- tool_mjcf must be a valid MJCF body snippet (single rigid body with geoms is fine).
- prioritize custom gripper-like geometries that can physically hook, scoop, cage, or push depending on task.
- waypoints length between 3 and 12.
- at least 1 mutation of top elite.
- at least 1 crossover between two elites or elite+memory design.
- include at least one aggressive strategy and one conservative strategy.
- no markdown, no code fence, JSON only.`;

  const response = await ai.models.generateContent({
    model,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: {
      temperature: 0.8,
      responseMimeType: 'application/json',
    },
  });

  return parseCandidates(response.text ?? '', count);
}

export async function runVLMgineer(
  sim: MujocoSim,
  task: string,
  options: VLMgineerRunOptions = {},
): Promise<VLMgineerRunSummary> {
  if (!sim.mjModel || !sim.mjData) {
    return {
      success: false,
      task,
      iterations: 0,
      bestFitness: 0,
      solved: false,
      iterationResults: [],
    };
  }

  const maxIterations = clamp(options.maxIterations ?? 3, 1, 8);
  const candidatesPerIteration = clamp(options.candidatesPerIteration ?? 4, 2, 10);
  const model = options.model ?? 'gemini-2.5-flash';
  const solveThreshold = clamp(options.solveThreshold ?? 0.78, 0.45, 0.98);

  const loader = new ToolLoader(sim);
  const sceneObjects = captureSceneObjects(sim);
  const memoryKey = taskKey(task);
  const priorMemory = taskMemory.get(memoryKey) ?? [];
  let elites: CandidateEval[] = [];
  let bestOverall: CandidateEval | null = null;
  const iterationResults: VLMgineerRunSummary['iterationResults'] = [];
  let noImprovementRounds = 0;
  let lastBest = -1;

  for (let iter = 0; iter < maxIterations; iter++) {
    const generated = await generateCandidates(
      task,
      sceneObjects,
      elites,
      priorMemory,
      candidatesPerIteration,
      model,
    );

    const evaluations: CandidateEval[] = [];
    for (const candidate of generated) {
      const evalResult = await loader.evaluate(candidate.tool_mjcf, candidate.waypoints, task);
      evaluations.push({
        candidate,
        fitness: evalResult.fitness,
        success: evalResult.success,
        error: evalResult.error,
      });
    }

    evaluations.sort((a, b) => b.fitness - a.fitness);
    elites = evaluations.slice(0, Math.min(3, evaluations.length));

    if (!bestOverall || (evaluations[0] && evaluations[0].fitness > bestOverall.fitness)) {
      bestOverall = evaluations[0] ?? bestOverall;
    }

    const avgFitness =
      evaluations.length > 0
        ? evaluations.reduce((sum, e) => sum + e.fitness, 0) / evaluations.length
        : 0;

    iterationResults.push({
      iteration: iter + 1,
      bestFitness: evaluations[0]?.fitness ?? 0,
      avgFitness,
      candidateCount: evaluations.length,
    });

    const iterBest = evaluations[0]?.fitness ?? 0;
    if (iterBest > lastBest + 0.02) {
      noImprovementRounds = 0;
      lastBest = iterBest;
    } else {
      noImprovementRounds++;
    }

    // Stop early when solved or plateauing for too long.
    if (iterBest >= solveThreshold || noImprovementRounds >= 2) {
      break;
    }
  }

  if (bestOverall) {
    useInventedToolsStore.getState().addInventedTool({
      id: uuidv4(),
      name: bestOverall.candidate.name,
      description: bestOverall.candidate.description,
      task,
      toolMjcf: bestOverall.candidate.tool_mjcf,
      waypoints: bestOverall.candidate.waypoints,
      bestFitness: bestOverall.fitness,
      iteration: iterationResults.length,
      createdAt: new Date().toISOString(),
    });
  }

  const summary: VLMgineerRunSummary = {
    success: Boolean(bestOverall && bestOverall.success),
    task,
    iterations: iterationResults.length,
    bestFitness: bestOverall?.fitness ?? 0,
    solved: (bestOverall?.fitness ?? 0) >= solveThreshold,
    bestToolName: bestOverall?.candidate.name,
    bestToolDescription: bestOverall?.candidate.description,
    bestToolMjcf: bestOverall?.candidate.tool_mjcf,
    bestWaypoints: bestOverall?.candidate.waypoints,
    iterationResults,
  };

  if (bestOverall) {
    const previous = taskMemory.get(memoryKey) ?? [];
    const merged = [...previous, ...elites, bestOverall]
      .sort((a, b) => b.fitness - a.fitness)
      .slice(0, 5);
    taskMemory.set(memoryKey, merged);
  }

  runHistory.unshift(summary);
  if (runHistory.length > 20) runHistory.pop();

  return summary;
}

export function getVLMgineerStatus() {
  return {
    runs: runHistory.slice(0, 10),
    lastRun: runHistory[0] ?? null,
    memoryTaskCount: taskMemory.size,
    taskMemoryKeys: Array.from(taskMemory.keys()).slice(0, 10),
  };
}
