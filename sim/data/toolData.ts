export interface ToolSuggestion {
  title: string;
  label: string;
  prompt: string;
}

export interface Tool {
  name: string;
  description: string;
  code: string;
  suggestion: ToolSuggestion;
}

export const PRIMITIVE_TOOLS: Tool[] = [
  {
    name: "move_to",
    description: "Move end-effector to target XYZ position",
    code: `def move_to(x: float, y: float, z: float) -> bool:\n    """Move robot end-effector to target position."""\n    return robot.ik_solve(target=[x, y, z])`,
    suggestion: {
      title: "Move Gripper",
      label: "Move the gripper to position (0.3, 0, 0.4)",
      prompt: "Move the gripper to position (0.3, 0, 0.4)",
    },
  },
  {
    name: "set_gripper",
    description: "Open or close the parallel gripper",
    code: `def set_gripper(state: str) -> bool:\n    """Set gripper state: 'open' or 'closed'."""\n    return robot.gripper.set(state)`,
    suggestion: {
      title: "Gripper Control",
      label: "Open the gripper to release the object",
      prompt: "Open the gripper to release the object",
    },
  },
  {
    name: "get_body_position",
    description: "Get the 3D position of a named body",
    code: `def get_body_position(name: str) -> tuple:\n    """Returns (x, y, z) world position of body."""\n    return sim.get_body_pos(name)`,
    suggestion: {
      title: "Find Object",
      label: "Where is the red cube on the table?",
      prompt: "Where is the red cube on the table?",
    },
  },
  {
    name: "detect_objects",
    description: "Run vision model to detect objects in scene",
    code: `def detect_objects(prompt: str) -> list:\n    """Detect objects matching prompt via Gemini."""\n    return vision.detect(prompt, mode="2d_bbox")`,
    suggestion: {
      title: "Scene Understanding",
      label: "What objects are on the table?",
      prompt: "What objects are on the table?",
    },
  },
  {
    name: "pick_and_place",
    description: "Pick object at source and place at target",
    code: `def pick_and_place(src: tuple, dst: tuple) -> bool:\n    """Pick from src (x,y,z) and place at dst."""\n    return robot.pick(src).place(dst)`,
    suggestion: {
      title: "Pick & Place",
      label: "Pick up the blue block and place it in the tray",
      prompt: "Pick up the blue block and place it in the tray",
    },
  },
];
