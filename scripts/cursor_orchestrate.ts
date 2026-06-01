/**
 * Cursor SDK orchestration layer for judge-prompt-lab.
 *
 * Starts a Claude agent that can inspect experiment state, run allowed Python
 * commands, and propose changes to allowed prompt files. The Python eval
 * harness remains the sole referee for accept/reject decisions.
 *
 * Usage:
 *   npm run orchestrate
 *   # or with a custom goal:
 *   ORCHESTRATE_GOAL="Focus on verbosity calibration." npm run orchestrate
 */

import Anthropic from "@anthropic-ai/sdk";
import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as dotenv from "dotenv";

dotenv.config({ path: ".env" });
dotenv.config({ path: ".env.cursor" }); // overrides .env if present

const ROOT = path.resolve(__dirname, "..");
const MODEL = process.env.CURSOR_MODEL ?? "claude-opus-4-8";
const MAX_AGENT_TURNS = 20;

// ---------------------------------------------------------------------------
// Guardrail lists — enforced in code, not just in the prompt.
// ---------------------------------------------------------------------------

const ALLOWED_READ = new Set([
  "README.md",
  "prompts/judge_prompt.md",
  "prompts/optimizer_prompt.md",
  "prompts/judge_prompt_candidate.md",
  "data/failed_cases.jsonl",
  "results/baseline_metrics.json",
  "results/candidate_metrics.json",
  "results/final_test_metrics.json",
  "results/experiment_log.jsonl",
]);

const ALLOWED_EDIT = new Set([
  "prompts/judge_prompt_candidate.md",
  "prompts/optimizer_prompt.md",
]);

// Commands are validated by prefix so the agent can vary safe flags.
const ALLOWED_COMMAND_PREFIXES = [
  "python src/summarize_experiment.py",
  "python src/optimize_loop.py",
  "python src/final_eval.py",
];

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

const TOOLS: Anthropic.Tool[] = [
  {
    name: "read_file",
    description:
      "Read a file from the project. Only files in the allowed read list are accessible. " +
      "JSONL files are previewed (first 5 lines).",
    input_schema: {
      type: "object" as const,
      properties: {
        path: { type: "string", description: "Path relative to project root." },
      },
      required: ["path"],
    },
  },
  {
    name: "run_command",
    description:
      "Run an allowed Python command. Only commands starting with an approved prefix are permitted.",
    input_schema: {
      type: "object" as const,
      properties: {
        command: {
          type: "string",
          description:
            "Full command to run. Must start with one of: " +
            ALLOWED_COMMAND_PREFIXES.join(", "),
        },
      },
      required: ["command"],
    },
  },
  {
    name: "edit_file",
    description:
      "Overwrite an allowed prompt file with new content. " +
      "Only prompts/judge_prompt_candidate.md and prompts/optimizer_prompt.md may be edited.",
    input_schema: {
      type: "object" as const,
      properties: {
        path: { type: "string", description: "Path relative to project root." },
        content: { type: "string", description: "Full new file content." },
      },
      required: ["path", "content"],
    },
  },
];

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

function toolReadFile(filePath: string): string {
  if (!ALLOWED_READ.has(filePath)) {
    return (
      `[BLOCKED] '${filePath}' is not in the allowed read list.\n` +
      `Allowed: ${[...ALLOWED_READ].join(", ")}`
    );
  }
  const abs = path.join(ROOT, filePath);
  if (!fs.existsSync(abs)) {
    return `[NOT FOUND] ${filePath}`;
  }
  const content = fs.readFileSync(abs, "utf-8");
  // Limit JSONL previews to avoid overwhelming context.
  if (filePath.endsWith(".jsonl")) {
    const lines = content.split("\n").filter(Boolean);
    const head = lines.slice(0, 5).join("\n");
    return lines.length > 5
      ? `${head}\n... (${lines.length} total lines, showing first 5)`
      : head;
  }
  return content;
}

function toolRunCommand(command: string): string {
  const allowed = ALLOWED_COMMAND_PREFIXES.some((prefix) =>
    command.startsWith(prefix)
  );
  if (!allowed) {
    return (
      `[BLOCKED] Command not permitted.\n` +
      `Allowed prefixes: ${ALLOWED_COMMAND_PREFIXES.join(", ")}`
    );
  }
  console.log(`  $ ${command}`);
  try {
    return execSync(command, {
      cwd: ROOT,
      encoding: "utf-8",
      timeout: 360_000, // 6 min — optimize_loop can be slow
    });
  } catch (e: any) {
    const out = [e.stdout, e.stderr].filter(Boolean).join("\n").trim();
    return `[ERROR] ${e.message}\n${out}`;
  }
}

function toolEditFile(filePath: string, content: string): string {
  if (!ALLOWED_EDIT.has(filePath)) {
    return (
      `[BLOCKED] '${filePath}' is not in the allowed edit list.\n` +
      `Allowed: ${[...ALLOWED_EDIT].join(", ")}`
    );
  }
  const abs = path.join(ROOT, filePath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf-8");
  return `[OK] Wrote ${content.length} chars to ${filePath}`;
}

async function executeTool(
  name: string,
  input: Record<string, string>
): Promise<string> {
  switch (name) {
    case "read_file":
      return toolReadFile(input.path);
    case "run_command":
      return toolRunCommand(input.command);
    case "edit_file":
      return toolEditFile(input.path, input.content);
    default:
      return `[ERROR] Unknown tool: ${name}`;
  }
}

// ---------------------------------------------------------------------------
// Main agentic loop
// ---------------------------------------------------------------------------

async function main() {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error(
      "Error: ANTHROPIC_API_KEY is not set. Add it to .env or .env.cursor."
    );
    process.exit(1);
  }

  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

  const systemPrompt = fs.readFileSync(
    path.join(ROOT, "agents/cursor_orchestrator.md"),
    "utf-8"
  );

  const userGoal =
    process.env.ORCHESTRATE_GOAL ??
    "Inspect the current experiment state and recommend or execute the next safe action. " +
      "Start by reading results/baseline_metrics.json and running python src/summarize_experiment.py.";

  console.log(`\njudge-prompt-lab Cursor orchestrator`);
  console.log(`Model : ${MODEL}`);
  console.log(`Goal  : ${userGoal}\n`);
  console.log("─".repeat(60));

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userGoal },
  ];

  for (let turn = 0; turn < MAX_AGENT_TURNS; turn++) {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 4096,
      system: systemPrompt,
      tools: TOOLS,
      messages,
    });

    messages.push({ role: "assistant", content: response.content });

    // Print any text the agent emits.
    for (const block of response.content) {
      if (block.type === "text" && block.text.trim()) {
        console.log(`\n[Agent]\n${block.text.trim()}`);
      }
    }

    if (response.stop_reason === "end_turn") {
      console.log("\n" + "─".repeat(60));
      console.log("Orchestrator finished.");
      break;
    }

    if (response.stop_reason === "tool_use") {
      const toolResults: Anthropic.ToolResultBlockParam[] = [];

      for (const block of response.content) {
        if (block.type !== "tool_use") continue;

        console.log(`\n[Tool: ${block.name}]`, JSON.stringify(block.input));
        const result = await executeTool(
          block.name,
          block.input as Record<string, string>
        );

        // Print a truncated preview so the terminal stays readable.
        const preview =
          result.length > 800 ? result.slice(0, 800) + "\n..." : result;
        console.log(`[Result]\n${preview}`);

        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
      }

      messages.push({ role: "user", content: toolResults });
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
