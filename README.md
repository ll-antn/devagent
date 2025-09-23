# AI Dev Agent

Proof-of-concept Python CLI for orchestrating LLM-assisted software development workflows. The agent can
plan work, draft Architecture Decision Records (ADRs), propose code diffs, and run local tests while keeping
humans in the loop for approvals.

## Installation

```bash
pip install -e .[dev]
```

## Configuration

Settings are resolved from environment variables prefixed with `DEVAGENT_` or from a TOML file located at
`.devagent.toml` in the project root (or `~/.config/devagent/config.toml`). Key options:

- `DEVAGENT_API_KEY`: API key for the configured LLM provider (DeepSeek by default).
- `DEVAGENT_MODEL`: Model name to use.
- `DEVAGENT_AUTO_APPROVE_PLAN`: `true` to auto-approve generated plans.
- `DEVAGENT_AUTO_APPROVE_CODE`: `true` to auto-apply code diffs without confirmation.
- `DEVAGENT_STRUCTURED_LOGGING`: `true` to emit JSON logs with correlation IDs (helpful for aggregators).

## CLI Overview

### Natural Language Routing

- Use `devagent "<request>"` to let the assistant choose the appropriate command via the LLM tool-calling API (DeepSeek/OpenAI compatible).
- Examples: `devagent "покажи содержимое директории docs"` lists the `docs/` folder, `devagent "где находится CLI входная точка"` routes to the repository search.
- Behind the scenes the router picks from the existing commands (`where`, `plan`, `ask`, `run-plan`, `review`) so all gating and state management stay consistent.

## Enhanced Planning Workflow

The planner generates comprehensive work breakdown structures with 8-15 structured tasks for any development goal. The enhanced planner is module-aware and creates execution-ready plans with full traceability.

### Planning Features

- **Module-Aware Planning**: Tasks reference specific DevAgent modules (cli, llm_provider, planning, approval, adr, code_edit, testing, utils, security)
- **Rich Task Structure**: Each task includes dependencies, effort estimates, RICE scoring (Reach, Impact, Confidence, Effort), and specific deliverables
- **Execution Guidance**: Tasks include suggested `devagent run --files ...` commands for immediate execution
- **Dependency Management**: Tasks are ordered with explicit dependencies to ensure logical execution flow
- **Full Lifecycle Coverage**: Plans span design → implementation → testing → documentation phases

### Task Categories

- **Design**: ADR creation and architectural decisions
- **Implementation**: Code changes and feature development  
- **Testing**: Test execution and QA validation
- **Documentation**: README updates and user guides

### RICE Prioritization

Tasks are automatically scored using the RICE framework:
- **Reach**: Number of users/systems affected (1-5)
- **Impact**: Importance to the goal (1-5) 
- **Confidence**: Certainty of estimates (0-1)
- **Effort**: Implementation complexity (1-5)

Tasks are ranked by RICE score to optimize execution order.

### Planning Commands

When you run `devagent plan "<goal>"`, review the printed deliverables/commands and save the plan.
Use those `devagent run --files …` instructions to execute each task in turn and monitor progress with
`devagent status`.

- `devagent plan "Implement feature"` – generates a work breakdown structure and persists it locally.
- `devagent status` – shows the last saved plan and task statuses.
- `devagent run [TASK_ID]` – executes the next pending task (or the specified task), handling ADR creation,
  code editing, and optional test execution.
- `devagent history` – lists recorded commands, with `--replay` support for quick reruns.
- `devagent metrics` – summarizes recent task outcomes and durations collected during `devagent run`.
- `devagent config` – prints the resolved configuration.
- `devagent hello` – prints a friendly greeting message for quick smoke tests.
- `devagent completion --shell <shell>` – prints shell completion scripts (pass `--eval` for helper snippet).

`devagent run` accepts options such as `--files`, `--instructions`, `--skip-tests`, `--test-command`,
`--hide-thinking`, and `--dry-run` to fine-tune execution. During execution, the CLI streams `[thinking]` logs that outline
each reasoning step and tool invocation; use `--hide-thinking` to suppress this trace. When proposing code
changes, the agent prints the unified diff before requesting approval and applying it with `git apply`.

### Git Workflow Helpers

The `git` subcommands keep version control tasks close to the agent workflow:

- `devagent git commit-message` – generates a conventional commit message using the staged/working diff.
- `devagent git start-feature` – creates a feature branch with consistent naming (supports `--dry-run`).
- `devagent git pr-description` – drafts a Markdown PR description from the current changes.

### Interactive & Persistent Modes

- `devagent shell` remains available for an interactive REPL-like session with persistent state.
- Command history (`devagent history`) records structured invocations and supports replaying past commands.

> **Persistent workflows:** standalone `devagent plan` and `devagent run` commands keep context only while the
> command is running. Launch `devagent shell` when you need the agent to maintain plan state across multiple
> interactive steps without restarting.

### Observability

- Logs now include a correlation ID per command, and structured JSON output can be enabled via
  `DEVAGENT_STRUCTURED_LOGGING=1`.
- Task executions emit metrics (duration, outcome, dry-run flag). Use `devagent metrics` to view recent data
  and aggregate success rates.

## Testing

Install the optional development dependencies and run the test suite:

```bash
pip install -e .[dev]
pytest
```

## Development Notes

- Generated plans, task status, and raw LLM responses are held in memory per process; use `devagent shell`
  to keep context alive over multiple interactions.
- ADR drafts can be stored under `docs/` (for example `docs/adr/`) if you choose to persist them.
  When generation fails, the agent falls back to a template-filled stub to keep progress moving.
- Code edits rely on unified diff output. Ensure `git` is available locally so `git apply` succeeds.
- Context gathering now enriches prompts with a Tree-sitter–derived project outline when the
  optional `tree-sitter` and `tree-sitter-languages` dependencies are available.

This PoC focuses on local flows; integrating with remote issue trackers or PR automation can be explored later as the project evolves.

## Hello Command

The `hello` command prints a friendly greeting and is useful for confirming the CLI is set up correctly.

```bash
devagent hello Codex
# Hello, Codex!
```
