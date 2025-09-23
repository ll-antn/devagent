# ReAct Execution Loop

The ReAct loop powers DevAgent's automated quality gate execution. Each run is
recorded as a sequence of **Think → Act → Observe → Evaluate** steps with
hard gates for correctness, safety, and traceability.

## Lifecycle

1. **Plan** – the CLI selects a plan task and constructs a `TaskSpec` with
   instructions, target files, and metadata.
2. **Act** – `ReactiveExecutor` requests an action, usually the
   `qa.pipeline` tool. The tool runs formatting, lint, type, compile, test, and
   coverage commands within the sandbox.
3. **Observe** – an `Observation` captures structured metrics, log snippets, and
   artefact references produced by the invoked tool.
4. **Evaluate** – `GateEvaluator` checks metrics against `GateConfig` thresholds
   (diff size, diff coverage, lint/type/test success, secret scan, sandbox
   violations, and flake detection). When all gates pass the run stops with
   `success`; repeated failures trigger a `blocked` status.
5. **Decide** – CLI consumers inspect the run result, update plan state, and
   optionally loop again with new instructions.

## CLI Integration

```
devagent react run [TASK_ID] --test-command "pytest --cov" --lint-command "ruff check"
devagent react plan                          # emit the canonical ReAct WBS
```

The run command resolves the next plan task, executes the quality pipeline, and
prints gate status (PASS/FAIL per gate) plus key metrics (diff footprint,
patch coverage, secrets, sandbox violations). Plan tasks store
`gates`, `last_run_metrics`, and `last_run_status` so CLI history and reports can surface recent outcomes.

### Configuration

Values live in `.devagent.toml` or environment variables (`DEVAGENT_*`):

- `diff_limit_lines`, `diff_limit_files`
- `patch_coverage_target`
- `stuck_threshold`, `steps_budget`
- `lint_command`, `typecheck_command`, `format_command`, `compile_command`,
  `test_command`
- `coverage_xml_path`
- `sandbox_allowlist`, `sandbox_cpu_time_limit`, `sandbox_memory_limit_mb`

Unset commands still produce gate failures (with explanations) so missing checks
cannot silently pass.

