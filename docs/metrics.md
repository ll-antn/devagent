# Metrics and Gates

DevAgent records a normalized `MetricsSnapshot` after every tool step. The
snapshot feeds both gate evaluation and observability dashboards.

## Core Metrics

| Field              | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `tests_passed`     | Final status of the configured test command.             |
| `lint_errors`      | Count of lint violations (non-zero return code).         |
| `type_errors`      | Count of type-check failures.                            |
| `format_errors`    | Formatter check failures (non-zero return code).         |
| `compile_errors`   | Build/packaging step failures.                           |
| `diff_lines/files` | Size of the working diff versus the configured ref.      |
| `diff_concentration` | Average changed lines per file (risk indicator).       |
| `patch_coverage`   | Coverage ratio for modified lines (diff coverage).       |
| `secrets_found`    | Findings reported by the secret scanner.                 |
| `sandbox_violations` | Disallowed commands or resource limit violations.     |
| `flaky_tests`      | Count of detected flaky tests (0 required).              |
| `tokens_cost`      | Optional LLM token cost metadata per step.               |
| `wall_time`        | Wall-clock duration of the tool action.                  |

`MetricsCollector` aggregates command results, diff statistics, patch coverage,
security scans, and sandbox stats into a single snapshot. Missing measurements
are annotated in `gate_notes` and prevent the associated gate from passing.

## Gate Rules

`GateEvaluator` enforces the following hard gates (configurable via `GateConfig`):

- Tests, lint, type, format, compile success
- Diff size limits (`diff_limit_lines`, `diff_limit_files`)
- Patch coverage ≥ `patch_coverage_target`
- `secrets_found == 0`
- `sandbox_violations == 0`
- `flaky_tests == 0`

The evaluator also implements **stuck detection**: `stuck_threshold` consecutive
steps without improvements across critical metrics (`tests_passed`, `lint_errors`,
`type_errors`, `diff_lines`, `patch_coverage`) triggers a `blocked` status.
`steps_budget` caps total iterations per task.

## Secret Scanning

`scan_for_secrets` examines modified files for AWS keys, Google API tokens,
Slack tokens, generic secret assignments, and high-entropy strings. Findings are
reported in `secret_findings` and counted in `secrets_found`.

## Patch Coverage

`compute_patch_coverage` cross-references the `git diff --unified=0` output with
`coverage.py` XML reports. Coverage is calculated for modified line numbers only
and returned as a `PatchCoverageResult` with per-file covered/uncovered lists.

## Diff Metrics

`compute_diff_metrics` summarises working tree changes vs the configured ref
(default `HEAD`). It reports total changed lines, number of files touched, and
average lines per file. Setting `include_untracked=True` accounts for new files.

## Iteration Tracking

`IterationTracker` records cumulative wall time and token usage across loop
iterations, enabling higher-level analytics (steps-to-green, trend analysis) on
stored traces.
