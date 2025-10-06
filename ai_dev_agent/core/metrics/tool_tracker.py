"""Tool success tracking and metrics system."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


@dataclass
class ToolExecution:
    """Record of a single tool execution."""

    tool_name: str
    timestamp: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    arguments: Optional[Dict] = None
    output_size: int = 0
    iteration: Optional[int] = None
    phase: Optional[str] = None


@dataclass
class ToolMetrics:
    """Aggregated metrics for a tool."""

    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    success_rate: float = 0.0
    avg_output_size: int = 0
    last_used: Optional[float] = None
    error_patterns: Dict[str, int] = field(default_factory=dict)
    phase_usage: Dict[str, int] = field(default_factory=dict)


class ToolTracker:
    """Track tool usage and performance metrics."""

    CACHE_FILE = ".devagent_cache/tool_metrics.json"

    def __init__(
        self,
        project_root: Optional[Path] = None,
        persist_metrics: bool = True
    ):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.persist_metrics = persist_metrics

        # Current session data
        self.executions: List[ToolExecution] = []
        self.tool_metrics: Dict[str, ToolMetrics] = {}

        # Historical data
        self.historical_metrics: Dict[str, ToolMetrics] = {}

        # Performance tracking
        self.consecutive_failures: Dict[str, int] = defaultdict(int)
        self.tool_patterns: Dict[str, List[str]] = defaultdict(list)

        # Load historical metrics
        if persist_metrics:
            self._load_metrics()

    def record_execution(
        self,
        tool_name: str,
        duration: float,
        success: bool,
        error_message: Optional[str] = None,
        arguments: Optional[Dict] = None,
        output_size: int = 0,
        iteration: Optional[int] = None,
        phase: Optional[str] = None
    ) -> None:
        """Record a tool execution."""

        execution = ToolExecution(
            tool_name=tool_name,
            timestamp=time.time(),
            duration=duration,
            success=success,
            error_message=error_message,
            arguments=arguments,
            output_size=output_size,
            iteration=iteration,
            phase=phase
        )

        self.executions.append(execution)

        # Update metrics
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolMetrics(name=tool_name)

        metrics = self.tool_metrics[tool_name]
        metrics.total_calls += 1

        if success:
            metrics.successful_calls += 1
            self.consecutive_failures[tool_name] = 0
        else:
            metrics.failed_calls += 1
            self.consecutive_failures[tool_name] += 1

            # Track error patterns
            if error_message:
                error_key = self._extract_error_pattern(error_message)
                metrics.error_patterns[error_key] = metrics.error_patterns.get(error_key, 0) + 1

        metrics.total_duration += duration
        metrics.avg_duration = metrics.total_duration / metrics.total_calls
        metrics.success_rate = metrics.successful_calls / metrics.total_calls
        metrics.last_used = execution.timestamp

        # Track output size
        if output_size > 0:
            current_avg = metrics.avg_output_size
            metrics.avg_output_size = int(
                (current_avg * (metrics.total_calls - 1) + output_size) / metrics.total_calls
            )

        # Track phase usage
        if phase:
            metrics.phase_usage[phase] = metrics.phase_usage.get(phase, 0) + 1

        # Update patterns
        self._update_patterns(tool_name)

    def _extract_error_pattern(self, error_message: str) -> str:
        """Extract a normalized error pattern from message."""

        # Common error patterns
        patterns = [
            ("file not found", r"(file|path).*not.*found"),
            ("permission denied", r"permission.*denied"),
            ("timeout", r"timeout|timed out"),
            ("syntax error", r"syntax.*error"),
            ("import error", r"import.*error"),
            ("connection error", r"connection.*error"),
            ("rate limit", r"rate.*limit"),
        ]

        import re
        error_lower = error_message.lower()

        for key, pattern in patterns:
            if re.search(pattern, error_lower):
                return key

        # Fallback: first few words
        words = error_message.split()[:3]
        return " ".join(words).lower()

    def _update_patterns(self, tool_name: str) -> None:
        """Update tool usage patterns."""

        # Track sequence of last N tools
        recent_tools = [ex.tool_name for ex in self.executions[-5:]]
        if len(recent_tools) >= 2:
            pattern = "->".join(recent_tools)
            self.tool_patterns[pattern].append(tool_name)

    def get_tool_recommendations(
        self,
        current_phase: str,
        recent_failures: List[str]
    ) -> List[Tuple[str, float]]:
        """Get recommended tools based on current context."""

        recommendations = []

        for tool_name, metrics in self.tool_metrics.items():
            score = 0.0

            # Base score on success rate
            score = metrics.success_rate * 5.0

            # Bonus for phase-appropriate tools
            if current_phase in metrics.phase_usage:
                phase_success = metrics.phase_usage[current_phase] / metrics.total_calls
                score += phase_success * 3.0

            # Penalty for consecutive failures
            consecutive = self.consecutive_failures.get(tool_name, 0)
            if consecutive > 0:
                score *= (0.5 ** consecutive)

            # Penalty for being in recent failures
            if tool_name in recent_failures:
                score *= 0.3

            # Bonus for efficiency (fast tools)
            if metrics.avg_duration > 0:
                efficiency_bonus = min(1.0, 1.0 / metrics.avg_duration)
                score += efficiency_bonus

            recommendations.append((tool_name, score))

        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def get_problematic_tools(self, threshold: float = 0.4) -> List[str]:
        """Get tools with low success rates."""

        problematic = []

        for tool_name, metrics in self.tool_metrics.items():
            if metrics.total_calls >= 3 and metrics.success_rate < threshold:
                problematic.append(tool_name)

        return problematic

    def get_tool_substitutions(self, tool_name: str) -> List[str]:
        """Suggest alternative tools based on usage patterns."""

        substitutions = []

        # Find tools often used after this one fails
        for execution in self.executions:
            if (execution.tool_name == tool_name
                and not execution.success
                and self.executions.index(execution) < len(self.executions) - 1):

                next_exec = self.executions[self.executions.index(execution) + 1]
                if next_exec.success:
                    substitutions.append(next_exec.tool_name)

        # Return unique substitutions ranked by frequency
        from collections import Counter
        counts = Counter(substitutions)
        return [tool for tool, _ in counts.most_common(3)]

    def get_phase_summary(self, phase: str) -> Dict:
        """Get summary of tool usage in a specific phase."""

        phase_tools = {}

        for tool_name, metrics in self.tool_metrics.items():
            if phase in metrics.phase_usage:
                usage = metrics.phase_usage[phase]
                phase_tools[tool_name] = {
                    "calls": usage,
                    "success_rate": metrics.success_rate,
                    "avg_duration": metrics.avg_duration
                }

        return phase_tools

    def get_efficiency_report(self) -> Dict:
        """Generate efficiency report for all tools."""

        report = {
            "total_executions": len(self.executions),
            "total_duration": sum(ex.duration for ex in self.executions),
            "overall_success_rate": 0.0,
            "tools": {}
        }

        if self.executions:
            successful = sum(1 for ex in self.executions if ex.success)
            report["overall_success_rate"] = successful / len(self.executions)

        # Per-tool efficiency
        for tool_name, metrics in self.tool_metrics.items():
            efficiency_score = self._calculate_efficiency_score(metrics)
            report["tools"][tool_name] = {
                "calls": metrics.total_calls,
                "success_rate": metrics.success_rate,
                "avg_duration": metrics.avg_duration,
                "efficiency_score": efficiency_score,
                "common_errors": list(metrics.error_patterns.keys())[:3]
            }

        return report

    def _calculate_efficiency_score(self, metrics: ToolMetrics) -> float:
        """Calculate efficiency score for a tool (0-100)."""

        if metrics.total_calls == 0:
            return 0.0

        # Components of efficiency
        success_weight = 0.5
        speed_weight = 0.3
        reliability_weight = 0.2

        # Success component (0-100)
        success_score = metrics.success_rate * 100

        # Speed component (0-100, based on duration)
        # Assume < 1s is excellent, > 10s is poor
        if metrics.avg_duration <= 0:
            speed_score = 100
        elif metrics.avg_duration < 1:
            speed_score = 100
        elif metrics.avg_duration < 10:
            speed_score = 100 - (metrics.avg_duration - 1) * 11.1
        else:
            speed_score = 0

        # Reliability component (0-100, based on consistency)
        # Fewer error patterns = more reliable
        error_diversity = len(metrics.error_patterns)
        if error_diversity == 0:
            reliability_score = 100
        elif error_diversity < 3:
            reliability_score = 80
        elif error_diversity < 5:
            reliability_score = 60
        else:
            reliability_score = 40

        # Calculate weighted score
        efficiency_score = (
            success_score * success_weight +
            speed_score * speed_weight +
            reliability_score * reliability_weight
        )

        return round(efficiency_score, 1)

    def should_skip_tool(self, tool_name: str, max_consecutive_failures: int = 3) -> bool:
        """Determine if a tool should be skipped due to failures."""

        consecutive = self.consecutive_failures.get(tool_name, 0)
        return consecutive >= max_consecutive_failures

    def reset_failure_count(self, tool_name: str) -> None:
        """Reset consecutive failure count for a tool."""
        self.consecutive_failures[tool_name] = 0

    def _load_metrics(self) -> None:
        """Load historical metrics from cache."""

        cache_path = self.project_root / self.CACHE_FILE

        if not cache_path.exists():
            return

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            for tool_data in data.get("tools", []):
                metrics = ToolMetrics(
                    name=tool_data["name"],
                    total_calls=tool_data.get("total_calls", 0),
                    successful_calls=tool_data.get("successful_calls", 0),
                    failed_calls=tool_data.get("failed_calls", 0),
                    total_duration=tool_data.get("total_duration", 0.0),
                    avg_duration=tool_data.get("avg_duration", 0.0),
                    success_rate=tool_data.get("success_rate", 0.0),
                    avg_output_size=tool_data.get("avg_output_size", 0),
                    last_used=tool_data.get("last_used"),
                    error_patterns=tool_data.get("error_patterns", {}),
                    phase_usage=tool_data.get("phase_usage", {})
                )
                self.historical_metrics[tool_data["name"]] = metrics

        except Exception:
            pass

    def save_metrics(self) -> None:
        """Save current metrics to cache."""

        if not self.persist_metrics:
            return

        cache_path = self.project_root / self.CACHE_FILE
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge current and historical metrics
        all_metrics = dict(self.historical_metrics)
        all_metrics.update(self.tool_metrics)

        data = {
            "updated": datetime.now().isoformat(),
            "tools": []
        }

        for metrics in all_metrics.values():
            data["tools"].append({
                "name": metrics.name,
                "total_calls": metrics.total_calls,
                "successful_calls": metrics.successful_calls,
                "failed_calls": metrics.failed_calls,
                "total_duration": metrics.total_duration,
                "avg_duration": metrics.avg_duration,
                "success_rate": metrics.success_rate,
                "avg_output_size": metrics.avg_output_size,
                "last_used": metrics.last_used,
                "error_patterns": dict(metrics.error_patterns),
                "phase_usage": dict(metrics.phase_usage)
            })

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass