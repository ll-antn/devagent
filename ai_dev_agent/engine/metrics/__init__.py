"""Metrics utilities for gate evaluation."""
from __future__ import annotations

from .collectors import MetricsCollector, build_metrics_snapshot
from .coverage import PatchCoverageResult, compute_patch_coverage
from .diff import DiffMetrics, compute_diff_metrics

__all__ = [
    "MetricsCollector",
    "build_metrics_snapshot",
    "DiffMetrics",
    "compute_diff_metrics",
    "PatchCoverageResult",
    "compute_patch_coverage",
]
