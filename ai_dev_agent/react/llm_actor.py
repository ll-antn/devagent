"""LLM-based action provider for intelligent ReAct execution."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..llm_provider.base import LLMClient, LLMError
from .types import ActionRequest, StepRecord, TaskSpec


@dataclass
class ThoughtProcess:
    """Result of LLM reasoning about current state."""
    current_state: str       # Where we are now
    goal_distance: str       # How far from completion (far/moderate/close/complete)
    obstacles: List[str]     # What's blocking progress
    next_step: str          # What to do next
    reasoning: str          # Why this action
    confidence: float       # Confidence in decision (0-1)


class LLMActionProvider:
    """
    Uses LLM for intelligent action selection in ReAct loop.
    Analyzes history, learns from mistakes, adapts strategy.
    """
    
    AVAILABLE_TOOLS = {
        "analyze_code": {
            "description": "Analyze code structure and identify issues",
            "args": ["files", "focus_areas"],
            "example": {"files": ["src/main.py"], "focus_areas": ["bugs", "coverage"]}
        },
        "write_patch": {
            "description": "Generate code changes to fix issues or add features",
            "args": ["files", "changes_description", "test_first"],
            "example": {"files": ["src/main.py"], "changes_description": "Fix bug X", "test_first": True}
        },
        "run_tests": {
            "description": "Execute tests and get detailed results",
            "args": ["test_pattern", "verbose", "coverage"],
            "example": {"test_pattern": "test_coverage", "verbose": True, "coverage": True}
        },
        "fix_tests": {
            "description": "Fix failing tests based on error output",
            "args": ["test_output", "files_to_fix"],
            "example": {"test_output": "TestFailed...", "files_to_fix": ["tests/test_main.py"]}
        },
        "add_tests": {
            "description": "Generate new tests to improve coverage",
            "args": ["target_files", "coverage_gaps", "test_style"],
            "example": {"target_files": ["src/main.py"], "coverage_gaps": ["function_x"], "test_style": "pytest"}
        },
        "run_linter": {
            "description": "Run linter and get style issues",
            "args": ["files", "fix_auto"],
            "example": {"files": ["src/"], "fix_auto": True}
        },
        "search_codebase": {
            "description": "Search for patterns or information in code",
            "args": ["query", "file_types", "include_tests"],
            "example": {"query": "def coverage", "file_types": ["py"], "include_tests": True}
        },
        "refactor_code": {
            "description": "Refactor code for better structure",
            "args": ["files", "refactoring_type", "preserve_tests"],
            "example": {"files": ["src/main.py"], "refactoring_type": "extract_method", "preserve_tests": True}
        },
        "check_dependencies": {
            "description": "Analyze and fix dependency issues",
            "args": ["check_security", "update_outdated"],
            "example": {"check_security": True, "update_outdated": False}
        },
        "generate_docs": {
            "description": "Generate or update documentation",
            "args": ["files", "doc_type", "include_examples"],
            "example": {"files": ["src/main.py"], "doc_type": "docstring", "include_examples": True}
        },
        "qa_pipeline": {
            "description": "Run full quality pipeline (tests, lint, format, etc.)",
            "args": ["run_tests", "fix_issues"],
            "example": {"run_tests": True, "fix_issues": False}
        }
    }
    
    def __init__(self, llm_client: LLMClient, verbose: bool = False):
        self.llm = llm_client
        self.verbose = verbose
        self.decision_history: List[ThoughtProcess] = []
        
    def __call__(self, task: TaskSpec, history: Sequence[StepRecord]) -> ActionRequest:
        """Main method - select next action based on ReAct reasoning."""
        
        # Convert to list for easier manipulation
        history_list = list(history)
        
        # 1. Analyze current state
        thought = self._think_about_state(task, history_list)
        
        # 2. Choose action based on analysis
        action = self._decide_action(thought, task, history_list)
        
        # 3. Save for learning
        self.decision_history.append(thought)
        
        # 4. Log if verbose
        if self.verbose:
            self._log_decision(thought, action)
            
        return action
    
    def _think_about_state(self, task: TaskSpec, history: List[StepRecord]) -> ThoughtProcess:
        """LLM analyzes current situation and plans next step."""
        
        context = self._build_context(task, history)
        
        prompt = f"""You are an AI agent using the ReAct pattern to complete a development task.

TASK: {task.goal}
Category: {task.category}
Instructions: {task.instructions or "None provided"}
Target files: {task.files or "Not specified"}

HISTORY OF ACTIONS TAKEN:
{self._format_history(history)}

CURRENT STATE ANALYSIS:
Based on the task and history, analyze:
1. What is the current state of progress?
2. How far are we from completing the goal?
3. What obstacles or issues are blocking progress?
4. What should be the logical next step?
5. Why is this the best action to take now?

Provide your analysis in JSON format:
{{
    "current_state": "description of where we are",
    "goal_distance": "far|moderate|close|complete",
    "obstacles": ["list", "of", "current", "problems"],
    "next_step": "what action to take next",
    "reasoning": "detailed explanation of why this action",
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = self.llm.complete(prompt, temperature=0.2)
            return self._parse_thought_process(response)
        except (LLMError, Exception) as e:
            # Fallback on LLM failure
            return ThoughtProcess(
                current_state=f"LLM analysis failed: {e}",
                goal_distance="unknown",
                obstacles=["LLM unavailable"],
                next_step="Continue with best guess",
                reasoning=f"Using fallback logic due to: {e}",
                confidence=0.3
            )
    
    def _decide_action(self, thought: ThoughtProcess, task: TaskSpec, history: List[StepRecord]) -> ActionRequest:
        """Based on analysis, choose specific action."""
        
        # Check if goal is complete
        if thought.goal_distance == "complete":
            raise StopIteration("Goal achieved according to LLM analysis")
        
        # Handle different scenarios
        if not history:
            return self._initial_action(task, thought)
        
        last_step = history[-1]
        
        if not last_step.observation.success:
            return self._recovery_action(last_step, thought)
        
        if self._is_stuck(history):
            return self._alternative_action(task, history, thought)
        
        return self._next_planned_action(thought, task, history)
    
    def _initial_action(self, task: TaskSpec, thought: ThoughtProcess) -> ActionRequest:
        """Choose first action for the task."""
        
        action_prompt = f"""Based on this analysis:
{json.dumps(thought.__dict__, indent=2)}

And the task: {task.goal}

Choose the FIRST action from available tools:
{json.dumps(self.AVAILABLE_TOOLS, indent=2)}

For a task about "{task.goal}", what should be the first step?

Return a JSON with:
{{
    "tool": "tool_name",
    "args": {{"arg1": "value1", ...}},
    "expected_outcome": "what we expect to achieve"
}}"""
        
        try:
            response = self.llm.complete(action_prompt, temperature=0.1)
            action_data = json.loads(response)
            
            return ActionRequest(
                step_id="S1",
                thought=thought.reasoning,
                tool=action_data["tool"],
                args=action_data["args"],
                metadata={"expected": action_data.get("expected_outcome", "")}
            )
        except (LLMError, json.JSONDecodeError, Exception):
            # Fallback to safe first action
            return ActionRequest(
                step_id="S1",
                thought="Fallback: Starting with code analysis",
                tool="analyze_code",
                args={"files": task.files or [], "focus_areas": ["structure", "issues"]},
                metadata={"fallback": True}
            )
    
    def _recovery_action(self, failed_step: StepRecord, thought: ThoughtProcess) -> ActionRequest:
        """Action to recover from error."""
        
        error = failed_step.observation.error or "Unknown error"
        
        recovery_prompt = f"""The last action failed with error:
{error}

Failed action was: {failed_step.action.tool}
With args: {json.dumps(failed_step.action.args)}

Current analysis:
{json.dumps(thought.__dict__, indent=2)}

What recovery action should we take? Consider:
1. Can we fix the error directly?
2. Should we try a different approach?
3. Do we need more information first?

Available tools:
{json.dumps(self.AVAILABLE_TOOLS, indent=2)}

Return JSON with recovery action:
{{
    "tool": "tool_name",
    "args": {{}},
    "recovery_strategy": "fix|workaround|investigate|alternative"
}}"""
        
        try:
            response = self.llm.complete(recovery_prompt, temperature=0.3)
            recovery_data = json.loads(response)
            
            step_num = len(self.decision_history) + 1
            
            return ActionRequest(
                step_id=f"S{step_num}",
                thought=f"Recovering from error: {recovery_data.get('recovery_strategy', 'fix')}",
                tool=recovery_data["tool"],
                args=recovery_data["args"],
                metadata={"recovery": True, "strategy": recovery_data.get("recovery_strategy")}
            )
        except (LLMError, json.JSONDecodeError, Exception):
            # Fallback recovery - try to analyze the problem
            step_num = len(self.decision_history) + 1
            return ActionRequest(
                step_id=f"S{step_num}",
                thought="Fallback recovery: Analyzing the issue",
                tool="search_codebase",
                args={"query": "error", "file_types": ["py"], "include_tests": True},
                metadata={"recovery": True, "fallback": True}
            )
    
    def _is_stuck(self, history: List[StepRecord], threshold: int = 3) -> bool:
        """Determine if we're stuck (repeating errors)."""
        
        if len(history) < threshold:
            return False
            
        # Check last N actions
        recent = history[-threshold:]
        
        # Stuck if all recent actions failed
        all_failed = all(not step.observation.success for step in recent)
        
        # Or if repeating same tool
        same_tool = len(set(step.action.tool for step in recent)) == 1
        
        return all_failed or (same_tool and len(recent) >= threshold)
    
    def _alternative_action(self, task: TaskSpec, history: List[StepRecord], thought: ThoughtProcess) -> ActionRequest:
        """Alternative approach when stuck."""
        
        failed_tools = [step.action.tool for step in history[-3:]]
        available_alternatives = [t for t in self.AVAILABLE_TOOLS if t not in failed_tools]
        
        alternative_prompt = f"""We are stuck. Recent failed attempts used:
{failed_tools}

Task: {task.goal}
Current thought: {json.dumps(thought.__dict__, indent=2)}

Suggest a DIFFERENT approach. Do NOT use these tools: {failed_tools}

Available alternatives:
{json.dumps({k: self.AVAILABLE_TOOLS[k] for k in available_alternatives}, indent=2)}

Return JSON with alternative strategy:
{{
    "tool": "different_tool",
    "args": {{}},
    "why_different": "explanation of new approach"
}}"""
        
        try:
            response = self.llm.complete(alternative_prompt, temperature=0.5)
            alt_data = json.loads(response)
            
            step_num = len(history) + 1
            
            return ActionRequest(
                step_id=f"S{step_num}",
                thought=f"Trying alternative: {alt_data.get('why_different', 'new approach')}",
                tool=alt_data["tool"],
                args=alt_data["args"],
                metadata={"alternative": True}
            )
        except (LLMError, json.JSONDecodeError, Exception):
            # Fallback alternative - try QA pipeline
            step_num = len(history) + 1
            return ActionRequest(
                step_id=f"S{step_num}",
                thought="Fallback alternative: Running quality pipeline",
                tool="qa_pipeline",
                args={"run_tests": True, "fix_issues": True},
                metadata={"alternative": True, "fallback": True}
            )
    
    def _next_planned_action(self, thought: ThoughtProcess, task: TaskSpec, history: List[StepRecord]) -> ActionRequest:
        """Regular next action according to plan."""
        
        next_prompt = f"""Continue with the task.

Task: {task.goal}
Progress: {thought.current_state}
Next step planned: {thought.next_step}

History summary:
{self._summarize_history(history)}

Choose the next action from tools:
{json.dumps(self.AVAILABLE_TOOLS, indent=2)}

Return JSON:
{{
    "tool": "tool_name",
    "args": {{}},
    "continues_from": "what previous step achieved"
}}"""
        
        try:
            response = self.llm.complete(next_prompt, temperature=0.2)
            next_data = json.loads(response)
            
            step_num = len(history) + 1
            
            return ActionRequest(
                step_id=f"S{step_num}",
                thought=thought.reasoning,
                tool=next_data["tool"],
                args=next_data["args"],
                metadata={"continues": next_data.get("continues_from", "")}
            )
        except (LLMError, json.JSONDecodeError, Exception):
            # Fallback - continue with QA
            step_num = len(history) + 1
            return ActionRequest(
                step_id=f"S{step_num}",
                thought="Fallback: Continuing with quality checks",
                tool="qa_pipeline",
                args={"run_tests": True, "fix_issues": False},
                metadata={"fallback": True}
            )
    
    def _build_context(self, task: TaskSpec, history: List[StepRecord]) -> str:
        """Build context string for LLM."""
        
        context_parts = [
            f"Task ID: {task.identifier}",
            f"Goal: {task.goal}",
            f"Category: {task.category}"
        ]
        
        if task.instructions:
            context_parts.append(f"Instructions: {task.instructions}")
        
        if task.files:
            context_parts.append(f"Files: {', '.join(task.files)}")
        
        if history:
            context_parts.append(f"Steps completed: {len(history)}")
            last = history[-1]
            context_parts.append(f"Last action: {last.action.tool} - {'✓' if last.observation.success else '✗'}")
        
        return "\n".join(context_parts)
    
    def _format_history(self, history: List[StepRecord]) -> str:
        """Format history for LLM display."""
        
        if not history:
            return "No actions taken yet"
        
        lines = []
        for i, step in enumerate(history, 1):
            status = "✓" if step.observation.success else "✗"
            lines.append(f"{i}. [{status}] {step.action.tool}")
            lines.append(f"   Thought: {step.action.thought}")
            lines.append(f"   Result: {step.observation.outcome}")
            
            if step.observation.error:
                lines.append(f"   Error: {step.observation.error}")
            
            if step.metrics:
                key_metrics = self._extract_key_metrics(step.metrics)
                if key_metrics:
                    lines.append(f"   Metrics: {key_metrics}")
        
        return "\n".join(lines)
    
    def _extract_key_metrics(self, metrics) -> str:
        """Extract key metrics for display."""
        
        key_items = []
        
        if hasattr(metrics, 'tests_passed') and metrics.tests_passed is not None:
            key_items.append(f"tests={'passed' if metrics.tests_passed else 'failed'}")
        
        if hasattr(metrics, 'patch_coverage') and metrics.patch_coverage is not None:
            key_items.append(f"coverage={metrics.patch_coverage:.1%}")
        
        if hasattr(metrics, 'lint_errors') and metrics.lint_errors is not None:
            key_items.append(f"lint_errors={metrics.lint_errors}")
        
        return ", ".join(key_items) if key_items else ""
    
    def _parse_thought_process(self, llm_response: str) -> ThoughtProcess:
        """Parse LLM response into ThoughtProcess."""
        
        try:
            # Try to extract JSON from response
            response = llm_response.strip()
            
            # Handle case where LLM wraps JSON in markdown
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            
            data = json.loads(response)
            return ThoughtProcess(
                current_state=data.get("current_state", "Unknown"),
                goal_distance=data.get("goal_distance", "unknown"),
                obstacles=data.get("obstacles", []),
                next_step=data.get("next_step", "Continue"),
                reasoning=data.get("reasoning", "No reasoning provided"),
                confidence=float(data.get("confidence", 0.5))
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback if LLM didn't return valid JSON
            return ThoughtProcess(
                current_state="Failed to parse LLM response",
                goal_distance="unknown",
                obstacles=["Parse error"],
                next_step="Continue with fallback logic",
                reasoning=f"Parse error: {e}. Response was: {llm_response[:200]}...",
                confidence=0.1
            )
    
    def _summarize_history(self, history: List[StepRecord]) -> str:
        """Brief summary of history for LLM."""
        
        if not history:
            return "Starting fresh"
        
        successful = [s for s in history if s.observation.success]
        failed = [s for s in history if not s.observation.success]
        
        return (f"Completed {len(successful)} successful and {len(failed)} failed actions. "
                f"Tools used: {', '.join(set(s.action.tool for s in history))}")
    
    def _log_decision(self, thought: ThoughtProcess, action: ActionRequest):
        """Log decision for debugging."""
        
        print(f"\n{'='*60}")
        print(f"REACT DECISION - Step {action.step_id}")
        print(f"{'='*60}")
        print(f"State: {thought.current_state}")
        print(f"Goal Distance: {thought.goal_distance}")
        print(f"Obstacles: {', '.join(thought.obstacles) if thought.obstacles else 'None'}")
        print(f"Confidence: {thought.confidence:.1%}")
        print(f"-" * 40)
        print(f"Action: {action.tool}")
        print(f"Reasoning: {thought.reasoning}")
        print(f"Args: {json.dumps(action.args, indent=2)}")
        print(f"{'='*60}\n")


__all__ = ["LLMActionProvider", "ThoughtProcess"]
