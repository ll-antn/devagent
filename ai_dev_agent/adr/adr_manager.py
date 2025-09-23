"""ADR generation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..llm_provider import LLMClient, LLMError, Message
from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are an expert software architect. Draft concise Architecture Decision Records (ADR) "
    "using the provided template. Keep language clear and actionable."
)

USER_TEMPLATE = """
Create an ADR for the following development task.
Task Title: {title}
Task Description: {description}
Overall Goal: {goal}
Template:
{template}

Fill the template fields with thoughtful content. Use the status 'Proposed'. Return only the markdown document.
"""


@dataclass
class ADRGenerationResult:
    path: Path
    fallback_reason: str | None
    content: str


@dataclass
class ADRManager:
    repo_root: Path
    llm_client: LLMClient
    template_path: Path

    def generate(
        self,
        goal: str,
        task_title: str,
        task_description: str,
        *,
        dry_run: bool = False,
    ) -> ADRGenerationResult:
        adr_dir = self.repo_root / "docs" / "adr"
        if not dry_run:
            adr_dir.mkdir(parents=True, exist_ok=True)
        template = self.template_path.read_text(encoding="utf-8")
        slug = _slugify(task_title)
        index = self._next_index(adr_dir)
        filename = f"ADR-{index:04d}-{slug}.md"
        filepath = adr_dir / filename
        prompt = USER_TEMPLATE.format(title=task_title, description=task_description, goal=goal, template=template)
        fallback_reason: str | None = None
        try:
            content = self.llm_client.complete(
                [
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=prompt),
                ],
                temperature=0.4,
            )
        except LLMError as exc:
            LOGGER.warning("Failed to generate ADR via LLM: %s. Falling back to template.", exc)
            fallback_reason = str(exc)
            content = template.format(
                title=task_title,
                status="Proposed",
                context=task_description,
                decision="Decision pending",
                consequences="Consequences TBD",
            )
        content = _strip_markdown_fence(content)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d")
        header = f"<!-- Generated on {timestamp} UTC -->\n"
        final_content = header + content.strip() + "\n"
        if dry_run:
            LOGGER.info("Dry run: would create ADR at %s", filepath)
        else:
            filepath.write_text(final_content, encoding="utf-8")
            LOGGER.info("ADR created at %s", filepath)
        return ADRGenerationResult(path=filepath, fallback_reason=fallback_reason, content=final_content)

    def _next_index(self, adr_dir: Path) -> int:
        existing = [path.name for path in adr_dir.glob("ADR-*.md")]
        numbers = []
        for name in existing:
            try:
                number = int(name.split("-", 1)[0].replace("ADR", ""))
                numbers.append(number)
            except ValueError:
                continue
        return max(numbers, default=0) + 1


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    cleaned = "-".join(filter(None, cleaned.split("-")))
    return cleaned or "adr"


def _strip_markdown_fence(content: str) -> str:
    if content.strip().startswith("```"):
        lines = content.strip().splitlines()
        # remove opening fence
        lines = lines[1:]
        # remove closing fence if present
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return content


__all__ = ["ADRManager", "ADRGenerationResult"]
