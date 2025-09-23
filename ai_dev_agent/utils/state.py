"""In-memory state helpers used to share context within a CLI process."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PlanSession:
    """Represents a plan execution session that lives for the current process."""

    session_id: str
    goal: str
    status: str
    current_task_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "status": self.status,
            "current_task_id": self.current_task_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanSession":
        return cls(
            session_id=data["session_id"],
            goal=data["goal"],
            status=data["status"],
            current_task_id=data.get("current_task_id"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )


@dataclass
class StateStore:
    """Simple in-memory state container.

    The object persists data only for the lifetime of the current Python process.
    The optional ``path`` argument is retained for backwards compatibility but is
    no longer used to touch the filesystem.
    """

    path: Optional[Path] = None
    _cache: Dict[str, Any] = field(default_factory=dict, init=False)

    def load(self) -> Dict[str, Any]:
        """Return the current state data, creating defaults when empty."""
        if not self._cache:
            self._cache = self._create_default_state()
        return self._cache

    def save(self, data: Dict[str, Any]) -> None:
        """Replace the in-memory state after validation."""
        self._validate_state(data)
        self._cache = data
        LOGGER.debug("State stored in memory (not persisted)")

    def update(self, **updates: Any) -> Dict[str, Any]:
        """Update state with automatic timestamping."""
        data = self.load().copy()
        data.update(updates)
        data["last_updated"] = datetime.utcnow().isoformat()
        self.save(data)
        return data

    def get_current_session(self) -> Optional[PlanSession]:
        """Get the current active plan session."""
        data = self.load()
        session_data = data.get("current_session")
        if session_data:
            return PlanSession.from_dict(session_data)
        return None

    def start_session(self, goal: str, session_id: Optional[str] = None) -> PlanSession:
        """Start a new plan session."""
        if session_id is None:
            session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        session = PlanSession(
            session_id=session_id,
            goal=goal,
            status="active"
        )

        self.update(
            current_session=session.to_dict(),
            session_history=self._get_session_history() + [session.to_dict()]
        )

        LOGGER.info("Started new in-memory session: %s", session_id)
        return session

    def update_session(self, **updates: Any) -> Optional[PlanSession]:
        """Update the current session."""
        session = self.get_current_session()
        if not session:
            LOGGER.warning("No active session to update")
            return None

        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)

        session.updated_at = datetime.utcnow().isoformat()

        self.update(current_session=session.to_dict())
        return session

    def end_session(self, status: str = "completed") -> None:
        """End the current session."""
        session = self.get_current_session()
        if session:
            session.status = status
            session.updated_at = datetime.utcnow().isoformat()

            self.update(
                current_session=None,
                session_history=self._get_session_history() + [session.to_dict()]
            )
            LOGGER.info("Ended in-memory session: %s with status: %s", session.session_id, status)

    def can_resume(self) -> bool:
        """Check if there's a session that can be resumed."""
        session = self.get_current_session()
        return session is not None and session.status in ["active", "paused", "interrupted"]

    def get_resumable_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks that can be resumed."""
        data = self.load()
        plan = data.get("last_plan", {})
        tasks = plan.get("tasks", [])

        return [
            task for task in tasks
            if task.get("status") in ["pending", "in_progress", "needs_attention"]
        ]

    def _create_default_state(self) -> Dict[str, Any]:
        """Create default state structure."""
        now = datetime.utcnow().isoformat()
        return {
            "version": "1.0",
            "created_at": now,
            "last_updated": now,
            "current_session": None,
            "session_history": [],
            "last_plan": None,
            "last_plan_raw": None,
            "command_history": [],
            "metrics": [],
        }

    def _validate_state(self, data: Dict[str, Any]) -> None:
        """Validate state structure."""
        if not isinstance(data, dict):
            raise ValueError("State must be a dictionary")

        if "last_updated" not in data:
            data["last_updated"] = datetime.utcnow().isoformat()

    def _get_session_history(self) -> List[Dict[str, Any]]:
        """Get session history list."""
        data = self.load()
        return data.get("session_history", [])

    def append_history(self, entry: Dict[str, Any], limit: int = 50) -> None:
        """Append an entry to the in-memory command history."""
        data = self.load()
        history = list(data.get("command_history", []))
        history.append(entry)
        if limit and len(history) > limit:
            history = history[-limit:]
        data["command_history"] = history
        data["last_updated"] = datetime.utcnow().isoformat()
        self.save(data)

    def record_metric(self, entry: Dict[str, Any], limit: int = 500) -> None:
        """Record a metrics entry while bounding total storage."""
        data = self.load()
        metrics = list(data.get("metrics", []))
        metrics.append(entry)
        if limit and len(metrics) > limit:
            metrics = metrics[-limit:]
        data["metrics"] = metrics
        data["last_updated"] = datetime.utcnow().isoformat()
        self.save(data)


__all__ = ["StateStore", "PlanSession"]
