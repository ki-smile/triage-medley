"""Audit Logger — append-only JSONL event logging.

Thread-safe, structured audit trail for all AI suggestions and human decisions.
"""

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.config import get_project_root

_AUDIT_DIR = get_project_root() / "data" / "audit"
_AUDIT_FILE = _AUDIT_DIR / "audit.jsonl"
_lock = threading.Lock()

# Session ID — unique per application run
_SESSION_ID = str(uuid.uuid4())[:8]


class AuditEvent(BaseModel):
    """A single audit log entry."""

    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str = Field(default_factory=lambda: _SESSION_ID)
    actor: str = Field(
        ...,
        description="Who: system | nurse | physician | model:{model_id}",
    )
    action: str = Field(
        ...,
        description=(
            "What: pretriage_result | vitals_entered | triage_result | "
            "override | differential_result | management_result | "
            "patient_registered | asr_disagreement_resolved"
        ),
    )
    stage: Optional[str] = Field(
        default=None, description="pretriage | triage | differential | management"
    )
    patient_id: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)


def log_event(
    actor: str,
    action: str,
    payload: dict[str, Any] | None = None,
    *,
    stage: str | None = None,
    patient_id: str | None = None,
) -> AuditEvent:
    """Log an audit event. Thread-safe, append-only.

    Returns the created AuditEvent.
    """
    event = AuditEvent(
        actor=actor,
        action=action,
        stage=stage,
        patient_id=patient_id,
        payload=payload or {},
    )

    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    with _lock:
        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    return event


def get_events(
    *,
    session_id: str | None = None,
    actor: str | None = None,
    action: str | None = None,
    patient_id: str | None = None,
    stage: str | None = None,
    limit: int | None = None,
) -> list[AuditEvent]:
    """Query audit events with optional filters.

    Returns events in chronological order (oldest first).
    """
    if not _AUDIT_FILE.exists():
        return []

    events: list[AuditEvent] = []

    with _lock:
        with open(_AUDIT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = AuditEvent.model_validate_json(line)
                except Exception:
                    continue

                # Apply filters
                if session_id and event.session_id != session_id:
                    continue
                if actor and event.actor != actor:
                    continue
                if action and event.action != action:
                    continue
                if patient_id and event.patient_id != patient_id:
                    continue
                if stage and event.stage != stage:
                    continue

                events.append(event)

    if limit:
        events = events[-limit:]

    return events


def get_current_session_events() -> list[AuditEvent]:
    """Get all events from the current session."""
    return get_events(session_id=_SESSION_ID)


def clear_audit_log() -> None:
    """Clear the audit log file. USE WITH CAUTION — for testing only."""
    with _lock:
        if _AUDIT_FILE.exists():
            _AUDIT_FILE.unlink()


def get_session_id() -> str:
    """Return the current session ID."""
    return _SESSION_ID
