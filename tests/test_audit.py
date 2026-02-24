"""Tests for Audit Logger."""

import pytest

from src.utils.audit import (
    AuditEvent,
    clear_audit_log,
    get_events,
    get_session_id,
    log_event,
)


@pytest.fixture(autouse=True)
def clean_audit_log():
    """Clear audit log before and after each test."""
    clear_audit_log()
    yield
    clear_audit_log()


class TestLogEvent:
    def test_basic_logging(self):
        event = log_event("system", "patient_registered", {"patient_id": "test"})
        assert isinstance(event, AuditEvent)
        assert event.actor == "system"
        assert event.action == "patient_registered"
        assert event.payload["patient_id"] == "test"

    def test_with_stage_and_patient(self):
        event = log_event(
            "model:retts_rules_engine", "triage_result",
            {"retts_level": "ORANGE"},
            stage="triage", patient_id="anders",
        )
        assert event.stage == "triage"
        assert event.patient_id == "anders"

    def test_session_id_assigned(self):
        event = log_event("system", "test_action")
        assert event.session_id == get_session_id()

    def test_timestamp_set(self):
        event = log_event("system", "test_action")
        assert event.timestamp is not None


class TestGetEvents:
    def test_read_back_events(self):
        log_event("system", "action1")
        log_event("nurse", "action2")
        log_event("system", "action3")

        events = get_events()
        assert len(events) == 3

    def test_filter_by_actor(self):
        log_event("system", "action1")
        log_event("nurse", "action2")
        log_event("system", "action3")

        events = get_events(actor="system")
        assert len(events) == 2

    def test_filter_by_action(self):
        log_event("system", "pretriage_result")
        log_event("nurse", "override")
        log_event("system", "triage_result")

        events = get_events(action="override")
        assert len(events) == 1
        assert events[0].actor == "nurse"

    def test_filter_by_patient(self):
        log_event("system", "action1", patient_id="anders")
        log_event("system", "action2", patient_id="ella")
        log_event("system", "action3", patient_id="anders")

        events = get_events(patient_id="anders")
        assert len(events) == 2

    def test_filter_by_stage(self):
        log_event("system", "action1", stage="pretriage")
        log_event("system", "action2", stage="triage")

        events = get_events(stage="pretriage")
        assert len(events) == 1

    def test_limit(self):
        for i in range(10):
            log_event("system", f"action_{i}")

        events = get_events(limit=3)
        assert len(events) == 3

    def test_empty_log(self):
        events = get_events()
        assert events == []


class TestEventStructure:
    def test_event_serialization(self):
        event = log_event(
            "nurse", "override",
            {"original": "ORANGE", "override_to": "YELLOW"},
            stage="triage", patient_id="test",
        )
        # Read back
        events = get_events()
        assert len(events) == 1
        restored = events[0]
        assert restored.actor == "nurse"
        assert restored.action == "override"
        assert restored.payload["original"] == "ORANGE"
        assert restored.stage == "triage"
        assert restored.patient_id == "test"
