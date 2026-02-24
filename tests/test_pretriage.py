"""Tests for Pre-Triage Engine."""

import pytest

from src.engines.pretriage_engine import evaluate
from src.models import (
    EHRSnapshot,
    FHIRCondition,
    FHIRMedication,
    PreTriageContext,
    QueuePriority,
)


def _make_context(
    speech: str,
    ehr: EHRSnapshot | None = None,
) -> PreTriageContext:
    return PreTriageContext(
        patient_id="test",
        speech_text=speech,
        ehr=ehr,
    )


class TestSpeechFlagDetection:
    def test_chest_pain_high(self):
        result = evaluate(_make_context("I have chest pain and it's hard to breathe"))
        assert result.queue_priority == QueuePriority.HIGH

    def test_breathing_difficulty_high(self):
        result = evaluate(_make_context("I can't breathe, please help"))
        assert result.queue_priority == QueuePriority.HIGH

    def test_fever_rash_high(self):
        result = evaluate(_make_context("My child has fever and rash"))
        assert result.queue_priority == QueuePriority.HIGH

    def test_fall_moderate(self):
        result = evaluate(_make_context("I fell and hurt my arm"))
        assert result.queue_priority == QueuePriority.MODERATE

    def test_mild_symptoms_standard(self):
        result = evaluate(_make_context("I have a mild sore throat"))
        assert result.queue_priority == QueuePriority.STANDARD

    def test_swedish_keywords(self):
        result = evaluate(_make_context("Jag har bröstsmärta och andnöd"))
        assert result.queue_priority == QueuePriority.HIGH

    def test_ess_hint_set(self):
        result = evaluate(_make_context("I have chest tightness"))
        assert result.ess_category_hint == "chest_pain"


class TestEHRRiskAmplification:
    def test_warfarin_fall_escalation(self):
        ehr = EHRSnapshot(
            patient_id="test", name="Test", age=81, sex="F",
            medications=[FHIRMedication(code="B01AA03", display="Warfarin")],
        )
        result = evaluate(_make_context(
            "I fell at home and hit my head", ehr=ehr
        ))
        assert result.queue_priority == QueuePriority.HIGH
        assert len(result.risk_amplifiers_detected) > 0

    def test_cardiac_history_chest_escalation(self):
        ehr = EHRSnapshot(
            patient_id="test", name="Test", age=68, sex="M",
            conditions=[FHIRCondition(code="I50", display="Heart failure")],
        )
        result = evaluate(_make_context("I have chest pain", ehr=ehr))
        # Already HIGH from speech keyword
        assert result.queue_priority == QueuePriority.HIGH

    def test_no_ehr_graceful(self):
        result = evaluate(_make_context("I fell and hurt my arm"))
        assert result.queue_priority == QueuePriority.MODERATE
        assert len(result.risk_amplifiers_detected) == 0


class TestScenarios:
    def test_anders_chest_tightness(self):
        ehr = EHRSnapshot(
            patient_id="anders", name="Anders", age=68, sex="M",
            conditions=[
                FHIRCondition(code="I50", display="Heart failure"),
                FHIRCondition(code="E11", display="Diabetes mellitus type 2"),
            ],
            medications=[FHIRMedication(code="B01AA03", display="Warfarin")],
        )
        result = evaluate(_make_context(
            "I have chest tightness and difficulty breathing since this morning",
            ehr=ehr,
        ))
        assert result.queue_priority == QueuePriority.HIGH

    def test_ella_fever_rash(self):
        result = evaluate(_make_context(
            "My daughter has fever and rash since yesterday",
            ehr=EHRSnapshot(patient_id="ella", name="Ella", age=4, sex="F"),
        ))
        assert result.queue_priority == QueuePriority.HIGH

    def test_margit_fall_warfarin(self):
        ehr = EHRSnapshot(
            patient_id="margit", name="Margit", age=81, sex="F",
            conditions=[FHIRCondition(code="M81", display="Osteoporosis")],
            medications=[FHIRMedication(code="B01AA03", display="Warfarin")],
        )
        result = evaluate(_make_context(
            "I fell at home and hit my head on the floor",
            ehr=ehr,
        ))
        assert result.queue_priority == QueuePriority.HIGH
        assert any("anticoagulant" in a.lower() for a in result.risk_amplifiers_detected)


class TestOutputStructure:
    def test_output_fields(self):
        result = evaluate(_make_context("I have chest pain"))
        assert result.model_id == "pretriage_engine"
        assert result.chief_complaint
        assert result.reasoning
        assert result.processing_time_ms is not None
        assert result.processing_time_ms >= 0
