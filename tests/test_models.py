"""Tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from src.models import (
    ConsciousnessLevel,
    FullTriageContext,
    PreTriageContext,
    QueuePriority,
    RETTSLevel,
    VitalSigns,
    EHRSnapshot,
    FHIRCondition,
    FHIRMedication,
    TriageOutput,
    PreTriageOutput,
    Confidence,
)


class TestRETTSLevel:
    def test_severity_ordering(self):
        assert RETTSLevel.RED < RETTSLevel.ORANGE
        assert RETTSLevel.ORANGE < RETTSLevel.YELLOW
        assert RETTSLevel.YELLOW < RETTSLevel.GREEN
        assert RETTSLevel.GREEN < RETTSLevel.BLUE

    def test_most_severe(self):
        assert RETTSLevel.most_severe(RETTSLevel.YELLOW, RETTSLevel.ORANGE) == RETTSLevel.ORANGE
        assert RETTSLevel.most_severe(RETTSLevel.BLUE, RETTSLevel.RED) == RETTSLevel.RED
        assert RETTSLevel.most_severe(RETTSLevel.GREEN, RETTSLevel.GREEN) == RETTSLevel.GREEN

    def test_string_values(self):
        assert RETTSLevel.RED.value == "RED"
        assert RETTSLevel("ORANGE") == RETTSLevel.ORANGE


class TestVitalSigns:
    def test_valid_vitals(self):
        v = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        assert v.heart_rate == 80
        assert v.consciousness == ConsciousnessLevel.ALERT

    def test_rejects_invalid_hr(self):
        with pytest.raises(ValidationError):
            VitalSigns(
                heart_rate=-5, systolic_bp=120, diastolic_bp=80,
                respiratory_rate=16, spo2=98, temperature=37.0,
                consciousness=ConsciousnessLevel.ALERT,
            )

    def test_rejects_invalid_spo2(self):
        with pytest.raises(ValidationError):
            VitalSigns(
                heart_rate=80, systolic_bp=120, diastolic_bp=80,
                respiratory_rate=16, spo2=105, temperature=37.0,
                consciousness=ConsciousnessLevel.ALERT,
            )


class TestPreTriageContext:
    def test_no_vitals_field(self):
        assert "vitals" not in PreTriageContext.model_fields

    def test_creates_with_speech_only(self):
        ctx = PreTriageContext(patient_id="test", speech_text="I feel unwell")
        assert ctx.patient_id == "test"
        assert ctx.ehr is None  # Graceful degradation

    def test_requires_speech(self):
        with pytest.raises(ValidationError):
            PreTriageContext(patient_id="test")


class TestFullTriageContext:
    def test_requires_vitals(self):
        with pytest.raises(ValidationError):
            FullTriageContext(patient_id="test", speech_text="test")

    def test_creates_with_vitals(self):
        v = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        ctx = FullTriageContext(
            patient_id="test", speech_text="test", vitals=v
        )
        assert ctx.vitals.heart_rate == 80

    def test_inherits_pretriage_fields(self):
        v = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        ctx = FullTriageContext(
            patient_id="test", speech_text="chest pain", vitals=v,
            ess_category="chest_pain",
        )
        assert ctx.speech_text == "chest pain"
        assert ctx.ess_category == "chest_pain"


class TestEHRSnapshot:
    def test_pediatric_detection(self):
        ehr = EHRSnapshot(patient_id="child", name="Test", age=4, sex="F")
        assert ehr.is_pediatric is True

        ehr_adult = EHRSnapshot(patient_id="adult", name="Test", age=30, sex="M")
        assert ehr_adult.is_pediatric is False

    def test_medication_matching(self):
        ehr = EHRSnapshot(
            patient_id="test", name="Test", age=70, sex="M",
            medications=[
                FHIRMedication(code="B01AA03", display="Warfarin"),
            ],
        )
        assert ehr.has_medication_class(["warfarin"]) is True
        assert ehr.has_medication_class(["aspirin"]) is False

    def test_condition_matching(self):
        ehr = EHRSnapshot(
            patient_id="test", name="Test", age=70, sex="M",
            conditions=[
                FHIRCondition(code="I50", display="Heart failure"),
            ],
        )
        assert ehr.has_condition_matching(["heart"]) is True
        assert ehr.has_condition_matching(["diabetes"]) is False
