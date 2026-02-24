"""Tests for RETTS Rules Engine."""

import pytest

from src.engines.retts_engine import evaluate
from src.models import (
    ConsciousnessLevel,
    FullTriageContext,
    PreTriageContext,
    RETTSLevel,
    VitalSigns,
    EHRSnapshot,
    FHIRCondition,
    FHIRMedication,
    RiskFlag,
)


def _make_context(
    vitals: VitalSigns,
    ess_category: str | None = None,
    ehr: EHRSnapshot | None = None,
) -> FullTriageContext:
    return FullTriageContext(
        patient_id="test",
        speech_text="Test complaint",
        vitals=vitals,
        ess_category=ess_category,
        ehr=ehr,
    )


def _normal_adult_vitals() -> VitalSigns:
    return VitalSigns(
        heart_rate=75, systolic_bp=120, diastolic_bp=80,
        respiratory_rate=16, spo2=98, temperature=37.0,
        consciousness=ConsciousnessLevel.ALERT,
    )


class TestRETTSEngineBasics:
    def test_rejects_pretriage_context(self):
        ctx = PreTriageContext(patient_id="test", speech_text="test")
        with pytest.raises(ValueError, match="FullTriageContext"):
            evaluate(ctx)

    def test_normal_vitals_green(self):
        result = evaluate(_make_context(_normal_adult_vitals()))
        assert result.retts_level == RETTSLevel.GREEN
        assert result.model_id == "retts_rules_engine"
        assert result.confidence.value == "HIGH"

    def test_returns_triage_output(self):
        result = evaluate(_make_context(_normal_adult_vitals()))
        assert result.processing_time_ms is not None
        assert result.processing_time_ms >= 0


class TestRETTSVitals:
    def test_tachycardia_triggers_red(self):
        vitals = VitalSigns(
            heart_rate=155, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.RED
        assert any("Heart rate" in c for c in result.vital_sign_concerns)

    def test_hypotension_triggers_red(self):
        vitals = VitalSigns(
            heart_rate=100, systolic_bp=70, diastolic_bp=40,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.RED

    def test_low_spo2_triggers_red(self):
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=82, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.RED

    def test_moderate_tachycardia_orange(self):
        vitals = VitalSigns(
            heart_rate=135, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=96, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.ORANGE

    def test_unconscious_triggers_red(self):
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.UNRESPONSIVE,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.RED

    def test_responds_to_pain_orange(self):
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.PAIN,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.ORANGE

    def test_fever_yellow(self):
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=39.5,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.YELLOW


class TestRETTSESSIntegration:
    def test_ess_chest_pain_yellow(self):
        result = evaluate(_make_context(_normal_adult_vitals(), ess_category="chest_pain"))
        assert result.retts_level == RETTSLevel.YELLOW

    def test_ess_overridden_by_worse_vitals(self):
        vitals = VitalSigns(
            heart_rate=155, systolic_bp=70, diastolic_bp=40,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals, ess_category="chest_pain"))
        assert result.retts_level == RETTSLevel.RED


class TestRETTSScenarios:
    """Test with the 3 demo scenario vitals."""

    def test_anders_vitals(self):
        """Anders: 68M, chest tightness. Mildly elevated HR/BP → ORANGE."""
        vitals = VitalSigns(
            heart_rate=92, systolic_bp=145, diastolic_bp=85,
            respiratory_rate=20, spo2=94, temperature=37.1,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals, ess_category="chest_pain"))
        # SpO2 94 is YELLOW, ESS chest_pain default is YELLOW → YELLOW overall
        assert result.retts_level in (RETTSLevel.YELLOW, RETTSLevel.ORANGE)

    def test_ella_vitals(self):
        """Ella: 4F, fever + rash. High HR and fever for age."""
        ehr = EHRSnapshot(patient_id="ella", name="Ella", age=4, sex="F")
        vitals = VitalSigns(
            heart_rate=185, systolic_bp=90, diastolic_bp=55,
            respiratory_rate=32, spo2=96, temperature=39.8,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals, ess_category="pediatric_fever", ehr=ehr))
        # Pediatric HR 185 → ORANGE (>=180), temp 39.8 → YELLOW
        assert result.retts_level in (RETTSLevel.ORANGE, RETTSLevel.RED)

    def test_margit_vitals(self):
        """Margit: 81F, fall on warfarin. Mild hypertension → YELLOW."""
        ehr = EHRSnapshot(
            patient_id="margit", name="Margit", age=81, sex="F",
            conditions=[FHIRCondition(code="M81", display="Osteoporosis")],
            medications=[FHIRMedication(code="B01AA03", display="Warfarin")],
            risk_flags=[RiskFlag(
                flag_type="anticoagulation_risk",
                description="On warfarin — bleeding risk",
                severity="high",
            )],
        )
        vitals = VitalSigns(
            heart_rate=78, systolic_bp=155, diastolic_bp=88,
            respiratory_rate=18, spo2=96, temperature=36.8,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals, ess_category="trauma_fall", ehr=ehr))
        assert result.retts_level == RETTSLevel.YELLOW
        assert len(result.risk_factors) > 0
