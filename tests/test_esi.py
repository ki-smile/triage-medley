"""Tests for ESI Rules Engine."""

import pytest

from src.engines.esi_engine import evaluate
from src.models import (
    ConsciousnessLevel,
    FullTriageContext,
    PreTriageContext,
    RETTSLevel,
    VitalSigns,
    EHRSnapshot,
    RiskFlag,
)


def _make_context(
    vitals: VitalSigns,
    ess_category: str | None = None,
    speech_text: str = "Test complaint",
    ehr: EHRSnapshot | None = None,
) -> FullTriageContext:
    return FullTriageContext(
        patient_id="test",
        speech_text=speech_text,
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


class TestESIEngineBasics:
    def test_rejects_pretriage_context(self):
        ctx = PreTriageContext(patient_id="test", speech_text="test")
        with pytest.raises(ValueError, match="FullTriageContext"):
            evaluate(ctx)

    def test_returns_triage_output(self):
        result = evaluate(_make_context(_normal_adult_vitals()))
        assert result.model_id == "esi_rules_engine"
        assert result.triage_system == "esi"
        assert result.confidence.value == "HIGH"
        assert result.processing_time_ms is not None

    def test_native_level_detail_present(self):
        result = evaluate(_make_context(_normal_adult_vitals()))
        assert result.native_level_detail is not None
        assert "esi_level" in result.native_level_detail
        assert "resources_predicted" in result.native_level_detail


class TestESI1:
    """ESI-1: Dying patient — immediate life-saving intervention."""

    def test_unresponsive_triggers_esi_1(self):
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.UNRESPONSIVE,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.RED
        assert result.native_level_detail["esi_level"] == 1

    def test_critical_bradycardia_triggers_esi_1(self):
        vitals = VitalSigns(
            heart_rate=25, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.RED
        assert result.native_level_detail["esi_level"] == 1

    def test_critical_hypotension_triggers_esi_1(self):
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=50, diastolic_bp=30,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.RED
        assert result.native_level_detail["esi_level"] == 1

    def test_critical_hypoxia_triggers_esi_1(self):
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=75, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.RED
        assert result.native_level_detail["esi_level"] == 1

    def test_intervention_keyword_triggers_esi_1(self):
        vitals = _normal_adult_vitals()
        result = evaluate(_make_context(
            vitals, speech_text="Patient in cardiac arrest, CPR in progress"
        ))
        assert result.retts_level == RETTSLevel.RED
        assert result.native_level_detail["esi_level"] == 1


class TestESI2:
    """ESI-2: High-risk / altered consciousness."""

    def test_altered_consciousness_pain(self):
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.PAIN,
        )
        result = evaluate(_make_context(vitals))
        assert result.retts_level == RETTSLevel.ORANGE
        assert result.native_level_detail["esi_level"] == 2

    def test_high_risk_keyword_chest_pain(self):
        vitals = _normal_adult_vitals()
        result = evaluate(_make_context(
            vitals, speech_text="I have severe chest pain since this morning"
        ))
        assert result.retts_level == RETTSLevel.ORANGE
        assert result.native_level_detail["esi_level"] == 2

    def test_high_risk_keyword_breathing_difficulty(self):
        vitals = _normal_adult_vitals()
        result = evaluate(_make_context(
            vitals, speech_text="I have breathing difficulty and feel faint"
        ))
        assert result.retts_level == RETTSLevel.ORANGE
        assert result.native_level_detail["esi_level"] == 2

    def test_dont_miss_populated_for_esi_2(self):
        vitals = _normal_adult_vitals()
        result = evaluate(_make_context(
            vitals, speech_text="Severe chest pain radiating to arm"
        ))
        assert len(result.dont_miss) > 0


class TestESI3to5:
    """ESI-3/4/5: Resource-based determination."""

    def test_chest_pain_category_esi_3(self):
        """Chest pain expects >=2 resources → ESI-3."""
        vitals = _normal_adult_vitals()
        result = evaluate(_make_context(
            vitals, ess_category="chest_pain",
            speech_text="Mild discomfort in chest area"
        ))
        # Not ESI-1 or ESI-2 (no keywords, normal vitals/consciousness)
        # chest_pain = 3 resources → ESI-3
        assert result.native_level_detail["esi_level"] == 3
        assert result.retts_level == RETTSLevel.YELLOW

    def test_skin_wound_category_esi_4(self):
        """Skin wound expects 1 resource → ESI-4."""
        vitals = _normal_adult_vitals()
        result = evaluate(_make_context(
            vitals, ess_category="skin_wound",
            speech_text="Small cut on my hand"
        ))
        assert result.native_level_detail["esi_level"] == 4
        assert result.retts_level == RETTSLevel.GREEN

    def test_medication_refill_esi_5(self):
        """Medication refill expects 0 resources → ESI-5."""
        vitals = _normal_adult_vitals()
        result = evaluate(_make_context(
            vitals, ess_category="medication_refill",
            speech_text="I need a refill of my prescription"
        ))
        assert result.native_level_detail["esi_level"] == 5
        assert result.retts_level == RETTSLevel.BLUE


class TestESIDangerZone:
    """ESI danger zone safety net: ESI-3 → ESI-2 if vitals abnormal."""

    def test_danger_zone_tachycardia_upgrades_esi_3(self):
        """High HR in ESI-3 patient → upgrade to ESI-2."""
        vitals = VitalSigns(
            heart_rate=140, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(
            vitals, ess_category="neurological",
            speech_text="I feel a bit dizzy today"
        ))
        assert result.native_level_detail["esi_level"] == 2
        assert result.retts_level == RETTSLevel.ORANGE

    def test_danger_zone_hypotension_upgrades_esi_3(self):
        """Low SBP in ESI-3 patient → upgrade to ESI-2."""
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=85, diastolic_bp=50,
            respiratory_rate=16, spo2=98, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(
            vitals, ess_category="neurological",
            speech_text="I feel a bit dizzy today"
        ))
        assert result.native_level_detail["esi_level"] == 2

    def test_danger_zone_hypoxia_upgrades_esi_3(self):
        """Low SpO2 in ESI-3 patient → upgrade to ESI-2."""
        vitals = VitalSigns(
            heart_rate=80, systolic_bp=120, diastolic_bp=80,
            respiratory_rate=16, spo2=90, temperature=37.0,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(
            vitals, ess_category="neurological",
            speech_text="I feel a bit dizzy today"
        ))
        assert result.native_level_detail["esi_level"] == 2


class TestESIScenarios:
    """Test with the 6 demo scenario vitals."""

    def test_anders_chest_tightness(self):
        """Anders: 68M, chest tightness + breathing difficulty.

        'chest tightness' doesn't match exact 'chest pain' keyword.
        'difficulty breathing' doesn't match exact 'breathing difficulty' keyword.
        → ESI-3 (3 resources from chest_pain category, no danger zone).
        This shows ESI's keyword-sensitivity limitation — a reason to use multi-engine.
        """
        vitals = VitalSigns(
            heart_rate=92, systolic_bp=145, diastolic_bp=85,
            respiratory_rate=20, spo2=94, temperature=37.1,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(
            vitals, ess_category="chest_pain",
            speech_text="I have chest tightness and difficulty breathing since this morning.",
        ))
        assert result.native_level_detail["esi_level"] == 3
        assert result.retts_level == RETTSLevel.YELLOW

    def test_ella_pediatric_fever(self):
        """Ella: 4F, fever + rash. HR 185 > 180 → ESI-1 critical tachycardia."""
        vitals = VitalSigns(
            heart_rate=185, systolic_bp=85, diastolic_bp=50,
            respiratory_rate=34, spo2=96, temperature=39.8,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(
            vitals, ess_category="pediatric_fever",
            speech_text="My daughter has had fever and rash since yesterday. She seems very tired.",
        ))
        # HR 185 > 180 critical threshold → ESI-1 (no age stratification in ESI)
        assert result.native_level_detail["esi_level"] == 1

    def test_sofia_chest_pain(self):
        """Sofia: 28F, 'pulled muscle'. ESI should flag chest pain keywords."""
        vitals = VitalSigns(
            heart_rate=102, systolic_bp=122, diastolic_bp=76,
            respiratory_rate=20, spo2=95, temperature=37.2,
            consciousness=ConsciousnessLevel.ALERT,
        )
        result = evaluate(_make_context(
            vitals, ess_category="chest_pain",
            speech_text="I think I pulled a muscle in my chest. It hurts when I take a deep breath.",
        ))
        # No high-risk keywords → resource-based
        # chest_pain = 3 resources → ESI-3
        # But SpO2 95 is NOT in danger zone (threshold is <92)
        assert result.native_level_detail["esi_level"] in (2, 3)
