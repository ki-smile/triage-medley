"""Integration tests — full pipeline through all 3 scenarios."""

import pytest

from src.engines.agreement_engine import (
    AgreementLevel,
    analyze_differential,
    analyze_management,
    analyze_triage,
)
from src.models import (
    ConsciousnessLevel,
    FullTriageContext,
    PreTriageContext,
    QueuePriority,
    RETTSLevel,
    VitalSigns,
)
from src.services.ehr_service import load_patient
from src.services.orchestrator import (
    run_full_pipeline,
    run_pretriage,
    run_triage_ensemble,
)
from src.utils.audit import clear_audit_log, get_events


@pytest.fixture(autouse=True)
def clean_audit():
    clear_audit_log()
    yield
    clear_audit_log()


# ---- Helper fixtures ----

def _anders_pretriage_ctx():
    ehr = load_patient("anders")
    return PreTriageContext(
        patient_id="anders",
        speech_text="I have chest tightness and difficulty breathing since this morning",
        ehr=ehr,
    )


def _anders_full_ctx():
    ehr = load_patient("anders")
    return FullTriageContext(
        patient_id="anders",
        speech_text="I have chest tightness and difficulty breathing since this morning",
        vitals=VitalSigns(
            heart_rate=92, systolic_bp=145, diastolic_bp=85,
            respiratory_rate=20, spo2=94, temperature=37.1,
            consciousness=ConsciousnessLevel.ALERT,
        ),
        ess_category="chest_pain",
        ehr=ehr,
    )


def _ella_full_ctx():
    ehr = load_patient("ella")
    return FullTriageContext(
        patient_id="ella",
        speech_text="My daughter has had fever and rash since yesterday",
        vitals=VitalSigns(
            heart_rate=185, systolic_bp=85, diastolic_bp=50,
            respiratory_rate=34, spo2=96, temperature=39.8,
            consciousness=ConsciousnessLevel.ALERT,
        ),
        ess_category="pediatric_fever",
        ehr=ehr,
    )


def _margit_full_ctx():
    ehr = load_patient("margit")
    return FullTriageContext(
        patient_id="margit",
        speech_text="I fell at home and hit my head on the floor",
        vitals=VitalSigns(
            heart_rate=78, systolic_bp=155, diastolic_bp=88,
            respiratory_rate=18, spo2=96, temperature=36.8,
            consciousness=ConsciousnessLevel.ALERT,
        ),
        ess_category="trauma_fall",
        ehr=ehr,
    )


class TestAndersPipeline:
    """Anders: 68M, chest tightness — consensus scenario."""

    def test_pretriage_high(self):
        result = run_pretriage(_anders_pretriage_ctx())
        assert result.engine_output.queue_priority == QueuePriority.HIGH
        assert result.model_output is not None
        assert result.model_output.queue_priority == QueuePriority.HIGH

    def test_triage_consensus_orange(self):
        ensemble = run_triage_ensemble(_anders_full_ctx())
        agreement = analyze_triage(ensemble)
        assert agreement.consensus_level == RETTSLevel.ORANGE
        assert agreement.final_level == RETTSLevel.ORANGE
        assert len(ensemble.failed_models) == 0

    def test_full_pipeline(self):
        t, d, m = run_full_pipeline(_anders_full_ctx())
        assert len(t.model_outputs) == 5
        assert len(d.model_outputs) == 5
        assert len(m.model_outputs) >= 2

    def test_audit_trail(self):
        run_full_pipeline(_anders_full_ctx())
        events = get_events(patient_id="anders")
        assert len(events) > 0
        actions = {e.action for e in events}
        assert "triage_result" in actions


class TestEllaPipeline:
    """Ella: 4F, fever + rash — disagreement scenario (minority catches meningococcal)."""

    def test_triage_disagreement(self):
        ensemble = run_triage_ensemble(_ella_full_ctx())
        agreement = analyze_triage(ensemble)

        # Majority says ORANGE, minority says RED
        assert agreement.consensus_level == RETTSLevel.ORANGE
        assert agreement.final_level == RETTSLevel.RED  # Escalated by minority
        assert agreement.requires_senior_review is True

    def test_biomistral_flags_red(self):
        ensemble = run_triage_ensemble(_ella_full_ctx())
        biomistral_outputs = [
            o for o in ensemble.model_outputs if o.model_id == "biomistral"
        ]
        assert len(biomistral_outputs) == 1
        assert biomistral_outputs[0].retts_level == RETTSLevel.RED

    def test_dont_miss_meningococcal(self):
        t, d, _ = run_full_pipeline(_ella_full_ctx())
        ta = analyze_triage(t)
        da = analyze_differential(d)

        # Meningococcal should appear in don't-miss alerts
        all_dont_miss = " ".join(ta.dont_miss_alerts + da.dont_miss_all).lower()
        assert "meningococcal" in all_dont_miss

    def test_differential_shows_disagreement(self):
        _, d, _ = run_full_pipeline(_ella_full_ctx())
        da = analyze_differential(d)

        # BioMistral's devil's advocate diagnoses should appear in minority
        assert len(da.devil_advocate_only) > 0 or len(da.some_agree) > 0
        assert len(da.dont_miss_all) > 0


class TestMargitPipeline:
    """Margit: 81F, fall on warfarin — escalation scenario."""

    def test_pretriage_escalated_high(self):
        ehr = load_patient("margit")
        ctx = PreTriageContext(
            patient_id="margit",
            speech_text="I fell at home and hit my head on the floor",
            ehr=ehr,
        )
        result = run_pretriage(ctx)
        assert result.engine_output.queue_priority == QueuePriority.HIGH

    def test_triage_split(self):
        ensemble = run_triage_ensemble(_margit_full_ctx())
        agreement = analyze_triage(ensemble)

        # Some models say YELLOW, some ORANGE
        assert "YELLOW" in agreement.vote_distribution or "ORANGE" in agreement.vote_distribution
        # At least one model should flag ORANGE (warfarin + head injury)
        levels = [o.retts_level for o in ensemble.model_outputs]
        assert RETTSLevel.ORANGE in levels

    def test_anticoag_in_risk_factors(self):
        """Warfarin risk should appear in model reasoning."""
        ensemble = run_triage_ensemble(_margit_full_ctx())
        all_risk_factors = []
        for o in ensemble.model_outputs:
            all_risk_factors.extend(o.risk_factors)
        risk_text = " ".join(all_risk_factors).lower()
        assert "warfarin" in risk_text or "anticoag" in risk_text

    def test_full_pipeline_completes(self):
        t, d, m = run_full_pipeline(_margit_full_ctx())
        assert len(t.model_outputs) == 5
        assert len(d.model_outputs) == 5
        assert len(m.model_outputs) >= 2


class TestGracefulDegradation:
    """Test that the pipeline degrades gracefully."""

    def test_unknown_patient_still_works(self):
        """Pipeline should work with unknown patient (no mock data)."""
        ctx = FullTriageContext(
            patient_id="unknown_patient",
            speech_text="I have a headache",
            vitals=VitalSigns(
                heart_rate=80, systolic_bp=120, diastolic_bp=80,
                respiratory_rate=16, spo2=98, temperature=37.0,
                consciousness=ConsciousnessLevel.ALERT,
            ),
        )
        t, d, m = run_full_pipeline(ctx)
        # Should get fallback responses, not crash
        assert t.retts_output is not None
        assert len(t.model_outputs) == 5
        assert len(t.failed_models) == 0

    def test_pretriage_without_ehr(self):
        """Pre-triage should work without EHR (speech-only mode)."""
        ctx = PreTriageContext(
            patient_id="walk_in",
            speech_text="I can't breathe, please help",
        )
        result = run_pretriage(ctx)
        assert result.engine_output.queue_priority == QueuePriority.HIGH


class TestASRService:
    """Test ASR service integration."""

    def test_anders_asr_disagreement(self):
        from src.services.asr_service import process_audio
        result = process_audio("anders")
        assert result.confidence_score > 0
        assert result.has_critical_disagreements  # warfarin/waran

    def test_ella_asr_clean(self):
        from src.services.asr_service import process_audio
        result = process_audio("ella")
        assert result.disagreement_count == 0
        assert result.confidence_score == 1.0

    def test_word_alignment_detection(self):
        from src.services.asr_service import (
            compute_word_alignment,
            detect_disagreements,
        )
        words1 = "patient takes warfarin daily".split()
        words2 = "patient takes waran daily".split()
        alignment = compute_word_alignment(words1, words2)
        disagreements = detect_disagreements(alignment)
        assert len(disagreements) >= 1
        assert any(d.clinical_significance == "high" for d in disagreements)
