"""Tests for multi-engine triage orchestration and agreement analysis."""

import pytest

from src.engines.agreement_engine import (
    AgreementLevel,
    EngineDisagreement,
    analyze_engine_disagreement,
    analyze_triage,
)
from src.engines.esi_engine import evaluate as esi_evaluate
from src.engines.mts_engine import evaluate as mts_evaluate
from src.engines.retts_engine import evaluate as retts_evaluate
from src.models import (
    ConsciousnessLevel,
    FullTriageContext,
    RETTSLevel,
    VitalSigns,
    EHRSnapshot,
    RiskFlag,
)
from src.models.outputs import TriageOutput
from src.services.orchestrator import (
    TriageEnsembleResult,
    _ENGINE_REGISTRY,
    DEFAULT_ACTIVE_ENGINES,
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


class TestEngineRegistry:
    def test_registry_has_three_engines(self):
        assert "retts" in _ENGINE_REGISTRY
        assert "esi" in _ENGINE_REGISTRY
        assert "mts" in _ENGINE_REGISTRY
        assert len(_ENGINE_REGISTRY) == 3

    def test_default_active_engines(self):
        assert DEFAULT_ACTIVE_ENGINES == ["retts"]

    def test_all_engines_produce_triage_output(self):
        ctx = _make_context(_normal_adult_vitals(), ess_category="chest_pain")
        for engine_id, evaluate_fn in _ENGINE_REGISTRY.items():
            result = evaluate_fn(ctx)
            assert isinstance(result, TriageOutput)
            assert result.triage_system in ("retts", "esi", "mts")


class TestTriageEnsembleResult:
    def test_backward_compat_retts_output(self):
        retts_out = retts_evaluate(
            _make_context(_normal_adult_vitals(), ess_category="chest_pain")
        )
        ensemble = TriageEnsembleResult(engine_outputs=[retts_out])
        assert ensemble.retts_output is not None
        assert ensemble.retts_output.model_id == "retts_rules_engine"

    def test_retts_output_none_when_retts_not_active(self):
        esi_out = esi_evaluate(
            _make_context(_normal_adult_vitals(), ess_category="chest_pain")
        )
        ensemble = TriageEnsembleResult(engine_outputs=[esi_out])
        assert ensemble.retts_output is None

    def test_multi_engine_outputs(self):
        ctx = _make_context(_normal_adult_vitals(), ess_category="chest_pain")
        outputs = [evaluate(ctx) for evaluate in _ENGINE_REGISTRY.values()]
        ensemble = TriageEnsembleResult(engine_outputs=outputs)
        assert len(ensemble.engine_outputs) == 3
        assert ensemble.retts_output is not None

    def test_failed_engines_tracked(self):
        ensemble = TriageEnsembleResult(
            engine_outputs=[],
            failed_engines=["bad_engine"],
        )
        assert "bad_engine" in ensemble.failed_engines


class TestCrossEngineAgreement:
    """Test all three engines on the same patient."""

    def test_normal_vitals_all_engines(self):
        """Normal vitals: RETTS and MTS should give GREEN, ESI resource-dependent."""
        ctx = _make_context(
            _normal_adult_vitals(),
            ess_category="chest_pain",
            speech_text="Mild discomfort",
        )
        retts_out = retts_evaluate(ctx)
        esi_out = esi_evaluate(ctx)
        mts_out = mts_evaluate(ctx)

        # RETTS: chest_pain ESS → YELLOW
        assert retts_out.retts_level == RETTSLevel.YELLOW
        # ESI: no high-risk keywords, chest_pain = 3 resources → ESI-3 = YELLOW
        assert esi_out.retts_level == RETTSLevel.YELLOW
        # MTS: no discriminator keywords → default GREEN
        assert mts_out.retts_level == RETTSLevel.GREEN

    def test_critical_patient_all_engines_agree(self):
        """Critical patient: all engines should agree on RED."""
        vitals = VitalSigns(
            heart_rate=25, systolic_bp=50, diastolic_bp=30,
            respiratory_rate=4, spo2=70, temperature=37.0,
            consciousness=ConsciousnessLevel.UNRESPONSIVE,
        )
        ctx = _make_context(vitals, ess_category="chest_pain")
        for evaluate_fn in _ENGINE_REGISTRY.values():
            result = evaluate_fn(ctx)
            assert result.retts_level == RETTSLevel.RED

    def test_ingrid_divergence(self):
        """Ingrid: deceptively normal vitals — engines should mostly agree on low acuity."""
        ctx = _make_context(
            VitalSigns(
                heart_rate=78, systolic_bp=118, diastolic_bp=68,
                respiratory_rate=20, spo2=96, temperature=36.4,
                consciousness=ConsciousnessLevel.ALERT,
            ),
            ess_category="neurological",
            speech_text="I'm just not myself today. I feel weak and a little dizzy.",
        )
        retts_out = retts_evaluate(ctx)
        mts_out = mts_evaluate(ctx)

        # Both should give low acuity — this is the undertriage trap
        # that LLMs with EHR context are meant to catch
        assert retts_out.retts_level in (RETTSLevel.GREEN, RETTSLevel.YELLOW)
        assert mts_out.retts_level == RETTSLevel.GREEN


class TestAnalyzeEngineDisagreement:
    def test_none_with_single_engine(self):
        retts_out = retts_evaluate(
            _make_context(_normal_adult_vitals(), ess_category="chest_pain")
        )
        result = analyze_engine_disagreement([retts_out])
        assert result is None

    def test_agreement_with_matching_levels(self):
        """Create two outputs with same level → engines_agree = True."""
        out1 = TriageOutput(
            model_id="engine_a", retts_level=RETTSLevel.YELLOW,
            chief_complaint="test", clinical_reasoning="test",
            triage_system="retts",
        )
        out2 = TriageOutput(
            model_id="engine_b", retts_level=RETTSLevel.YELLOW,
            chief_complaint="test", clinical_reasoning="test",
            triage_system="esi",
        )
        result = analyze_engine_disagreement([out1, out2])
        assert result is not None
        assert result.engines_agree is True
        assert "agree" in result.disagreement_explanation.lower()

    def test_disagreement_with_different_levels(self):
        """Create two outputs with different levels → engines_agree = False."""
        out1 = TriageOutput(
            model_id="engine_a", retts_level=RETTSLevel.YELLOW,
            chief_complaint="test", clinical_reasoning="test",
            triage_system="retts",
        )
        out2 = TriageOutput(
            model_id="engine_b", retts_level=RETTSLevel.ORANGE,
            chief_complaint="test", clinical_reasoning="test",
            triage_system="esi",
        )
        result = analyze_engine_disagreement([out1, out2])
        assert result is not None
        assert result.engines_agree is False
        assert result.most_severe_engine == "ESI"
        assert "disagreement" in result.disagreement_explanation.lower()

    def test_three_engine_disagreement(self):
        ctx = _make_context(
            VitalSigns(
                heart_rate=92, systolic_bp=145, diastolic_bp=85,
                respiratory_rate=20, spo2=94, temperature=37.1,
                consciousness=ConsciousnessLevel.ALERT,
            ),
            ess_category="chest_pain",
            speech_text="I have chest tightness and difficulty breathing since this morning.",
        )
        retts_out = retts_evaluate(ctx)
        esi_out = esi_evaluate(ctx)
        mts_out = mts_evaluate(ctx)

        result = analyze_engine_disagreement([retts_out, esi_out, mts_out])
        assert result is not None
        assert len(result.engine_levels) == 3


class TestAnalyzeTriageWithMultipleEngines:
    """Test the agreement engine with multi-engine TriageEnsembleResult."""

    def test_agreement_with_engines_list(self):
        """analyze_triage should work with engine_outputs instead of retts_output."""
        ctx = _make_context(_normal_adult_vitals(), ess_category="chest_pain")
        retts_out = retts_evaluate(ctx)
        esi_out = esi_evaluate(ctx)

        ensemble = TriageEnsembleResult(
            engine_outputs=[retts_out, esi_out],
            model_outputs=[],
        )
        agreement = analyze_triage(ensemble)
        assert agreement.total_voters == 2  # 2 engines, 0 models
        assert agreement.agreement_level in (
            AgreementLevel.FULL, AgreementLevel.PARTIAL, AgreementLevel.NONE
        )

    def test_agreement_engines_plus_models(self):
        """Agreement analysis should count both engines and models."""
        ctx = _make_context(_normal_adult_vitals(), ess_category="chest_pain")
        retts_out = retts_evaluate(ctx)
        esi_out = esi_evaluate(ctx)

        # Simulate a model output
        model_out = TriageOutput(
            model_id="mock_model_a",
            retts_level=RETTSLevel.YELLOW,
            chief_complaint="test",
            clinical_reasoning="test reasoning",
        )

        ensemble = TriageEnsembleResult(
            engine_outputs=[retts_out, esi_out],
            model_outputs=[model_out],
        )
        agreement = analyze_triage(ensemble)
        assert agreement.total_voters == 3  # 2 engines + 1 model
