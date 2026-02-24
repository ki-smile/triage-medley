"""Tests for HuggingFace adapters — uses mocked API responses.

Tests cover:
1. Prompt builder renders correct templates
2. HF base adapter parses JSON responses (clean, code-block, embedded)
3. Factory creates HF adapters when config says "huggingface"
4. All 5 concrete adapters produce valid output types
5. Mock vs HF adapters produce structurally compatible outputs
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.base import ModelAdapter
from src.adapters.hf_base import HFBaseAdapter, _extract_json, _safe_enum
from src.adapters.hf_ensemble import BioMistralAdapter, MeditronAdapter
from src.adapters.hf_medgemma import MedGemma4BAdapter, MedGemma27BAdapter
from src.adapters.prompt_builder import (
    build_differential_prompt,
    build_management_prompt,
    build_pretriage_prompt,
    build_triage_prompt,
)
from src.models import (
    ConsciousnessLevel,
    FullTriageContext,
    PreTriageContext,
    QueuePriority,
    RETTSLevel,
    VitalSigns,
)
from src.models.clinical import EHRSnapshot, FHIRCondition, FHIRMedication, RiskFlag
from src.models.enums import Confidence
from src.models.outputs import (
    DifferentialOutput,
    ManagementOutput,
    PreTriageOutput,
    TriageOutput,
)


# ---- Test fixtures ----

@pytest.fixture
def pretriage_ctx():
    return PreTriageContext(
        patient_id="test_patient",
        speech_text="I have chest pain and difficulty breathing",
        ehr=EHRSnapshot(
            patient_id="test_patient",
            name="Test Patient",
            age=65,
            sex="M",
            conditions=[
                FHIRCondition(code="I50", display="Heart failure", status="active"),
            ],
            medications=[
                FHIRMedication(code="B01", display="Warfarin 5mg", status="active"),
            ],
            allergies=[],
            risk_flags=[
                RiskFlag(
                    flag_type="anticoagulation_risk",
                    description="Patient on anticoagulant therapy",
                    source="ehr",
                    severity="high",
                ),
            ],
        ),
    )


@pytest.fixture
def full_ctx(pretriage_ctx):
    return FullTriageContext(
        patient_id=pretriage_ctx.patient_id,
        speech_text=pretriage_ctx.speech_text,
        ehr=pretriage_ctx.ehr,
        vitals=VitalSigns(
            heart_rate=110,
            systolic_bp=90,
            diastolic_bp=60,
            respiratory_rate=24,
            spo2=93,
            temperature=38.5,
            consciousness=ConsciousnessLevel.ALERT,
        ),
    )


# ---- Mock HF API responses ----

MOCK_PRETRIAGE_RESPONSE = json.dumps({
    "queue_priority": "HIGH",
    "chief_complaint": "Chest pain with dyspnoea",
    "reasoning": "Red-flag: chest pain in patient on anticoagulant with heart failure history",
    "ess_category_hint": "chest_pain",
    "risk_amplifiers_detected": ["anticoagulation_risk", "cardiac_history"],
})

MOCK_TRIAGE_RESPONSE = json.dumps({
    "retts_level": "ORANGE",
    "ess_category": "chest_pain",
    "chief_complaint": "Chest pain with dyspnoea",
    "clinical_reasoning": "Tachycardic, hypotensive, desaturating with cardiac history",
    "vital_sign_concerns": ["Tachycardia HR 110", "Hypotension 90/60", "SpO2 93%"],
    "risk_factors": ["Anticoagulation", "Heart failure"],
    "confidence": "HIGH",
    "dont_miss": ["Pulmonary embolism", "Acute coronary syndrome"],
})

MOCK_DIFFERENTIAL_RESPONSE = json.dumps({
    "candidates": [
        {
            "diagnosis": "Acute coronary syndrome",
            "probability": 0.45,
            "supporting_evidence": ["Chest pain", "Cardiac history", "Tachycardia"],
            "is_dont_miss": True,
        },
        {
            "diagnosis": "Pulmonary embolism",
            "probability": 0.25,
            "supporting_evidence": ["Dyspnoea", "Desaturation", "Anticoagulation"],
            "is_dont_miss": True,
        },
        {
            "diagnosis": "Decompensated heart failure",
            "probability": 0.2,
            "supporting_evidence": ["Known HF", "Dyspnoea", "Tachycardia"],
            "is_dont_miss": False,
        },
    ],
    "reasoning": "Acute presentation with cardiac risk factors",
    "confidence": "MODERATE",
})

MOCK_MANAGEMENT_RESPONSE = json.dumps({
    "investigations": ["Troponin", "D-dimer", "BNP", "CBC", "BMP"],
    "imaging": ["Chest X-ray", "CT pulmonary angiography if D-dimer elevated"],
    "medications": ["Aspirin 300mg loading (check anticoagulation status)"],
    "disposition": "admission",
    "contraindications_flagged": ["Dual anticoagulation risk with warfarin"],
    "reasoning": "Requires monitoring and serial troponins",
    "confidence": "HIGH",
})


def _mock_chat_response(content: str):
    """Create a mock HF InferenceClient chat_completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


# ---- Prompt Builder Tests ----

class TestPromptBuilder:
    def test_pretriage_prompt_has_system_and_user(self, pretriage_ctx):
        msgs = build_pretriage_prompt(pretriage_ctx)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_pretriage_prompt_contains_speech(self, pretriage_ctx):
        msgs = build_pretriage_prompt(pretriage_ctx)
        assert "chest pain" in msgs[1]["content"]

    def test_pretriage_prompt_contains_ehr(self, pretriage_ctx):
        msgs = build_pretriage_prompt(pretriage_ctx)
        assert "Heart failure" in msgs[1]["content"]
        assert "Warfarin" in msgs[1]["content"]

    def test_pretriage_prompt_no_ehr(self):
        ctx = PreTriageContext(patient_id="no_ehr", speech_text="headache")
        msgs = build_pretriage_prompt(ctx)
        assert "No EHR data available" in msgs[1]["content"]

    def test_triage_prompt_contains_vitals(self, full_ctx):
        msgs = build_triage_prompt(full_ctx, "medgemma_4b")
        content = msgs[1]["content"]
        assert "110" in content  # heart rate
        assert "90" in content   # systolic BP
        assert "93" in content   # SpO2

    def test_differential_prompt_has_risk_flags(self, full_ctx):
        msgs = build_differential_prompt(full_ctx, "medgemma_4b")
        content = msgs[1]["content"]
        assert "anticoagulant" in content.lower()

    def test_management_prompt_structure(self, full_ctx):
        msgs = build_management_prompt(full_ctx, "medgemma_4b")
        assert len(msgs) == 2
        assert "management" in msgs[0]["content"].lower()


# ---- JSON Extraction Tests ----

class TestJSONExtraction:
    def test_clean_json(self):
        text = '{"retts_level": "ORANGE", "confidence": "HIGH"}'
        result = _extract_json(text)
        assert result["retts_level"] == "ORANGE"

    def test_code_block_json(self):
        text = 'Here is my assessment:\n```json\n{"retts_level": "RED"}\n```\nEnd.'
        result = _extract_json(text)
        assert result["retts_level"] == "RED"

    def test_embedded_json(self):
        text = 'Based on analysis, {"retts_level": "YELLOW", "confidence": "LOW"} is my result.'
        result = _extract_json(text)
        assert result["retts_level"] == "YELLOW"

    def test_no_json_returns_empty(self):
        text = "I think the patient should be triaged as Orange."
        result = _extract_json(text)
        assert result == {}

    def test_malformed_json_returns_empty(self):
        text = '{"retts_level": "RED", broken}'
        result = _extract_json(text)
        assert result == {}


# ---- Safe Enum Tests ----

class TestSafeEnum:
    def test_valid_value(self):
        assert _safe_enum(RETTSLevel, "RED", RETTSLevel.YELLOW) == RETTSLevel.RED

    def test_invalid_value_returns_default(self):
        assert _safe_enum(RETTSLevel, "PURPLE", RETTSLevel.YELLOW) == RETTSLevel.YELLOW

    def test_confidence_enum(self):
        assert _safe_enum(Confidence, "HIGH", Confidence.MODERATE) == Confidence.HIGH


# ---- HF Base Adapter Tests (mocked API) ----

class TestHFBaseAdapter:
    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_pretriage_parsing(self, pretriage_ctx):
        adapter = MedGemma4BAdapter()
        with patch.object(adapter, "_chat_completion", return_value=MOCK_PRETRIAGE_RESPONSE):
            result = adapter.pretriage(pretriage_ctx)

        assert isinstance(result, PreTriageOutput)
        assert result.model_id == "medgemma_4b"
        assert result.queue_priority == QueuePriority.HIGH
        assert "chest pain" in result.chief_complaint.lower()
        assert result.processing_time_ms >= 0

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_triage_parsing(self, full_ctx):
        adapter = MedGemma4BAdapter()
        with patch.object(adapter, "_chat_completion", return_value=MOCK_TRIAGE_RESPONSE):
            result = adapter.triage(full_ctx)

        assert isinstance(result, TriageOutput)
        assert result.retts_level == RETTSLevel.ORANGE
        assert result.confidence == Confidence.HIGH
        assert len(result.dont_miss) == 2

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_differential_parsing(self, full_ctx):
        adapter = MedGemma4BAdapter()
        with patch.object(adapter, "_chat_completion", return_value=MOCK_DIFFERENTIAL_RESPONSE):
            result = adapter.differential(full_ctx)

        assert isinstance(result, DifferentialOutput)
        assert len(result.candidates) == 3
        assert result.candidates[0].probability == 0.45
        assert result.candidates[0].is_dont_miss is True

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_management_parsing(self, full_ctx):
        adapter = MedGemma4BAdapter()
        with patch.object(adapter, "_chat_completion", return_value=MOCK_MANAGEMENT_RESPONSE):
            result = adapter.management(full_ctx)

        assert isinstance(result, ManagementOutput)
        assert result.disposition == "admission"
        assert len(result.investigations) == 5
        assert len(result.contraindications_flagged) == 1

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_graceful_nonstandard_retts_value(self, full_ctx):
        """Model returns lowercase retts level — should still parse."""
        adapter = MedGemma4BAdapter()
        response = json.dumps({"retts_level": "orange", "clinical_reasoning": "test"})
        with patch.object(adapter, "_chat_completion", return_value=response):
            result = adapter.triage(full_ctx)
        assert result.retts_level == RETTSLevel.ORANGE

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_graceful_no_json_response(self, full_ctx):
        """Model returns free text — should still produce valid output."""
        adapter = MedGemma4BAdapter()
        response = "I think this patient should be triaged as Yellow based on stable vitals."
        with patch.object(adapter, "_chat_completion", return_value=response):
            result = adapter.triage(full_ctx)
        # Falls back to YELLOW default when parsing fails
        assert result.retts_level == RETTSLevel.YELLOW
        assert result.model_id == "medgemma_4b"

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_code_block_json_response(self, full_ctx):
        """Model wraps JSON in code block — should extract correctly."""
        adapter = MedGemma4BAdapter()
        response = '```json\n{"retts_level": "RED", "clinical_reasoning": "Critical", "confidence": "HIGH"}\n```'
        with patch.object(adapter, "_chat_completion", return_value=response):
            result = adapter.triage(full_ctx)
        assert result.retts_level == RETTSLevel.RED
        assert result.confidence == Confidence.HIGH

    def test_missing_hf_token_raises(self, pretriage_ctx):
        """No HF_TOKEN → RuntimeError on first API call."""
        adapter = MedGemma4BAdapter()
        with patch.dict("os.environ", {}, clear=True):
            # Remove HF_TOKEN
            import os
            os.environ.pop("HF_TOKEN", None)
            with pytest.raises(RuntimeError, match="HF_TOKEN"):
                adapter._get_client()

    def test_unsupported_stage_raises(self, pretriage_ctx):
        """27B doesn't support pretriage."""
        adapter = MedGemma27BAdapter()
        with pytest.raises(NotImplementedError, match="pretriage"):
            adapter.pretriage(pretriage_ctx)


# ---- Concrete Adapter Tests ----

class TestConcreteAdapters:
    def test_medgemma_4b_protocol_compliance(self):
        adapter = MedGemma4BAdapter()
        assert isinstance(adapter, ModelAdapter)
        assert adapter.model_id == "medgemma_4b"
        assert "pretriage" in adapter.supported_stages

    def test_medgemma_27b_stages(self):
        adapter = MedGemma27BAdapter()
        assert "pretriage" not in adapter.supported_stages
        assert "triage" in adapter.supported_stages
        assert "management" in adapter.supported_stages

    def test_meditron_stages(self):
        adapter = MeditronAdapter()
        assert adapter.model_id == "meditron_7b"
        assert adapter.hf_model_id == "OpenMeditron/Meditron3-Qwen2.5-7B"

    def test_biomistral_stages(self):
        adapter = BioMistralAdapter()
        assert adapter.model_id == "biomistral"
        assert "differential" in adapter.supported_stages


# ---- Factory Integration Tests ----

class TestFactoryHFIntegration:
    def test_factory_mock_still_works(self):
        """Default config (all mock) still works after factory update."""
        from src.adapters.factory import create_all_adapters
        adapters = create_all_adapters(adapter_type_override="mock")
        assert len(adapters) == 4
        assert all(
            type(a).__name__ == "MockAdapter"
            for a in adapters.values()
        )

    def test_factory_creates_hf_adapter(self):
        """Factory creates HF adapter when config says huggingface."""
        from src.adapters.factory import _build_hf_adapter

        adapter = _build_hf_adapter(
            model_id="medgemma_4b",
            model_name="MedGemma 4B",
            stages=["pretriage", "triage", "differential", "management"],
            model_config={
                "hf_id": "google/medgemma-4b-it",
                "timeout_seconds": 10,
            },
        )
        assert isinstance(adapter, MedGemma4BAdapter)
        assert adapter.hf_model_id == "google/medgemma-4b-it"

    def test_factory_hf_unknown_model_raises(self):
        from src.adapters.factory import _build_hf_adapter

        with pytest.raises(ValueError, match="No HuggingFace adapter"):
            _build_hf_adapter(
                model_id="unknown_model",
                model_name="Unknown",
                stages=["triage"],
                model_config={"hf_id": "unknown/model"},
            )


# ---- Mock vs HF Output Compatibility Tests ----

class TestMockHFCompatibility:
    """Verify mock and HF adapters produce structurally identical outputs."""

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_pretriage_output_fields_match(self, pretriage_ctx):
        from src.adapters.mock_adapter import MockAdapter

        mock = MockAdapter("medgemma_4b", "MedGemma 4B",
                           ["pretriage", "triage", "differential", "management"])
        mock_result = mock.pretriage(pretriage_ctx)

        hf = MedGemma4BAdapter()
        with patch.object(hf, "_chat_completion", return_value=MOCK_PRETRIAGE_RESPONSE):
            hf_result = hf.pretriage(pretriage_ctx)

        # Both return PreTriageOutput with same fields
        assert type(mock_result) is type(hf_result)
        assert type(mock_result).model_fields.keys() == type(hf_result).model_fields.keys()

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_triage_output_fields_match(self, full_ctx):
        from src.adapters.mock_adapter import MockAdapter

        mock = MockAdapter("medgemma_4b", "MedGemma 4B",
                           ["pretriage", "triage", "differential", "management"])
        mock_result = mock.triage(full_ctx)

        hf = MedGemma4BAdapter()
        with patch.object(hf, "_chat_completion", return_value=MOCK_TRIAGE_RESPONSE):
            hf_result = hf.triage(full_ctx)

        assert type(mock_result) is type(hf_result)
        assert isinstance(mock_result.retts_level, RETTSLevel)
        assert isinstance(hf_result.retts_level, RETTSLevel)

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_differential_output_fields_match(self, full_ctx):
        from src.adapters.mock_adapter import MockAdapter

        mock = MockAdapter("medgemma_4b", "MedGemma 4B",
                           ["pretriage", "triage", "differential", "management"])
        mock_result = mock.differential(full_ctx)

        hf = MedGemma4BAdapter()
        with patch.object(hf, "_chat_completion", return_value=MOCK_DIFFERENTIAL_RESPONSE):
            hf_result = hf.differential(full_ctx)

        assert type(mock_result) is type(hf_result)
        # Both have candidates list
        assert isinstance(mock_result.candidates, list)
        assert isinstance(hf_result.candidates, list)
