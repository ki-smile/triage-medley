"""Tests for HuggingFace Space adapters — uses mocked Gradio responses.

Tests cover:
1. Message flattening (chat messages → single prompt string)
2. Space adapter parses JSON responses (all 4 stages)
3. Factory creates Space adapters when config says "space"
4. SpaceMedGemma4BAdapter produces valid output types
5. Missing space_id raises ValueError
6. Existing mock factory is unaffected
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.base import ModelAdapter
from src.adapters.space_base import SpaceBaseAdapter
from src.adapters.space_medgemma import SpaceMedGemma4BAdapter
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


# ---- Mock responses (same as HF adapter tests) ----

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
    ],
    "reasoning": "Acute presentation with cardiac risk factors",
    "confidence": "MODERATE",
})

MOCK_MANAGEMENT_RESPONSE = json.dumps({
    "investigations": ["Troponin", "D-dimer", "BNP"],
    "imaging": ["Chest X-ray"],
    "medications": ["Aspirin 300mg loading"],
    "disposition": "admission",
    "contraindications_flagged": ["Dual anticoagulation risk with warfarin"],
    "reasoning": "Requires monitoring and serial troponins",
    "confidence": "HIGH",
})


# ---- Message Flattening Tests ----

class TestMessageFlattening:
    def test_system_and_user_messages(self):
        messages = [
            {"role": "system", "content": "You are a triage AI."},
            {"role": "user", "content": "Patient has chest pain."},
        ]
        result = SpaceBaseAdapter._flatten_messages(messages)
        assert "[System]" in result
        assert "[User]" in result
        assert "You are a triage AI." in result
        assert "Patient has chest pain." in result

    def test_single_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = SpaceBaseAdapter._flatten_messages(messages)
        assert result == "[User]\nHello"

    def test_empty_messages(self):
        result = SpaceBaseAdapter._flatten_messages([])
        assert result == ""

    def test_messages_separated_by_double_newline(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        result = SpaceBaseAdapter._flatten_messages(messages)
        assert "\n\n" in result
        parts = result.split("\n\n")
        assert len(parts) == 2

    def test_role_capitalized(self):
        messages = [{"role": "assistant", "content": "response"}]
        result = SpaceBaseAdapter._flatten_messages(messages)
        assert "[Assistant]" in result


# ---- Space Adapter Concrete Tests ----

class TestSpaceMedGemma4BAdapter:
    def test_protocol_compliance(self):
        adapter = SpaceMedGemma4BAdapter()
        assert isinstance(adapter, ModelAdapter)

    def test_default_properties(self):
        adapter = SpaceMedGemma4BAdapter()
        assert adapter.model_id == "medgemma_4b"
        assert adapter.space_id == ""
        assert adapter.hf_model_id == "google/medgemma-4b-it"
        assert "pretriage" in adapter.supported_stages
        assert "triage" in adapter.supported_stages
        assert "differential" in adapter.supported_stages
        assert "management" in adapter.supported_stages

    def test_custom_space_id(self):
        adapter = SpaceMedGemma4BAdapter(space_id="other/space")
        assert adapter.space_id == "other/space"


# ---- Space Adapter Parsing Tests (mocked _chat_completion) ----

class TestSpaceAdapterParsing:
    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_pretriage_parsing(self, pretriage_ctx):
        adapter = SpaceMedGemma4BAdapter()
        with patch.object(adapter, "_chat_completion", return_value=MOCK_PRETRIAGE_RESPONSE):
            result = adapter.pretriage(pretriage_ctx)

        assert isinstance(result, PreTriageOutput)
        assert result.model_id == "medgemma_4b"
        assert result.queue_priority == QueuePriority.HIGH
        assert "chest pain" in result.chief_complaint.lower()
        assert result.processing_time_ms >= 0

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_triage_parsing(self, full_ctx):
        adapter = SpaceMedGemma4BAdapter()
        with patch.object(adapter, "_chat_completion", return_value=MOCK_TRIAGE_RESPONSE):
            result = adapter.triage(full_ctx)

        assert isinstance(result, TriageOutput)
        assert result.retts_level == RETTSLevel.ORANGE
        assert result.confidence == Confidence.HIGH
        assert len(result.dont_miss) == 2

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_differential_parsing(self, full_ctx):
        adapter = SpaceMedGemma4BAdapter()
        with patch.object(adapter, "_chat_completion", return_value=MOCK_DIFFERENTIAL_RESPONSE):
            result = adapter.differential(full_ctx)

        assert isinstance(result, DifferentialOutput)
        assert len(result.candidates) == 2
        assert result.candidates[0].probability == 0.45

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_management_parsing(self, full_ctx):
        adapter = SpaceMedGemma4BAdapter()
        with patch.object(adapter, "_chat_completion", return_value=MOCK_MANAGEMENT_RESPONSE):
            result = adapter.management(full_ctx)

        assert isinstance(result, ManagementOutput)
        assert result.disposition == "admission"
        assert len(result.investigations) == 3

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_graceful_no_json_response(self, full_ctx):
        """Space returns free text — should still produce valid output."""
        adapter = SpaceMedGemma4BAdapter()
        response = "Patient should be triaged as Yellow."
        with patch.object(adapter, "_chat_completion", return_value=response):
            result = adapter.triage(full_ctx)
        assert result.retts_level == RETTSLevel.YELLOW  # default fallback


# ---- Space Client Initialization Tests ----

class TestSpaceClientInit:
    def test_missing_token_raises(self):
        """No HF_TOKEN → RuntimeError before creating Gradio client."""
        adapter = SpaceMedGemma4BAdapter()
        mock_gradio = MagicMock()
        with patch.dict("os.environ", {}, clear=True), \
             patch.dict("sys.modules", {"gradio_client": mock_gradio}):
            import os, sys
            os.environ.pop("HF_TOKEN", None)
            with pytest.raises(RuntimeError, match="HF_TOKEN"):
                adapter._get_client()

    @patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"})
    def test_client_created_with_space_id(self):
        """Client is created with correct space_id and token."""
        adapter = SpaceMedGemma4BAdapter()
        mock_gradio = MagicMock()
        mock_client_instance = MagicMock()
        mock_gradio.Client.return_value = mock_client_instance
        with patch.dict("sys.modules", {"gradio_client": mock_gradio}):
            client = adapter._get_client()
            mock_gradio.Client.assert_called_once_with(
                "test-user/test-space",
                hf_token="hf_test_token_123",
            )
            assert client is mock_client_instance


# ---- Factory Integration Tests ----

class TestFactorySpaceIntegration:
    def test_factory_creates_space_adapter(self):
        from src.adapters.factory import _build_space_adapter

        adapter = _build_space_adapter(
            model_id="medgemma_4b",
            model_name="MedGemma 4B",
            stages=["pretriage", "triage", "differential", "management"],
            model_config={
                "hf_id": "google/medgemma-4b-it",
                "space_id": "test-user/test-space",
                "api_name": "/predict",
            },
        )
        assert isinstance(adapter, SpaceMedGemma4BAdapter)
        assert adapter.space_id == "test-user/test-space"

    def test_missing_space_id_raises(self):
        from src.adapters.factory import _build_space_adapter

        with pytest.raises(ValueError, match="space_id"):
            _build_space_adapter(
                model_id="medgemma_4b",
                model_name="MedGemma 4B",
                stages=["triage"],
                model_config={"hf_id": "google/medgemma-4b-it"},
            )

    def test_unknown_model_raises(self):
        from src.adapters.factory import _build_space_adapter

        with pytest.raises(ValueError, match="No Space adapter"):
            _build_space_adapter(
                model_id="unknown_model",
                model_name="Unknown",
                stages=["triage"],
                model_config={"hf_id": "unknown/model", "space_id": "x/y"},
            )

    def test_factory_mock_still_works(self):
        """Adding space support doesn't break mock factory."""
        from src.adapters.factory import create_all_adapters
        adapters = create_all_adapters(adapter_type_override="mock")
        assert len(adapters) == 4
        assert all(
            type(a).__name__ == "MockAdapter"
            for a in adapters.values()
        )

    def test_build_adapter_dispatches_space(self):
        """_build_adapter routes 'space' to _build_space_adapter."""
        from src.adapters.factory import _build_adapter

        adapter = _build_adapter(
            "medgemma_4b",
            {
                "adapter": "space",
                "name": "MedGemma 4B",
                "hf_id": "google/medgemma-4b-it",
                "space_id": "test-user/test-space",
                "api_name": "/predict",
                "stages": ["pretriage", "triage", "differential", "management"],
            },
        )
        assert isinstance(adapter, SpaceMedGemma4BAdapter)
