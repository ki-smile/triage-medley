"""Tests for PDF report generation."""

from datetime import datetime

import pytest

from src.engines.agreement_engine import (
    analyze_differential,
    analyze_management,
    analyze_triage,
)
from src.models.context import FullTriageContext
from src.models.enums import ArrivalPathway, ConsciousnessLevel
from src.models.vitals import VitalSigns
from src.services.asr_service import process_audio
from src.services.ehr_service import load_patient
from src.services.orchestrator import run_full_pipeline, run_pretriage
from src.services.pdf_service import generate_physician_pdf, generate_triage_pdf
from src.services.session_manager import PatientSession
from src.models.context import PreTriageContext


# --------------------------------------------------------------------------- #
# Fixture: build a full PatientSession for any demo scenario
# --------------------------------------------------------------------------- #

_SCENARIO_VITALS = {
    "anders": VitalSigns(
        heart_rate=92, systolic_bp=145, diastolic_bp=85,
        respiratory_rate=20, spo2=94, temperature=37.1,
        consciousness=ConsciousnessLevel.ALERT,
    ),
    "ella": VitalSigns(
        heart_rate=185, systolic_bp=85, diastolic_bp=50,
        respiratory_rate=34, spo2=96, temperature=39.8,
        consciousness=ConsciousnessLevel.ALERT,
    ),
    "margit": VitalSigns(
        heart_rate=78, systolic_bp=155, diastolic_bp=88,
        respiratory_rate=18, spo2=96, temperature=36.8,
        consciousness=ConsciousnessLevel.ALERT,
    ),
    "ingrid": VitalSigns(
        heart_rate=78, systolic_bp=118, diastolic_bp=68,
        respiratory_rate=20, spo2=96, temperature=36.4,
        consciousness=ConsciousnessLevel.ALERT,
    ),
    "erik": VitalSigns(
        heart_rate=98, systolic_bp=152, diastolic_bp=88,
        respiratory_rate=18, spo2=97, temperature=36.8,
        consciousness=ConsciousnessLevel.ALERT,
    ),
    "sofia": VitalSigns(
        heart_rate=102, systolic_bp=122, diastolic_bp=76,
        respiratory_rate=20, spo2=95, temperature=37.2,
        consciousness=ConsciousnessLevel.ALERT,
    ),
}

_SCENARIO_ESS = {
    "anders": "chest_pain",
    "ella": "pediatric_fever",
    "margit": "trauma_fall",
    "ingrid": "neurological",
    "erik": "neurological",
    "sofia": "chest_pain",
}


def _build_session(patient_id: str) -> PatientSession:
    """Build a complete PatientSession for a demo scenario."""
    ehr = load_patient(patient_id)
    asr = process_audio(patient_id)

    session = PatientSession(
        patient_id=patient_id,
        name=ehr.name if ehr else patient_id,
        age=ehr.age if ehr else 0,
        sex=ehr.sex if ehr else "",
        speech_text=f"Test speech for {patient_id}",
        asr_result=asr,
    )

    vitals = _SCENARIO_VITALS.get(patient_id)
    ess = _SCENARIO_ESS.get(patient_id, "general")

    if vitals:
        ctx = FullTriageContext(
            patient_id=patient_id,
            speech_text=session.speech_text,
            ehr=ehr,
            asr_disagreements=asr.disagreements if asr else [],
            vitals=vitals,
            ess_category=ess,
        )
        session.full_context = ctx
        t, d, m = run_full_pipeline(ctx)
        session.triage_ensemble = t
        session.triage_agreement = analyze_triage(t)
        session.differential_ensemble = d
        session.differential_agreement = analyze_differential(d)
        session.management_ensemble = m
        session.management_agreement = analyze_management(m)

    return session


# --------------------------------------------------------------------------- #
# Triage PDF tests
# --------------------------------------------------------------------------- #

class TestTriagePDF:
    def test_generates_bytes(self):
        session = _build_session("anders")
        result = generate_triage_pdf(session)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_starts_with_pdf_header(self):
        session = _build_session("anders")
        result = generate_triage_pdf(session)
        assert result[:5] == b"%PDF-"

    @pytest.mark.parametrize("patient_id", ["anders", "ella", "margit", "ingrid", "erik", "sofia"])
    def test_all_six_scenarios(self, patient_id):
        session = _build_session(patient_id)
        result = generate_triage_pdf(session)
        assert result[:5] == b"%PDF-", f"Failed for {patient_id}"
        assert len(result) > 100

    def test_handles_missing_optional_data(self):
        """PDF should work even with minimal data."""
        session = PatientSession(
            patient_id="minimal",
            name="Test Patient",
            age=30,
            sex="M",
            speech_text="I have a headache",
        )
        result = generate_triage_pdf(session)
        assert result[:5] == b"%PDF-"

    def test_nurse_override_included(self):
        session = _build_session("anders")
        session.overrides.append({
            "original": "ORANGE",
            "override_to": "RED",
            "reason": "Clinical concern",
        })
        result = generate_triage_pdf(session)
        assert result[:5] == b"%PDF-"
        assert len(result) > 100


# --------------------------------------------------------------------------- #
# Physician PDF tests
# --------------------------------------------------------------------------- #

class TestPhysicianPDF:
    def test_generates_bytes(self):
        session = _build_session("anders")
        session.physician_approved_investigations = ["CBC", "Troponin"]
        session.physician_disposition = "admission"
        session.physician_sign_off = datetime.now()
        session.physician_name = "Dr. Nilsson"
        result = generate_physician_pdf(session)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_unsigned_report_still_generates(self):
        """Physician report should generate even without sign-off."""
        session = _build_session("anders")
        result = generate_physician_pdf(session)
        assert result[:5] == b"%PDF-"

    def test_with_physician_notes(self):
        session = _build_session("ella")
        session.physician_notes = "Patient requires pediatric ICU admission"
        session.physician_sign_off = datetime.now()
        session.physician_name = "Dr. Berg"
        result = generate_physician_pdf(session)
        assert result[:5] == b"%PDF-"
        assert len(result) > 100

    @pytest.mark.parametrize("patient_id", ["anders", "ella", "margit", "ingrid", "erik", "sofia"])
    def test_all_six_scenarios(self, patient_id):
        session = _build_session(patient_id)
        session.physician_approved_investigations = ["CBC"]
        session.physician_disposition = "observation"
        session.physician_sign_off = datetime.now()
        session.physician_name = "Dr. Test"
        result = generate_physician_pdf(session)
        assert result[:5] == b"%PDF-", f"Failed for {patient_id}"
