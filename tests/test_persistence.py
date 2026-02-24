"""Persistence tests — verify SQLite storage and shared state."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import streamlit as st

from src.services import db_service
from src.services.session_manager import (
    PatientSession, add_patient, get_patient, get_patients, 
    _load_patients_from_db, DotDict, StringProxy
)
from src.models.enums import QueuePriority, RETTSLevel, ConsciousnessLevel

@pytest.fixture(autouse=True)
def mock_session_state():
    """Mock streamlit session state for each test."""
    with patch("streamlit.session_state", MagicMock()) as mock_state:
        mock_state.patients = {}
        mock_state.demo_loaded = False
        # Setup .get() to return defaults if keys are missing
        def mock_get(key, default=None):
            return getattr(mock_state, key, default)
        mock_state.get.side_effect = mock_get
        yield mock_state

@pytest.fixture(autouse=True)
def clean_db():
    db_service.clear_db()
    yield
    db_service.clear_db()

def test_db_save_load_basic():
    """Verify basic save/load of a patient session."""
    session = PatientSession(patient_id="test_p1", name="Test Patient", age=45, sex="M")
    db_service.save_patient(session)
    
    loaded = db_service.load_all_patients()
    assert "test_p1" in loaded
    assert loaded["test_p1"]["name"] == "Test Patient"
    assert loaded["test_p1"]["age"] == 45

def test_session_manager_persistence():
    """Verify session manager correctly bridges to DB."""
    session = PatientSession(patient_id="test_p2", name="Persist Test")
    add_patient(session)
    
    # Force reload from DB
    reloaded_patients = _load_patients_from_db()
    assert "test_p2" in reloaded_patients
    assert reloaded_patients["test_p2"].name == "Persist Test"

def test_enum_rehydration_simulated():
    """Verify that strings behave like enums via StringProxy/DotDict."""
    data = {
        "patient_id": "test_p3",
        "queue_priority": "HIGH",
        "triage_agreement": {
            "final_level": "RED",
            "agreement_level": "FULL"
        }
    }
    
    # Simulate reconstruction logic
    session = PatientSession(patient_id=data["patient_id"])
    session.pretriage_result = DotDict({"engine_output": {"queue_priority": "HIGH"}})
    session.triage_agreement = DotDict(data["triage_agreement"])
    
    # Test property getters we made robust
    assert session.queue_priority == QueuePriority.HIGH
    assert session.retts_level == RETTSLevel.RED
    
    # Test .value access (StringProxy)
    assert session.retts_level.value == "RED"

def test_nested_list_rehydration():
    """Verify that lists of dicts are rehydrated into lists of DotDicts."""
    session = PatientSession(patient_id="test_p4")
    session.overrides = [{"original": "YELLOW", "new": "RED"}]
    add_patient(session)
    
    reloaded = _load_patients_from_db()["test_p4"]
    assert isinstance(reloaded.overrides[0], DotDict)
    assert reloaded.overrides[0].original == "YELLOW"

def test_complex_field_rehydration():
    """Verify critical context objects are rehydrated into real models or DotDicts."""
    from src.models.vitals import VitalSigns
    from src.models.context import FullTriageContext
    
    vitals = VitalSigns(heart_rate=80, systolic_bp=120, diastolic_bp=80, 
                        respiratory_rate=16, spo2=98, temperature=37.0,
                        consciousness=ConsciousnessLevel.ALERT)
    ctx = FullTriageContext(patient_id="test_p5", speech_text="hello", vitals=vitals)
    
    session = PatientSession(patient_id="test_p5")
    session.full_context = ctx
    add_patient(session)
    
    reloaded = _load_patients_from_db()["test_p5"]
    # rehydrated FullTriageContext has vitals as DotDict or real object
    assert reloaded.full_context.vitals.heart_rate == 80
    assert reloaded.full_context.speech_text == "hello"

def test_db_clear():
    """Verify database clear functionality."""
    session = PatientSession(patient_id="test_p6")
    add_patient(session)
    assert len(db_service.load_all_patients()) == 1
    
    db_service.clear_db()
    assert len(db_service.load_all_patients()) == 0
