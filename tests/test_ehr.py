"""Tests for EHR Service."""

import pytest

from src.services.ehr_service import load_patient, list_available_patients


class TestPatientLoading:
    def test_list_available(self):
        patients = list_available_patients()
        assert "anders" in patients
        assert "ella" in patients
        assert "margit" in patients

    def test_nonexistent_returns_none(self):
        result = load_patient("nonexistent_patient")
        assert result is None


class TestAnders:
    def test_demographics(self):
        p = load_patient("anders")
        assert p is not None
        assert p.name == "Anders Lindqvist"
        assert p.sex == "M"
        assert p.age > 60
        assert p.is_pediatric is False

    def test_conditions(self):
        p = load_patient("anders")
        cond_names = [c.display for c in p.active_conditions]
        assert any("Heart failure" in c for c in cond_names)
        assert any("Diabetes" in c for c in cond_names)
        assert any("Atrial fibrillation" in c for c in cond_names)

    def test_medications(self):
        p = load_patient("anders")
        med_names = [m.display for m in p.active_medications]
        assert any("Warfarin" in m for m in med_names)
        assert any("Metformin" in m for m in med_names)

    def test_allergies(self):
        p = load_patient("anders")
        assert any("Penicillin" in a.substance for a in p.allergies)

    def test_risk_flags(self):
        p = load_patient("anders")
        flag_types = [f.flag_type for f in p.risk_flags]
        assert "anticoagulation_risk" in flag_types
        assert "diabetes_hypoglycemia_risk" in flag_types

    def test_medication_class_matching(self):
        p = load_patient("anders")
        assert p.has_medication_class(["warfarin"]) is True
        assert p.has_medication_class(["aspirin"]) is False


class TestElla:
    def test_demographics(self):
        p = load_patient("ella")
        assert p is not None
        assert p.name == "Ella Johansson"
        assert p.sex == "F"
        assert p.age < 10
        assert p.is_pediatric is True

    def test_minimal_history(self):
        p = load_patient("ella")
        assert len(p.active_conditions) <= 2
        assert len(p.risk_flags) == 0


class TestMargit:
    def test_demographics(self):
        p = load_patient("margit")
        assert p is not None
        assert p.name == "Margit Eriksson"
        assert p.sex == "F"
        assert p.age > 75

    def test_anticoagulation_flag(self):
        p = load_patient("margit")
        flag_types = [f.flag_type for f in p.risk_flags]
        assert "anticoagulation_risk" in flag_types
        assert "fall_risk_elderly" in flag_types

    def test_warfarin(self):
        p = load_patient("margit")
        assert p.has_medication_class(["warfarin"]) is True

    def test_osteoporosis(self):
        p = load_patient("margit")
        assert p.has_condition_matching(["osteoporosis"]) is True
