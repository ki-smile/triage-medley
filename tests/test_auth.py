"""Tests for the mock authentication service."""

import pytest

from src.services.auth_service import (
    DEMO_CREDENTIALS,
    ROLE_PAGES,
    Role,
    authenticate,
    has_page_access,
)


class TestAuthenticate:
    def test_valid_nurse(self):
        result = authenticate("nurse_anna", "1234")
        assert result.success is True
        assert result.role == Role.TRIAGE_NURSE
        assert "Anna" in result.display_name

    def test_valid_physician(self):
        result = authenticate("dr_nilsson", "5678")
        assert result.success is True
        assert result.role == Role.PHYSICIAN

    def test_valid_admin(self):
        result = authenticate("admin", "0000")
        assert result.success is True
        assert result.role == Role.ADMIN

    def test_wrong_pin(self):
        result = authenticate("nurse_anna", "9999")
        assert result.success is False
        assert result.error is not None

    def test_unknown_user(self):
        result = authenticate("nobody", "1234")
        assert result.success is False

    def test_case_insensitive_username(self):
        result = authenticate("NURSE_ANNA", "1234")
        assert result.success is True

    def test_whitespace_trimmed(self):
        result = authenticate("  nurse_anna  ", "1234")
        assert result.success is True


class TestPageAccess:
    def test_patient_kiosk_only(self):
        assert has_page_access(Role.PATIENT, "kiosk") is True
        assert has_page_access(Role.PATIENT, "queue_view") is False
        assert has_page_access(Role.PATIENT, "audit_log") is False

    def test_nurse_pages(self):
        assert has_page_access(Role.TRIAGE_NURSE, "queue_view") is True
        assert has_page_access(Role.TRIAGE_NURSE, "triage_view") is True
        assert has_page_access(Role.TRIAGE_NURSE, "physician_view") is False

    def test_physician_pages(self):
        assert has_page_access(Role.PHYSICIAN, "physician_view") is True
        assert has_page_access(Role.PHYSICIAN, "queue_view") is True

    def test_admin_all_pages(self):
        for page_id in [
            "kiosk",
            "queue_view",
            "triage_view",
            "physician_view",
            "prompt_editor",
            "audit_log",
        ]:
            assert has_page_access(Role.ADMIN, page_id) is True


class TestRoleEnum:
    def test_all_roles_have_page_mapping(self):
        for role in Role:
            assert role in ROLE_PAGES

    def test_all_credentials_have_valid_roles(self):
        for username, cred in DEMO_CREDENTIALS.items():
            assert isinstance(cred["role"], Role)
            assert len(cred["pin"]) == 4

    def test_role_values(self):
        assert Role.PATIENT.value == "patient"
        assert Role.TRIAGE_NURSE.value == "triage_nurse"
        assert Role.PHYSICIAN.value == "physician"
        assert Role.ADMIN.value == "admin"
