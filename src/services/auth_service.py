"""Mock authentication service for role-based access.

Demo credentials hardcoded. In production, replace with hospital SSO/LDAP.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Role(str, Enum):
    """User roles in the triage system."""

    PATIENT = "patient"
    TRIAGE_NURSE = "triage_nurse"
    PHYSICIAN = "physician"
    ADMIN = "admin"


# Page access control matrix
ROLE_PAGES: dict[Role, list[str]] = {
    Role.PATIENT: ["kiosk"],
    Role.TRIAGE_NURSE: ["queue_view", "triage_view"],
    Role.PHYSICIAN: ["queue_view", "physician_view"],
    Role.ADMIN: [
        "kiosk",
        "queue_view",
        "triage_view",
        "physician_view",
        "prompt_editor",
        "audit_log",
        "engine_config",
    ],
}

# Demo credentials: username → {pin, role, display_name}
DEMO_CREDENTIALS: dict[str, dict] = {
    "nurse_anna": {
        "pin": "1234",
        "role": Role.TRIAGE_NURSE,
        "display_name": "Anna Lindberg, RN",
    },
    "nurse_erik": {
        "pin": "1234",
        "role": Role.TRIAGE_NURSE,
        "display_name": "Erik Holm, RN",
    },
    "dr_nilsson": {
        "pin": "5678",
        "role": Role.PHYSICIAN,
        "display_name": "Dr. Sara Nilsson",
    },
    "dr_berg": {
        "pin": "5678",
        "role": Role.PHYSICIAN,
        "display_name": "Dr. Magnus Berg",
    },
    "admin": {
        "pin": "0000",
        "role": Role.ADMIN,
        "display_name": "System Admin",
    },
}


@dataclass
class AuthResult:
    """Result of an authentication attempt."""

    success: bool
    role: Optional[Role] = None
    display_name: Optional[str] = None
    error: Optional[str] = None


def authenticate(username: str, pin: str) -> AuthResult:
    """Validate username + PIN against demo credentials."""
    user = DEMO_CREDENTIALS.get(username.lower().strip())
    if not user:
        return AuthResult(success=False, error="Unknown user")
    if user["pin"] != pin:
        return AuthResult(success=False, error="Incorrect PIN")
    return AuthResult(
        success=True,
        role=user["role"],
        display_name=user["display_name"],
    )


def has_page_access(role: Role, page_id: str) -> bool:
    """Check if a role has access to a specific page."""
    return page_id in ROLE_PAGES.get(role, [])
