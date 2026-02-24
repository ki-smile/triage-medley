"""Database Service — Persistent shared storage for patient sessions using SQLite.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.session_manager import PatientSession

from src.utils.config import get_project_root

_DB_PATH = get_project_root() / "data" / "triage.db"

def init_db():
    """Initialize the SQLite database and create the patients table."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                payload TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()

def save_patient(session: "PatientSession"):
    """Save or update a patient session in the database."""
    import dataclasses
    
    # Custom recursive converter to handle dataclasses and Pydantic models
    def _to_dict(obj):
        if obj is None:
            return None
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "value"): # Enums
            return obj.value
        if hasattr(obj, "model_dump"): # Pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"): # Pydantic v1
            return obj.dict()
        if dataclasses.is_dataclass(obj):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, list):
            return [_to_dict(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj

    def _encoder(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    payload = json.dumps(_to_dict(session), default=_encoder)
    
    conn = sqlite3.connect(_DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO patients (patient_id, payload, updated_at) VALUES (?, ?, ?)",
            (session.patient_id, payload, datetime.now().isoformat())
        )
        conn.commit()
    finally:
        conn.close()

def load_all_patients() -> dict[str, Any]:
    """Load all patient session payloads from the database."""
    if not _DB_PATH.exists():
        return {}
        
    conn = sqlite3.connect(_DB_PATH)
    try:
        cursor = conn.execute("SELECT patient_id, payload FROM patients")
        rows = cursor.fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}
    finally:
        conn.close()

def clear_db():
    """Clear all patient records and reset the database file."""
    if _DB_PATH.exists():
        try:
            _DB_PATH.unlink() # Physically delete the file
            init_db() # Re-create empty
        except Exception:
            # Fallback to DELETE if file is locked
            conn = sqlite3.connect(_DB_PATH)
            try:
                conn.execute("DELETE FROM patients")
                conn.execute("VACUUM")
                conn.commit()
            finally:
                conn.close()
