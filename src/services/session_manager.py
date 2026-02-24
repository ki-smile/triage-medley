"""Session Manager — Streamlit session state for patient pipeline data.

Manages the flow of patient data across pages:
Queue View → Triage View → Physician View.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import streamlit as st

from src.engines.agreement_engine import (
    DifferentialAgreement,
    ManagementAgreement,
    TriageAgreement,
)
from src.models.context import FullTriageContext, PreTriageContext
from src.models.enums import ArrivalPathway, QueuePriority, RETTSLevel
from src.models.outputs import PreTriageOutput
from src.services.asr_service import ASRResult
from src.services.orchestrator import (
    DifferentialEnsembleResult,
    ManagementEnsembleResult,
    PreTriageResult,
    TriageEnsembleResult,
)
import src.services.db_service as db_service


@dataclass
class PatientSession:
    """All data for a single patient flowing through the pipeline."""
    patient_id: str
    name: str = ""
    age: int = 0
    sex: str = ""
    arrival_pathway: ArrivalPathway = ArrivalPathway.WALK_IN
    arrival_time: datetime = field(default_factory=datetime.now)

    # Stage A
    speech_text: str = ""
    asr_result: Optional[ASRResult] = None
    pretriage_context: Optional[PreTriageContext] = None
    pretriage_result: Optional[PreTriageResult] = None

    # Stage B
    full_context: Optional[FullTriageContext] = None
    triage_ensemble: Optional[TriageEnsembleResult] = None
    triage_agreement: Optional[TriageAgreement] = None
    differential_ensemble: Optional[DifferentialEnsembleResult] = None
    differential_agreement: Optional[DifferentialAgreement] = None
    management_ensemble: Optional[ManagementEnsembleResult] = None
    management_agreement: Optional[ManagementAgreement] = None

    # Nurse overrides
    overrides: list[dict] = field(default_factory=list)

    # Physician workflow
    physician_approved_investigations: list[str] = field(default_factory=list)
    physician_approved_imaging: list[str] = field(default_factory=list)
    physician_approved_medications: list[str] = field(default_factory=list)
    physician_disposition: Optional[str] = None
    physician_notes: str = ""
    physician_sign_off: Optional[datetime] = None
    physician_name: Optional[str] = None

    @property
    def queue_priority(self) -> Optional[QueuePriority]:
        if self.pretriage_result:
            # Handle both object and DotDict
            res = self.pretriage_result
            eo = res.get("engine_output") if isinstance(res, dict) else getattr(res, "engine_output", None)
            if eo:
                prio = eo.get("queue_priority") if isinstance(eo, dict) else getattr(eo, "queue_priority", None)
                return QueuePriority(prio) if prio else None
        return None

    @property
    def retts_level(self) -> Optional[RETTSLevel]:
        if self.triage_agreement:
            res = self.triage_agreement
            lvl = res.get("final_level") if isinstance(res, dict) else getattr(res, "final_level", None)
            return RETTSLevel(lvl) if lvl else None
        return None

    @property
    def is_triaged(self) -> bool:
        return self.triage_ensemble is not None

    @property
    def is_signed_off(self) -> bool:
        return self.physician_sign_off is not None


def init_session_state() -> None:
    """Initialize session state if not already set."""
    db_service.init_db()
    
    # Handle manual DB clear if requested (via UI button)
    if st.session_state.get("_clear_db_requested"):
        db_service.clear_db()
        st.session_state.patients = {}
        st.session_state.demo_loaded = False
        st.session_state._clear_db_requested = False
        st.rerun()

    if "patients" not in st.session_state:
        # Initial load from DB to session state
        st.session_state.patients = _load_patients_from_db()
    
    if "selected_patient" not in st.session_state:
        st.session_state.selected_patient = None
    
    if "demo_loaded" not in st.session_state:
        # Check if DB has patients to determine if demo is effectively loaded
        st.session_state.demo_loaded = len(st.session_state.patients) > 0


class StringProxy(str):
    """A string that also has a .value property, simulating an Enum."""
    @property
    def value(self):
        return str(self)


class DotDict(dict):
    """A dictionary that allows dot notation access and simulates objects."""
    def __getattr__(self, name):
        try:
            val = self[name]
            if isinstance(val, dict):
                return DotDict(val)
            if isinstance(val, list):
                return [DotDict(i) if isinstance(i, dict) else (StringProxy(i) if isinstance(i, str) else i) for i in val]
            if isinstance(val, str):
                return StringProxy(val)
            return val
        except KeyError:
            # EHRSnapshot methods/properties needed by engines
            if name == "is_pediatric":
                return self.get("age", 0) < 16
            if name == "active_conditions":
                return [DotDict(c) for c in self.get("conditions", []) if c.get("status") == "active"]
            if name == "active_medications":
                return [DotDict(m) for m in self.get("medications", []) if m.get("status") == "active"]
            
            # TriageAgreement properties
            if name == "agreement_ratio":
                if "consensus_level" in self and "total_voters" in self:
                    cv = self.get("vote_distribution", {}).get(self["consensus_level"], 0)
                    tv = self["total_voters"]
                    return cv / tv if tv > 0 else 0.0
                return 0.0
            
            # ASRResult properties
            if name == "has_critical_disagreements":
                return any(d.get("clinical_significance") == "high" for d in self.get("disagreements", []))
            if name == "disagreement_count":
                return len(self.get("disagreements", []))
            if name == "unresolved_count":
                return sum(1 for d in self.get("disagreements", []) if not d.get("resolved", False))
            
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def has_medication_class(self, keywords: list[str]) -> bool:
        med_names = [m.get("display", "").lower() for m in self.get("medications", []) if m.get("status") == "active"]
        return any(kw.lower() in name for name in med_names for kw in keywords)

    def has_condition_matching(self, keywords: list[str]) -> bool:
        cond_names = [c.get("display", "").lower() for c in self.get("conditions", []) if c.get("status") == "active"]
        return any(kw.lower() in name for name in cond_names for kw in keywords)


def _load_patients_from_db() -> dict[str, Any]:
    """Internal helper to reconstruct PatientSession objects from DB JSON."""
    from src.models.clinical import EHRSnapshot
    from src.models.context import FullTriageContext, PreTriageContext
    from src.models.vitals import VitalSigns
    
    raw_data = db_service.load_all_patients()
    patients = {}
    
    # Critical clinical models that need real classes for logic engines
    CLINICAL_MODELS = {
        "pretriage_context": PreTriageContext,
        "full_context": FullTriageContext,
    }
    
    # Fields that should use DotDict for recursive UI access
    RESULT_FIELDS = {
        "pretriage_result", "triage_ensemble", "triage_agreement",
        "differential_ensemble", "differential_agreement",
        "management_ensemble", "management_agreement", "asr_result"
    }

    for pid, data in raw_data.items():
        try:
            # Create a fresh session object
            session = PatientSession(patient_id=pid)
            
            # Map attributes
            for key, value in data.items():
                if hasattr(session, key) and value is not None:
                    if isinstance(value, dict):
                        try:
                            if key in CLINICAL_MODELS:
                                cls = CLINICAL_MODELS[key]
                                if value.get("ehr"):
                                    value["ehr"] = EHRSnapshot(**value["ehr"])
                                if key == "full_context" and value.get("vitals"):
                                    value["vitals"] = VitalSigns(**value["vitals"])
                                value = cls.model_validate(value)
                            elif key in RESULT_FIELDS:
                                # FORCED DotDict for all result types
                                value = DotDict(value)
                            else:
                                value = DotDict(value)
                        except Exception:
                            value = DotDict(value)
                        setattr(session, key, value)
                    elif isinstance(value, list):
                        # Ensure nested lists of dicts are also handled
                        processed_list = []
                        for i in value:
                            if isinstance(i, dict):
                                processed_list.append(DotDict(i))
                            elif isinstance(i, str):
                                processed_list.append(StringProxy(i))
                            else:
                                processed_list.append(i)
                        setattr(session, key, processed_list)
                    elif isinstance(value, str):
                        setattr(session, key, StringProxy(value))
                    else:
                        setattr(session, key, value)
            
            # Handle stringified datetimes
            if isinstance(session.arrival_time, str):
                session.arrival_time = datetime.fromisoformat(session.arrival_time)
            if isinstance(session.physician_sign_off, str):
                session.physician_sign_off = datetime.fromisoformat(session.physician_sign_off)
            
            patients[pid] = session
        except Exception:
            continue
    return patients


def logout() -> None:
    """Clear authentication state."""
    st.session_state.role = None
    st.session_state.user_display_name = None
    st.session_state.username = None


def get_patients() -> dict[str, PatientSession]:
    """Get all patient sessions, refreshed from DB."""
    # We always refresh from DB to catch updates from other devices/tabs
    st.session_state.patients = _load_patients_from_db()
    return st.session_state.patients


def get_patient(patient_id: str) -> Optional[PatientSession]:
    """Get a specific patient session."""
    return get_patients().get(patient_id)


def add_patient(session: PatientSession) -> None:
    """Add or update a patient session in DB and session state."""
    db_service.save_patient(session)
    st.session_state.patients[session.patient_id] = session


def select_patient(patient_id: str) -> None:
    """Set the currently selected patient."""
    st.session_state.selected_patient = patient_id


def get_selected_patient() -> Optional[PatientSession]:
    """Get the currently selected patient session."""
    init_session_state()
    pid = st.session_state.selected_patient
    if pid:
        return st.session_state.patients.get(pid)
    return None


def get_queue_ordered() -> list[PatientSession]:
    """Get patients ordered by queue priority (HIGH first, then by arrival time)."""
    patients = list(get_patients().values())
    priority_order = {"HIGH": 0, "MODERATE": 1, "STANDARD": 2}

    def sort_key(p: PatientSession):
        prio = priority_order.get(
            p.queue_priority.value if p.queue_priority else "STANDARD", 2
        )
        return (prio, p.arrival_time)

    return sorted(patients, key=sort_key)


def register_kiosk_patient(
    patient_id: str, speech_text: str, language: str = "sv"
) -> PatientSession:
    """Register a single walk-in patient through the kiosk.

    Runs Stage A only (speech + EHR → queue priority). No vitals.
    Returns the created PatientSession.
    """
    from src.services.asr_service import process_audio
    from src.services.ehr_service import load_patient
    from src.services.orchestrator import run_pretriage
    from src.utils.audit import log_event

    ehr = load_patient(patient_id)
    asr = process_audio(patient_id)

    session = PatientSession(
        patient_id=patient_id,
        name=ehr.name if ehr else patient_id,
        age=ehr.age if ehr else 0,
        sex=ehr.sex if ehr else "",
        speech_text=speech_text,
        asr_result=asr,
    )

    pretriage_ctx = PreTriageContext(
        patient_id=patient_id,
        speech_text=speech_text,
        ehr=ehr,
        asr_disagreements=asr.disagreements if asr else [],
        language=language,
    )
    session.pretriage_context = pretriage_ctx
    session.pretriage_result = run_pretriage(pretriage_ctx)

    log_event(
        "kiosk", "patient_registered",
        {"name": session.name, "age": session.age, "pathway": "walk_in"},
        patient_id=patient_id,
    )

    add_patient(session)
    return session


def load_demo_scenarios() -> None:
    """Load the 3 demo scenarios into session state."""
    from src.models.enums import ConsciousnessLevel
    from src.models.vitals import VitalSigns
    from src.services.asr_service import process_audio
    from src.services.ehr_service import load_patient
    from src.services.orchestrator import run_pretriage
    from src.engines.agreement_engine import (
        analyze_triage,
        analyze_differential,
        analyze_management,
    )
    from src.services.orchestrator import run_full_pipeline
    from src.utils.audit import log_event

    now = datetime.now()
    scenarios = [
        {
            "patient_id": "anders",
            "speech": "I have chest tightness and difficulty breathing since this morning. I take warfarin and metoprolol for my heart condition.",
            "vitals": VitalSigns(
                heart_rate=92, systolic_bp=145, diastolic_bp=85,
                respiratory_rate=20, spo2=94, temperature=37.1,
                consciousness=ConsciousnessLevel.ALERT,
            ),
            "ess_category": "chest_pain",
            "wait_minutes": 12,  # HIGH — arrived recently
        },
        {
            "patient_id": "ella",
            "speech": "My daughter has had fever and rash since yesterday. She seems very tired and doesn't want to eat. The rash appeared this morning.",
            "vitals": VitalSigns(
                heart_rate=185, systolic_bp=85, diastolic_bp=50,
                respiratory_rate=34, spo2=96, temperature=39.8,
                consciousness=ConsciousnessLevel.ALERT,
            ),
            "ess_category": "pediatric_fever",
            "wait_minutes": 8,   # HIGH — paediatric, fast-tracked
        },
        {
            "patient_id": "margit",
            "speech": "I fell at home and hit my head on the floor. My hip hurts and I feel a bit dizzy. I take warfarin for my heart.",
            "vitals": VitalSigns(
                heart_rate=78, systolic_bp=155, diastolic_bp=88,
                respiratory_rate=18, spo2=96, temperature=36.8,
                consciousness=ConsciousnessLevel.ALERT,
            ),
            "ess_category": "trauma_fall",
            "wait_minutes": 35,  # MODERATE — head trauma on warfarin
        },
        # ---- Tricky Cases (from research) ----
        {
            "patient_id": "ingrid",
            "speech": "I'm just not myself today. I feel weak and a little dizzy. My daughter made me come in — I told her I'm fine, just tired. I might have slipped getting out of bed this morning but I caught myself.",
            "vitals": VitalSigns(
                heart_rate=78, systolic_bp=118, diastolic_bp=68,
                respiratory_rate=20, spo2=96, temperature=36.4,
                consciousness=ConsciousnessLevel.ALERT,
            ),
            "ess_category": "neurological",
            "wait_minutes": 52,  # Looks STANDARD — deceptively normal vitals
        },
        {
            "patient_id": "erik",
            "speech": "I had a funny turn about two hours ago — everything went blurry and I couldn't think straight for a few minutes, then it went away. Then I had some loose stools that were real dark, almost black. My wife's worried so she drove me in.",
            "vitals": VitalSigns(
                heart_rate=98, systolic_bp=152, diastolic_bp=88,
                respiratory_rate=18, spo2=97, temperature=36.8,
                consciousness=ConsciousnessLevel.ALERT,
            ),
            "ess_category": "neurological",
            "wait_minutes": 22,  # HIGH — TIA symptoms + melena
        },
        {
            "patient_id": "sofia",
            "speech": "I think I pulled a muscle in my chest. It started two days ago after I helped my friend move apartments. It hurts when I take a deep breath. I took ibuprofen and it helped a little but it woke me up last night.",
            "vitals": VitalSigns(
                heart_rate=102, systolic_bp=122, diastolic_bp=76,
                respiratory_rate=20, spo2=95, temperature=37.2,
                consciousness=ConsciousnessLevel.ALERT,
            ),
            "ess_category": "chest_pain",
            "wait_minutes": 41,  # Looks STANDARD — PE masquerading as muscle pain
        },
    ]

    for sc in scenarios:
        pid = sc["patient_id"]
        ehr = load_patient(pid)
        asr = process_audio(pid)

        session = PatientSession(
            patient_id=pid,
            name=ehr.name if ehr else pid,
            age=ehr.age if ehr else 0,
            sex=ehr.sex if ehr else "",
            speech_text=sc["speech"],
            asr_result=asr,
            arrival_time=now - timedelta(minutes=sc["wait_minutes"]),
        )

        # Stage A: Pre-triage
        pretriage_ctx = PreTriageContext(
            patient_id=pid,
            speech_text=sc["speech"],
            ehr=ehr,
            asr_disagreements=asr.disagreements if asr else [],
        )
        session.pretriage_context = pretriage_ctx
        session.pretriage_result = run_pretriage(pretriage_ctx)

        log_event("system", "patient_registered",
                  {"name": session.name, "age": session.age},
                  patient_id=pid)

        # Stage B: Full triage pipeline
        full_ctx = FullTriageContext(
            patient_id=pid,
            speech_text=sc["speech"],
            ehr=ehr,
            asr_disagreements=asr.disagreements if asr else [],
            vitals=sc["vitals"],
            ess_category=sc["ess_category"],
        )
        session.full_context = full_ctx

        t, d, m = run_full_pipeline(full_ctx)
        session.triage_ensemble = t
        session.triage_agreement = analyze_triage(t)
        session.differential_ensemble = d
        session.differential_agreement = analyze_differential(d)
        session.management_ensemble = m
        session.management_agreement = analyze_management(m)

        add_patient(session)

    st.session_state.demo_loaded = True
