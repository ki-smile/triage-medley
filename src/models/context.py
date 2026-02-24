"""Triage context models — type-enforced stage separation.

PreTriageContext has NO vitals field (Stage A).
FullTriageContext extends it with required VitalSigns (Stage B).
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.models.clinical import ASRDisagreement, EHRSnapshot
from src.models.enums import ArrivalPathway
from src.models.vitals import VitalSigns


class PreTriageContext(BaseModel):
    """Stage A context — speech + EHR only, NO vitals.

    This is type-enforced: there is no vitals field on this class.
    Any code attempting to access .vitals will fail at the type level.
    """

    patient_id: str
    arrival_pathway: ArrivalPathway = ArrivalPathway.WALK_IN
    arrival_time: datetime = Field(default_factory=datetime.now)
    speech_text: str = Field(..., description="Patient speech transcript")
    ehr: Optional[EHRSnapshot] = Field(
        default=None, description="EHR data if available (graceful degradation)"
    )
    asr_disagreements: list[ASRDisagreement] = Field(default_factory=list)
    language: str = Field(default="sv", description="ISO 639-1 language code")


class FullTriageContext(PreTriageContext):
    """Stage B context — extends PreTriageContext with REQUIRED vitals.

    The vitals field is required (no default). RETTS engine refuses
    to run without vitals — enforced at both the type level and runtime.
    """

    vitals: VitalSigns = Field(..., description="Measured vital signs (required)")
    ess_category: Optional[str] = Field(
        default=None, description="ESS code from pre-triage hint or NLP"
    )
