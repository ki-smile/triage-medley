"""Output models for all pipeline stages."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.models.enums import Confidence, QueuePriority, RETTSLevel


class PreTriageOutput(BaseModel):
    """Stage A output — queue priority assignment."""

    model_id: str = Field(..., description="Model that produced this output")
    queue_priority: QueuePriority
    chief_complaint: str
    reasoning: str
    ess_category_hint: Optional[str] = None
    risk_amplifiers_detected: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = None


class TriageOutput(BaseModel):
    """Stage B output — triage level from a single model or rules engine."""

    model_id: str = Field(..., description="Model that produced this output")
    retts_level: RETTSLevel
    ess_category: Optional[str] = None
    chief_complaint: str
    clinical_reasoning: str
    vital_sign_concerns: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.MODERATE
    dont_miss: list[str] = Field(
        default_factory=list,
        description="Don't-miss diagnoses flagged by this model",
    )
    triage_system: Optional[str] = Field(
        default=None,
        description="Triage system: retts | esi | mts | None (for LLM models)",
    )
    native_level_detail: Optional[dict] = Field(
        default=None,
        description="Engine-specific metadata: ESI resources_predicted, MTS max_wait_minutes, etc.",
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = None


class DifferentialCandidate(BaseModel):
    """A single differential diagnosis candidate."""

    diagnosis: str
    probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Estimated probability"
    )
    supporting_evidence: list[str] = Field(default_factory=list)
    is_dont_miss: bool = Field(
        default=False, description="Flagged as don't-miss by any model"
    )


class DifferentialOutput(BaseModel):
    """Differential diagnosis output from a single model."""

    model_id: str
    candidates: list[DifferentialCandidate] = Field(default_factory=list)
    reasoning: str = ""
    confidence: Confidence = Confidence.MODERATE
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = None


class ManagementOutput(BaseModel):
    """Management plan output."""

    model_id: str
    investigations: list[str] = Field(
        default_factory=list, description="Lab/diagnostic investigations"
    )
    imaging: list[str] = Field(
        default_factory=list, description="Imaging recommendations"
    )
    medications: list[str] = Field(
        default_factory=list, description="Medication recommendations"
    )
    disposition: str = Field(
        default="observation",
        description="discharge | admission | icu | observation",
    )
    contraindications_flagged: list[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: Confidence = Confidence.MODERATE
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[int] = None
