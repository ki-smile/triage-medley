"""Clinical data models — EHR, symptoms, ASR, risk flags."""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class FHIRCondition(BaseModel):
    """A patient condition extracted from FHIR Bundle."""
    code: str = Field(..., description="SNOMED or ICD-10 code")
    display: str = Field(..., description="Human-readable condition name")
    onset_date: Optional[date] = None
    status: str = Field(default="active", description="active | resolved | inactive")


class FHIRMedication(BaseModel):
    """A medication extracted from FHIR Bundle."""
    code: str = Field(..., description="ATC or RxNorm code")
    display: str = Field(..., description="Medication name")
    dosage: Optional[str] = None
    status: str = Field(default="active", description="active | stopped")


class FHIRAllergy(BaseModel):
    """An allergy/intolerance from FHIR Bundle."""
    substance: str = Field(..., description="Allergen substance")
    reaction: Optional[str] = None
    severity: Optional[str] = None


class RiskFlag(BaseModel):
    """EHR-derived risk amplification flag."""
    flag_type: str = Field(..., description="Risk flag category")
    description: str = Field(..., description="Human-readable risk description")
    source: str = Field(
        default="ehr", description="Source: ehr | speech | computed"
    )
    severity: str = Field(default="moderate", description="low | moderate | high")


class EHRSnapshot(BaseModel):
    """Structured patient EHR data extracted from FHIR Bundle."""
    patient_id: str
    name: str
    age: int = Field(..., ge=0, le=150)
    sex: str = Field(..., description="M | F")
    conditions: list[FHIRCondition] = Field(default_factory=list)
    medications: list[FHIRMedication] = Field(default_factory=list)
    allergies: list[FHIRAllergy] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)

    @property
    def is_pediatric(self) -> bool:
        return self.age < 16

    @property
    def active_conditions(self) -> list[FHIRCondition]:
        return [c for c in self.conditions if c.status == "active"]

    @property
    def active_medications(self) -> list[FHIRMedication]:
        return [m for m in self.medications if m.status == "active"]

    def has_medication_class(self, keywords: list[str]) -> bool:
        """Check if patient has any medication matching keywords."""
        med_names = [m.display.lower() for m in self.active_medications]
        return any(kw.lower() in name for name in med_names for kw in keywords)

    def has_condition_matching(self, keywords: list[str]) -> bool:
        """Check if patient has any condition matching keywords."""
        cond_names = [c.display.lower() for c in self.active_conditions]
        return any(kw.lower() in name for name in cond_names for kw in keywords)


class Symptom(BaseModel):
    """Extracted symptom from speech transcript."""
    text: str = Field(..., description="Symptom as described")
    ess_category: Optional[str] = None
    severity: Optional[str] = None


class ASRDisagreement(BaseModel):
    """Word-level ASR disagreement between MedASR and Whisper."""
    word_index: int
    medasr_word: str
    whisper_word: str
    clinical_significance: str = Field(
        default="low", description="low | moderate | high"
    )
    resolved: bool = Field(default=False)
    resolved_to: Optional[str] = None
