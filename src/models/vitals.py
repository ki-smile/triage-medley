"""Vital signs data model."""

from pydantic import BaseModel, Field

from src.models.enums import ConsciousnessLevel


class VitalSigns(BaseModel):
    """Measured vital signs — required for Stage B (full triage)."""

    heart_rate: int = Field(..., ge=0, le=300, description="Heart rate in bpm")
    systolic_bp: int = Field(..., ge=0, le=350, description="Systolic blood pressure mmHg")
    diastolic_bp: int = Field(..., ge=0, le=250, description="Diastolic blood pressure mmHg")
    respiratory_rate: int = Field(..., ge=0, le=80, description="Respiratory rate /min")
    spo2: int = Field(..., ge=0, le=100, description="Oxygen saturation %")
    temperature: float = Field(..., ge=25.0, le=45.0, description="Temperature in Celsius")
    consciousness: ConsciousnessLevel = Field(
        ..., description="AVPU consciousness level"
    )
