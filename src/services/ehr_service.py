"""EHR Service — FHIR Bundle parser and risk amplifier.

Loads synthetic Synthea FHIR R4 bundles and extracts structured EHRSnapshot.
Computes risk flags from medication + condition combinations.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from src.models.clinical import (
    EHRSnapshot,
    FHIRAllergy,
    FHIRCondition,
    FHIRMedication,
    RiskFlag,
)
from src.utils.config import get_project_root

_EHR_DIR = get_project_root() / "data" / "ehr"

# Risk amplification rules: medication class + condition → risk flag
_RISK_RULES = [
    {
        "name": "anticoagulation_risk",
        "description": "Patient on anticoagulant therapy — bleeding risk with trauma",
        "medication_keywords": ["warfarin", "apixaban", "rivaroxaban", "dabigatran",
                                "edoxaban", "heparin", "enoxaparin"],
        "severity": "high",
    },
    {
        "name": "cardiac_polypharmacy",
        "description": "Multiple cardiac medications — complex drug interactions",
        "medication_keywords": ["metoprolol", "enalapril", "furosemide", "digoxin",
                                "amlodipine", "ramipril"],
        "min_matches": 3,
        "severity": "moderate",
    },
    {
        "name": "diabetes_hypoglycemia_risk",
        "description": "Diabetic patient — hypoglycemia risk with altered consciousness",
        "condition_keywords": ["diabetes"],
        "severity": "moderate",
    },
    {
        "name": "immunosuppression",
        "description": "Immunosuppressed — infection risk elevated",
        "medication_keywords": ["methotrexate", "azathioprine", "cyclosporine",
                                "tacrolimus", "prednisolone", "prednisone"],
        "severity": "high",
    },
    {
        "name": "fall_risk_elderly",
        "description": "Elderly patient (>75) with osteoporosis — fracture risk",
        "condition_keywords": ["osteoporosis"],
        "min_age": 75,
        "severity": "moderate",
    },
    {
        "name": "beta_blocker_masking",
        "description": "Beta-blocker may mask tachycardia — vital signs can be deceptively normal",
        "medication_keywords": ["metoprolol", "atenolol", "bisoprolol", "propranolol",
                                "carvedilol", "labetalol", "sotalol"],
        "severity": "moderate",
    },
    {
        "name": "aortic_valve_disease",
        "description": "Known aortic valve/root pathology — dissection risk elevated",
        "condition_keywords": ["bicuspid", "aortic valve", "aortic root", "aortic dilation",
                               "marfan"],
        "severity": "high",
    },
    {
        "name": "vte_risk_hormonal",
        "description": "Hormonal contraceptive use — venous thromboembolism risk",
        "medication_keywords": ["drospirenone", "ethinylestradiol", "yasmin", "yaz",
                                "levonorgestrel", "desogestrel", "gestodene"],
        "severity": "moderate",
    },
    {
        "name": "chronic_steroid_adrenal",
        "description": "Chronic corticosteroid use — adrenal insufficiency risk if stressed",
        "medication_keywords": ["prednisone", "prednisolone", "hydrocortisone",
                                "dexamethasone", "methylprednisolone"],
        "severity": "moderate",
    },
]


def load_patient(patient_id: str) -> Optional[EHRSnapshot]:
    """Load a patient's EHR from a FHIR Bundle JSON file.

    Returns None if the patient file doesn't exist (graceful degradation).
    """
    path = _EHR_DIR / f"{patient_id}.json"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    return _parse_bundle(patient_id, bundle)


def _parse_bundle(patient_id: str, bundle: dict) -> EHRSnapshot:
    """Parse a FHIR R4 Bundle into an EHRSnapshot."""
    entries = bundle.get("entry", [])

    patient_resource = None
    conditions: list[FHIRCondition] = []
    medications: list[FHIRMedication] = []
    allergies: list[FHIRAllergy] = []

    for entry in entries:
        resource = entry.get("resource", {})
        res_type = resource.get("resourceType")

        if res_type == "Patient":
            patient_resource = resource
        elif res_type == "Condition":
            conditions.append(_parse_condition(resource))
        elif res_type == "MedicationStatement":
            medications.append(_parse_medication(resource))
        elif res_type == "AllergyIntolerance":
            allergies.append(_parse_allergy(resource))

    name = _extract_name(patient_resource) if patient_resource else patient_id
    age = _compute_age(patient_resource) if patient_resource else 0
    sex = _extract_sex(patient_resource) if patient_resource else "M"

    snapshot = EHRSnapshot(
        patient_id=patient_id,
        name=name,
        age=age,
        sex=sex,
        conditions=conditions,
        medications=medications,
        allergies=allergies,
        risk_flags=[],
    )

    # Compute risk flags after building the snapshot
    snapshot.risk_flags = _compute_risk_flags(snapshot)

    return snapshot


def _parse_condition(resource: dict) -> FHIRCondition:
    coding = _first_coding(resource.get("code", {}))
    status_coding = resource.get("clinicalStatus", {}).get("coding", [{}])
    status = status_coding[0].get("code", "active") if status_coding else "active"
    onset = resource.get("onsetDateTime")
    onset_date = None
    if onset:
        try:
            onset_date = datetime.fromisoformat(onset).date()
        except (ValueError, TypeError):
            pass

    return FHIRCondition(
        code=coding.get("code", "unknown"),
        display=coding.get("display", "Unknown condition"),
        onset_date=onset_date,
        status=status,
    )


def _parse_medication(resource: dict) -> FHIRMedication:
    coding = _first_coding(resource.get("medicationCodeableConcept", {}))
    dosage_list = resource.get("dosage", [])
    dosage = dosage_list[0].get("text") if dosage_list else None
    status = resource.get("status", "active")

    return FHIRMedication(
        code=coding.get("code", "unknown"),
        display=coding.get("display", "Unknown medication"),
        dosage=dosage,
        status=status,
    )


def _parse_allergy(resource: dict) -> FHIRAllergy:
    coding = _first_coding(resource.get("code", {}))
    reactions = resource.get("reaction", [])
    reaction_text = None
    severity = None
    if reactions:
        manifestations = reactions[0].get("manifestation", [])
        if manifestations:
            man_coding = _first_coding(manifestations[0])
            reaction_text = man_coding.get("display")
        severity = reactions[0].get("severity")

    return FHIRAllergy(
        substance=coding.get("display", "Unknown"),
        reaction=reaction_text,
        severity=severity,
    )


def _first_coding(codeable_concept: dict) -> dict:
    """Extract the first coding entry from a FHIR CodeableConcept."""
    codings = codeable_concept.get("coding", [])
    return codings[0] if codings else {}


def _extract_name(patient: dict) -> str:
    names = patient.get("name", [])
    if not names:
        return "Unknown"
    name_entry = names[0]
    given = " ".join(name_entry.get("given", []))
    family = name_entry.get("family", "")
    return f"{given} {family}".strip()


def _extract_sex(patient: dict) -> str:
    gender = patient.get("gender", "unknown")
    return {"male": "M", "female": "F"}.get(gender, "M")


def _compute_age(patient: dict) -> int:
    birth_str = patient.get("birthDate")
    if not birth_str:
        return 0
    try:
        birth = datetime.fromisoformat(birth_str).date()
        today = date.today()
        age = today.year - birth.year
        if (today.month, today.day) < (birth.month, birth.day):
            age -= 1
        return max(0, age)
    except (ValueError, TypeError):
        return 0


def _compute_risk_flags(snapshot: EHRSnapshot) -> list[RiskFlag]:
    """Compute risk flags from medication + condition combinations."""
    flags: list[RiskFlag] = []
    med_names = [m.display.lower() for m in snapshot.active_medications]
    cond_names = [c.display.lower() for c in snapshot.active_conditions]

    for rule in _RISK_RULES:
        # Check age requirement
        if "min_age" in rule and snapshot.age < rule["min_age"]:
            continue

        triggered = False

        # Medication-based rules
        if "medication_keywords" in rule:
            min_matches = rule.get("min_matches", 1)
            matches = sum(
                1 for kw in rule["medication_keywords"]
                if any(kw.lower() in mn for mn in med_names)
            )
            if matches >= min_matches:
                triggered = True

        # Condition-based rules
        if "condition_keywords" in rule:
            if any(
                kw.lower() in cn
                for kw in rule["condition_keywords"]
                for cn in cond_names
            ):
                triggered = True

        if triggered:
            flags.append(RiskFlag(
                flag_type=rule["name"],
                description=rule["description"],
                source="ehr",
                severity=rule["severity"],
            ))

    return flags


def list_available_patients() -> list[str]:
    """List patient IDs with available FHIR bundles."""
    if not _EHR_DIR.exists():
        return []
    return [p.stem for p in _EHR_DIR.glob("*.json")]
