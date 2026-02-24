"""RETTS Rules Engine — deterministic vitals → RETTS colour.

Takes FullTriageContext only (raises ValueError on PreTriageContext).
Evaluates vitals against age-stratified thresholds, maps chief complaint
to ESS code, and returns the HIGHER of vitals assessment and ESS assessment.
"""

import time
from typing import Optional

from src.models.context import FullTriageContext, PreTriageContext
from src.models.enums import Confidence, ConsciousnessLevel, RETTSLevel
from src.models.outputs import TriageOutput
from src.utils.config import load_ess_codes, load_vitals_cutoffs

MODEL_ID = "retts_rules_engine"

# RETTS severity ordering for comparisons
_LEVEL_ORDER = [RETTSLevel.RED, RETTSLevel.ORANGE, RETTSLevel.YELLOW,
                RETTSLevel.GREEN, RETTSLevel.BLUE]


def evaluate(context: FullTriageContext) -> TriageOutput:
    """Run the RETTS rules engine on a full triage context.

    Raises ValueError if given a PreTriageContext (no vitals).
    """
    if not isinstance(context, FullTriageContext):
        raise ValueError(
            "RETTS engine requires FullTriageContext with vitals. "
            "PreTriageContext (Stage A) does not have vitals — "
            "use the pre-triage engine instead."
        )

    start = time.monotonic()

    vitals_level, vitals_concerns = _assess_vitals(context)
    ess_level = _assess_ess(context)

    # RETTS takes the higher (more severe) of vitals and ESS
    final_level = RETTSLevel.most_severe(vitals_level, ess_level)

    risk_factors = _gather_risk_factors(context)
    reasoning = _build_reasoning(vitals_level, ess_level, final_level,
                                 vitals_concerns, context)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return TriageOutput(
        model_id=MODEL_ID,
        retts_level=final_level,
        ess_category=context.ess_category,
        chief_complaint=_extract_chief_complaint(context),
        clinical_reasoning=reasoning,
        vital_sign_concerns=vitals_concerns,
        risk_factors=risk_factors,
        confidence=Confidence.HIGH,  # Deterministic engine
        triage_system="retts",
        timestamp=context.arrival_time,
        processing_time_ms=elapsed_ms,
    )


def _assess_vitals(context: FullTriageContext) -> tuple[RETTSLevel, list[str]]:
    """Assess vital signs against RETTS thresholds."""
    cutoffs = load_vitals_cutoffs()
    age_group = "pediatric" if (context.ehr and context.ehr.is_pediatric) else "adult"
    thresholds = cutoffs[age_group]

    concerns: list[str] = []
    worst_level = RETTSLevel.BLUE

    vitals = context.vitals

    # Heart rate
    hr_level = _check_numeric_vital(
        vitals.heart_rate, thresholds["heart_rate"], "Heart rate", "bpm", concerns
    )
    worst_level = RETTSLevel.most_severe(worst_level, hr_level)

    # Systolic BP
    bp_level = _check_numeric_vital(
        vitals.systolic_bp, thresholds["systolic_bp"],
        "Systolic BP", "mmHg", concerns
    )
    worst_level = RETTSLevel.most_severe(worst_level, bp_level)

    # Respiratory rate
    rr_level = _check_numeric_vital(
        vitals.respiratory_rate, thresholds["respiratory_rate"],
        "Respiratory rate", "/min", concerns
    )
    worst_level = RETTSLevel.most_severe(worst_level, rr_level)

    # SpO2
    spo2_level = _check_numeric_vital(
        vitals.spo2, thresholds["spo2"], "SpO2", "%", concerns
    )
    worst_level = RETTSLevel.most_severe(worst_level, spo2_level)

    # Temperature
    temp_level = _check_numeric_vital(
        vitals.temperature, thresholds["temperature"],
        "Temperature", "°C", concerns
    )
    worst_level = RETTSLevel.most_severe(worst_level, temp_level)

    # Consciousness (AVPU)
    consciousness_level = _check_consciousness(
        vitals.consciousness, thresholds["consciousness"], concerns
    )
    worst_level = RETTSLevel.most_severe(worst_level, consciousness_level)

    return worst_level, concerns


def _check_numeric_vital(
    value: float,
    thresholds: dict,
    name: str,
    unit: str,
    concerns: list[str],
) -> RETTSLevel:
    """Check a numeric vital sign against RETTS thresholds.

    Returns the most severe matching level.
    """
    for level_str in ["RED", "ORANGE", "YELLOW"]:
        rules = thresholds.get(level_str, {})
        triggered = False

        if "high" in rules and value >= rules["high"]:
            concerns.append(f"{name} {value} {unit} >= {rules['high']} ({level_str})")
            triggered = True
        if "low_below" in rules and value < rules["low_below"]:
            concerns.append(f"{name} {value} {unit} < {rules['low_below']} ({level_str})")
            triggered = True

        if triggered:
            return RETTSLevel(level_str)

    return RETTSLevel.GREEN


def _check_consciousness(
    level: ConsciousnessLevel,
    thresholds: dict,
    concerns: list[str],
) -> RETTSLevel:
    """Check AVPU consciousness level against RETTS thresholds."""
    for level_str in ["RED", "ORANGE", "YELLOW"]:
        matching = thresholds.get(level_str, [])
        if level.value in matching:
            concerns.append(f"Consciousness: {level.value} ({level_str})")
            return RETTSLevel(level_str)
    return RETTSLevel.GREEN


def _assess_ess(context: FullTriageContext) -> RETTSLevel:
    """Assess ESS category severity. Returns default if no match."""
    ess_category = context.ess_category
    if not ess_category:
        return RETTSLevel.GREEN

    ess_codes = load_ess_codes()
    ess_data = ess_codes.get(ess_category)
    if not ess_data:
        return RETTSLevel.GREEN

    default = ess_data.get("default_level", "GREEN")
    return RETTSLevel(default)


def _gather_risk_factors(context: FullTriageContext) -> list[str]:
    """Collect EHR-based risk factors."""
    if not context.ehr:
        return []
    return [f"{rf.flag_type}: {rf.description}" for rf in context.ehr.risk_flags]


def _extract_chief_complaint(context: FullTriageContext) -> str:
    """Extract a brief chief complaint from speech text."""
    text = context.speech_text.strip()
    if len(text) > 120:
        return text[:117] + "..."
    return text


def _build_reasoning(
    vitals_level: RETTSLevel,
    ess_level: RETTSLevel,
    final_level: RETTSLevel,
    concerns: list[str],
    context: FullTriageContext,
) -> str:
    """Build human-readable clinical reasoning."""
    parts = [
        f"Vitals assessment: {vitals_level.value}",
        f"ESS category assessment ({context.ess_category or 'none'}): {ess_level.value}",
        f"Final RETTS level: {final_level.value} (higher of vitals and ESS)",
    ]
    if concerns:
        parts.append("Vital sign concerns: " + "; ".join(concerns))
    if context.ehr and context.ehr.risk_flags:
        flags = [rf.description for rf in context.ehr.risk_flags]
        parts.append("EHR risk flags: " + "; ".join(flags))
    return " | ".join(parts)
