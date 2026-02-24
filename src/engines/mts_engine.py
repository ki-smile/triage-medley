"""MTS (Manchester Triage System) Engine — discriminator flowchart evaluation.

Takes FullTriageContext only (raises ValueError on PreTriageContext).
Selects a complaint-specific flowchart based on ESS category, then walks
top-down through discriminators (RED -> ORANGE -> YELLOW -> GREEN).
The first discriminator that triggers determines the triage level.
Falls back to general discriminators if no complaint-specific flowchart matches.
"""

import time
from typing import Optional

from src.models.context import FullTriageContext, PreTriageContext
from src.models.enums import Confidence, ConsciousnessLevel, RETTSLevel
from src.models.outputs import TriageOutput
from src.utils.config import load_config

MODEL_ID = "mts_rules_engine"

# MTS priority level ordering — maps MTS levels to RETTS colour equivalents
_MTS_TO_RETTS = {
    "RED": RETTSLevel.RED,
    "ORANGE": RETTSLevel.ORANGE,
    "YELLOW": RETTSLevel.YELLOW,
    "GREEN": RETTSLevel.GREEN,
    "BLUE": RETTSLevel.BLUE,
}


def evaluate(context: FullTriageContext) -> TriageOutput:
    """Run the MTS rules engine on a full triage context.

    Raises ValueError if given a PreTriageContext (no vitals).
    """
    if not isinstance(context, FullTriageContext):
        raise ValueError(
            "MTS engine requires FullTriageContext with vitals. "
            "PreTriageContext (Stage A) does not have vitals — "
            "use the pre-triage engine instead."
        )

    start = time.monotonic()

    flowcharts = load_config("mts/flowcharts.json")
    general = load_config("mts/general_discriminators.json")

    # Step 1: Select flowchart
    flowchart_name, discriminators, dont_miss = _select_flowchart(
        context.ess_category, flowcharts, general
    )

    # Step 2: Evaluate discriminators top-down
    result = _evaluate_discriminators(discriminators, context)

    # Step 3: Build vital sign concerns from any vital-based triggers
    vital_concerns = _collect_vital_concerns(discriminators, context)

    # Step 4: Build reasoning
    if result is not None:
        level = _MTS_TO_RETTS[result["level"]]
        reasoning = (
            f"MTS flowchart: {flowchart_name} | "
            f"Triggered discriminator: {result['name']} ({result['level']}) | "
            f"Max wait: {result['max_wait_minutes']} min"
        )
        native_detail = {
            "flowchart": flowchart_name,
            "triggered_discriminator": result["name"],
            "max_wait_minutes": result["max_wait_minutes"],
        }
    else:
        # Nothing triggered — default to BLUE (non-urgent)
        level = RETTSLevel.BLUE
        reasoning = (
            f"MTS flowchart: {flowchart_name} | "
            f"No discriminator triggered — defaulting to BLUE (non-urgent)"
        )
        native_detail = {
            "flowchart": flowchart_name,
            "triggered_discriminator": None,
            "max_wait_minutes": 240,
        }

    # Add EHR context to reasoning if available
    if context.ehr and context.ehr.risk_flags:
        flags = [rf.description for rf in context.ehr.risk_flags]
        reasoning += " | EHR risk flags: " + "; ".join(flags)

    risk_factors = _gather_risk_factors(context)
    chief_complaint = _extract_chief_complaint(context)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return TriageOutput(
        model_id=MODEL_ID,
        retts_level=level,
        ess_category=context.ess_category,
        chief_complaint=chief_complaint,
        clinical_reasoning=reasoning,
        vital_sign_concerns=vital_concerns,
        risk_factors=risk_factors,
        confidence=Confidence.HIGH,  # Deterministic engine
        dont_miss=dont_miss,
        triage_system="mts",
        native_level_detail=native_detail,
        timestamp=context.arrival_time,
        processing_time_ms=elapsed_ms,
    )


def _select_flowchart(
    ess_category: Optional[str],
    flowcharts: dict,
    general: dict,
) -> tuple[str, list[dict], list[str]]:
    """Select the complaint-specific flowchart or fall back to general.

    Returns (flowchart_name, discriminators_list, dont_miss_list).
    """
    # Skip the "meta" key — it contains metadata, not a flowchart
    if ess_category and ess_category != "meta" and ess_category in flowcharts:
        fc = flowcharts[ess_category]
        flowchart_name = fc.get("flowchart_name", ess_category)
        discriminators = fc.get("discriminators", [])
        dont_miss = fc.get("dont_miss", [])
        return flowchart_name, discriminators, dont_miss

    # Fall back to general discriminators.
    # Supports two config shapes:
    #   1. {"general": {"flowchart_name": ..., "discriminators": [...], "dont_miss": [...]}}
    #   2. {"general_discriminators": [...]}  (flat list, no nested object)
    if "general" in general and isinstance(general["general"], dict):
        gen_data = general["general"]
        flowchart_name = gen_data.get("flowchart_name", "General Discriminators")
        discriminators = gen_data.get("discriminators", [])
        dont_miss = gen_data.get("dont_miss", [])
    elif "general_discriminators" in general:
        flowchart_name = "General Discriminators"
        discriminators = general["general_discriminators"]
        dont_miss = general.get("dont_miss", [])
    else:
        flowchart_name = "General Discriminators"
        discriminators = []
        dont_miss = []

    return flowchart_name, discriminators, dont_miss


def _evaluate_discriminators(
    discriminators: list[dict],
    context: FullTriageContext,
) -> Optional[dict]:
    """Walk top-down through discriminators; return first triggered.

    Discriminators should already be ordered by severity (RED first,
    then ORANGE, YELLOW, GREEN). Returns None if nothing triggers.
    """
    for disc in discriminators:
        if _check_discriminator(disc, context):
            return {
                "level": disc["level"],
                "name": disc["name"],
                "max_wait_minutes": disc.get("max_wait_minutes", 0),
            }
    return None


def _check_discriminator(disc: dict, context: FullTriageContext) -> bool:
    """Evaluate a single discriminator against the context.

    Discriminator check types:
      - consciousness: context.vitals.consciousness.value in check["values"]
      - vital_threshold: vital value < below or > above
      - keyword_match: any keyword in context.speech_text (case-insensitive)
      - keyword_and_vital: both keyword match AND vital threshold
      - default: always matches (fallback)
    """
    check = disc.get("check", {})
    check_type = check.get("type", "")

    if check_type == "consciousness":
        return _check_consciousness(check, context)

    if check_type == "vital_threshold":
        return _check_vital_threshold(check, context)

    if check_type == "keyword_match":
        return _check_keyword_match(check, context)

    if check_type == "keyword_and_vital":
        keyword_ok = _check_keyword_match(check, context)
        vital_ok = _check_vital_threshold(check, context)
        return keyword_ok and vital_ok

    if check_type == "default":
        return True

    return False


def _check_consciousness(check: dict, context: FullTriageContext) -> bool:
    """Check if consciousness level is in the specified values."""
    target_values = check.get("values", [])
    return context.vitals.consciousness.value in target_values


def _check_vital_threshold(check: dict, context: FullTriageContext) -> bool:
    """Check if a vital sign is below or above a threshold."""
    vital_name = check.get("vital", "")
    vitals = context.vitals

    # Map vital name to actual value
    vital_value = _get_vital_value(vital_name, vitals)
    if vital_value is None:
        return False

    below = check.get("below")
    above = check.get("above")

    if below is not None and vital_value < below:
        return True
    if above is not None and vital_value > above:
        return True

    return False


def _check_keyword_match(check: dict, context: FullTriageContext) -> bool:
    """Check if any keyword appears in the speech text (case-insensitive)."""
    keywords = check.get("keywords", [])
    speech_lower = context.speech_text.lower()
    return any(kw.lower() in speech_lower for kw in keywords)


def _get_vital_value(vital_name: str, vitals) -> Optional[float]:
    """Get a vital sign value by name, returning None if not found."""
    vital_map = {
        "heart_rate": vitals.heart_rate,
        "systolic_bp": vitals.systolic_bp,
        "diastolic_bp": vitals.diastolic_bp,
        "respiratory_rate": vitals.respiratory_rate,
        "spo2": vitals.spo2,
        "temperature": vitals.temperature,
    }
    return vital_map.get(vital_name)


def _collect_vital_concerns(
    discriminators: list[dict],
    context: FullTriageContext,
) -> list[str]:
    """Collect human-readable concern strings from vital-based discriminators."""
    concerns: list[str] = []
    vitals = context.vitals

    for disc in discriminators:
        check = disc.get("check", {})
        check_type = check.get("type", "")

        if check_type in ("vital_threshold", "keyword_and_vital"):
            vital_name = check.get("vital", "")
            vital_value = _get_vital_value(vital_name, vitals)
            if vital_value is None:
                continue

            below = check.get("below")
            above = check.get("above")
            triggered = False

            if below is not None and vital_value < below:
                concerns.append(
                    f"{vital_name} {vital_value} < {below} ({disc['level']}: {disc['name']})"
                )
                triggered = True
            if above is not None and vital_value > above:
                concerns.append(
                    f"{vital_name} {vital_value} > {above} ({disc['level']}: {disc['name']})"
                )
                triggered = True

        if check_type == "consciousness":
            target_values = check.get("values", [])
            if vitals.consciousness.value in target_values:
                concerns.append(
                    f"Consciousness: {vitals.consciousness.value} ({disc['level']}: {disc['name']})"
                )

    return concerns


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
