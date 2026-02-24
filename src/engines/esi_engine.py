"""ESI Rules Engine — deterministic ESI 5-level triage decision tree.

Takes FullTriageContext only (raises ValueError on PreTriageContext).
Implements the Emergency Severity Index algorithm:
  ESI-1: Dying patient (immediate life-saving intervention)
  ESI-2: High-risk / confused / severe distress
  ESI-3: Multiple resources expected (with vitals danger zone safety net)
  ESI-4: One resource expected
  ESI-5: No resources expected

ESI levels are mapped to RETTSLevel for unified pipeline output:
  ESI-1 -> RED, ESI-2 -> ORANGE, ESI-3 -> YELLOW, ESI-4 -> GREEN, ESI-5 -> BLUE
"""

import time
from typing import Optional

from src.models.context import FullTriageContext, PreTriageContext
from src.models.enums import Confidence, ConsciousnessLevel, RETTSLevel
from src.models.outputs import TriageOutput
from src.utils.config import load_config

MODEL_ID = "esi_rules_engine"

# ESI level -> RETTSLevel mapping
_ESI_TO_RETTS = {
    1: RETTSLevel.RED,
    2: RETTSLevel.ORANGE,
    3: RETTSLevel.YELLOW,
    4: RETTSLevel.GREEN,
    5: RETTSLevel.BLUE,
}

# Default resource count when ESS category is unknown
_DEFAULT_RESOURCES = 2


def evaluate(context: FullTriageContext) -> TriageOutput:
    """Run the ESI rules engine on a full triage context.

    Raises ValueError if given a PreTriageContext (no vitals).
    """
    if not isinstance(context, FullTriageContext):
        raise ValueError(
            "ESI engine requires FullTriageContext with vitals. "
            "PreTriageContext (Stage A) does not have vitals — "
            "use the pre-triage engine instead."
        )

    start = time.monotonic()

    decision_config = load_config("esi/decision_tree.json")
    resource_config = load_config("esi/resource_rules.json")

    reasoning_steps: list[str] = []
    vital_sign_concerns: list[str] = []
    dont_miss: list[str] = []

    # --- Step 1: Is the patient dying? (ESI-1) ---
    if _is_esi_1(context, decision_config, vital_sign_concerns, reasoning_steps):
        esi_level = 1
        reasoning_steps.insert(0, "ESI-1: Patient meets immediate life-threat criteria")
    # --- Step 2: High-risk situation? (ESI-2) ---
    elif _is_esi_2_high_risk(context, decision_config, dont_miss, reasoning_steps):
        esi_level = 2
        reasoning_steps.insert(0, "ESI-2: High-risk situation identified")
    else:
        # --- Step 3-5: Resource-based estimation ---
        resources = _estimate_resources(context, resource_config, reasoning_steps)

        if resources >= 2:
            # Check vitals danger zone safety net
            danger_zone = _vitals_danger_zone(
                context, decision_config, vital_sign_concerns, reasoning_steps
            )
            if danger_zone:
                esi_level = 2
                reasoning_steps.insert(
                    0,
                    "ESI-3 -> ESI-2 UPGRADE: Vitals in danger zone with "
                    f"{resources} predicted resources",
                )
            else:
                esi_level = 3
                reasoning_steps.insert(
                    0,
                    f"ESI-3: {resources} resources predicted, "
                    "vitals outside danger zone",
                )
        elif resources == 1:
            esi_level = 4
            reasoning_steps.insert(0, f"ESI-4: {resources} resource predicted")
        else:
            esi_level = 5
            reasoning_steps.insert(0, "ESI-5: No resources predicted")

    retts_level = _ESI_TO_RETTS[esi_level]
    resources_predicted = _estimate_resources(context, resource_config, [])

    # Build native level detail
    native_detail: dict = {
        "esi_level": esi_level,
        "resources_predicted": resources_predicted,
    }
    # Include danger zone flag if ESI-3 was evaluated (resources >= 2)
    if resources_predicted >= 2 and esi_level not in (1,):
        danger_zone_triggered = (
            esi_level == 2
            and resources_predicted >= 2
            and not _is_esi_2_high_risk(context, decision_config, [], [])
        )
        native_detail["danger_zone_triggered"] = danger_zone_triggered

    # Add dont_miss from config for ESI-2 criteria if not already populated
    if not dont_miss:
        esi_2_config = decision_config.get("esi_2_criteria", {})
        dont_miss = esi_2_config.get("dont_miss_conditions", [])

    clinical_reasoning = _build_reasoning(reasoning_steps, context)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    return TriageOutput(
        model_id=MODEL_ID,
        retts_level=retts_level,
        ess_category=context.ess_category,
        chief_complaint=_extract_chief_complaint(context),
        clinical_reasoning=clinical_reasoning,
        vital_sign_concerns=vital_sign_concerns,
        risk_factors=_gather_risk_factors(context),
        confidence=Confidence.HIGH,  # Deterministic engine
        dont_miss=dont_miss,
        triage_system="esi",
        native_level_detail=native_detail,
        timestamp=context.arrival_time,
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# ESI Decision Tree Steps
# ---------------------------------------------------------------------------

def _is_esi_1(
    context: FullTriageContext,
    config: dict,
    concerns: list[str],
    reasoning: list[str],
) -> bool:
    """ESI-1: Is the patient dying? Requires immediate life-saving intervention.

    Checks for:
    - Unresponsive consciousness
    - Critical vital sign thresholds (HR, SBP, RR, SpO2)
    - Intervention keywords in speech text (intubation, cardiac arrest, etc.)
    """
    criteria = config.get("esi_1_criteria", {})
    vitals = context.vitals
    triggered = False

    # Check consciousness — Unresponsive = ESI-1
    consciousness_cfg = criteria.get("consciousness", {})
    # Support both formats: {"values": [...]} or plain list
    if isinstance(consciousness_cfg, dict):
        esi_1_consciousness = consciousness_cfg.get("values", ["Unresponsive"])
    else:
        esi_1_consciousness = consciousness_cfg

    if vitals.consciousness.value in esi_1_consciousness:
        concerns.append(
            f"Consciousness: {vitals.consciousness.value} (ESI-1 critical)"
        )
        reasoning.append(
            f"Patient is {vitals.consciousness.value} — meets ESI-1 consciousness criterion"
        )
        triggered = True

    # Check critical vitals
    critical_vitals = criteria.get("critical_vitals", {})

    # Heart rate
    hr_rules = critical_vitals.get("heart_rate", {})
    if "below" in hr_rules and vitals.heart_rate < hr_rules["below"]:
        concerns.append(
            f"Heart rate {vitals.heart_rate} bpm < {hr_rules['below']} (ESI-1 critical)"
        )
        reasoning.append(
            f"HR {vitals.heart_rate} < {hr_rules['below']} — critical bradycardia"
        )
        triggered = True
    if "above" in hr_rules and vitals.heart_rate > hr_rules["above"]:
        concerns.append(
            f"Heart rate {vitals.heart_rate} bpm > {hr_rules['above']} (ESI-1 critical)"
        )
        reasoning.append(
            f"HR {vitals.heart_rate} > {hr_rules['above']} — critical tachycardia"
        )
        triggered = True

    # Systolic BP
    bp_rules = critical_vitals.get("systolic_bp", {})
    if "below" in bp_rules and vitals.systolic_bp < bp_rules["below"]:
        concerns.append(
            f"Systolic BP {vitals.systolic_bp} mmHg < {bp_rules['below']} (ESI-1 critical)"
        )
        reasoning.append(
            f"SBP {vitals.systolic_bp} < {bp_rules['below']} — critical hypotension"
        )
        triggered = True

    # Respiratory rate
    rr_rules = critical_vitals.get("respiratory_rate", {})
    if "below" in rr_rules and vitals.respiratory_rate < rr_rules["below"]:
        concerns.append(
            f"Respiratory rate {vitals.respiratory_rate}/min < {rr_rules['below']} (ESI-1 critical)"
        )
        reasoning.append(
            f"RR {vitals.respiratory_rate} < {rr_rules['below']} — critical bradypnea"
        )
        triggered = True
    if "above" in rr_rules and vitals.respiratory_rate > rr_rules["above"]:
        concerns.append(
            f"Respiratory rate {vitals.respiratory_rate}/min > {rr_rules['above']} (ESI-1 critical)"
        )
        reasoning.append(
            f"RR {vitals.respiratory_rate} > {rr_rules['above']} — critical tachypnea"
        )
        triggered = True

    # SpO2
    spo2_rules = critical_vitals.get("spo2", {})
    if "below" in spo2_rules and vitals.spo2 < spo2_rules["below"]:
        concerns.append(
            f"SpO2 {vitals.spo2}% < {spo2_rules['below']}% (ESI-1 critical)"
        )
        reasoning.append(
            f"SpO2 {vitals.spo2}% < {spo2_rules['below']}% — critical hypoxemia"
        )
        triggered = True

    # Check intervention keywords in speech text
    intervention_keywords = criteria.get("intervention_keywords", [])
    if intervention_keywords:
        speech_lower = context.speech_text.lower()
        matched = [kw for kw in intervention_keywords if kw.lower() in speech_lower]
        if matched:
            reasoning.append(
                f"Life-threat keywords detected in speech: {', '.join(matched)}"
            )
            triggered = True

    return triggered


def _is_esi_2_high_risk(
    context: FullTriageContext,
    config: dict,
    dont_miss: list[str],
    reasoning: list[str],
) -> bool:
    """ESI-2: High-risk situation check.

    Checks for:
    - Altered consciousness (Pain or Voice on AVPU)
    - High-risk keywords in speech text
    """
    criteria = config.get("esi_2_criteria", {})
    triggered = False

    # Check consciousness — Pain or Voice = ESI-2
    consciousness_cfg = criteria.get("altered_consciousness", {})
    if isinstance(consciousness_cfg, dict):
        esi_2_consciousness = consciousness_cfg.get("values", ["Pain", "Voice"])
    else:
        esi_2_consciousness = consciousness_cfg

    if context.vitals.consciousness.value in esi_2_consciousness:
        reasoning.append(
            f"Consciousness: {context.vitals.consciousness.value} — "
            "altered mental status meets ESI-2"
        )
        triggered = True

    # Check speech text for high-risk keywords
    high_risk_keywords = criteria.get("high_risk_speech_keywords", [])
    speech_lower = context.speech_text.lower()
    matched_keywords = [kw for kw in high_risk_keywords if kw.lower() in speech_lower]
    if matched_keywords:
        reasoning.append(
            f"High-risk keywords detected in speech: {', '.join(matched_keywords)}"
        )
        triggered = True

    # Populate dont_miss from config
    if triggered:
        config_dont_miss = criteria.get("dont_miss_conditions", [])
        dont_miss.extend(config_dont_miss)

    return triggered


def _estimate_resources(
    context: FullTriageContext,
    resource_config: dict,
    reasoning: list[str],
) -> int:
    """Estimate the number of resources the patient will require.

    Resources include: labs, imaging, IV fluids/meds, procedures, specialist consults.
    Uses ESS category from context to look up resource rules.
    Falls back to _default entry if category is unknown.
    """
    ess_category = context.ess_category

    if ess_category and ess_category in resource_config:
        entry = resource_config[ess_category]
        resources = entry.get("min_resources", _DEFAULT_RESOURCES)
        detail = entry.get("detail", [])
        if reasoning is not None:
            reasoning.append(
                f"Resource estimation for '{ess_category}': {resources} "
                f"({', '.join(detail) if detail else 'none specified'})"
            )
    else:
        # Fallback: use _default entry or hard default
        default_entry = resource_config.get("_default", {})
        resources = default_entry.get("min_resources", _DEFAULT_RESOURCES)
        category_label = ess_category if ess_category else "none"
        if reasoning is not None:
            reasoning.append(
                f"ESS category '{category_label}' not found in resource rules — "
                f"defaulting to {resources} resources"
            )

    return resources


def _vitals_danger_zone(
    context: FullTriageContext,
    config: dict,
    concerns: list[str],
    reasoning: list[str],
) -> bool:
    """ESI danger zone check — safety net for ESI-3 patients.

    If any vital sign falls in the danger zone, the patient is bumped
    from ESI-3 to ESI-2. Thresholds loaded from esi_2_criteria.vitals_danger_zone.
    """
    # Danger zone is nested under esi_2_criteria in the config
    esi_2_criteria = config.get("esi_2_criteria", {})
    danger_zone = esi_2_criteria.get("vitals_danger_zone", {})
    vitals = context.vitals
    triggered = False

    # Heart rate
    hr_rules = danger_zone.get("heart_rate", {})
    if "below" in hr_rules and vitals.heart_rate < hr_rules["below"]:
        concerns.append(
            f"Heart rate {vitals.heart_rate} bpm < {hr_rules['below']} (danger zone)"
        )
        reasoning.append(
            f"HR {vitals.heart_rate} in danger zone (< {hr_rules['below']})"
        )
        triggered = True
    if "above" in hr_rules and vitals.heart_rate > hr_rules["above"]:
        concerns.append(
            f"Heart rate {vitals.heart_rate} bpm > {hr_rules['above']} (danger zone)"
        )
        reasoning.append(
            f"HR {vitals.heart_rate} in danger zone (> {hr_rules['above']})"
        )
        triggered = True

    # Systolic BP
    bp_rules = danger_zone.get("systolic_bp", {})
    if "below" in bp_rules and vitals.systolic_bp < bp_rules["below"]:
        concerns.append(
            f"Systolic BP {vitals.systolic_bp} mmHg < {bp_rules['below']} (danger zone)"
        )
        reasoning.append(
            f"SBP {vitals.systolic_bp} in danger zone (< {bp_rules['below']})"
        )
        triggered = True
    if "above" in bp_rules and vitals.systolic_bp > bp_rules["above"]:
        concerns.append(
            f"Systolic BP {vitals.systolic_bp} mmHg > {bp_rules['above']} (danger zone)"
        )
        reasoning.append(
            f"SBP {vitals.systolic_bp} in danger zone (> {bp_rules['above']})"
        )
        triggered = True

    # Respiratory rate
    rr_rules = danger_zone.get("respiratory_rate", {})
    if "below" in rr_rules and vitals.respiratory_rate < rr_rules["below"]:
        concerns.append(
            f"Respiratory rate {vitals.respiratory_rate}/min < {rr_rules['below']} (danger zone)"
        )
        reasoning.append(
            f"RR {vitals.respiratory_rate} in danger zone (< {rr_rules['below']})"
        )
        triggered = True
    if "above" in rr_rules and vitals.respiratory_rate > rr_rules["above"]:
        concerns.append(
            f"Respiratory rate {vitals.respiratory_rate}/min > {rr_rules['above']} (danger zone)"
        )
        reasoning.append(
            f"RR {vitals.respiratory_rate} in danger zone (> {rr_rules['above']})"
        )
        triggered = True

    # SpO2
    spo2_rules = danger_zone.get("spo2", {})
    if "below" in spo2_rules and vitals.spo2 < spo2_rules["below"]:
        concerns.append(
            f"SpO2 {vitals.spo2}% < {spo2_rules['below']}% (danger zone)"
        )
        reasoning.append(
            f"SpO2 {vitals.spo2}% in danger zone (< {spo2_rules['below']}%)"
        )
        triggered = True

    # Temperature
    temp_rules = danger_zone.get("temperature", {})
    if "below" in temp_rules and vitals.temperature < temp_rules["below"]:
        concerns.append(
            f"Temperature {vitals.temperature} C < {temp_rules['below']} (danger zone)"
        )
        reasoning.append(
            f"Temp {vitals.temperature} in danger zone (< {temp_rules['below']})"
        )
        triggered = True
    if "above" in temp_rules and vitals.temperature > temp_rules["above"]:
        concerns.append(
            f"Temperature {vitals.temperature} C > {temp_rules['above']} (danger zone)"
        )
        reasoning.append(
            f"Temp {vitals.temperature} in danger zone (> {temp_rules['above']})"
        )
        triggered = True

    return triggered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _build_reasoning(steps: list[str], context: FullTriageContext) -> str:
    """Build human-readable clinical reasoning from decision path."""
    parts = list(steps)
    if context.ehr and context.ehr.risk_flags:
        flags = [rf.description for rf in context.ehr.risk_flags]
        parts.append("EHR risk flags: " + "; ".join(flags))
    return " | ".join(parts)
