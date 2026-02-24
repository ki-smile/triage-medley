"""Pre-Triage Engine — speech + EHR → queue priority.

Stage A only. No vitals. Must complete in < 60 seconds.
Uses keyword matching against red-flag patterns and EHR risk amplification.
"""

import time

from src.models.clinical import EHRSnapshot
from src.models.context import PreTriageContext
from src.models.enums import QueuePriority
from src.models.outputs import PreTriageOutput
from src.utils.config import load_pretriage_config

MODEL_ID = "pretriage_engine"


def evaluate(context: PreTriageContext) -> PreTriageOutput:
    """Evaluate pre-triage priority from speech + EHR.

    Returns PreTriageOutput with queue priority and reasoning.
    """
    start = time.monotonic()
    config = load_pretriage_config()

    # Step 1: Scan speech for red-flag keywords
    matched_flags = _match_speech_flags(context.speech_text, config["red_flags"])

    # Step 2: Determine initial priority from speech matches
    priority, ess_hint = _determine_priority(matched_flags, config)

    # Step 3: Apply EHR risk amplification
    amplifiers_detected: list[str] = []
    if context.ehr and priority != QueuePriority.HIGH:
        priority, amplifiers_detected = _apply_risk_amplifiers(
            priority, matched_flags, context.ehr,
            config.get("risk_amplifiers", {}), context.speech_text,
        )

    # Step 4: Build reasoning
    reasoning = _build_reasoning(matched_flags, amplifiers_detected, priority)
    chief_complaint = _extract_chief_complaint(matched_flags, context.speech_text)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return PreTriageOutput(
        model_id=MODEL_ID,
        queue_priority=priority,
        chief_complaint=chief_complaint,
        reasoning=reasoning,
        ess_category_hint=ess_hint,
        risk_amplifiers_detected=amplifiers_detected,
        timestamp=context.arrival_time,
        processing_time_ms=elapsed_ms,
    )


def _match_speech_flags(
    speech_text: str, red_flags: dict
) -> list[dict]:
    """Match speech text against red-flag keyword patterns.

    Returns list of matched flag entries with their metadata.
    """
    text_lower = speech_text.lower()
    matched = []

    for flag_name, flag_data in red_flags.items():
        keywords = flag_data.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append({
                    "flag_name": flag_name,
                    "keyword": keyword,
                    "priority": flag_data.get("priority", "STANDARD"),
                    "ess_hint": flag_data.get("ess_hint"),
                })
                break  # One match per flag category is enough

    return matched


def _determine_priority(
    matched_flags: list[dict], config: dict
) -> tuple[QueuePriority, str | None]:
    """Determine queue priority from matched flags."""
    if not matched_flags:
        default = config.get("default_priority", "STANDARD")
        return QueuePriority(default), None

    # Take the highest priority from all matches
    priorities = [f["priority"] for f in matched_flags]
    if "HIGH" in priorities:
        priority = QueuePriority.HIGH
    elif "MODERATE" in priorities:
        priority = QueuePriority.MODERATE
    else:
        priority = QueuePriority.STANDARD

    # Use the ESS hint from the highest-priority match
    ess_hint = None
    for flag in matched_flags:
        if flag["priority"] == priority.value and flag.get("ess_hint"):
            ess_hint = flag["ess_hint"]
            break

    return priority, ess_hint


def _apply_risk_amplifiers(
    current_priority: QueuePriority,
    matched_flags: list[dict],
    ehr: EHRSnapshot,
    amplifiers: dict,
    full_speech_text: str = "",
) -> tuple[QueuePriority, list[str]]:
    """Apply EHR risk amplification rules.

    If speech triggers MODERATE + EHR matches amplifier conditions → escalate.
    Checks the full speech text (not just matched keywords) for condition matches.
    """
    detected: list[str] = []
    flag_names = {f["flag_name"] for f in matched_flags}
    speech_lower = full_speech_text.lower()

    for amp_name, amp_data in amplifiers.items():
        # Check medication-based amplifiers
        if "medications" in amp_data and "conditions" in amp_data:
            med_keywords = amp_data["medications"]
            cond_keywords = amp_data["conditions"]
            if ehr.has_medication_class(med_keywords):
                # Check if full speech text or EHR conditions match
                if any(kw.lower() in speech_lower for kw in cond_keywords) or \
                   ehr.has_condition_matching(cond_keywords):
                    detected.append(amp_data.get("description", amp_name))

        # Check condition + speech flag amplifiers
        if "conditions" in amp_data and "speech_flags" in amp_data:
            speech_flag_names = amp_data["speech_flags"]
            if any(sf in flag_names for sf in speech_flag_names):
                cond_keywords = amp_data["conditions"]
                if ehr.has_condition_matching(cond_keywords):
                    desc = amp_data.get("description", amp_name)
                    if desc not in detected:
                        detected.append(desc)

    if detected:
        escalate_to = QueuePriority.HIGH
        return escalate_to, detected

    return current_priority, detected


def _build_reasoning(
    matched_flags: list[dict],
    amplifiers: list[str],
    priority: QueuePriority,
) -> str:
    """Build human-readable reasoning string."""
    parts = []

    if matched_flags:
        flag_strs = [f"{f['flag_name']} ('{f['keyword']}')" for f in matched_flags]
        parts.append(f"Speech flags detected: {', '.join(flag_strs)}")
    else:
        parts.append("No red-flag keywords detected in speech")

    if amplifiers:
        parts.append(f"EHR risk amplifiers: {', '.join(amplifiers)}")

    parts.append(f"Queue priority: {priority.value}")
    return ". ".join(parts) + "."


def _extract_chief_complaint(
    matched_flags: list[dict], speech_text: str
) -> str:
    """Extract chief complaint from matched flags or speech."""
    if matched_flags:
        # Use the highest-priority flag as chief complaint
        return matched_flags[0]["flag_name"].replace("_", " ").title()

    # Fallback: first sentence of speech
    text = speech_text.strip()
    if "." in text:
        return text[: text.index(".")]
    if len(text) > 100:
        return text[:97] + "..."
    return text
