"""Prompt builder — renders YAML prompt templates with clinical context.

Loads prompt templates from config/prompts/ and fills in patient data
from PreTriageContext or FullTriageContext. Used by HuggingFace adapters
to construct the messages sent to the Inference API.
"""

from src.models.context import FullTriageContext, PreTriageContext
from src.utils.config import load_prompt_template


def build_pretriage_prompt(context: PreTriageContext) -> list[dict[str, str]]:
    """Build chat messages for Stage A pre-triage.

    Returns a list of {"role": ..., "content": ...} dicts.
    """
    template = load_prompt_template("pretriage")
    system_prompt = template["system_prompt"].strip()

    # Build EHR summary
    ehr_summary = "No EHR data available"
    if context.ehr:
        parts = []
        active_conds = context.ehr.active_conditions
        if active_conds:
            parts.append(f"Active conditions: {', '.join(c.display for c in active_conds)}")
        active_meds = context.ehr.active_medications
        if active_meds:
            parts.append(f"Current medications: {', '.join(m.display for m in active_meds)}")
        if context.ehr.allergies:
            parts.append(f"Allergies: {', '.join(a.substance for a in context.ehr.allergies)}")
        if context.ehr.risk_flags:
            parts.append(f"Risk flags: {', '.join(f.description for f in context.ehr.risk_flags)}")
        ehr_summary = "\n".join(parts) if parts else "No significant history"

    user_content = template["user_template"].format(
        speech_text=context.speech_text,
        ehr_summary=ehr_summary,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content.strip()},
    ]


def build_triage_prompt(
    context: FullTriageContext, model_id: str
) -> list[dict[str, str]]:
    """Build chat messages for Stage B full triage."""
    template = load_prompt_template("triage")
    system_prompt = template["system_prompt"].strip()

    conditions, medications, allergies = _format_ehr(context)

    user_content = template["user_template"].format(
        speech_text=context.speech_text,
        conditions=conditions,
        medications=medications,
        allergies=allergies,
        heart_rate=context.vitals.heart_rate,
        systolic_bp=context.vitals.systolic_bp,
        diastolic_bp=context.vitals.diastolic_bp,
        respiratory_rate=context.vitals.respiratory_rate,
        spo2=context.vitals.spo2,
        temperature=context.vitals.temperature,
        consciousness=context.vitals.consciousness.value,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content.strip()},
    ]


def build_differential_prompt(
    context: FullTriageContext, model_id: str
) -> list[dict[str, str]]:
    """Build chat messages for differential diagnosis."""
    template = load_prompt_template("differential")
    system_prompt = template["system_prompt"].strip()

    conditions, medications, allergies = _format_ehr(context)
    risk_flags = _format_risk_flags(context)

    user_content = template["user_template"].format(
        speech_text=context.speech_text,
        conditions=conditions,
        medications=medications,
        allergies=allergies,
        risk_flags=risk_flags,
        heart_rate=context.vitals.heart_rate,
        systolic_bp=context.vitals.systolic_bp,
        diastolic_bp=context.vitals.diastolic_bp,
        respiratory_rate=context.vitals.respiratory_rate,
        spo2=context.vitals.spo2,
        temperature=context.vitals.temperature,
        consciousness=context.vitals.consciousness.value,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content.strip()},
    ]


def build_management_prompt(
    context: FullTriageContext, model_id: str
) -> list[dict[str, str]]:
    """Build chat messages for management plan."""
    template = load_prompt_template("management")
    system_prompt = template["system_prompt"].strip()

    conditions, medications, allergies = _format_ehr(context)
    risk_flags = _format_risk_flags(context)

    user_content = template["user_template"].format(
        speech_text=context.speech_text,
        conditions=conditions,
        medications=medications,
        allergies=allergies,
        risk_flags=risk_flags,
        heart_rate=context.vitals.heart_rate,
        systolic_bp=context.vitals.systolic_bp,
        diastolic_bp=context.vitals.diastolic_bp,
        respiratory_rate=context.vitals.respiratory_rate,
        spo2=context.vitals.spo2,
        temperature=context.vitals.temperature,
        consciousness=context.vitals.consciousness.value,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content.strip()},
    ]


def _format_ehr(context: FullTriageContext) -> tuple[str, str, str]:
    """Extract formatted EHR strings for prompt templates."""
    if not context.ehr:
        return ("None available", "None available", "None available")

    conditions = ", ".join(
        c.display for c in context.ehr.active_conditions
    ) or "None"
    medications = ", ".join(
        m.display for m in context.ehr.active_medications
    ) or "None"
    allergies = ", ".join(
        a.substance for a in context.ehr.allergies
    ) or "NKDA"

    return conditions, medications, allergies


def _format_risk_flags(context: FullTriageContext) -> str:
    """Format EHR risk flags for prompt templates."""
    if not context.ehr or not context.ehr.risk_flags:
        return "None identified"
    return ", ".join(f.description for f in context.ehr.risk_flags)
