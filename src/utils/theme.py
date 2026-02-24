"""M3 Expressive theme utilities for Streamlit with KI colour palette."""

from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


class KIColors:
    """Karolinska Institutet colour palette mapped to M3 Expressive tokens."""

    PRIMARY = "#4F0433"
    ON_PRIMARY = "#FFFFFF"
    SECONDARY = "#870052"
    TERTIARY = "#4DB5BC"
    SURFACE = "#EDF4F4"
    SURFACE_VARIANT = "#FEEEEB"
    BACKGROUND = "#F1F1F1"
    ON_SURFACE = "#000000"
    ON_SURFACE_VARIANT = "#666666"
    ERROR = "#B84145"
    OUTLINE = "#E0E0E0"

    # RETTS Triage Levels
    RETTS_RED = "#B84145"
    RETTS_ORANGE = "#FF876F"
    RETTS_YELLOW = "#F59A00"
    RETTS_GREEN = "#54B986"
    RETTS_BLUE = "#4DB5BC"

    @classmethod
    def retts_color(cls, level: str) -> str:
        return {
            "RED": cls.RETTS_RED,
            "ORANGE": cls.RETTS_ORANGE,
            "YELLOW": cls.RETTS_YELLOW,
            "GREEN": cls.RETTS_GREEN,
            "BLUE": cls.RETTS_BLUE,
        }.get(level.upper(), cls.ON_SURFACE_VARIANT)

    @classmethod
    def retts_text_color(cls, level: str) -> str:
        return "#FFFFFF" if level.upper() in ("RED",) else "#000000"

    @classmethod
    def priority_color(cls, priority: str) -> str:
        return {
            "HIGH": cls.RETTS_RED,
            "MODERATE": cls.RETTS_YELLOW,
            "STANDARD": cls.RETTS_GREEN,
        }.get(priority.upper(), cls.ON_SURFACE_VARIANT)


_COPYRIGHT = (
    "\u00a9 2026 SMAILE (Stockholm Medical Artificial Intelligence "
    "and Learning Environments), Karolinska Institutet."
)


def inject_custom_css() -> None:
    """Inject M3 Expressive custom CSS into the Streamlit app with dynamic font scaling."""
    css_path = _ASSETS_DIR / "style.css"
    base_css = ""
    if css_path.exists():
        base_css = css_path.read_text(encoding="utf-8")

    # Dynamic Font Size Logic
    size_map = {
        "Small": "14px",
        "Medium": "16px",
        "Large": "18px",
        "X-Large": "20px",
    }
    current_size = st.session_state.get("font_size_selector", "Medium")
    root_font_size = size_map.get(current_size, "16px")

    dynamic_css = f"""
    <style>
    :root {{
        font-size: {root_font_size} !important;
    }}
    /* Apply to standard elements but avoid breaking layout-critical containers */
    body, p, div, span, label, button, input {{
        font-size: inherit;
    }}
    </style>
    """
    
    st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)
    st.markdown(dynamic_css, unsafe_allow_html=True)


def render_footer() -> None:
    """Render the SMAILE copyright footer at the bottom of any page."""
    st.markdown(
        f'<div style="text-align:center; padding:2rem 0 1rem; '
        f'color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.8rem; '
        f'border-top:1px solid {KIColors.OUTLINE}; margin-top:2rem;">'
        f'{_COPYRIGHT}</div>',
        unsafe_allow_html=True,
    )


def retts_badge(level: str) -> str:
    """Return an HTML span styled as a RETTS colour badge."""
    if not level:
        return ""
    css_class = f"retts-{level.lower()}"
    return f'<span class="retts-badge {css_class}">{level}</span>'


def priority_badge(priority: str) -> str:
    """Return an HTML span styled as a queue priority badge."""
    if not priority:
        return ""
    css_class = f"priority-{priority.lower()}"
    return f'<span class="priority-badge {css_class}">{priority}</span>'


def m3_card(content: str, elevated: bool = False) -> str:
    """Wrap content in an M3 card div."""
    css_class = "m3-card-elevated" if elevated else "m3-card"
    return f'<div class="{css_class}">{content}</div>'


# ---- Mobile-First Clinical UI Components ----

_RETTS_ORDER = ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE"]


def retts_banner(
    level: str,
    patient_info: str = "",
    agreement: str = "",
    vote_distribution: dict[str, int] | None = None,
    pre_triage: str = "",
) -> str:
    """Generate a full-width RETTS colour banner with embedded metadata.

    Parameters
    ----------
    level : str
        RETTS level (RED/ORANGE/YELLOW/GREEN/BLUE).
    patient_info : str
        Patient name + demographics string.
    agreement : str
        Agreement level text (e.g. "FULL 100%").
    vote_distribution : dict
        Mapping of RETTS level -> vote count (for embedded compact vote bar).
    pre_triage : str
        Pre-triage queue priority (HIGH/MODERATE/STANDARD) or empty.
    """
    bg = KIColors.retts_color(level)
    fg = KIColors.retts_text_color(level)

    badges_parts: list[str] = []
    if agreement:
        badges_parts.append(
            f'<span class="retts-badge" style="background:rgba(255,255,255,0.8);'
            f'color:#000; font-size:0.8rem; font-weight:800; border:1px solid rgba(0,0,0,0.1);">{agreement}</span>'
        )
    if pre_triage:
        badges_parts.append(priority_badge(pre_triage))

    badges_html = "".join(badges_parts)

    # Compact vote bar inside banner (mobile)
    vote_html = ""
    if vote_distribution:
        total = sum(vote_distribution.values())
        if total > 0:
            vote_html = (
                f'<div class="mobile-only" style="width:100%;margin-top:0.25rem;">'
                f'{vote_distribution_bar(vote_distribution, total, compact=True)}'
                f'</div>'
            )

    return (
        f'<div class="retts-banner" style="background-color:{bg};color:{fg};">'
        f'<span class="retts-banner-level">RETTS: {level.upper()}</span>'
        f'<span class="retts-banner-patient">{patient_info}</span>'
        f'<div class="retts-banner-badges">{badges_html}</div>'
        f'{vote_html}'
        f'</div>'
    )


def vote_distribution_bar(
    vote_distribution: dict[str, int],
    total_voters: int,
    compact: bool = True,
) -> str:
    """Generate a segmented horizontal vote distribution bar.

    Parameters
    ----------
    vote_distribution : dict
        Mapping of RETTS level -> vote count.
    total_voters : int
        Total number of voters.
    compact : bool
        True for compact segmented bar (mobile/tablet),
        False for labelled bar chart (desktop right column).
    """
    if total_voters <= 0:
        return ""

    sorted_levels = sorted(
        vote_distribution.items(),
        key=lambda x: _RETTS_ORDER.index(x[0]) if x[0] in _RETTS_ORDER else 99,
    )

    if compact:
        segments = []
        for level_str, count in sorted_levels:
            if count <= 0:
                continue
            pct = count / total_voters * 100
            bg = KIColors.retts_color(level_str)
            fg = KIColors.retts_text_color(level_str)
            segments.append(
                f'<div class="vote-bar-segment" '
                f'style="flex-grow:{pct}; background:{bg}; color:{fg}; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.75rem; overflow:hidden; white-space:nowrap;">'
                f'{count}/{total_voters}</div>'
            )
        return f'<div class="vote-bar" style="display:flex; width:100%; height:24px; border-radius:12px; overflow:hidden; margin:5px 0;">{"".join(segments)}</div>'

    # Desktop: labelled bar rows
    rows = []
    for level_str, count in sorted_levels:
        if count <= 0:
            continue
        pct = count / total_voters * 100
        bg = KIColors.retts_color(level_str)
        rows.append(
            f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px; width:100%;">'
            f'<div style="width:70px; flex-shrink:0;">{retts_badge(level_str)}</div>'
            f'<div style="flex-grow:1; background:#F0F0F0; border-radius:6px; height:20px; overflow:hidden;">'
            f'<div style="width:{pct}%; background:{bg}; height:100%; transition:width 0.5s ease;"></div>'
            f'</div>'
            f'<div style="width:40px; flex-shrink:0; text-align:right; font-weight:700; font-size:0.9rem; color:#444;">{count}</div>'
            f'</div>'
        )
    return f'<div style="padding:5px 0;">{"".join(rows)}</div>'


def dont_miss_card(alerts: list[str]) -> str:
    """Generate a prominent don't-miss alert card.

    Parameters
    ----------
    alerts : list[str]
        List of don't-miss diagnosis/alert strings.
    """
    if not alerts:
        return ""
    items = "".join(
        f'<div class="dont-miss-card-item">{a}</div>' for a in alerts
    )
    return (
        f'<div class="dont-miss-card">'
        f'<div class="dont-miss-card-title">'
        f"\u26a0\ufe0f Don't-Miss Diagnoses</div>"
        f'{items}</div>'
    )


def model_assessment_card(
    model_id: str,
    retts_level: str,
    confidence: str = "",
    reasoning: str = "",
    dont_miss: list[str] | None = None,
    is_dissenter: bool = False,
    is_engine: bool = False,
    native_detail: str = "",
) -> str:
    """Generate a colour-coded model result card.

    Parameters
    ----------
    model_id : str
        Model or engine identifier.
    retts_level : str
        RETTS level this model assigned.
    confidence : str
        Confidence level string (e.g. "high", "moderate").
    reasoning : str
        Clinical reasoning text.
    dont_miss : list[str] | None
        Don't-miss diagnoses flagged by this model.
    is_dissenter : bool
        Whether this model dissents from consensus.
    is_engine : bool
        Whether this is a deterministic rules engine.
    native_detail : str
        Additional native triage system detail string.
    """
    border_color = KIColors.RETTS_RED if is_dissenter else KIColors.retts_color(retts_level)

    dissent_tag = (
        ' <span class="model-card-dont-miss" '
        'style="background:rgba(184,65,69,0.15); color:var(--retts-red);">DISSENT</span>'
        if is_dissenter else ""
    )

    engine_tag = (
        ' <span style="color:var(--ki-on-surface-variant);font-size:0.8rem;">'
        '(deterministic)</span>'
        if is_engine else ""
    )

    confidence_html = (
        f' <span style="color:var(--ki-on-surface-variant);font-size:0.8rem;">'
        f'({confidence})</span>'
        if confidence else ""
    )

    native_html = (
        f'<div style="color:var(--ki-on-surface-variant);font-size:0.8rem;'
        f'margin-top:0.15rem;">{native_detail}</div>'
        if native_detail else ""
    )

    dm_html = ""
    if dont_miss:
        items = ", ".join(dont_miss)
        dm_html = (
            f' <span class="model-card-dont-miss">'
            f"Don't-miss: {items}</span>"
        )

    reasoning_html = (
        f'<div class="model-card-reasoning">{reasoning}</div>'
        if reasoning else ""
    )

    return (
        f'<div class="model-card" style="border-left:4px solid {border_color};">'
        f'<div class="model-card-header">'
        f'<span class="model-card-name">{model_id}</span>'
        f'{engine_tag}{dissent_tag}'
        f' {retts_badge(retts_level)}'
        f'{confidence_html}'
        f'{dm_html}'
        f'</div>'
        f'{native_html}'
        f'{reasoning_html}'
        f'</div>'
    )


# ---- Physician View: Differential Diagnosis Card ----

_TIER_BORDER = {
    "consensus": KIColors.RETTS_GREEN,
    "partial": KIColors.RETTS_YELLOW,
    "minority": KIColors.RETTS_ORANGE,
}


def _model_count_color(ratio: float) -> str:
    """Return badge colour based on model agreement ratio."""
    if ratio >= 0.8:
        return KIColors.RETTS_GREEN
    if ratio >= 0.4:
        return KIColors.RETTS_YELLOW
    return KIColors.RETTS_ORANGE


def _prob_bar_color(probability: float) -> str:
    """Return bar fill colour based on probability value."""
    if probability > 0.6:
        return KIColors.RETTS_RED
    if probability >= 0.3:
        return KIColors.RETTS_YELLOW
    return KIColors.RETTS_GREEN


def diagnosis_card(
    diagnosis: str,
    probability: float | None,
    supporting_evidence: list[str],
    is_dont_miss: bool,
    model_count: int,
    total_models: int,
    tier: str,
) -> str:
    """Generate a rich differential diagnosis card with probability bar and model count.

    Parameters
    ----------
    diagnosis : str
        Diagnosis name.
    probability : float | None
        Estimated probability (0.0–1.0).
    supporting_evidence : list[str]
        Evidence bullets (top 3 shown directly).
    is_dont_miss : bool
        Whether this is a don't-miss diagnosis.
    model_count : int
        How many models flagged this diagnosis.
    total_models : int
        Total models in ensemble.
    tier : str
        "consensus" | "partial" | "minority"
    """
    border_color = _TIER_BORDER.get(tier, KIColors.ON_SURFACE_VARIANT)

    # Don't-miss badge
    dm_html = ' <span class="dx-dont-miss">DON\'T MISS</span>' if is_dont_miss else ""

    # Model count badge
    ratio = model_count / total_models if total_models > 0 else 0
    mc_color = _model_count_color(ratio)
    mc_html = (
        f'<span class="dx-model-count" style="color:{mc_color};">'
        f'{model_count}/{total_models} models</span>'
    )

    # Probability bar
    prob_html = ""
    if probability is not None:
        pct = probability * 100
        bar_color = _prob_bar_color(probability)
        prob_html = (
            f'<div class="dx-prob-text">{pct:.0f}%</div>'
            f'<div class="dx-prob-bar">'
            f'<div class="dx-prob-fill" style="width:{pct}%;background:{bar_color};"></div>'
            f'</div>'
        )

    # Evidence bullets (top 3, no expander)
    ev_html = ""
    if supporting_evidence:
        items = "".join(f"<li>{e}</li>" for e in supporting_evidence[:3])
        ev_html = f'<ul class="dx-evidence">{items}</ul>'

    return (
        f'<div class="dx-card" style="border-left:4px solid {border_color};">'
        f'<div class="dx-card-header">'
        f'<span class="dx-card-name">{diagnosis}</span>{dm_html}{mc_html}'
        f'</div>'
        f'{prob_html}'
        f'{ev_html}'
        f'</div>'
    )


# ---- Physician View: Management Item with Consensus Dots ----


def consensus_dots(model_count: int, total_models: int) -> str:
    """Generate filled/empty circle dots showing model agreement.

    Returns HTML like: ●●●○○ 3/5
    """
    filled = '<span class="consensus-dot-filled">●</span>' * model_count
    empty = '<span class="consensus-dot-empty">○</span>' * (total_models - model_count)
    return (
        f'<span class="consensus-dots">{filled}{empty}'
        f' <span style="font-size:0.7rem;">{model_count}/{total_models}</span></span>'
    )


def management_item(
    item: str,
    model_count: int,
    total_models: int,
    is_contraindicated: bool = False,
) -> str:
    """Generate HTML for a consensus-aware management plan item.

    Parameters
    ----------
    item : str
        The investigation/imaging/medication text.
    model_count : int
        How many models recommended this item.
    total_models : int
        Total models in ensemble.
    is_contraindicated : bool
        If True, render with red warning background.
    """
    ci_class = " mgmt-item-contraindicated" if is_contraindicated else ""
    dots = consensus_dots(model_count, total_models)
    return (
        f'<div class="mgmt-item{ci_class}">'
        f'<span>{item}</span>'
        f'<span style="margin-left:auto;">{dots}</span>'
        f'</div>'
    )


def patient_context_strip(
    vitals_text: str,
    conditions_text: str = "",
    medications_text: str = "",
) -> str:
    """Generate a compact patient context strip with vitals and EHR summary.

    Parameters
    ----------
    vitals_text : str
        Formatted vitals string (e.g. "HR 78 · BP 118/68 · SpO2 96%").
    conditions_text : str
        Comma-separated active conditions.
    medications_text : str
        Comma-separated active medications.
    """
    parts = [vitals_text]
    if conditions_text:
        parts.append(f"Conditions: {conditions_text}")
    if medications_text:
        parts.append(f"Meds: {medications_text}")
    sep = '<span class="ctx-sep">|</span>'
    inner = sep.join(parts)
    return f'<div class="patient-context-strip">{inner}</div>'
