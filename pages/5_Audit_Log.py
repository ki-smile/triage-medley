"""Audit Log — Compliance: full decision trail."""

import streamlit as st

from src.utils.audit import get_events
from src.utils.theme import KIColors

st.markdown(
    f'<h1 style="color:{KIColors.PRIMARY};">Audit Log</h1>',
    unsafe_allow_html=True,
)
st.caption("Compliance: immutable decision audit trail")

# ---- Filters ----
filter_cols = st.columns(4)

with filter_cols[0]:
    actor_filter = st.text_input("Filter by actor", placeholder="e.g., nurse, system")
with filter_cols[1]:
    action_filter = st.text_input("Filter by action", placeholder="e.g., override, triage_result")
with filter_cols[2]:
    patient_filter = st.text_input("Filter by patient ID", placeholder="e.g., anders")
with filter_cols[3]:
    stage_filter = st.selectbox(
        "Filter by stage",
        options=["All", "pretriage", "triage", "differential", "management"],
    )

# Build filter kwargs
kwargs = {}
if actor_filter:
    kwargs["actor"] = actor_filter
if action_filter:
    kwargs["action"] = action_filter
if patient_filter:
    kwargs["patient_id"] = patient_filter
if stage_filter != "All":
    kwargs["stage"] = stage_filter

events = get_events(**kwargs)

# ---- Metrics ----
st.markdown(f"**{len(events)} events** matching filters")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

if not events:
    st.info("No audit events found. Load demo scenarios to generate events.")
    st.stop()

# ---- Events Table ----
# Show most recent first
for event in reversed(events):
    # Color-code by actor type
    if event.actor.startswith("model:"):
        actor_color = KIColors.TERTIARY
    elif event.actor == "nurse":
        actor_color = KIColors.SECONDARY
    elif event.actor == "physician":
        actor_color = KIColors.PRIMARY
    else:
        actor_color = KIColors.ON_SURFACE_VARIANT

    # Action badge color
    action_color = KIColors.ON_SURFACE
    if event.action == "override":
        action_color = KIColors.RETTS_ORANGE
    elif "error" in event.action:
        action_color = KIColors.ERROR
    elif "result" in event.action:
        action_color = KIColors.RETTS_GREEN

    timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]

    # Payload summary
    payload_str = ""
    if event.payload:
        items = []
        for k, v in event.payload.items():
            val_str = str(v)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            items.append(f"<strong>{k}:</strong> {val_str}")
        payload_str = " &middot; ".join(items)

    st.markdown(
        f'<div class="m3-card" style="padding:0.75rem 1rem; margin-bottom:0.5rem;">'
        f'<div style="display:flex; align-items:center; gap:1rem;">'
        f'<span style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.8rem; min-width:80px;">{timestamp}</span>'
        f'<span style="color:{actor_color}; font-weight:600; min-width:140px;">{event.actor}</span>'
        f'<span style="color:{action_color}; min-width:120px;">{event.action}</span>'
        f'<span style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.85rem;">'
        f'{event.patient_id or ""}</span>'
        f'<span style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.85rem;">'
        f'{event.stage or ""}</span>'
        f'</div>'
        f'{"<div style=" + chr(34) + "margin-top:4px; font-size:0.8rem; color:" + KIColors.ON_SURFACE_VARIANT + ";" + chr(34) + ">" + payload_str + "</div>" if payload_str else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ---- Export ----
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
with st.expander("Export"):
    import json
    export_data = [e.model_dump(mode="json") for e in events]
    st.download_button(
        "Download as JSON",
        data=json.dumps(export_data, indent=2, default=str),
        file_name="audit_log.json",
        mime="application/json",
    )
