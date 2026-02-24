"""Queue View — High-Density Clinical Command Center.
Optimized for rapid situational awareness of the waiting room.
"""

from datetime import datetime
import streamlit as st

from src.services.auth_service import Role
from src.services.session_manager import get_queue_ordered, select_patient
from src.utils.theme import KIColors, priority_badge, retts_badge

# --- Live Auto-Refresh (New) ---
# This causes the page to rerun every 15 seconds to check for new patients in SQLite
if "last_auto_refresh" not in st.session_state:
    st.session_state.last_auto_refresh = datetime.now()

# We use a fragment to allow the UI to update without losing focus on other elements
@st.fragment(run_every="15s")
def auto_refresh_monitor():
    current_count = len(get_queue_ordered())
    if "prev_count" not in st.session_state:
        st.session_state.prev_count = current_count
        
    if current_count != st.session_state.prev_count:
        st.session_state.prev_count = current_count
        st.rerun()

    st.session_state.last_auto_refresh = datetime.now()
    st.markdown(
        f'<div style="text-align:right; font-size:0.75rem; color:#444; margin-top:-10px; font-weight:600;">'
        f'● LIVE AUTO-SYNC (Last: {st.session_state.last_auto_refresh.strftime("%H:%M:%S")})'
        f'</div>', 
        unsafe_allow_html=True
    )

# Get current role for permission checks
role = st.session_state.get("role")
is_nurse = role in (Role.TRIAGE_NURSE, Role.ADMIN)
is_physician = role in (Role.PHYSICIAN, Role.ADMIN)

# Dashboard Metrics Bar
patients = get_queue_ordered()
if not patients:
    st.info("No patients in queue.")
    
    # Allow loading demo scenarios directly from here if empty
    if not st.session_state.get("demo_loaded"):
        from src.services.session_manager import load_demo_scenarios
        if st.button("Load Mockup Patients (Demo Scenarios)", type="primary"):
            with st.spinner("Running pipeline..."):
                load_demo_scenarios()
            st.rerun()
            
    st.stop()

st.markdown('<div class="dashboard-header"><div class="dashboard-title">Emergency Department Command Center</div></div>', unsafe_allow_html=True)
auto_refresh_monitor()

# Top KPI Row (Compact)
m1, m2, m3, m4 = st.columns(4)
total = len(patients)
high = sum(1 for p in patients if p.queue_priority and (p.queue_priority.value if hasattr(p.queue_priority, "value") else p.queue_priority) == "HIGH")
triaged = sum(1 for p in patients if p.is_triaged)
avg_wait = sum(int((datetime.now() - p.arrival_time).total_seconds() / 60) for p in patients) // total

m1.metric("TOTAL QUEUE", total)
m2.metric("CRITICAL (HIGH)", high, delta_color="inverse")
m3.metric("TRIAGE COMPLETE", f"{triaged}/{total}")
m4.metric("AVG WAIT", f"{avg_wait}m")

# --- CUSTOM QUEUE TABLE (Responsive & High Density) ---
st.markdown("""
<div style="display:grid; grid-template-columns: 80px 1fr 100px 100px 100px 150px; gap:10px; padding:10px; background:#4F0433; color:white; font-size:0.75rem; font-weight:700; border-radius:8px 8px 0 0; margin-top:20px;">
    <div>PRIO</div>
    <div>PATIENT / COMPLAINT</div>
    <div>WAIT</div>
    <div>RETTS</div>
    <div>AGREEMENT</div>
    <div>ACTIONS</div>
</div>
""", unsafe_allow_html=True)

for p in patients:
    wait = int((datetime.now() - p.arrival_time).total_seconds() / 60)
    wait_color = "#B84145" if wait > 30 else "#666"
    
    # Robust enum access
    p_prio = p.queue_priority.value if hasattr(p.queue_priority, "value") else p.queue_priority
    p_retts = p.retts_level.value if hasattr(p.retts_level, "value") else p.retts_level

    # Card Class based on level
    border_color = "#E0E0E0"
    if p_retts: border_color = KIColors.retts_color(p_retts)
    elif p_prio == "HIGH": border_color = "#B84145"

    agreement_txt = "N/A"
    if p.triage_agreement:
        lvl = p.triage_agreement.agreement_level
        agreement_txt = f"{lvl.value if hasattr(lvl, 'value') else lvl} ({p.triage_agreement.agreement_ratio:.0%})"

    # Row Item
    cols = st.columns([1, 6, 1.5, 1.5, 1.5, 2.5])
    
    with cols[0]: st.markdown(priority_badge(p_prio) if p_prio else "", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div style="line-height:1.2;">
            <div style="font-weight:700; color:{KIColors.PRIMARY};">{p.name} <span style="font-weight:600; color:#333; font-size:0.8rem;">({p.age}y {p.sex})</span></div>
            <div style="font-size:0.75rem; color:#444; font-style:italic;">"{p.speech_text[:80]}..."</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]: st.markdown(f'<div style="font-weight:700; color:{wait_color};">{wait}m</div>', unsafe_allow_html=True)
    with cols[3]: st.markdown(retts_badge(p_retts) if p_retts else '<span style="color:#CCC;">Pending</span>', unsafe_allow_html=True)
    with cols[4]: st.markdown(f'<div style="font-size:0.75rem; font-weight:600;">{agreement_txt}</div>', unsafe_allow_html=True)
    
    with cols[5]:
        btn_col1, btn_col2 = st.columns(2)
        # Only show Triage button if user has access to Triage View
        if is_nurse:
            if btn_col1.button("⚕️ Triage", key=f"t_{p.patient_id}", use_container_width=True):
                select_patient(p.patient_id)
                st.switch_page("pages/2_Triage_View.py")
        
        # Only show Review button if user has access to Physician View
        if is_physician:
            if btn_col2.button("👁️ Rev", key=f"r_{p.patient_id}", use_container_width=True):
                select_patient(p.patient_id)
                st.switch_page("pages/3_Physician_View.py")
    
    st.markdown(f'<div style="height:1px; background:{border_color}33; margin:2px 0;"></div>', unsafe_allow_html=True)

# Legend Sticky

st.markdown(
    '<div class="sticky-footer">'
    '<div style="display:flex; gap:20px; font-size:0.75rem; color:#333; font-weight:600;">'
    '<span><strong>🟥 HIGH</strong>: Immediate Attention</span>'
    '<span><strong>🟨 MODERATE</strong>: 15-30m</span>'
    '<span><strong>🟩 STANDARD</strong>: 30m+</span>'
    '</div></div>',
    unsafe_allow_html=True,
)
