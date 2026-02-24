"""Triage View — Advanced Responsive Clinical Dashboard.
Optimized for one-glance situational awareness and minimal scrolling.
"""

from datetime import datetime
import streamlit as st

from src.engines.agreement_engine import (
    analyze_differential,
    analyze_engine_disagreement,
    analyze_management,
    analyze_triage,
)
from src.models.context import FullTriageContext
from src.models.enums import ConsciousnessLevel
from src.models.vitals import VitalSigns
from src.services.orchestrator import run_full_pipeline
from src.services.session_manager import (
    get_patients,
    get_selected_patient,
    select_patient,
)
from src.services.pdf_service import generate_triage_pdf
from src.utils.audit import log_event
from src.utils.theme import KIColors, retts_badge, priority_badge, vote_distribution_bar

# Set page to wide mode
# st.set_page_config(layout="wide")

patients = get_patients()
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

# Header / Selection
st.markdown('<div class="dashboard-header"><div class="dashboard-title">Clinical Decision Support System</div></div>', unsafe_allow_html=True)

hcol1, hcol2 = st.columns([1, 2])
with hcol1:
    patient_options = {pid: f"{p.name} ({p.age}y {p.sex})" for pid, p in patients.items()}
    selected_pid = st.selectbox(
        "Current Patient",
        options=list(patient_options.keys()),
        format_func=lambda x: patient_options[x],
        index=list(patient_options.keys()).index(st.session_state.selected_patient)
        if st.session_state.get("selected_patient") in patient_options
        else 0,
        label_visibility="collapsed"
    )
    select_patient(selected_pid)

patient = get_selected_patient()
if not patient:
    st.warning("Patient not found.")
    st.stop()

ta = patient.triage_agreement
engine_disagree = None
if patient.triage_ensemble and len(patient.triage_ensemble.engine_outputs) >= 2:
    engine_disagree = analyze_engine_disagreement(patient.triage_ensemble.engine_outputs)

# ===================================================================
# ADVANCED RESPONSIVE GRID
# ===================================================================

# Main Content Area
c1, c2, c3 = st.columns([3, 5, 4]) # Adaptive layout ratios

# -------------------------------------------------------------------
# COLUMN 1: PATIENT CONTEXT (Static info)
# -------------------------------------------------------------------
with c1:
    # Patient Summary Card
    st.markdown(f"""
    <div class="ui-card">
        <div class="ui-card-header">🪪 Patient Identity</div>
        <div style="display:flex; align-items:center; gap:15px; margin-bottom:10px;">
            <div style="font-size:2.5rem; background:#F0F0F0; border-radius:50%; width:60px; height:60px; display:flex; align-items:center; justify-content:center;">👤</div>
            <div>
                <div style="font-size:1.2rem; font-weight:700; color:{KIColors.PRIMARY};">{patient.name}</div>
                <div style="font-size:0.85rem; color:#333;">{patient.age}y {patient.sex} &middot; ID: <strong>{patient.patient_id[:8]}...</strong></div>
            </div>
        </div>
        <div style="display:flex; gap:5px;">
            {priority_badge(patient.queue_priority.value) if patient.queue_priority else ""}
            {retts_badge(patient.retts_level.value) if patient.retts_level else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # EHR Quick Look
    ehr = patient.full_context.ehr if patient.full_context else (patient.pretriage_context.ehr if patient.pretriage_context else None)
    if ehr:
        st.markdown('<div class="ui-card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-header">🔎 EHR Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
        st.markdown("**Active Conditions**")
        conds = "".join(f"<li>{c.display}</li>" for c in ehr.active_conditions[:4]) or "<li>None</li>"
        st.markdown(f'<ul class="compact-list">{conds}</ul>', unsafe_allow_html=True)
        
        st.markdown("**Current Medications**")
        meds = "".join(f"<li>{m.display}</li>" for m in ehr.active_medications[:4]) or "<li>None</li>"
        st.markdown(f'<ul class="compact-list">{meds}</ul>', unsafe_allow_html=True)
        
        if ehr.risk_flags:
            st.markdown("**Risk Flags**")
            for rf in ehr.risk_flags:
                st.markdown(f'<div class="badge-ui" style="background:rgba(184,65,69,0.1); color:var(--ki-error); margin-bottom:4px;">⚠️ {rf.description}</div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# COLUMN 2: VITAL SIGNS & PRIMARY WORKFLOW
# -------------------------------------------------------------------
with c2:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown('<div class="ui-card-header">⚡ Clinical Vitals & Triage</div>', unsafe_allow_html=True)
    
    _has_vitals = patient.full_context is not None
    _v = patient.full_context.vitals if _has_vitals else None

    # Vitals Data Grid (Read Only / Visual)
    if _v:
        v1, v2, v3, v4 = st.columns(4)
        v1.markdown(f'<div class="vital-box {"critical" if _v.heart_rate > 100 or _v.heart_rate < 50 else "normal"}"><div class="vital-val">{_v.heart_rate}</div><div class="vital-label">HR</div></div>', unsafe_allow_html=True)
        v2.markdown(f'<div class="vital-box {"critical" if _v.systolic_bp > 160 or _v.systolic_bp < 90 else "normal"}"><div class="vital-val">{_v.systolic_bp}/{_v.diastolic_bp}</div><div class="vital-label">BP</div></div>', unsafe_allow_html=True)
        v3.markdown(f'<div class="vital-box {"warning" if _v.spo2 < 95 else "normal"}"><div class="vital-val">{_v.spo2}%</div><div class="vital-label">SpO2</div></div>', unsafe_allow_html=True)
        v4.markdown(f'<div class="vital-box {"warning" if _v.temperature > 38.0 else "normal"}"><div class="vital-val">{_v.temperature}°</div><div class="vital-label">Temp</div></div>', unsafe_allow_html=True)

    # Input Form (Compact)
    with st.form(key="vitals_form_dashboard", border=False):
        f1, f2, f3 = st.columns(3)
        inp_hr = f1.number_input("HR", 0, 300, value=_v.heart_rate if _v else 80, step=1)
        inp_sbp = f2.number_input("SBP", 0, 300, value=_v.systolic_bp if _v else 120, step=1)
        inp_rr = f3.number_input("RR", 0, 80, value=_v.respiratory_rate if _v else 18, step=1)
        
        f4, f5, f6 = st.columns(3)
        inp_spo2 = f4.number_input("SpO2", 0, 100, value=_v.spo2 if _v else 98, step=1)
        inp_temp = f5.number_input("Temp", 30.0, 45.0, value=_v.temperature if _v else 37.0, step=0.1)
        inp_ess = f6.selectbox("ESS", options=["chest_pain", "respiratory", "abdominal", "neurological", "trauma_fall"], index=0)
        
        if st.form_submit_button("RE-TRIAGE PATIENT", type="primary", use_container_width=True):
            new_vitals = VitalSigns(heart_rate=inp_hr, systolic_bp=inp_sbp, diastolic_bp=90, respiratory_rate=inp_rr, spo2=inp_spo2, temperature=inp_temp, consciousness=ConsciousnessLevel.ALERT)
            ctx = FullTriageContext(patient_id=patient.patient_id, speech_text=patient.speech_text, ehr=ehr, asr_disagreements=[], vitals=new_vitals, ess_category=inp_ess)
            patient.full_context = ctx
            with st.spinner("Processing..."):
                t, d, m = run_full_pipeline(ctx)
                patient.triage_ensemble, patient.triage_agreement = t, analyze_triage(t)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Engine Evidence (Below Form)
    if ta and patient.triage_ensemble:
        st.markdown('<div class="ui-card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-header">🦴 Logic Engine Evidence</div>', unsafe_allow_html=True)
        for engine_out in patient.triage_ensemble.engine_outputs:
            # SAFETY: Skip if engine_out is a string/corrupted
            if not hasattr(engine_out, "retts_level"):
                continue
                
            st.markdown(f"""
            <div style="border-left:3px solid {KIColors.retts_color(engine_out.retts_level.value)}; padding-left:10px; margin-bottom:10px;">
                <div style="font-size:0.8rem; font-weight:700;">{engine_out.triage_system.upper()} &rarr; {retts_badge(engine_out.retts_level.value)}</div>
                <div style="font-size:0.8rem; color:#444;">{engine_out.clinical_reasoning}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# COLUMN 3: AI INSIGHTS & OVERRIDE
# -------------------------------------------------------------------
with c3:
    # Senior Review / Alerts
    if ta and ta.requires_senior_review:
        st.markdown('<div class="safety-alert">⚠️ DISAGREEMENT DETECTED: Senior Review Recommended</div>', unsafe_allow_html=True)

    # Vote Distribution
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown('<div class="ui-card-header">📊 Ensemble Consensus</div>', unsafe_allow_html=True)
    if ta:
        st.markdown(vote_distribution_bar(ta.vote_distribution, ta.total_voters, compact=False), unsafe_allow_html=True)
        
        # Actionable insights (Don't miss)
        if ta.dont_miss_alerts:
            st.markdown('<div style="margin-top:15px; background:rgba(184,65,69,0.05); border:1px solid rgba(184,65,69,0.2); padding:10px; border-radius:8px;">', unsafe_allow_html=True)
            st.markdown('<div style="font-weight:700; color:var(--ki-error); font-size:0.8rem;">DON\'T MISS DIAGNOSES:</div>', unsafe_allow_html=True)
            for alert in ta.dont_miss_alerts:
                st.markdown(f'<div style="font-size:0.85rem; padding:2px 0;">• {alert}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Nurse Override Area
        st.markdown('<div style="margin-top:20px; border-top:1px solid #EEE; padding-top:10px;">', unsafe_allow_html=True)
        st.markdown('<div class="ui-card-header">Decision Override</div>', unsafe_allow_html=True)
        over_col1, over_col2 = st.columns(2)
        
        # Robustly get current level value
        current_lvl = ta.final_level.value if hasattr(ta.final_level, "value") else ta.final_level
        level_options = ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE"]
        try:
            default_idx = level_options.index(current_lvl)
        except (ValueError, KeyError):
            default_idx = 2 # YELLOW fallback
            
        override_level = over_col1.selectbox("Final Level", options=level_options, index=default_idx)
        if over_col2.button("COMMIT", type="primary", use_container_width=True, help="Submit final decision and handoff"):
            st.success("Decision Signed")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Awaiting pipeline results...")
    st.markdown('</div>', unsafe_allow_html=True)

# Final Handoff Sticky
st.markdown("""
<div class="sticky-footer-placeholder" style="height:80px;"></div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="sticky-footer">', unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        st.markdown(f"""
            <div style="display:flex; align-items:center; gap:15px; height:100%;">
                <span style="font-size:0.8rem; color:#666;">Time since arrival: <strong>24m</strong></span>
                <span class="badge-ui" style="background:#EEE; color:#333;">Audit Logged</span>
            </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        if patient.triage_agreement:
            try:
                pdf_bytes = generate_triage_pdf(patient)
                st.download_button(
                    "📄 Download Triage Report",
                    data=pdf_bytes,
                    file_name=f"triage_{patient.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF Error: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)
