"""Physician View — Advanced Dashboard.
Optimized for high-density clinical reasoning and rapid sign-off.
"""

from datetime import datetime
import streamlit as st

from src.services.orchestrator import DifferentialEnsembleResult, ManagementEnsembleResult
from src.services.pdf_service import generate_physician_pdf
from src.services.session_manager import (
    get_patients,
    get_selected_patient,
    select_patient,
)
from src.utils.audit import log_event
from src.utils.theme import (
    KIColors,
    consensus_dots,
    diagnosis_card,
    patient_context_strip,
    retts_badge,
    retts_banner,
    vote_distribution_bar,
)

# Helpers for model counting
def _get_items_consensus(items: list[str], ensemble, field: str) -> list[tuple[str, int]]:
    counts = {}
    for item in items:
        counts[item] = 0
    if ensemble:
        for out in ensemble.model_outputs:
            for i in getattr(out, field, []):
                counts[i] = counts.get(i, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)

patients = get_patients()
if not patients:
    st.info("No patients available.")
    
    # Allow loading demo scenarios directly from here if empty
    if not st.session_state.get("demo_loaded"):
        from src.services.session_manager import load_demo_scenarios
        if st.button("Load Mockup Patients (Demo Scenarios)", type="primary"):
            with st.spinner("Running pipeline..."):
                load_demo_scenarios()
            st.rerun()
            
    st.stop()

# Header
st.markdown('<div class="dashboard-header"><div class="dashboard-title">Physician Decision Console</div></div>', unsafe_allow_html=True)

hcol1, hcol2 = st.columns([1, 2])
with hcol1:
    patient_options = {pid: f"{p.name} ({p.age}y {p.sex})" for pid, p in patients.items()}
    selected_pid = st.selectbox("Patient", options=list(patient_options.keys()), format_func=lambda x: patient_options[x], 
                                index=list(patient_options.keys()).index(st.session_state.selected_patient) if st.session_state.get("selected_patient") in patient_options else 0,
                                label_visibility="collapsed")
    select_patient(selected_pid)

patient = get_selected_patient()
if not patient:
    st.stop()

# Context Bar (Sticky-like top row)
if patient.full_context:
    v = patient.full_context.vitals
    v_text = f"HR {v.heart_rate} | BP {v.systolic_bp}/{v.diastolic_bp} | SpO2 {v.spo2}% | T {v.temperature}°"
    
    ehr = patient.full_context.ehr
    cond_text = ", ".join([c.display for c in ehr.active_conditions]) if ehr else "None"
    med_text = ", ".join([m.display for m in ehr.active_medications]) if ehr else "None"
    
    st.markdown(patient_context_strip(v_text, cond_text, med_text), unsafe_allow_html=True)

# ===================================================================
# DUAL-COLUMN CLINICAL WORKSPACE
# ===================================================================
dx_col, mgmt_col = st.columns([1, 1])

# --- LEFT: DIFFERENTIAL ---
with dx_col:
    with st.container(border=True):
        st.markdown('<div class="ui-card-header">🩺 Differential Diagnosis Analysis</div>', unsafe_allow_html=True)
        
        da = patient.differential_agreement
        de = patient.differential_ensemble
        
        if da:
            with st.container(height=500, border=False):
                # Tiers
                has_consensus = False
                for tier_name, candidates, tier_key in [("Consensus (≥80%)", da.all_agree, "consensus"), 
                                                       ("Alternatives (40-79%)", da.some_agree, "partial")]:
                    if candidates:
                        has_consensus = True
                        st.markdown(f'<div style="font-size:0.75rem; font-weight:700; color:#888; margin:10px 0 5px;">{tier_name.upper()}</div>', unsafe_allow_html=True)
                        for c in candidates:
                            mc = sum(1 for out in de.model_outputs if any(cx.diagnosis.lower() == c.diagnosis.lower() for cx in out.candidates)) if de else 0
                            st.markdown(diagnosis_card(c.diagnosis, c.probability, c.supporting_evidence, c.is_dont_miss, mc, len(de.model_outputs) if de else 1, tier_key), unsafe_allow_html=True)
                
                if not has_consensus:
                    st.warning("No clear consensus reached between models.")

                if da.devil_advocate_only:
                    with st.expander("Minority Opinions / Devil's Advocate", expanded=not has_consensus):
                        for c in da.devil_advocate_only:
                            mc = sum(1 for out in de.model_outputs if any(cx.diagnosis.lower() == c.diagnosis.lower() for cx in out.candidates)) if de else 0
                            st.markdown(diagnosis_card(c.diagnosis, c.probability, c.supporting_evidence, c.is_dont_miss, mc, len(de.model_outputs) if de else 1, "minority"), unsafe_allow_html=True)
        else:
            st.info("Generating analysis...")

# --- RIGHT: MANAGEMENT ---
with mgmt_col:
    with st.container(border=True):
        st.markdown('<div class="ui-card-header">⚕️ Evidence-Based Management Plan</div>', unsafe_allow_html=True)
        
        ma = patient.management_agreement
        me = patient.management_ensemble
        
        if ma:
            # Action area at the TOP
            with st.form("mgmt_signoff", border=False):
                sc1, sc2 = st.columns([1, 1.5])
                with sc1:
                    disp_opt = ["discharge", "observation", "admission", "icu"]
                    st.selectbox("Disposition", options=disp_opt, index=disp_opt.index(ma.consensus_disposition) if ma.consensus_disposition in disp_opt else 1)
                with sc2:
                    st.form_submit_button("SIGN OFF & COMMIT PLAN", type="primary", use_container_width=True)
                
                st.text_area("Physician Notes", placeholder="Clinical justification...", height=65, label_visibility="collapsed")
                
                # Use a scrollable container for the items
                with st.container(height=350, border=False):
                    if ma.contraindications:
                        st.markdown('<div style="background:rgba(184,65,69,0.08); border:1px solid var(--retts-red); padding:10px; border-radius:8px; margin-bottom:15px;">', unsafe_allow_html=True)
                        st.markdown('<div style="color:var(--retts-red); font-weight:700; font-size:0.8rem;">⚠️ CONTRAINDICATIONS ALERT</div>', unsafe_allow_html=True)
                        for ci in ma.contraindications:
                            st.markdown(f'<div style="font-size:0.85rem;">• {ci}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Combined grid for items
                    has_items = False
                    for label, items, field in [("Investigations", ma.common_investigations, "investigations"),
                                               ("Imaging", ma.common_imaging, "imaging"),
                                               ("Medications", ma.common_medications, "medications")]:
                        if items:
                            has_items = True
                            st.markdown(f'<div style="font-size:0.75rem; font-weight:700; color:#888; margin:10px 0 5px;">{label.upper()}</div>', unsafe_allow_html=True)
                            item_counts = _get_items_consensus(items, me, field)
                            for item, count in item_counts:
                                ic1, ic2 = st.columns([0.8, 0.2])
                                with ic1:
                                    st.checkbox(item, value=True, key=f"p_{field}_{item}")
                                with ic2:
                                    st.markdown(consensus_dots(count, len(me.model_outputs) if me else 1), unsafe_allow_html=True)
                    
                    if not has_items:
                        st.info("No common management items reached consensus (≥50%).")
                    
                    # Minority Suggestions
                    minority_labels = [("Minority Investigations", ma.minority_investigations, "investigations"),
                                       ("Minority Imaging", ma.minority_imaging, "imaging"),
                                       ("Minority Medications", ma.minority_medications, "medications")]
                    
                    if any(m[1] for m in minority_labels):
                        with st.expander("Minority Suggestions / Dissenting Opinions", expanded=not has_items):
                            for label, items, field in minority_labels:
                                if items:
                                    st.markdown(f'<div style="font-size:0.7rem; font-weight:700; color:#AAA; margin:5px 0 2px;">{label.upper()}</div>', unsafe_allow_html=True)
                                    item_counts = _get_items_consensus(items, me, field)
                                    for item, count in item_counts:
                                        ic1, ic2 = st.columns([0.8, 0.2])
                                        with ic1:
                                            st.checkbox(item, value=False, key=f"p_min_{field}_{item}")
                                        with ic2:
                                            st.markdown(consensus_dots(count, len(me.model_outputs) if me else 1), unsafe_allow_html=True)
        else:
            st.info("Generating management consensus...")

# Final Handoff Sticky
st.markdown("""
<div class="sticky-footer-placeholder" style="height:80px;"></div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="sticky-footer">', unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        status_color = KIColors.RETTS_GREEN if patient.is_signed_off else KIColors.RETTS_YELLOW
        st.markdown(f"""
            <div style="display:flex; align-items:center; gap:15px; height:100%;">
                <span style="font-size:0.85rem; color:#333;">Status: <strong style="color:{status_color};">{'SIGNED OFF' if patient.is_signed_off else 'PENDING REVIEW'}</strong></span>
                <span class="badge-ui" style="background:#EEE; color:#333;">Physician Console</span>
            </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        if patient.triage_agreement:
            from datetime import datetime as _dt
            try:
                pdf_bytes = generate_physician_pdf(patient)
                st.download_button(
                    "📄 Download Physician Report",
                    data=pdf_bytes,
                    file_name=f"physician_{patient.patient_id}_{_dt.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF Error: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)

# Remove the temporary screenshot
import os
if os.path.exists("physician_scroll_issue.png"):
    os.remove("physician_scroll_issue.png")
