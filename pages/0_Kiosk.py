"""Kiosk View — Patient-facing walk-in self-arrival screen.

3-step wizard: Identify -> Describe complaint -> Confirmation + queue priority.
Optimized for ZERO TYPING: Expanded Quick-Select symptoms + Guided refinement.
"""

import base64
import io
import streamlit as st

from src.services.ehr_service import list_available_patients, load_patient
from src.services.session_manager import (
    get_patients,
    get_queue_ordered,
    register_kiosk_patient,
)
from src.utils.theme import KIColors, priority_badge

def _generate_qr_code(url: str) -> str:
    try:
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=6, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#4F0433", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except ImportError:
        return ""

# Symptom Definitions with Guided Questions
SYMPTOMS_MAP = {
    "chest_pain": {
        "icon": "🫀", "label": "Chest Pain",
        "questions": ["Started suddenly?", "Pressure/Crushing?", "Pain in arm/jaw?"]
    },
    "respiratory": {
        "icon": "🫁", "label": "Breathing Trouble",
        "questions": ["Short of breath at rest?", "Wheezing?", "Worse when lying down?"]
    },
    "trauma_fall": {
        "icon": "🩹", "label": "Injury / Fall",
        "questions": ["Hit your head?", "Unable to walk?", "Visible bleeding?"]
    },
    "neurological": {
        "icon": "🧠", "label": "Headache / Dizzy",
        "questions": ["Worst headache ever?", "Numbness/Weakness?", "Blurred vision?"]
    },
    "abdominal": {
        "icon": "🤢", "label": "Stomach Pain",
        "questions": ["Severe cramping?", "Vomiting?", "Diarrhea?"]
    },
    "allergy": {
        "icon": "🐝", "label": "Allergy",
        "questions": ["Swelling of lips/face?", "Itchy rash?", "Trouble swallowing?"]
    },
    "infection": {
        "icon": "🤒", "label": "Fever / Sick",
        "questions": ["Chills/Shivering?", "Coughing?", "Skin rash?"]
    },
    "wound": {
        "icon": "🩸", "label": "Bleeding/Wound",
        "questions": ["Bleeding won't stop?", "Deep cut?", "Painful swelling?"]
    }
}

# ---- Kiosk-specific CSS ----
st.markdown(
    """
    <style>
    .big-input input {
        font-size: 2.2rem !important;
        height: 5.5rem !important;
        text-align: center !important;
        border-radius: 20px !important;
        border: 3px solid #4F0433 !important;
        background: #FFF9FC !important;
    }
    .stButton > button {
        height: 4.5rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border-radius: 2.25rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    .step-label {
        font-weight: 800;
        text-transform: uppercase;
        color: #4F0433;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .refinement-chip {
        padding: 10px 20px;
        border-radius: 30px;
        border: 2px solid #4DB5BC;
        background: white;
        color: #4DB5BC;
        font-weight: 600;
        display: inline-block;
        margin: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "kiosk_step" not in st.session_state: st.session_state.kiosk_step = 1
if "kiosk_patient_id" not in st.session_state: st.session_state.kiosk_patient_id = None
if "kiosk_ehr" not in st.session_state: st.session_state.kiosk_ehr = None
if "kiosk_result_session" not in st.session_state: st.session_state.kiosk_result_session = None
if "selected_symptom_id" not in st.session_state: st.session_state.selected_symptom_id = None
if "selected_refinements" not in st.session_state: st.session_state.selected_refinements = []

step = st.session_state.kiosk_step

# Progress Dots
dots_html = "".join(f'<div class="kiosk-step-dot {"active" if i == step else ""}"></div>' for i in range(1, 4))
st.markdown(f'<div class="kiosk-step-indicator">{dots_html}</div>', unsafe_allow_html=True)

_, col_main, _ = st.columns([1, 6, 1])

with col_main:
    # ==================================================
    # STEP 1: Identification
    # ==================================================
    if step == 1:
        with st.container(border=True):
            st.markdown('<div class="kiosk-title">Welcome</div>', unsafe_allow_html=True)
            st.markdown('<div class="kiosk-subtitle">Enter your 12-digit personnummer to start</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="big-input">', unsafe_allow_html=True)
            typed_id = st.text_input("PN", placeholder="YYYYMMDDXXXX", key="kiosk_typed_id", label_visibility="collapsed")
            st.markdown('</div><br/>', unsafe_allow_html=True)
            
            if st.button("START CHECK-IN", type="primary", use_container_width=True):
                patient_id = typed_id.strip().lower()
                if not patient_id:
                    st.warning("Please enter your ID.")
                else:
                    ehr = load_patient(patient_id)
                    st.session_state.kiosk_patient_id = patient_id
                    st.session_state.kiosk_ehr = ehr
                    st.session_state.kiosk_step = 2
                    st.rerun()

        qr_uri = _generate_qr_code("http://localhost:8501") 
        st.markdown(f"""
            <div class="qr-section" style="border: 3px dashed #4DB5BC; background: #F0F9FA;">
                <div style="font-weight:800; color:#4DB5BC; font-size:1.2rem; margin-bottom:0.5rem;">📱 HATE TYPING?</div>
                <div style="font-size:1rem; color:#444; margin-bottom:1rem;">Scan to check-in privately on your phone.</div>
                <img src="{qr_uri}" style="width:150px; border-radius:12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"/>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("Testing / Demo (Staff)"):
            available = list_available_patients()
            cols = st.columns(3)
            for idx, pid in enumerate(sorted(available)):
                if cols[idx % 3].button(pid.capitalize(), key=f"demo_{pid}", use_container_width=True):
                    st.session_state.kiosk_patient_id = pid
                    st.session_state.kiosk_ehr = load_patient(pid)
                    st.session_state.kiosk_step = 2
                    st.rerun()

    # ==================================================
    # STEP 2: Complaint (Guided Refinement + Voice)
    # ==================================================
    elif step == 2:
        ehr = st.session_state.kiosk_ehr
        name = ehr.name if ehr else "Valued Patient"
        
        st.markdown(f'<div class="kiosk-title">Hello, {name}</div>', unsafe_allow_html=True)
        st.markdown('<div class="kiosk-subtitle">Tap what you are feeling</div>', unsafe_allow_html=True)

        # A: EXPANDED QUICK SELECT
        qcols = st.columns(4)
        for i, (sid, data) in enumerate(SYMPTOMS_MAP.items()):
            with qcols[i % 4]:
                active = st.session_state.selected_symptom_id == sid
                btn_type = "primary" if active else "secondary"
                if st.button(f"{data['icon']}\n{data['label']}", key=f"btn_{sid}", use_container_width=True, type=btn_type):
                    st.session_state.selected_symptom_id = sid
                    st.session_state.selected_refinements = []
                    st.rerun()

        # B: GUIDED REFINEMENT QUESTIONS
        if st.session_state.selected_symptom_id:
            sid = st.session_state.selected_symptom_id
            st.markdown("<br/>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f'<div class="step-label">Refine: {SYMPTOMS_MAP[sid]["label"]}</div>', unsafe_allow_html=True)
                
                # Show follow-up questions as toggle-buttons
                ref_cols = st.columns(len(SYMPTOMS_MAP[sid]["questions"]))
                for i, q in enumerate(SYMPTOMS_MAP[sid]["questions"]):
                    is_on = q in st.session_state.selected_refinements
                    if ref_cols[i].button(q, key=f"q_{i}", use_container_width=True, type="primary" if is_on else "secondary"):
                        if is_on: st.session_state.selected_refinements.remove(q)
                        else: st.session_state.selected_refinements.append(q)
                        st.rerun()

        st.markdown("<br/>", unsafe_allow_html=True)
        
        # C: VOICE RECORDING / FINAL DESC
        with st.container(border=True):
            st.markdown('<div class="step-label">Add more detail (Optional)</div>', unsafe_allow_html=True)
            audio_data = st.audio_input("Speak naturally about your symptoms")
            
            # Construct the descriptive text from choices
            base_text = ""
            if st.session_state.selected_symptom_id:
                base_text = f"Primary complaint: {SYMPTOMS_MAP[st.session_state.selected_symptom_id]['label']}. "
                if st.session_state.selected_refinements:
                    base_text += "Patient reports: " + ", ".join(st.session_state.selected_refinements) + ". "

            if audio_data is not None:
                from src.services.asr_service import process_audio
                with st.spinner("Listening..."):
                    asr_result = process_audio(st.session_state.kiosk_patient_id)
                    final_text = st.text_area("Final Summary:", value=f"{base_text}{asr_result.merged_transcript}", height=120)
            else:
                final_text = st.text_area("Final Summary:", value=base_text, height=100)

            st.markdown("<br/>", unsafe_allow_html=True)
            
            c_back, c_done = st.columns([1, 2])
            if c_back.button("BACK"):
                st.session_state.kiosk_step = 1
                st.rerun()
            if c_done.button("SUBMIT & GET TICKET", type="primary"):
                if not final_text.strip():
                    st.warning("Please select a symptom or provide a description.")
                else:
                    with st.spinner("Registering..."):
                        session = register_kiosk_patient(st.session_state.kiosk_patient_id, final_text.strip(), "en")
                        st.session_state.kiosk_result_session = session
                        st.session_state.kiosk_step = 3
                        st.rerun()

    # ==================================================
    # STEP 3: Confirmation
    # ==================================================
    elif step == 3:
        session = st.session_state.kiosk_result_session
        prio = session.queue_priority.value if session and session.queue_priority else "STANDARD"
        color = KIColors.priority_color(prio)
        queue = get_queue_ordered()
        pos = next((i for i, p in enumerate(queue) if p.patient_id == session.patient_id), 0) + 1

        with st.container(border=True):
            st.markdown("""
                <div style="text-align:center;">
                    <div style="font-size:5rem;margin-bottom:1rem;">✅</div>
                    <div class="kiosk-title">You're Registered</div>
                    <div class="kiosk-subtitle">Take a seat. We will call you soon.</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="text-align:center; padding:1.5rem 0;">
                    <div style="background:{color}; color:white; padding:2rem; border-radius:24px; display:inline-block; min-width:300px; margin-bottom:2rem; box-shadow:0 8px 25px rgba(0,0,0,0.15);">
                        <div style="font-size:1rem;text-transform:uppercase;opacity:0.9;font-weight:800;letter-spacing:1px;">Priority</div>
                        <div style="font-size:3.5rem;font-weight:900;">{prio}</div>
                    </div>
                    <div style="font-size:1.8rem; font-weight:700; color:var(--ki-on-surface);">
                        Position: <strong>#{pos}</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("FINISH", type="primary", use_container_width=True):
                st.session_state.kiosk_step = 1
                st.session_state.kiosk_patient_id = None
                st.session_state.kiosk_ehr = None
                st.session_state.kiosk_result_session = None
                st.session_state.selected_symptom_id = None
                st.session_state.selected_refinements = []
                st.rerun()
