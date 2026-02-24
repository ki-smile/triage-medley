"""Triage-Medley — Streamlit entry point with role-based navigation.

Central router: role selector → mock auth → role-filtered st.navigation().
"""

import streamlit as st

from src.services.auth_service import DEMO_CREDENTIALS, Role, ROLE_PAGES, authenticate
from src.services.session_manager import init_session_state, load_demo_scenarios, logout
from src.utils.theme import KIColors, inject_custom_css, render_footer

# ---- Page config: called ONCE before anything else ----
st.set_page_config(
    page_title="Triage-Medley",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
init_session_state()


# ====================================================================
# Page Definitions
# ====================================================================

PAGE_REGISTRY = {
    "kiosk": st.Page(
        "pages/0_Kiosk.py", title="Walk-in Kiosk", icon=":material/person:"
    ),
    "queue_view": st.Page(
        "pages/1_Queue_View.py", title="Queue View", icon=":material/queue:"
    ),
    "triage_view": st.Page(
        "pages/2_Triage_View.py", title="Triage View", icon=":material/vital_signs:"
    ),
    "physician_view": st.Page(
        "pages/3_Physician_View.py",
        title="Physician View",
        icon=":material/stethoscope:",
    ),
    "prompt_editor": st.Page(
        "pages/4_Prompt_Editor.py", title="Prompt Editor", icon=":material/edit_note:"
    ),
    "audit_log": st.Page(
        "pages/5_Audit_Log.py", title="Audit Log", icon=":material/history:"
    ),
    "engine_config": st.Page(
        "pages/6_Engine_Config.py",
        title="Engine Config",
        icon=":material/tune:",
    ),
}


# ====================================================================
# Inline Page Functions (role selector + login)
# ====================================================================


def _role_selector_page():
    """Full-screen landing page with 4 role cards."""
    st.markdown(
        f"""<div style="text-align:center; padding:2rem 0 1rem 0;">
        <h1 style="color:{KIColors.PRIMARY}; font-size:2.4rem; font-weight:700; margin-bottom:0.3rem;">
            Triage-Medley
        </h1>
        <p style="color:{KIColors.SECONDARY}; font-size:1.15rem; margin-bottom:0.5rem;">
            Human-in-the-Loop AI-Powered Triage Decision Support System
        </p>
        <p style="color:{KIColors.ERROR}; font-size:0.85rem; font-weight:600;
            background:rgba(184,65,69,0.08); border:1px solid rgba(184,65,69,0.25);
            border-radius:8px; padding:0.4rem 1rem; display:inline-block; margin-top:0.5rem;">
            &#9888; Proof-of-concept demonstrator only &mdash; not developed or validated for clinical use
        </p>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<h3 style="text-align:center; color:{KIColors.PRIMARY}; margin-bottom:1.5rem;">'
        f"Select Your Role</h3>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""<div class="m3-card-elevated" style="text-align:center; min-height:200px; padding:1.5rem;">
            <div style="font-size:2.5rem;">&#128100;</div>
            <h3 style="color:{KIColors.PRIMARY};">Patient</h3>
            <p style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.85rem;">
            Walk-in self-arrival kiosk.<br/>No login required.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Enter as Patient", use_container_width=True, type="primary"):
            st.session_state.role = Role.PATIENT
            st.session_state.user_display_name = "Patient"
            st.rerun()

    with c2:
        st.markdown(
            f"""<div class="m3-card-elevated" style="text-align:center; min-height:200px; padding:1.5rem;">
            <div style="font-size:2.5rem;">&#129658;</div>
            <h3 style="color:{KIColors.PRIMARY};">Triage Nurse</h3>
            <p style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.85rem;">
            Queue management +<br/>ensemble triage review.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Login as Nurse", use_container_width=True):
            st.session_state._login_target_role = "triage_nurse"
            st.rerun()

    with c3:
        st.markdown(
            f"""<div class="m3-card-elevated" style="text-align:center; min-height:200px; padding:1.5rem;">
            <div style="font-size:2.5rem;">&#129657;</div>
            <h3 style="color:{KIColors.PRIMARY};">Physician</h3>
            <p style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.85rem;">
            Differential diagnosis +<br/>management plan.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Login as Physician", use_container_width=True):
            st.session_state._login_target_role = "physician"
            st.rerun()

    with c4:
        st.markdown(
            f"""<div class="m3-card-elevated" style="text-align:center; min-height:200px; padding:1.5rem;">
            <div style="font-size:2.5rem;">&#9881;&#65039;</div>
            <h3 style="color:{KIColors.PRIMARY};">Admin</h3>
            <p style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.85rem;">
            Full access to all<br/>pages and dev tools.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Login as Admin", use_container_width=True):
            st.session_state._login_target_role = "admin"
            st.rerun()

    # Demo credentials hint
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    with st.expander("Demo Credentials & Setup"):
        c_setup1, c_setup2 = st.columns(2)
        if not st.session_state.get("demo_loaded"):
            if c_setup1.button("PRE-LOAD DEMO SCENARIOS", type="primary", use_container_width=True):
                with st.spinner("Running pipeline..."):
                    load_demo_scenarios()
                st.rerun()
        
        if c_setup2.button("RESET DATABASE", type="secondary", use_container_width=True, help="Clears all persistent patient data"):
            st.session_state._clear_db_requested = True
            st.rerun()
            
        st.markdown("---")
            
        st.markdown(
            """
| Username | PIN | Role |
|----------|-----|------|
| `nurse_anna` | `1234` | Triage Nurse |
| `nurse_erik` | `1234` | Triage Nurse |
| `dr_nilsson` | `5678` | Physician |
| `dr_berg` | `5678` | Physician |
| `admin` | `0000` | Admin |
"""
        )
    render_footer()


def _login_page():
    """Staff login form: username + 4-digit PIN."""
    target = st.session_state.get("_login_target_role", "")
    role_label = target.replace("_", " ").title() if target else "Staff"

    st.markdown(
        f"""<div style="text-align:center; padding:2rem 0 1rem 0;">
        <h2 style="color:{KIColors.PRIMARY};">Staff Login</h2>
        <p style="color:{KIColors.ON_SURFACE_VARIANT};">
        Logging in as: <strong>{role_label}</strong></p>
        </div>""",
        unsafe_allow_html=True,
    )

    _, col_form, _ = st.columns([1, 2, 1])
    with col_form:
        with st.form("login_form"):
            username = st.text_input(
                "Username", placeholder="e.g. nurse_anna, dr_nilsson"
            )
            pin = st.text_input(
                "PIN", type="password", max_chars=4, placeholder="4-digit PIN"
            )
            submitted = st.form_submit_button(
                "Login", type="primary", use_container_width=True
            )

            if submitted:
                result = authenticate(username, pin)
                if result.success:
                    if target and result.role.value != target:
                        st.error(
                            f"This account is for "
                            f"**{result.role.value.replace('_', ' ').title()}**, "
                            f"not {role_label}."
                        )
                    else:
                        st.session_state.role = result.role
                        st.session_state.user_display_name = result.display_name
                        st.session_state.username = username.lower().strip()
                        st.session_state._login_target_role = None
                        
                        # Auto-load demo scenarios for clinical staff
                        if not st.session_state.get("demo_loaded"):
                            load_demo_scenarios()
                            
                        st.rerun()
                else:
                    st.error(result.error)

        if st.button("Back to Role Selection", use_container_width=True):
            st.session_state._login_target_role = None
            st.rerun()

    render_footer()


# ====================================================================
# Navigation Router
# ====================================================================

role = st.session_state.get("role")

# Check if we need to show the login form (staff clicked a role card)
_show_login = st.session_state.get("_login_target_role") is not None and role is None

if role is None and not _show_login:
    # ---- No role: show role selector (hide sidebar) ----
    pg = st.navigation(
        [st.Page(_role_selector_page, title="Welcome", icon=":material/home:")],
        position="hidden",
    )
    pg.run()

elif role is None and _show_login:
    # ---- Staff login form (hide sidebar) ----
    pg = st.navigation(
        [st.Page(_login_page, title="Login", icon=":material/login:")],
        position="hidden",
    )
    pg.run()

elif role == Role.PATIENT:
    # ---- Patient: kiosk only, no sidebar ----
    pg = st.navigation(
        [PAGE_REGISTRY["kiosk"]],
        position="hidden",
    )
    pg.run()

else:
    # ---- Staff roles: sidebar with role-filtered pages ----
    allowed_ids = ROLE_PAGES[role]

    if role == Role.ADMIN:
        pg = st.navigation(
            {
                "Clinical": [
                    PAGE_REGISTRY["kiosk"],
                    PAGE_REGISTRY["queue_view"],
                    PAGE_REGISTRY["triage_view"],
                    PAGE_REGISTRY["physician_view"],
                ],
                "Development": [
                    PAGE_REGISTRY["prompt_editor"],
                    PAGE_REGISTRY["audit_log"],
                    PAGE_REGISTRY["engine_config"],
                ],
            },
            position="sidebar",
        )
    else:
        allowed_pages = [
            PAGE_REGISTRY[pid] for pid in allowed_ids if pid in PAGE_REGISTRY
        ]
        pg = st.navigation(allowed_pages, position="sidebar")

    with st.sidebar:
        st.markdown(
            f'<div class="ki-brand">Triage-Medley</div>'
            f'<div class="ki-brand-sub">Karolinska Institutet / KTH</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(f"**{st.session_state.user_display_name}**")
        st.caption(f"Role: {role.value.replace('_', ' ').title()}")
        if st.button("Logout", use_container_width=True):
            logout()
            st.rerun()
        st.markdown("---")

        # Demo status
        if not st.session_state.get("demo_loaded"):
            st.warning("Demo data not loaded")
        else:
            st.success("Demo loaded")
            from src.services.session_manager import get_patients

            patients = get_patients()
            st.caption(f"{len(patients)} patients in queue")

        # Sidebar footer
        st.markdown("---")
        
        # Font Size Selector
        st.caption("🖼️ Font Size")
        st.radio(
            "Font Size",
            options=["Small", "Medium", "Large", "X-Large"],
            index=1,
            horizontal=False,
            label_visibility="collapsed",
            key="font_size_selector",
        )
        
        st.markdown("---")
        from src.utils.theme import _COPYRIGHT
        st.caption(_COPYRIGHT)

    pg.run()
