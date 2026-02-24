"""Engine Config — Admin page for configuring active triage engines and LLM models.

Allows admins to select which rule-based engines (RETTS, ESI, MTS) and which
LLM models participate in the MEDLEY ensemble. Changes take effect on next
demo reload or new patient triage.
"""

import streamlit as st

from src.services.session_manager import init_session_state, load_demo_scenarios
from src.utils.config import load_config
from src.utils.theme import KIColors

init_session_state()

# ---- Header ----
st.markdown(
    f'<h1 style="color:{KIColors.PRIMARY};">Engine Configuration</h1>',
    unsafe_allow_html=True,
)
st.caption("Admin: configure which triage engines and LLM models participate in the ensemble")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================================
# Section 0: HuggingFace API Key
# ====================================================================
st.subheader("HuggingFace API Key")
st.markdown(
    f'<p style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.9rem;">'
    f"For demo servers without a pre-configured <code>HF_TOKEN</code> environment variable, "
    f"provide your own HuggingFace API key here. The env var takes priority if set.</p>",
    unsafe_allow_html=True,
)

# Status indicator
import os as _os
_env_token = _os.environ.get("HF_TOKEN")
_session_token = st.session_state.get("hf_api_key")

if _env_token:
    st.markdown(
        f'<div style="color:{KIColors.RETTS_GREEN}; font-size:0.9rem; font-weight:600;">'
        f'&#x2705; Set via environment variable</div>',
        unsafe_allow_html=True,
    )
elif _session_token:
    st.markdown(
        f'<div style="color:{KIColors.RETTS_GREEN}; font-size:0.9rem; font-weight:600;">'
        f'&#x2705; Set via admin panel</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div style="color:{KIColors.RETTS_ORANGE}; font-size:0.9rem; font-weight:600;">'
        f'&#x26A0;&#xFE0F; Not configured &mdash; models will use mock adapters</div>',
        unsafe_allow_html=True,
    )

# API key & Space ID inputs
_new_key = st.text_input(
    "HuggingFace API Token",
    value=_session_token or "",
    type="password",
    placeholder="hf_...",
    help="Paste your HuggingFace token. Stored in session only.",
)
if _new_key != (_session_token or ""):
    st.session_state.hf_api_key = _new_key if _new_key else None
    st.rerun()

_current_space = st.session_state.get("hf_space_id", "")
_new_space = st.text_input(
    "Global Space ID (optional)",
    value=_current_space,
    placeholder="your-username/your-space-name",
    help="Override the default Space ID for all models using 'space' adapter.",
)
if _new_space != _current_space:
    st.session_state.hf_space_id = _new_space if _new_space else None
    st.rerun()

# Test connection buttons
_test_col1, _test_col2 = st.columns(2)

with _test_col1:
    if st.button("Test Inference API", key="test_hf_connection", use_container_width=True):
        _active_token = _env_token or _session_token
        if not _active_token:
            st.error("No API key available. Enter a key above or set `HF_TOKEN` env var.")
        else:
            # Check if user has explicitly set medgemma to 'space' in the UI
            is_space_mode = st.session_state.get("adapter_type_medgemma_4b") == "space"
            
            if is_space_mode:
                st.warning("MedGemma 4B is in SPACE mode. Use 'Test Space Connection' on the right to verify.")
            
            test_model = "meta-llama/Llama-3.2-1B-Instruct" 
            with st.spinner(f"Verifying token validity with {test_model}..."):
                try:
                    from huggingface_hub import InferenceClient
                    _test_client = InferenceClient(
                        model=test_model,
                        token=_active_token,
                        timeout=15,
                    )
                    _resp = _test_client.chat_completion(
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=20,
                    )
                    st.success("HuggingFace Token is VALID.")
                    if not is_space_mode:
                        st.info("Note: 'google/medgemma-4b-it' is NOT supported by the Inference API. Please switch it to 'space' mode below.")
                except Exception as e:
                    st.error(f"Token verification failed: {e}")

with _test_col2:
    if st.button("Test Space Connection", key="test_space_connection", use_container_width=True):
        _active_token = _env_token or _session_token
        if not _active_token:
            st.error("No API key available. Enter a key above or set `HF_TOKEN` env var.")
        else:
            _space_id = st.session_state.get("hf_space_id") or _os.environ.get("HF_SPACE_ID", "")
            with st.spinner(f"Testing Space: {_space_id}..."):
                try:
                    from gradio_client import Client as _GradioClient
                    _test_space = _GradioClient(_space_id, token=_active_token)
                    _resp = _test_space.predict(
                        model_choice="MedGemma 4B",
                        image=None,
                        text_prompt="Hello",
                        api_name="/doctor_infer"
                    )
                    st.success(f"Space OK: {str(_resp)[:100]}")
                except Exception as e:
                    st.error(f"Space connection failed: {e}")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================================
# Section 1: Rule-Based Triage Engines
# ====================================================================
st.subheader("Rule-Based Triage Engines")
st.markdown(
    f'<p style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.9rem;">'
    f"Each engine applies a different triage philosophy. Cross-system disagreement "
    f"is clinically informative — it's the MEDLEY diversity principle in action.</p>",
    unsafe_allow_html=True,
)

engines_config = load_config("engines.yaml")
engines = engines_config.get("engines", {})

active_engines = st.session_state.get("active_engines", ["retts", "esi", "mts"])

for engine_id, engine_data in engines.items():
    name = engine_data.get("name", engine_id.upper())
    description = engine_data.get("description", "")
    philosophy = engine_data.get("philosophy", "")
    is_default = engine_data.get("enabled_by_default", False)

    # RETTS is always enabled (can't be disabled)
    is_retts = engine_id == "retts"
    is_active = engine_id in active_engines

    border_color = KIColors.RETTS_GREEN if is_active else KIColors.OUTLINE
    opacity = "1" if is_active else "0.7"

    col_check, col_info = st.columns([1, 6])
    with col_check:
        if is_retts:
            st.checkbox(
                "Active",
                value=True,
                disabled=True,
                key=f"engine_{engine_id}",
                help="RETTS is always active as the Swedish national standard",
            )
        else:
            new_val = st.checkbox(
                "Active",
                value=is_active,
                key=f"engine_{engine_id}",
            )
            if new_val and engine_id not in active_engines:
                active_engines.append(engine_id)
                st.session_state.active_engines = active_engines
            elif not new_val and engine_id in active_engines:
                active_engines.remove(engine_id)
                st.session_state.active_engines = active_engines

    with col_info:
        default_badge = (
            f'<span style="color:{KIColors.RETTS_GREEN}; font-size:0.8rem; margin-left:8px;">DEFAULT</span>'
            if is_default else ""
        )
        st.markdown(
            f'<div class="m3-card" style="border-left:4px solid {border_color}; opacity:{opacity};">'
            f'<strong style="color:{KIColors.PRIMARY};">{name}</strong>'
            f'{default_badge}'
            f'<div style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.85rem; margin-top:4px;">'
            f'{description.strip()}</div>'
            f'<div style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.8rem; font-style:italic; margin-top:4px;">'
            f'{philosophy.strip()}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================================
# Section 2: LLM Models
# ====================================================================
st.subheader("LLM Models")
st.markdown(
    f'<p style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.9rem;">'
    f"AI models that complement the rule-based engines. Each brings a different "
    f"training background and clinical perspective.</p>",
    unsafe_allow_html=True,
)

models_config = load_config("models.yaml")
all_models = models_config.get("models", {})

# Filter to only LLM models (exclude deterministic engines)
llm_models = {
    k: v for k, v in all_models.items()
    if v.get("adapter") != "deterministic"
}

active_models = st.session_state.get("active_models", None)
# If None, all models are active
if active_models is None:
    active_models = list(llm_models.keys())

for model_id, model_data in llm_models.items():
    name = model_data.get("name", model_id)
    hf_id = model_data.get("hf_id", "")
    role = model_data.get("role", "")
    stages = model_data.get("stages", [])
    priority = model_data.get("priority", "")
    adapter = model_data.get("adapter", "mock")

    is_active = model_id in active_models
    border_color = KIColors.RETTS_GREEN if is_active else KIColors.OUTLINE
    opacity = "1" if is_active else "0.7"

    # Priority badge color
    prio_colors = {"P1": KIColors.RETTS_RED, "P2": KIColors.RETTS_ORANGE, "P3": KIColors.RETTS_YELLOW}
    prio_color = prio_colors.get(priority, KIColors.ON_SURFACE_VARIANT)

    col_check, col_info = st.columns([1, 6])
    with col_check:
        new_val = st.checkbox(
            "Active",
            value=is_active,
            key=f"model_{model_id}",
        )
        if new_val and model_id not in active_models:
            active_models.append(model_id)
            st.session_state.active_models = active_models
        elif not new_val and model_id in active_models:
            active_models.remove(model_id)
            st.session_state.active_models = active_models

    with col_info:
        # Adapter Type Selection
        current_adapter = st.session_state.get(f"adapter_type_{model_id}") or adapter
        adapter_options = ["mock", "huggingface", "space"]
        new_adapter = st.selectbox(
            "Adapter Mode",
            options=adapter_options,
            index=adapter_options.index(current_adapter) if current_adapter in adapter_options else 0,
            key=f"select_adapter_{model_id}",
            label_visibility="collapsed"
        )
        if new_adapter != current_adapter:
            st.session_state[f"adapter_type_{model_id}"] = new_adapter
            st.rerun()

        # Per-model Space ID if in space mode
        if new_adapter == "space":
            model_space = st.session_state.get(f"space_id_{model_id}") or model_data.get("space_id", "")
            new_model_space = st.text_input(
                "Model Space ID",
                value=model_space,
                key=f"space_input_{model_id}",
                placeholder="author/space-name",
                label_visibility="collapsed"
            )
            if new_model_space != model_space:
                st.session_state[f"space_id_{model_id}"] = new_model_space
                st.rerun()

        stages_str = ", ".join(s.capitalize() for s in stages)
        adapter_badge = (
            f'<span style="color:{KIColors.RETTS_GREEN}; font-size:0.75rem; '
            f'border:1px solid {KIColors.RETTS_GREEN}; border-radius:8px; padding:1px 6px;">'
            f'{new_adapter}</span>'
        )

        st.markdown(
            f'<div class="m3-card" style="border-left:4px solid {border_color}; opacity:{opacity};">'
            f'<strong style="color:{KIColors.PRIMARY};">{name}</strong>'
            f' <span style="color:{prio_color}; font-weight:600; font-size:0.8rem;">{priority}</span>'
            f' {adapter_badge}'
            f'<div style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.85rem; margin-top:4px;">'
            f'{role}</div>'
            f'<div style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.8rem; margin-top:2px;">'
            f'<code>{hf_id}</code> &middot; Stages: {stages_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================================
# Section 3: Ensemble Summary
# ====================================================================
st.subheader("Ensemble Summary")

n_engines = len(st.session_state.get("active_engines", ["retts", "esi", "mts"]))
current_active_models = st.session_state.get("active_models", list(llm_models.keys()))
n_models = len(current_active_models) if current_active_models else len(llm_models)
total_voters = n_engines + n_models

summary_cols = st.columns(4)
with summary_cols[0]:
    st.metric("Rule Engines", f"{n_engines}/3")
with summary_cols[1]:
    st.metric("LLM Models", f"{n_models}/{len(llm_models)}")
with summary_cols[2]:
    st.metric("Total Voters", total_voters)
with summary_cols[3]:
    diversity = "HIGH" if n_engines >= 2 and n_models >= 3 else (
        "MODERATE" if n_engines >= 1 and n_models >= 2 else "LOW"
    )
    diversity_color = (
        KIColors.RETTS_GREEN if diversity == "HIGH"
        else KIColors.RETTS_YELLOW if diversity == "MODERATE"
        else KIColors.RETTS_RED
    )
    st.markdown(
        f'<div style="text-align:center;">'
        f'<div style="font-size:0.8rem; color:{KIColors.ON_SURFACE_VARIANT};">Diversity</div>'
        f'<div style="font-size:1.5rem; font-weight:700; color:{diversity_color};">{diversity}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Active engine list
engine_names = {
    "retts": "RETTS",
    "esi": "ESI",
    "mts": "MTS",
}
active_list = [engine_names.get(e, e.upper()) for e in st.session_state.get("active_engines", ["retts", "esi", "mts"])]
st.markdown(
    f'<div style="color:{KIColors.ON_SURFACE_VARIANT}; font-size:0.9rem; margin-top:0.5rem;">'
    f'Active engines: <strong>{" + ".join(active_list)}</strong> '
    f'+ {n_models} LLM model{"s" if n_models != 1 else ""} '
    f'= <strong>{total_voters} total voters</strong> in the MEDLEY ensemble'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ====================================================================
# Section 4: Reload with New Configuration
# ====================================================================
st.subheader("Apply Configuration")

if st.button(
    "Reload Demo with Current Configuration",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Re-running pipeline with updated engine configuration..."):
        st.session_state.demo_loaded = False
        st.session_state.patients = {}
        load_demo_scenarios()
    st.success("Demo reloaded with new engine configuration.")
    st.rerun()
