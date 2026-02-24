"""Prompt Editor — Dev tool for live YAML prompt editing."""

from pathlib import Path

import streamlit as st
import yaml

from src.utils.config import get_project_root
from src.utils.theme import KIColors

st.markdown(
    f'<h1 style="color:{KIColors.PRIMARY};">Prompt Editor</h1>',
    unsafe_allow_html=True,
)
st.caption("Developer tool: edit YAML prompt templates for AI models")

PROMPTS_DIR = get_project_root() / "config" / "prompts"

# ---- File selection ----
prompt_files = sorted(PROMPTS_DIR.glob("*.yaml"))
if not prompt_files:
    st.warning("No prompt files found in config/prompts/")
    st.stop()

file_options = {str(f): f.stem for f in prompt_files}
selected_file = st.selectbox(
    "Select prompt template",
    options=list(file_options.keys()),
    format_func=lambda x: file_options[x],
)

# ---- Load and display ----
file_path = Path(selected_file)
content = file_path.read_text(encoding="utf-8")

try:
    parsed = yaml.safe_load(content)
except yaml.YAMLError as e:
    parsed = None
    st.error(f"YAML parse error: {e}")

# Show parsed structure
if parsed:
    info_cols = st.columns(3)
    with info_cols[0]:
        st.markdown(f"**Stage:** {parsed.get('stage', '—')}")
    with info_cols[1]:
        st.markdown(f"**Model:** {parsed.get('model_id', '—')}")
    with info_cols[2]:
        st.markdown(f"**File:** `{file_path.name}`")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ---- Editor sections ----
if parsed:
    tab_full, tab_system, tab_user, tab_schema = st.tabs(
        ["Full YAML", "System Prompt", "User Template", "Output Schema"]
    )

    with tab_system:
        system_prompt = parsed.get("system_prompt", "")
        new_system = st.text_area(
            "System Prompt", value=system_prompt, height=250,
            key="system_prompt_editor",
        )

    with tab_user:
        user_template = parsed.get("user_template", "")
        new_user = st.text_area(
            "User Template", value=user_template, height=350,
            key="user_template_editor",
        )

    with tab_schema:
        schema = parsed.get("output_schema", {})
        schema_str = yaml.dump(schema, default_flow_style=False) if schema else ""
        new_schema_str = st.text_area(
            "Output Schema (YAML)", value=schema_str, height=200,
            key="schema_editor",
        )

    with tab_full:
        new_content = st.text_area(
            "Full YAML", value=content, height=500,
            key="full_yaml_editor",
        )

    # ---- Save ----
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    save_col1, save_col2 = st.columns([1, 4])
    with save_col1:
        if st.button("Save Changes", type="primary"):
            # Determine which tab was edited — use full YAML for simplicity
            try:
                # Validate YAML
                updated = yaml.safe_load(new_content)
                if updated is None:
                    st.error("Invalid YAML — empty document")
                else:
                    file_path.write_text(new_content, encoding="utf-8")
                    st.success(f"Saved to {file_path.name}")
                    # Clear config cache
                    from src.utils.config import clear_cache
                    clear_cache()
            except yaml.YAMLError as e:
                st.error(f"YAML validation failed: {e}")
    with save_col2:
        st.caption("Changes take effect on next model inference run.")

else:
    # Fallback: plain text editor
    new_content = st.text_area("Raw YAML", value=content, height=500)
    if st.button("Save"):
        file_path.write_text(new_content, encoding="utf-8")
        st.success("Saved.")
