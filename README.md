# Triage-Medley

**Human-in-the-Loop AI-Powered Triage Decision Support System**

A multi-model ensemble triage decision support system for Swedish emergency departments, built on the [MEDLEY framework](https://arxiv.org/abs/2508.21648) (Medical Ensemble Diagnostic system with Leveraged diversitY).

Built for the **MedGemma Impact Challenge** (February 2026).

## Features

- **Multi-Model Ensemble:** Combines MedGemma (4B & 27B), BioMistral, and rule engines.
- **Shared Persistence:** Uses SQLite to synchronize patient data across multiple devices/browsers.
- **Live Auto-Sync:** Queue View updates automatically when new patients arrive at the kiosk.
- **Role-Based Workflows:** Optimized dashboards for Patients, Nurses, Physicians, and Admins.
- **Audit Compliance:** Every AI suggestion and human decision is logged to `data/audit/audit.jsonl`.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App (Mock Mode -- No API Keys Required)

```bash
streamlit run app.py
```

The app runs with **mock adapters** by default -- all AI model responses are simulated using pre-built scenario data. No HuggingFace account or API keys are needed to explore the full UI.

### 3. Run Tests

```bash
# All tests
pytest tests/ -v

# HuggingFace adapter tests only (uses mocked API -- no token needed)
pytest tests/test_hf_adapters.py -v

# Space adapter tests (uses mocked Gradio client -- no token needed)
pytest tests/test_space_adapters.py -v

# PDF report tests
pytest tests/test_pdf.py -v
```

## Demo Credentials

| Username | PIN | Role |
|----------|-----|------|
| `nurse_anna` | `1234` | Triage Nurse |
| `nurse_erik` | `1234` | Triage Nurse |
| `dr_nilsson` | `5678` | Physician |
| `dr_berg` | `5678` | Physician |
| `admin` | `0000` | Admin |

Patient role requires no login.

---

## HuggingFace Integration

The system supports two modes of live model inference — the [HuggingFace Inference API](https://huggingface.co/docs/huggingface_hub/guides/inference) (direct model access) and [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces) (custom Gradio deployments). Adapter switching is entirely config-driven -- no code changes required.

| Mode | Library | Config Value | Best For |
|------|---------|-------------|----------|
| **Mock** | N/A (local JSON) | `adapter: "mock"` | Development, demos, no API key needed |
| **Inference API** | `huggingface_hub.InferenceClient` | `adapter: "huggingface"` | Direct model access via HF infrastructure |
| **Space** | `gradio_client.Client` | `adapter: "space"` | Custom deployments with full control over GPU/config |

### Model Registry

| Model ID | HuggingFace Repo | Stages | Role |
|----------|-------------------|--------|------|
| `medgemma_4b` | `google/medgemma-4b-it` | Pre-triage + Triage + Differential + Management | Anchor model (fast, multimodal) |
| `medgemma_27b` | `google/medgemma-27b-text-it` | Triage + Differential + Management | Deep analysis (87.7% MedQA) [Disabled by default] |
| `meditron_7b` | `OpenMeditron/Meditron3-Qwen2.5-7B` | Triage + Differential | Medical specialized Qwen2.5-7B variant |
| `biomistral` | `BioMistral/BioMistral-7B` | Triage + Differential | Devil's advocate / don't-miss |

### Step 1: Create a HuggingFace Account and Token

1. Sign up at [huggingface.co](https://huggingface.co/join)
2. Go to **Settings** > **Access Tokens** ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
3. Create a new token with **Read** access (fine-grained: `Make calls to the serverless Inference API`)
4. Copy the token (starts with `hf_...`)

### Step 2: Request Model Access (Gated Models)

Some models require accepting a license agreement before use:

1. Visit each model page on HuggingFace and click **"Agree and access repository"** if prompted:
   - [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
   - [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it)
2. Community models (Meditron3-7B, BioMistral) are typically open access -- no approval needed.

### Alternative: Provide API Key via Admin Panel

If `HF_TOKEN` is not set as an environment variable, admin users can provide a HuggingFace API key through the **Engine Config** page. Login as `admin` (PIN: `0000`) → Engine Config → enter your API key. The key is stored in the session only and not persisted.

### Step 3: Set the HF_TOKEN Environment Variable

```bash
# Linux / macOS
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Windows (PowerShell)
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Or add to .env / .bashrc for persistence
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' >> ~/.bashrc
```

### Step 4: Switch from Mock to Live Adapters

Edit `config/models.yaml` and change `adapter` from `"mock"` to `"huggingface"` or `"space"`:

```yaml
# config/models.yaml -- Mock mode (default)
models:
  medgemma_4b:
    name: "MedGemma 4B"
    hf_id: "google/medgemma-4b-it"
    adapter: "mock"                    # <-- change this
    space_id: "your-username/your-space"
    api_name: "/predict"
    stages: ["pretriage", "triage", "differential", "management"]
    timeout_seconds: 3
```

**Option A: HuggingFace Inference API** (direct model access)

```yaml
    adapter: "huggingface"             # Direct model via InferenceClient
    timeout_seconds: 30                # increase for API latency
```

**Option B: HuggingFace Space** (custom Gradio deployment)

```yaml
    adapter: "space"                   # Gradio Space via gradio_client
    space_id: "your-username/your-space"  # your deployed Space
    api_name: "/predict"               # Gradio endpoint name
    timeout_seconds: 30                # Spaces may have cold-start latency
```

You can switch models individually -- e.g. enable only `medgemma_4b` via Space while keeping the rest on mock. Both live modes use the same `hf_` token.

### Step 5: Test the Connection

```bash
# Test Inference API connection
python -c "
from huggingface_hub import InferenceClient
import os
client = InferenceClient(model='google/medgemma-4b-it', token=os.environ['HF_TOKEN'])
response = client.chat_completion(messages=[{'role': 'user', 'content': 'Hello'}], max_tokens=50)
print('Inference API OK:', response.choices[0].message.content[:100])
"

# Test Space connection
python -c "
from gradio_client import Client
import os
client = Client('your-username/your-space', hf_token=os.environ['HF_TOKEN'])
result = client.predict(message='Hello', api_name='/predict')
print('Space OK:', str(result)[:100])
"

# Run adapter tests (uses mocked responses -- validates parsing logic)
pytest tests/test_hf_adapters.py tests/test_space_adapters.py -v
```

You can also test connections via the **Engine Config** admin page (login as `admin` / `0000`), which has both "Test Inference API" and "Test Space Connection" buttons.

### Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: HF_TOKEN environment variable not set` | Token not exported | Run `export HF_TOKEN=hf_...` before starting the app |
| `401 Unauthorized` | Invalid or expired token | Regenerate token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `403 Forbidden` on MedGemma | Model access not granted | Visit model page and click "Agree and access repository" |
| `timeout` errors | Model cold start or large model | Increase `timeout_seconds` in `config/models.yaml` (30-60s for 27B models) |
| `ModuleNotFoundError: gradio_client` | Space dependency not installed | Run `pip install gradio_client>=1.0.0` |
| Space cold-start (30-60s delay) | Free-tier Space sleeping | First request wakes the Space; increase `timeout_seconds` to 60 |
| `No Space adapter registered` | Model has no Space adapter class | Only `medgemma_4b` currently has a Space adapter |
| Graceful degradation | One model fails | System continues with remaining models; failed model excluded from ensemble |

---

## Project Structure

```
triage-medley/
├── app.py                       # Streamlit entry point (role-based navigation)
├── requirements.txt             # Python dependencies
├── assets/
│   └── style.css                # Custom Streamlit CSS theme
├── config/
│   ├── engines.yaml             # Multi-engine configuration (RETTS, ESI, MTS)
│   ├── models.yaml              # Model registry + adapter selection (mock | huggingface | space)
│   ├── pretriage.yaml           # Pre-triage priority rules
│   ├── esi/                     # ESI triage rules
│   │   ├── decision_tree.json   # ESI 5-level decision tree
│   │   └── resource_rules.json  # ESI resource prediction rules
│   ├── mts/                     # Manchester Triage System rules
│   │   ├── flowcharts.json      # MTS clinical flowcharts
│   │   └── general_discriminators.json
│   ├── prompts/                 # YAML prompt templates (per stage)
│   │   ├── pretriage.yaml
│   │   ├── triage.yaml
│   │   ├── differential.yaml
│   │   └── management.yaml
│   └── retts/                   # RETTS clinical rules
│       ├── ess_codes.json       # ESS category flowcharts
│       └── vitals_cutoffs.json  # Vital sign thresholds per RETTS level
├── src/
│   ├── models/                  # Pydantic data classes
│   │   ├── clinical.py          # Clinical data models
│   │   ├── context.py           # PreTriageContext, FullTriageContext
│   │   ├── enums.py             # RETTS levels, priorities, roles
│   │   ├── outputs.py           # Triage/differential/management outputs
│   │   └── vitals.py            # VitalSigns model
│   ├── adapters/                # ModelAdapter protocol + implementations
│   │   ├── base.py              # ModelAdapter protocol definition
│   │   ├── mock_adapter.py      # Mock responses from JSON scenario files
│   │   ├── hf_base.py           # HuggingFace Inference API base adapter
│   │   ├── hf_medgemma.py       # MedGemma 4B / 27B adapters (Inference API)
│   │   ├── hf_ensemble.py       # QwenMed, BioMistral adapters (Inference API)
│   │   ├── space_base.py        # HuggingFace Space base adapter (Gradio client)
│   │   ├── space_medgemma.py    # MedGemma 4B adapter via Space
│   │   ├── prompt_builder.py    # Prompt template rendering
│   │   └── factory.py           # Config-driven adapter creation (mock/hf/space)
│   ├── engines/                 # Deterministic triage rule engines
│   │   ├── retts_engine.py      # RETTS vitals + ESS → colour level
│   │   ├── esi_engine.py        # ESI 5-level acuity engine
│   │   ├── mts_engine.py        # Manchester Triage System engine
│   │   ├── pretriage_engine.py  # Pre-triage queue priority engine
│   │   └── agreement_engine.py  # Ensemble agreement analysis
│   ├── services/                # Application services
│   │   ├── asr_service.py       # Dual-ASR speech pipeline
│   │   ├── auth_service.py      # Role-based authentication
│   │   ├── ehr_service.py       # FHIR EHR data retrieval
│   │   ├── orchestrator.py      # Two-stage pipeline orchestrator
│   │   ├── pdf_service.py       # PDF triage report generation
│   │   └── session_manager.py   # Streamlit session state management
│   └── utils/                   # Shared utilities
│       ├── audit.py             # Append-only audit logger
│       ├── config.py            # YAML/JSON config loader
│       └── theme.py             # UI theme constants
├── pages/                       # Streamlit multi-page app
│   ├── 0_Kiosk.py               # Patient self-arrival kiosk
│   ├── 1_Queue_View.py          # Charge nurse: priority-ordered waiting room
│   ├── 2_Triage_View.py         # Triage nurse: vitals entry → ensemble result
│   ├── 3_Physician_View.py      # Physician: differential + management handoff
│   ├── 4_Prompt_Editor.py       # Dev tool: live YAML prompt editing
│   ├── 5_Audit_Log.py           # Compliance: full decision trail
│   └── 6_Engine_Config.py       # Admin: engine selection + API key + Space config
├── data/
│   ├── audit/                   # Audit log output (audit.jsonl)
│   ├── ehr/                     # Synthetic FHIR patient bundles (6 scenarios)
│   └── scenarios/               # Mock JSON responses (6 patients × 4 stages × models)
├── tests/                       # Pytest test suite
│   ├── test_adapters.py         # Mock adapter tests
│   ├── test_hf_adapters.py      # HuggingFace Inference API adapter tests (mocked)
│   ├── test_space_adapters.py   # HuggingFace Space adapter tests (mocked)
│   ├── test_pipeline.py         # End-to-end pipeline tests
│   ├── test_retts.py            # RETTS engine tests
│   ├── test_esi.py              # ESI engine tests
│   ├── test_mts.py              # MTS engine tests
│   ├── test_multi_engine.py     # Multi-engine agreement tests
│   ├── test_pretriage.py        # Pre-triage engine tests
│   ├── test_models.py           # Pydantic model tests
│   ├── test_ehr.py              # EHR service tests
│   ├── test_auth.py             # Authentication tests
│   ├── test_audit.py            # Audit logger tests
│   └── test_pdf.py              # PDF report tests
├── docs/
│   └── ARCHITECTURE.md          # Detailed architecture documentation
├── presentation/                # PPTX slide deck + build tooling
│   ├── Triage-Medley.pptx     # Generated presentation
│   ├── build.js                 # Slide builder script
│   ├── create-assets.js         # Asset generation script
│   └── slides/                  # HTML slide sources + image assets
└── ref/                         # Reference documents (papers, specs)
```

## Contributors

### SMAILE Team

- Farhad Abtahi
- Fardin Afdideh
- Eduardo Illueca Fernandez
- Abdolamir Karbalaie
- Fernando Seoane

### Clinical Lead

- Olof Silfver

## License

(c) 2026 SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments), Karolinska Institutet.
