# HuggingFace Integration Guide

> Detailed setup instructions for connecting Triage Medley to live HuggingFace models.  
> **Not required for demos** — the system runs with mock adapters by default.

---

## Adapter Modes

The system supports three modes of model inference. Adapter switching is entirely config-driven — no code changes required.

| Mode | Library | Config Value | Best For |
|------|---------|-------------|----------|
| **Mock** | N/A (local JSON) | `adapter: "mock"` | Development, demos, no API key needed |
| **Inference API** | `huggingface_hub.InferenceClient` | `adapter: "huggingface"` | Direct model access via HF infrastructure |
| **Space** | `gradio_client.Client` | `adapter: "space"` | Custom deployments with full control over GPU/config |

---

## Model Registry

| Model ID | HuggingFace Repo | Stages | Role |
|----------|-------------------|--------|------|
| `medgemma_4b` | `google/medgemma-4b-it` | Pre-triage + Triage + Differential + Management | Anchor model (fast, multimodal) |
| `medgemma_27b` | `google/medgemma-27b-text-it` | Triage + Differential + Management | Deep analysis (87.7% MedQA) [Disabled by default] |
| `meditron_7b` | `OpenMeditron/Meditron3-Qwen2.5-7B` | Triage + Differential | Medical specialized Qwen2.5-7B variant |
| `biomistral` | `BioMistral/BioMistral-7B` | Triage + Differential | Devil's advocate / don't-miss |

---

## Step 1: Create a HuggingFace Account and Token

1. Sign up at [huggingface.co](https://huggingface.co/join)
2. Go to **Settings** > **Access Tokens** ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
3. Create a new token with **Read** access (fine-grained: `Make calls to the serverless Inference API`)
4. Copy the token (starts with `hf_...`)

## Step 2: Request Model Access (Gated Models)

Some models require accepting a license agreement before use:

1. Visit each model page on HuggingFace and click **"Agree and access repository"** if prompted:
   - [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
   - [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it)
2. Community models (Meditron3-7B, BioMistral) are typically open access — no approval needed.

### Alternative: Provide API Key via Admin Panel

If `HF_TOKEN` is not set as an environment variable, admin users can provide a HuggingFace API key through the **Engine Config** page. Login as `admin` (PIN: `0000`) → Engine Config → enter your API key. The key is stored in the session only and not persisted.

## Step 3: Set the HF_TOKEN Environment Variable

```bash
# Linux / macOS
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Windows (PowerShell)
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Or add to .env / .bashrc for persistence
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' >> ~/.bashrc
```

## Step 4: Switch from Mock to Live Adapters

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

You can switch models individually — e.g. enable only `medgemma_4b` via Space while keeping the rest on mock. Both live modes use the same `hf_` token.

## Step 5: Test the Connection

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

---

## Troubleshooting

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
