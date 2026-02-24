# HuggingFace Space Setup Guide

This folder contains the files needed to deploy the **Triage-Medley Triage Space** on HuggingFace.

## Overview

The Space hosts three medical LLMs behind a single Gradio API:
- **MedGemma 4B** (`google/medgemma-4b-it`) — anchor model
- **Qwen 2.5-Med** / Meditron3 (`OpenMeditron/Meditron3-Qwen2.5-7B`) — medical Qwen variant
- **BioMistral 7B** (`BioMistral/BioMistral-7B`) — diversity model

The main Triage-Medley app calls this Space via `gradio_client` for inference.

---

## Step 1: Create a HuggingFace Account & Token

1. Go to [huggingface.co](https://huggingface.co) and create an account
2. Navigate to **Settings > Access Tokens** ([direct link](https://huggingface.co/settings/tokens))
3. Click **New token** and create a token with:
   - **Type**: Fine-grained
   - **Permissions**: at minimum `read` access to models
   - **Name**: e.g. `triage-medley`
4. Copy the token (starts with `hf_...`)

> **Important**: Never commit your token to git. Use environment variables.

---

## Step 2: Accept Model Licenses

Some models require accepting their license on HuggingFace before use:

| Model | License Page |
|-------|-------------|
| MedGemma 4B | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) |
| Meditron3 | [OpenMeditron/Meditron3-Qwen2.5-7B](https://huggingface.co/OpenMeditron/Meditron3-Qwen2.5-7B) |
| BioMistral | [BioMistral/BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B) |

Visit each page and click **"Agree and access repository"** if prompted.

---

## Step 3: Create the HuggingFace Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Configure:
   - **Name**: e.g. `triage`
   - **SDK**: Gradio
   - **Hardware**: GPU (L40S recommended for 3 models)
   - **Visibility**: Private (recommended)
3. Upload the files from this folder:
   - `app.py` — Gradio inference server
   - `requirements.txt` — Python dependencies
4. Add your HuggingFace token as a Space Secret:
   - Go to **Settings > Variables and Secrets**
   - Add secret: `HF_TOKEN` = your token from Step 1

---

## Step 4: Configure the Triage-Medley App

Set these environment variables for the main Streamlit app (either in `.env` or your hosting platform):

```bash
# Your HuggingFace token
export HF_TOKEN=hf_your_token_here

# Your Space ID (format: username/space-name)
export HF_SPACE_ID=your-username/triage
```

### Option A: Local Development (`.env` file)

Edit the `.env` file in the project root:

```
HF_TOKEN=hf_your_token_here
HF_SPACE_ID=your-username/triage
```

### Option B: DigitalOcean App Platform

In the App Settings > Environment Variables, add:
- `HF_TOKEN` = your token
- `HF_SPACE_ID` = your Space ID

### Option C: Streamlit Cloud

In the app dashboard > Settings > Secrets:

```toml
HF_TOKEN = "hf_your_token_here"
HF_SPACE_ID = "your-username/triage"
```

### Option D: Admin UI (Runtime)

The **Engine Config** page in the app allows entering the HF token and Space ID at runtime without environment variables.

---

## Step 5: Verify

Run the Streamlit app and navigate to **Engine Config** page:

1. The HF API Key field should show your token (from env or manual entry)
2. Click **Test Space Connection** to verify the Space is reachable
3. Navigate to **Queue View** and run a patient through the pipeline

---

## Space API Reference

The Space exposes a single endpoint:

```
POST /doctor_infer

Parameters:
  - model_choice (str): "MedGemma 4B" | "Qwen 2.5-Med" | "BioMistral 7B"
  - image (file | None): Optional medical image
  - text_prompt (str): The clinical prompt text

Returns:
  - str: Model's text response
```

### Test with Python:

```python
from gradio_client import Client
import os

client = Client(
    os.environ["HF_SPACE_ID"],
    token=os.environ["HF_TOKEN"]
)

result = client.predict(
    "MedGemma 4B",       # model_choice
    None,                 # image (optional)
    "What is the triage level for a 68-year-old male with chest pain?",
    api_name="/doctor_infer"
)
print(result)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `401 Unauthorized` | Token is invalid or expired. Generate a new one. |
| `403 Forbidden` | You haven't accepted the model license. See Step 2. |
| Space is "Sleeping" | GPU Spaces sleep after inactivity. First request wakes it (~30s). |
| `Connection refused` | Space may be building. Check the Space Logs tab on HuggingFace. |
| Timeout errors | Increase `timeout_seconds` in `config/models.yaml` (default: 180s). |
