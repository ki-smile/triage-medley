"""HuggingFace Space — Multi-Model Medical Triage Inference Server.

Serves MedGemma 4B, Qwen 2.5-Med (Meditron3), and BioMistral 7B
behind a single Gradio API endpoint (/doctor_infer).

Deploy this file as the main app.py in a HuggingFace Space with GPU hardware.
"""

import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# --- Model Registry ---
MODELS = {
    "MedGemma 4B": {
        "id": "google/medgemma-4b-it",
        "model": None,
        "tokenizer": None,
    },
    "Qwen 2.5-Med": {
        "id": "OpenMeditron/Meditron3-Qwen2.5-7B",
        "model": None,
        "tokenizer": None,
    },
    "BioMistral 7B": {
        "id": "BioMistral/BioMistral-7B",
        "model": None,
        "tokenizer": None,
    },
}


def load_model(name: str):
    """Lazy-load a model and its tokenizer."""
    entry = MODELS[name]
    if entry["model"] is None:
        print(f"Loading {name} ({entry['id']})...")
        entry["tokenizer"] = AutoTokenizer.from_pretrained(
            entry["id"], token=HF_TOKEN, trust_remote_code=True
        )
        entry["model"] = AutoModelForCausalLM.from_pretrained(
            entry["id"],
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"{name} loaded.")
    return entry["model"], entry["tokenizer"]


def doctor_infer(model_choice: str, image, text_prompt: str) -> str:
    """Run inference on the selected model.

    Args:
        model_choice: One of "MedGemma 4B", "Qwen 2.5-Med", "BioMistral 7B"
        image: Optional uploaded image (unused for text-only models)
        text_prompt: The clinical prompt text

    Returns:
        Model's generated text response
    """
    if model_choice not in MODELS:
        return f"Unknown model: {model_choice}"

    model, tokenizer = load_model(model_choice)

    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# --- Gradio Interface ---
demo = gr.Interface(
    fn=doctor_infer,
    inputs=[
        gr.Radio(
            choices=list(MODELS.keys()),
            label="Model Choice",
            value="MedGemma 4B",
        ),
        gr.Image(type="filepath", label="Medical Image (optional)"),
        gr.Textbox(label="Clinical Prompt", lines=10),
    ],
    outputs=gr.Textbox(label="Model Response", lines=20),
    title="Triage-Medley Medical Triage Space",
    description="Multi-model medical inference endpoint for the Triage-Medley triage system.",
)

if __name__ == "__main__":
    demo.launch()
