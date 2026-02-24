"""MedGemma Space adapter — MedGemma 4B via Gradio Space.

Wraps MedGemma 4B deployed at a HuggingFace Space.
Same stages and behaviour as MedGemma4BAdapter but calls a Space
instead of the Inference API.
"""

from src.adapters.space_base import SpaceBaseAdapter


class SpaceMedGemma4BAdapter(SpaceBaseAdapter):
    """MedGemma 4B via HuggingFace Space (Gradio /predict endpoint)."""

    def __init__(
        self,
        model_id: str = "medgemma_4b",
        model_name: str = "MedGemma 4B (Space)",
        supported_stages: list[str] | None = None,
        hf_model_id: str = "google/medgemma-4b-it",
        space_id: str = "",
        api_name: str = "/doctor_infer",
        timeout_seconds: int = 30,
    ):
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            supported_stages=supported_stages or [
                "pretriage", "triage", "differential", "management"
            ],
            hf_model_id=hf_model_id,
            space_id=space_id,
            api_name=api_name,
            timeout_seconds=timeout_seconds,
            max_tokens=2048,
            space_model_name="MedGemma 4B",
        )
