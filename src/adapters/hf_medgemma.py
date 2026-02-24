"""MedGemma HuggingFace adapters — 4B and 27B variants.

MedGemma 4B: Anchor model, fast, runs in both Stage A and B.
MedGemma 27B: Deep analysis (87.7% MedQA), Stage B only.
"""

from src.adapters.hf_base import HFBaseAdapter


class MedGemma4BAdapter(HFBaseAdapter):
    """MedGemma 4B — fast multimodal anchor model.

    Runs in both Stage A (pretriage) and Stage B (triage, differential, management).
    Uses google/medgemma-4b-it via HuggingFace Inference API.
    """

    def __init__(
        self,
        model_id: str = "medgemma_4b",
        model_name: str = "MedGemma 4B",
        supported_stages: list[str] | None = None,
        hf_model_id: str = "google/medgemma-4b-it",
        timeout_seconds: int = 10,
    ):
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            supported_stages=supported_stages or [
                "pretriage", "triage", "differential", "management"
            ],
            hf_model_id=hf_model_id,
            timeout_seconds=timeout_seconds,
            max_tokens=2048,
        )


class MedGemma27BAdapter(HFBaseAdapter):
    """MedGemma 27B — deep analysis model, Stage B only.

    Text-only model with 87.7% MedQA accuracy. Too slow for pretriage.
    Uses google/medgemma-27b-text-it via HuggingFace Inference API.
    """

    def __init__(
        self,
        model_id: str = "medgemma_27b",
        model_name: str = "MedGemma 27B",
        supported_stages: list[str] | None = None,
        hf_model_id: str = "google/medgemma-27b-text-it",
        timeout_seconds: int = 30,
    ):
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            supported_stages=supported_stages or [
                "triage", "differential", "management"
            ],
            hf_model_id=hf_model_id,
            timeout_seconds=timeout_seconds,
            max_tokens=4096,
        )
