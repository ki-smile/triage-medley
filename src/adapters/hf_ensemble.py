"""Ensemble diversity HuggingFace adapters — Meditron, BioMistral.

These models provide training diversity for the MEDLEY ensemble:
- Meditron3 7B: Medical specialized Qwen2.5-7B variant (OpenMeditron)
- BioMistral: Devil's advocate / don't-miss flagging
"""

from src.adapters.hf_base import HFBaseAdapter
from src.adapters.space_base import SpaceBaseAdapter


class MeditronAdapter(HFBaseAdapter):
    """Meditron3 7B — Medical specialized Qwen2.5-7B variant.

    Stage B only: triage + differential.
    Uses OpenMeditron/Meditron3-Qwen2.5-7B via HuggingFace Inference API.
    """

    def __init__(
        self,
        model_id: str = "meditron_7b",
        model_name: str = "Meditron3 7B",
        supported_stages: list[str] | None = None,
        hf_model_id: str = "OpenMeditron/Meditron3-Qwen2.5-7B",
        timeout_seconds: int = 15,
    ):
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            supported_stages=supported_stages or ["triage", "differential"],
            hf_model_id=hf_model_id,
            timeout_seconds=timeout_seconds,
            max_tokens=2048,
        )


class SpaceMeditronAdapter(SpaceBaseAdapter):
    """Meditron3 7B via HuggingFace Space."""

    def __init__(
        self,
        model_id: str = "meditron_7b",
        model_name: str = "Meditron3 7B (Space)",
        supported_stages: list[str] | None = None,
        hf_model_id: str = "OpenMeditron/Meditron3-Qwen2.5-7B",
        space_id: str = "",
        api_name: str = "/doctor_infer",
        timeout_seconds: int = 30,
    ):
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            supported_stages=supported_stages or ["triage", "differential"],
            hf_model_id=hf_model_id,
            space_id=space_id,
            api_name=api_name,
            timeout_seconds=timeout_seconds,
            max_tokens=2048,
            space_model_name="Qwen 2.5-Med"
        )


class BioMistralAdapter(HFBaseAdapter):
    """BioMistral — devil's advocate for don't-miss diagnosis flagging.

    Stage B only: triage + differential.
    Uses BioMistral/BioMistral-7B via HuggingFace Inference API.
    """

    def __init__(
        self,
        model_id: str = "biomistral",
        model_name: str = "BioMistral",
        supported_stages: list[str] | None = None,
        hf_model_id: str = "BioMistral/BioMistral-7B",
        timeout_seconds: int = 15,
    ):
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            supported_stages=supported_stages or ["triage", "differential"],
            hf_model_id=hf_model_id,
            timeout_seconds=timeout_seconds,
            max_tokens=2048,
        )


class SpaceBioMistralAdapter(SpaceBaseAdapter):
    """BioMistral via HuggingFace Space."""

    def __init__(
        self,
        model_id: str = "biomistral",
        model_name: str = "BioMistral (Space)",
        supported_stages: list[str] | None = None,
        hf_model_id: str = "BioMistral/BioMistral-7B",
        space_id: str = "",
        api_name: str = "/doctor_infer",
        timeout_seconds: int = 30,
    ):
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            supported_stages=supported_stages or ["triage", "differential"],
            hf_model_id=hf_model_id,
            space_id=space_id,
            api_name=api_name,
            timeout_seconds=timeout_seconds,
            max_tokens=2048,
            space_model_name="BioMistral 7B"
        )