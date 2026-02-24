"""Adapter factory — creates model adapters from config/models.yaml.

Reads the model registry and instantiates the correct adapter type
(mock, huggingface, or space) for each model. Adapter swapping is
config-driven: change ``adapter: "mock"`` to ``adapter: "huggingface"``
or ``adapter: "space"`` in models.yaml.
"""

import logging

from src.adapters.base import BaseAdapter
from src.adapters.mock_adapter import MockAdapter
from src.utils.config import load_models_config

logger = logging.getLogger(__name__)

# HuggingFace adapter class registry — maps model_id → HF adapter class.
# Lazy-imported to avoid pulling in huggingface_hub when using mock mode.
_HF_ADAPTER_CLASSES: dict[str, tuple[str, str]] = {
    "medgemma_4b": ("src.adapters.hf_medgemma", "MedGemma4BAdapter"),
    "claude_opus_4_5": ("src.adapters.hf_medgemma", "MedGemma27BAdapter"),
    "meditron_7b": ("src.adapters.hf_ensemble", "MeditronAdapter"),
    "biomistral": ("src.adapters.hf_ensemble", "BioMistralAdapter"),
}

# HuggingFace Space adapter class registry — maps model_id → Space adapter class.
# Lazy-imported to avoid pulling in gradio_client when using mock/HF mode.
_SPACE_ADAPTER_CLASSES: dict[str, tuple[str, str]] = {
    "medgemma_4b": ("src.adapters.space_medgemma", "SpaceMedGemma4BAdapter"),
    "meditron_7b": ("src.adapters.hf_ensemble", "SpaceMeditronAdapter"),
    "biomistral": ("src.adapters.hf_ensemble", "SpaceBioMistralAdapter"),
}


def create_adapter(model_id: str, adapter_type_override: str | None = None) -> BaseAdapter:
    """Create a single adapter by model ID from config."""
    config = load_models_config()
    models = config.get("models", {})

    if model_id not in models:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(models.keys())}")

    model_config = models[model_id].copy()
    if adapter_type_override:
        model_config["adapter"] = adapter_type_override
        
    return _build_adapter(model_id, model_config)


def create_all_adapters(adapter_type_override: str | None = None) -> dict[str, BaseAdapter]:
    """Create adapters for all models in the registry."""
    config = load_models_config()
    adapters = {}

    for model_id, model_config in config.get("models", {}).items():
        # Skip the RETTS rules engine — it's deterministic, not an adapter
        if model_config.get("adapter") == "deterministic":
            continue
        # Skip explicitly disabled models
        if model_config.get("enabled") is False:
            continue
        
        m_config = model_config.copy()
        if adapter_type_override:
            m_config["adapter"] = adapter_type_override
            
        adapters[model_id] = _build_adapter(model_id, m_config)

    return adapters


def create_stage_adapters(stage: str) -> dict[str, BaseAdapter]:
    """Create adapters for all models that support a given stage."""
    all_adapters = create_all_adapters()
    return {
        mid: adapter
        for mid, adapter in all_adapters.items()
        if stage in adapter.supported_stages
    }


def _build_adapter(model_id: str, model_config: dict) -> BaseAdapter:
    """Build a single adapter from its config entry."""
    # Check for Streamlit session state overrides (allows UI-driven configuration)
    try:
        import streamlit as st
        # Check for per-model overrides
        s_adapter = st.session_state.get(f"adapter_type_{model_id}")
        if s_adapter:
            model_config["adapter"] = s_adapter
            
        s_space = st.session_state.get(f"space_id_{model_id}")
        if s_space:
            model_config["space_id"] = s_space
            
        # Check for global space override
        global_space = st.session_state.get("hf_space_id")
        if global_space and model_config.get("adapter") == "space":
            model_config["space_id"] = global_space
    except (ImportError, RuntimeError, TypeError):
        # Not running in Streamlit context or streamlit import failed (protobuf conflict)
        pass

    adapter_type = model_config.get("adapter", "mock")
    model_name = model_config.get("name", model_id)
    stages = model_config.get("stages", [])

    if adapter_type == "mock":
        return MockAdapter(model_id, model_name, stages)
    elif adapter_type == "huggingface":
        return _build_hf_adapter(model_id, model_name, stages, model_config)
    elif adapter_type == "space":
        return _build_space_adapter(model_id, model_name, stages, model_config)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


def _build_hf_adapter(
    model_id: str,
    model_name: str,
    stages: list[str],
    model_config: dict,
) -> BaseAdapter:
    """Build a HuggingFace adapter using lazy-imported class."""
    if model_id not in _HF_ADAPTER_CLASSES:
        raise ValueError(
            f"No HuggingFace adapter registered for model '{model_id}'. "
            f"Available: {list(_HF_ADAPTER_CLASSES.keys())}"
        )

    module_path, class_name = _HF_ADAPTER_CLASSES[model_id]

    import importlib
    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)

    hf_model_id = model_config.get("hf_id", "")
    timeout = model_config.get("timeout_seconds", 30)

    logger.info("Creating HF adapter: %s (%s) via %s", model_id, hf_model_id, class_name)

    return adapter_class(
        model_id=model_id,
        model_name=model_name,
        supported_stages=stages,
        hf_model_id=hf_model_id,
        timeout_seconds=timeout,
    )


def _build_space_adapter(
    model_id: str,
    model_name: str,
    stages: list[str],
    model_config: dict,
) -> BaseAdapter:
    """Build a HuggingFace Space adapter using lazy-imported class."""
    if model_id not in _SPACE_ADAPTER_CLASSES:
        raise ValueError(
            f"No Space adapter registered for model '{model_id}'. "
            f"Available: {list(_SPACE_ADAPTER_CLASSES.keys())}"
        )

    module_path, class_name = _SPACE_ADAPTER_CLASSES[model_id]

    import importlib
    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)

    hf_model_id = model_config.get("hf_id", "")
    space_id = model_config.get("space_id", "")
    api_name = model_config.get("api_name", "/predict")
    timeout = model_config.get("timeout_seconds", 30)

    if not space_id:
        raise ValueError(
            f"Model '{model_id}' uses adapter 'space' but has no 'space_id' "
            f"in config. Add space_id to config/models.yaml."
        )

    logger.info("Creating Space adapter: %s (%s) via %s", model_id, space_id, class_name)

    return adapter_class(
        model_id=model_id,
        model_name=model_name,
        supported_stages=stages,
        hf_model_id=hf_model_id,
        space_id=space_id,
        api_name=api_name,
        timeout_seconds=timeout,
    )
