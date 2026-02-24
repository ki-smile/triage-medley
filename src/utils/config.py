"""Configuration loader for YAML and JSON config files."""

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} references in config strings."""
    if isinstance(obj, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(i) for i in obj]
    return obj

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

_cache: dict[str, Any] = {}


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(relative_path: str, *, use_cache: bool = True) -> dict:
    """Load a config file relative to the config/ directory.

    Supports .yaml, .yml, and .json files.
    Results are cached by default.
    """
    if use_cache and relative_path in _cache:
        return _cache[relative_path]

    path = _CONFIG_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = _load_yaml(path)
    elif suffix == ".json":
        data = _load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")

    data = _expand_env_vars(data)

    if use_cache:
        _cache[relative_path] = data
    return data


def load_models_config() -> dict:
    return load_config("models.yaml")


def load_pretriage_config() -> dict:
    return load_config("pretriage.yaml")


def load_prompt_template(stage: str) -> dict:
    return load_config(f"prompts/{stage}.yaml")


def load_ess_codes() -> dict:
    return load_config("retts/ess_codes.json")


def load_vitals_cutoffs() -> dict:
    return load_config("retts/vitals_cutoffs.json")


def clear_cache() -> None:
    _cache.clear()


def get_project_root() -> Path:
    return _PROJECT_ROOT
