"""HuggingFace adapter base — shared inference logic for all HF models.

Uses huggingface_hub InferenceClient for chat completions.
Concrete adapters (MedGemma, Qwen-Med, etc.) inherit and override
model-specific response parsing if needed.
"""

import json
import logging
import os
import re
import time
from typing import Optional

from huggingface_hub import InferenceClient

from src.adapters.base import BaseAdapter
from src.adapters.prompt_builder import (
    build_differential_prompt,
    build_management_prompt,
    build_pretriage_prompt,
    build_triage_prompt,
)
from src.models.context import FullTriageContext, PreTriageContext
from src.models.enums import Confidence, QueuePriority, RETTSLevel
from src.models.outputs import (
    DifferentialCandidate,
    DifferentialOutput,
    ManagementOutput,
    PreTriageOutput,
    TriageOutput,
)

logger = logging.getLogger(__name__)


def _resolve_hf_token() -> str | None:
    """Resolve HF token: env var takes priority, then session state."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        import streamlit as st
        return st.session_state.get("hf_api_key")
    except Exception:
        return None


class HFBaseAdapter(BaseAdapter):
    """Base adapter for HuggingFace Inference API models.

    Handles:
    - InferenceClient initialization with token
    - Chat completion calls with timeout
    - JSON response extraction from model output
    - Fallback parsing for non-JSON outputs

    Subclasses override parse methods for model-specific quirks.
    """

    def __init__(
        self,
        model_id: str,
        model_name: str,
        supported_stages: list[str],
        hf_model_id: str,
        timeout_seconds: int = 30,
        max_tokens: int = 2048,
    ):
        super().__init__(model_id, model_name, supported_stages)
        self._hf_model_id = hf_model_id
        self._timeout_seconds = timeout_seconds
        self._max_tokens = max_tokens
        self._client: Optional[InferenceClient] = None
        self._current_token: Optional[str] = None

    @property
    def hf_model_id(self) -> str:
        return self._hf_model_id

    def _get_client(self) -> InferenceClient:
        """Lazily initialize the InferenceClient.

        Re-creates client if the resolved token has changed (e.g. admin
        entered a new key via Engine Config).
        """
        token = _resolve_hf_token()
        if not token:
            raise RuntimeError(
                "HF_TOKEN not set. Set HF_TOKEN env var or provide "
                "via Engine Config admin panel."
            )
        # Re-create client if token changed
        if self._client is None or token != self._current_token:
            self._client = InferenceClient(
                model=self._hf_model_id,
                token=token,
                timeout=self._timeout_seconds,
            )
            self._current_token = token
        return self._client

    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Send messages to the HF Inference API and return raw text response."""
        client = self._get_client()
        response = client.chat_completion(
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content

    # ---- Stage implementations ----

    def pretriage(self, context: PreTriageContext) -> PreTriageOutput:
        if "pretriage" not in self.supported_stages:
            raise NotImplementedError(f"{self.model_id} does not support pretriage")

        start = time.monotonic()
        messages = build_pretriage_prompt(context)
        raw = self._chat_completion(messages)
        elapsed = int((time.monotonic() - start) * 1000)

        return self._parse_pretriage(raw, elapsed)

    def triage(self, context: FullTriageContext) -> TriageOutput:
        if "triage" not in self.supported_stages:
            raise NotImplementedError(f"{self.model_id} does not support triage")

        start = time.monotonic()
        messages = build_triage_prompt(context, self.model_id)
        raw = self._chat_completion(messages)
        elapsed = int((time.monotonic() - start) * 1000)

        return self._parse_triage(raw, elapsed)

    def differential(self, context: FullTriageContext) -> DifferentialOutput:
        if "differential" not in self.supported_stages:
            raise NotImplementedError(f"{self.model_id} does not support differential")

        start = time.monotonic()
        messages = build_differential_prompt(context, self.model_id)
        raw = self._chat_completion(messages)
        elapsed = int((time.monotonic() - start) * 1000)

        return self._parse_differential(raw, elapsed)

    def management(self, context: FullTriageContext) -> ManagementOutput:
        if "management" not in self.supported_stages:
            raise NotImplementedError(f"{self.model_id} does not support management")

        start = time.monotonic()
        messages = build_management_prompt(context, self.model_id)
        raw = self._chat_completion(messages)
        elapsed = int((time.monotonic() - start) * 1000)

        return self._parse_management(raw, elapsed)

    # ---- Response parsers (overridable by subclasses) ----

    def _parse_pretriage(self, raw: str, elapsed_ms: int) -> PreTriageOutput:
        """Parse pre-triage response. Override for model-specific formats."""
        data = _extract_json(raw)
        if not data:
            logger.info("Falling back to text extraction for pretriage (%s)", self.model_id)
            data = _extract_pretriage_from_text(raw)

        priority_str = _get_str(data, "queue_priority", "MODERATE").upper()
        priority = _safe_enum(QueuePriority, priority_str, QueuePriority.MODERATE)

        return PreTriageOutput(
            model_id=self.model_id,
            queue_priority=priority,
            chief_complaint=_get_str(data, "chief_complaint", ""),
            reasoning=_get_str(data, "reasoning", raw[:500]),
            ess_category_hint=data.get("ess_category_hint"),
            risk_amplifiers_detected=_get_list(data, "risk_amplifiers_detected"),
            processing_time_ms=elapsed_ms,
        )

    def _parse_triage(self, raw: str, elapsed_ms: int) -> TriageOutput:
        """Parse triage response. Override for model-specific formats."""
        data = _extract_json(raw)
        if not data:
            logger.info("Falling back to text extraction for triage (%s)", self.model_id)
            data = _extract_triage_from_text(raw)

        retts_str = _get_str(data, "retts_level", "YELLOW").upper()
        retts = _safe_enum(RETTSLevel, retts_str, RETTSLevel.YELLOW)
        confidence_str = _get_str(data, "confidence", "MODERATE").upper()
        confidence = _safe_enum(Confidence, confidence_str, Confidence.MODERATE)

        return TriageOutput(
            model_id=self.model_id,
            retts_level=retts,
            ess_category=data.get("ess_category"),
            chief_complaint=_get_str(data, "chief_complaint", ""),
            clinical_reasoning=_get_str(data, "clinical_reasoning", raw[:500]),
            vital_sign_concerns=_get_list(data, "vital_sign_concerns"),
            risk_factors=_get_list(data, "risk_factors"),
            confidence=confidence,
            dont_miss=_get_list(data, "dont_miss"),
            processing_time_ms=elapsed_ms,
        )

    def _parse_differential(self, raw: str, elapsed_ms: int) -> DifferentialOutput:
        """Parse differential diagnosis response."""
        data = _extract_json(raw)
        if not data:
            logger.info("Falling back to text extraction for differential (%s)", self.model_id)
            data = _extract_differential_from_text(raw)

        candidates = []
        for c in _get_list(data, "candidates"):
            if isinstance(c, dict):
                prob = c.get("probability")
                if isinstance(prob, str):
                    try:
                        prob = float(prob)
                    except (ValueError, TypeError):
                        prob = None
                candidates.append(DifferentialCandidate(
                    diagnosis=c.get("diagnosis", "Unknown"),
                    probability=prob,
                    supporting_evidence=c.get("supporting_evidence", []),
                    is_dont_miss=c.get("is_dont_miss", False),
                ))

        confidence_str = _get_str(data, "confidence", "MODERATE").upper()
        confidence = _safe_enum(Confidence, confidence_str, Confidence.MODERATE)

        return DifferentialOutput(
            model_id=self.model_id,
            candidates=candidates,
            reasoning=_get_str(data, "reasoning", raw[:500]),
            confidence=confidence,
            processing_time_ms=elapsed_ms,
        )

    def _parse_management(self, raw: str, elapsed_ms: int) -> ManagementOutput:
        """Parse management plan response."""
        data = _extract_json(raw)
        if not data:
            logger.info("Falling back to text extraction for management (%s)", self.model_id)
            data = _extract_management_from_text(raw)

        confidence_str = _get_str(data, "confidence", "MODERATE").upper()
        confidence = _safe_enum(Confidence, confidence_str, Confidence.MODERATE)

        return ManagementOutput(
            model_id=self.model_id,
            investigations=_get_list(data, "investigations"),
            imaging=_get_list(data, "imaging"),
            medications=_get_list(data, "medications"),
            disposition=_get_str(data, "disposition", "observation"),
            contraindications_flagged=_get_list(data, "contraindications_flagged"),
            reasoning=_get_str(data, "reasoning", raw[:500]),
            confidence=confidence,
            processing_time_ms=elapsed_ms,
        )


# ---- JSON extraction helpers ----

def _extract_json(text: str) -> dict:
    """Extract JSON from model response text.

    Models in Spaces often return [Prompt] + [Generation].
    We must strip structural hints from the prompt to avoid parsing example JSON.
    """
    if not text or not isinstance(text, str):
        return {}

    # 1. Aggressive cleaning: Remove known prompt structural hints
    # This prevents parsing the 'JSON structure: {"retts_level": "..."}' part of the prompt
    clean_text = text
    if 'JSON structure:' in clean_text:
        # Keep everything AFTER the structural hint
        idx = clean_text.rfind('JSON structure:')
        # If there's a lot of text after it, it's likely the generation
        if len(clean_text) - idx > 100:
            clean_text = clean_text[idx + 50:]

    # 2. Try parsing the whole response as JSON
    try:
        return json.loads(clean_text.strip())
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. Try extracting from code blocks - find ALL code blocks
    code_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", clean_text, re.DOTALL)
    for block in reversed(code_blocks): # Try latest block first
        if '...' in block: continue # Skip structural hints
        try:
            return json.loads(block.strip())
        except (json.JSONDecodeError, TypeError):
            continue

    # 3b. Handle TRUNCATED code blocks (opening ``` but no closing ```)
    # This happens when Space models hit max_new_tokens limit
    trunc_match = re.search(r"```(?:json)?\s*\n?(.*)", clean_text, re.DOTALL)
    if trunc_match:
        truncated = trunc_match.group(1).strip()
        if truncated and '{' in truncated:
            repaired = _repair_truncated_json(truncated)
            if repaired:
                return repaired

    # 4. Fallback: Find all potential { ... } structures
    potential_objects = []
    stack = []
    start_idx = -1
    
    for i, char in enumerate(clean_text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    potential_objects.append(clean_text[start_idx:i+1])
    
    # Try parsing from newest to oldest
    for obj_str in reversed(potential_objects):
        if '...' in obj_str: continue # Skip structural hints
        try:
            parsed = json.loads(obj_str)
            if isinstance(parsed, dict) and len(parsed) > 0:
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue

    logger.warning("Could not extract JSON from model response (length %d). First 300 chars: %s",
                   len(text), text[:300])
    return {}


def _repair_truncated_json(text: str) -> dict:
    """Attempt to repair truncated JSON by closing unclosed braces/brackets.

    When models hit max_new_tokens, JSON gets cut off mid-stream.
    Strategy: find the last comma after a complete JSON value, truncate there,
    then close all unclosed structures in correct stack order.
    """
    idx = text.find('{')
    if idx < 0:
        return {}

    fragment = text[idx:]

    # Find all comma positions that follow a complete JSON value
    safe_commas = [
        m.end() - 1
        for m in re.finditer(r'(?:"|true|false|null|\d|\]|\})\s*,', fragment)
    ]

    for comma_pos in reversed(safe_commas):
        candidate = fragment[:comma_pos].rstrip().rstrip(',')

        # Build closing sequence using a stack (order matters for nested JSON)
        stack = []
        in_string = False
        escape = False
        for ch in candidate:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch in '}]' and stack:
                stack.pop()

        if not stack:
            continue

        # Close in reverse stack order (innermost first)
        closing = ''.join(reversed(stack))
        closed = candidate + closing

        try:
            parsed = json.loads(closed)
            if isinstance(parsed, dict) and len(parsed) > 0:
                logger.info("Repaired truncated JSON (%d keys recovered)", len(parsed))
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue

    return {}


def _extract_triage_from_text(text: str) -> dict:
    """Extract triage fields from free-text response using keyword matching."""
    data = {}
    t = text.upper()

    # RETTS level
    for level in ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE"]:
        if re.search(rf'\b{level}\b', t):
            data["retts_level"] = level
            break

    # Confidence
    for conf in ["HIGH", "MODERATE", "LOW"]:
        if re.search(rf'\bCONFIDENCE[:\s]*{conf}\b', t) or re.search(rf'\b{conf}\s+CONFIDENCE\b', t):
            data["confidence"] = conf
            break

    # Use the full text as clinical reasoning
    data["clinical_reasoning"] = text.strip()[:1000]

    # Extract don't-miss items
    dont_miss = []
    for pattern in [r"don'?t[- ]miss[:\s]*([^\n]+)", r"cannot miss[:\s]*([^\n]+)",
                    r"must not miss[:\s]*([^\n]+)", r"rule out[:\s]*([^\n]+)"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dont_miss.extend(m.strip() for m in matches)
    if dont_miss:
        data["dont_miss"] = dont_miss

    return data


def _extract_differential_from_text(text: str) -> dict:
    """Extract differential diagnosis from free-text response."""
    data = {"candidates": [], "reasoning": text.strip()[:1000]}

    # Look for numbered diagnoses like "1. Diagnosis name" or "- Diagnosis name"
    diag_patterns = re.findall(
        r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-•]\s*)([A-Z][A-Za-z\s/\-]+?)(?:\s*[-–:(\n])',
        text
    )
    seen = set()
    for diag in diag_patterns:
        diag = diag.strip().rstrip(':-')
        if len(diag) > 3 and diag not in seen:
            seen.add(diag)
            is_dm = bool(re.search(r"don'?t[- ]miss|cannot miss|must not miss|rule out",
                                   text[max(0, text.find(diag)-50):text.find(diag)+len(diag)+50],
                                   re.IGNORECASE))
            data["candidates"].append({
                "diagnosis": diag,
                "probability": None,
                "supporting_evidence": [],
                "is_dont_miss": is_dm,
            })

    # Confidence
    for conf in ["HIGH", "MODERATE", "LOW"]:
        if re.search(rf'\b{conf}\b', text.upper()):
            data["confidence"] = conf
            break

    return data


def _extract_management_from_text(text: str) -> dict:
    """Extract management plan from free-text response."""
    data = {"reasoning": text.strip()[:1000]}
    t_lower = text.lower()

    # Disposition
    for disp in ["icu", "admission", "observation", "discharge"]:
        if disp in t_lower:
            data["disposition"] = disp
            break

    # Try to extract list items after common headers
    for field, keywords in [
        ("investigations", ["lab", "investigation", "blood test", "CBC", "CRP", "troponin", "BMP"]),
        ("imaging", ["x-ray", "xray", "CT", "MRI", "ultrasound", "imaging", "radiograph"]),
        ("medications", ["medication", "drug", "prescri", "administer", "paracetamol", "morphine"]),
    ]:
        items = []
        for kw in keywords:
            if kw.lower() in t_lower:
                # Find the line containing the keyword and grab it
                for line in text.split('\n'):
                    if kw.lower() in line.lower() and len(line.strip()) > 5:
                        items.append(line.strip().lstrip('-•*0123456789.) '))
                        break
        if items:
            data[field] = items

    return data


def _extract_pretriage_from_text(text: str) -> dict:
    """Extract pretriage fields from free-text response."""
    data = {"reasoning": text.strip()[:500]}
    t = text.upper()

    for priority in ["HIGH", "MODERATE", "STANDARD"]:
        if re.search(rf'\b{priority}\b', t):
            data["queue_priority"] = priority
            break

    return data


def _get_str(data: dict, key: str, default: str = "") -> str:
    """Safely get a string value from parsed data."""
    val = data.get(key, default)
    return str(val) if val is not None else default


def _get_list(data: dict, key: str) -> list:
    """Safely get a list value from parsed data."""
    val = data.get(key, [])
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [item.strip() for item in val.split(",") if item.strip()]
    return []


def _safe_enum(enum_class, value: str, default):
    """Safely convert a string to an enum, returning default on failure."""
    try:
        return enum_class(value)
    except (ValueError, KeyError):
        logger.warning("Unknown %s value '%s', using default %s",
                       enum_class.__name__, value, default)
        return default
