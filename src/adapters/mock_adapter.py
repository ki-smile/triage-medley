"""Mock model adapter — loads pre-computed JSON responses from data/scenarios/.

Used during Sprint 1-3 development. Swappable to HF adapter via config/models.yaml.
"""

import json
import time
from pathlib import Path
from typing import Optional

from src.adapters.base import BaseAdapter
from src.models.context import FullTriageContext, PreTriageContext
from src.models.enums import Confidence, QueuePriority, RETTSLevel
from src.models.outputs import (
    DifferentialCandidate,
    DifferentialOutput,
    ManagementOutput,
    PreTriageOutput,
    TriageOutput,
)
from src.utils.config import get_project_root

_SCENARIOS_DIR = get_project_root() / "data" / "scenarios"


class MockAdapter(BaseAdapter):
    """Mock adapter that loads responses from JSON files.

    File lookup: data/scenarios/{patient_id}/{stage}/{model_id}.json
    Falls back to a default response if no file exists.
    """

    def __init__(self, model_id: str, model_name: str, supported_stages: list[str]):
        super().__init__(model_id, model_name, supported_stages)

    def pretriage(self, context: PreTriageContext) -> PreTriageOutput:
        if "pretriage" not in self.supported_stages:
            raise NotImplementedError(f"{self.model_id} does not support pretriage")

        data = self._load_scenario(context.patient_id, "pretriage")
        if data:
            return PreTriageOutput(
                model_id=self.model_id,
                queue_priority=QueuePriority(data["queue_priority"]),
                chief_complaint=data.get("chief_complaint", ""),
                reasoning=data.get("reasoning", ""),
                ess_category_hint=data.get("ess_category_hint"),
                risk_amplifiers_detected=data.get("risk_amplifiers_detected", []),
                processing_time_ms=data.get("processing_time_ms", 0),
            )

        return self._default_pretriage(context)

    def triage(self, context: FullTriageContext) -> TriageOutput:
        if "triage" not in self.supported_stages:
            raise NotImplementedError(f"{self.model_id} does not support triage")

        data = self._load_scenario(context.patient_id, "triage")
        if data:
            return TriageOutput(
                model_id=self.model_id,
                retts_level=RETTSLevel(data["retts_level"]),
                ess_category=data.get("ess_category"),
                chief_complaint=data.get("chief_complaint", ""),
                clinical_reasoning=data.get("clinical_reasoning", ""),
                vital_sign_concerns=data.get("vital_sign_concerns", []),
                risk_factors=data.get("risk_factors", []),
                confidence=Confidence(data.get("confidence", "MODERATE")),
                dont_miss=data.get("dont_miss", []),
                processing_time_ms=data.get("processing_time_ms", 0),
            )

        return self._default_triage(context)

    def differential(self, context: FullTriageContext) -> DifferentialOutput:
        if "differential" not in self.supported_stages:
            raise NotImplementedError(f"{self.model_id} does not support differential")

        data = self._load_scenario(context.patient_id, "differential")
        if data:
            candidates = [
                DifferentialCandidate(
                    diagnosis=c["diagnosis"],
                    probability=c.get("probability"),
                    supporting_evidence=c.get("supporting_evidence", []),
                    is_dont_miss=c.get("is_dont_miss", False),
                )
                for c in data.get("candidates", [])
            ]
            return DifferentialOutput(
                model_id=self.model_id,
                candidates=candidates,
                reasoning=data.get("reasoning", ""),
                confidence=Confidence(data.get("confidence", "MODERATE")),
                processing_time_ms=data.get("processing_time_ms", 0),
            )

        return self._default_differential(context)

    def management(self, context: FullTriageContext) -> ManagementOutput:
        if "management" not in self.supported_stages:
            raise NotImplementedError(f"{self.model_id} does not support management")

        data = self._load_scenario(context.patient_id, "management")
        if data:
            return ManagementOutput(
                model_id=self.model_id,
                investigations=data.get("investigations", []),
                imaging=data.get("imaging", []),
                medications=data.get("medications", []),
                disposition=data.get("disposition", "observation"),
                contraindications_flagged=data.get("contraindications_flagged", []),
                reasoning=data.get("reasoning", ""),
                confidence=Confidence(data.get("confidence", "MODERATE")),
                processing_time_ms=data.get("processing_time_ms", 0),
            )

        return self._default_management(context)

    # ---- File loading ----

    def _load_scenario(self, patient_id: str, stage: str) -> Optional[dict]:
        """Load mock JSON for a specific patient + stage + model."""
        path = _SCENARIOS_DIR / patient_id / stage / f"{self.model_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- Default fallbacks for unknown patients ----

    def _default_pretriage(self, context: PreTriageContext) -> PreTriageOutput:
        return PreTriageOutput(
            model_id=self.model_id,
            queue_priority=QueuePriority.MODERATE,
            chief_complaint=context.speech_text[:100],
            reasoning=f"Mock default: no scenario data for patient {context.patient_id}",
            processing_time_ms=100,
        )

    def _default_triage(self, context: FullTriageContext) -> TriageOutput:
        return TriageOutput(
            model_id=self.model_id,
            retts_level=RETTSLevel.YELLOW,
            chief_complaint=context.speech_text[:100],
            clinical_reasoning=f"Mock default: no scenario data for patient {context.patient_id}",
            confidence=Confidence.LOW,
            processing_time_ms=100,
        )

    def _default_differential(self, context: FullTriageContext) -> DifferentialOutput:
        return DifferentialOutput(
            model_id=self.model_id,
            candidates=[],
            reasoning=f"Mock default: no scenario data for patient {context.patient_id}",
            confidence=Confidence.LOW,
            processing_time_ms=100,
        )

    def _default_management(self, context: FullTriageContext) -> ManagementOutput:
        return ManagementOutput(
            model_id=self.model_id,
            reasoning=f"Mock default: no scenario data for patient {context.patient_id}",
            confidence=Confidence.LOW,
            processing_time_ms=100,
        )
