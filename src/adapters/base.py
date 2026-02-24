"""ModelAdapter protocol — common interface for all model adapters.

Every AI model (mock or real) implements this protocol. The orchestrator
dispatches to adapters without knowing whether they are mock or HuggingFace.
Swapping mock → HF is a config change in models.yaml, not a code change.
"""

from typing import Protocol, runtime_checkable

from src.models.context import FullTriageContext, PreTriageContext
from src.models.outputs import (
    DifferentialOutput,
    ManagementOutput,
    PreTriageOutput,
    TriageOutput,
)


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for all model adapters (mock and real)."""

    @property
    def model_id(self) -> str:
        """Unique identifier for this model."""
        ...

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    def supported_stages(self) -> list[str]:
        """Stages this model supports: pretriage, triage, differential, management."""
        ...

    def pretriage(self, context: PreTriageContext) -> PreTriageOutput:
        """Stage A: pre-triage assessment (no vitals).

        Raises NotImplementedError if model doesn't support pretriage.
        """
        ...

    def triage(self, context: FullTriageContext) -> TriageOutput:
        """Stage B: full triage with vitals → RETTS level.

        Raises NotImplementedError if model doesn't support triage.
        """
        ...

    def differential(self, context: FullTriageContext) -> DifferentialOutput:
        """Differential diagnosis from full context.

        Raises NotImplementedError if model doesn't support differential.
        """
        ...

    def management(self, context: FullTriageContext) -> ManagementOutput:
        """Management plan from full context.

        Raises NotImplementedError if model doesn't support management.
        """
        ...


class BaseAdapter:
    """Base class with default not-implemented methods.

    Concrete adapters inherit from this and override supported stages.
    """

    def __init__(self, model_id: str, model_name: str, supported_stages: list[str]):
        self._model_id = model_id
        self._model_name = model_name
        self._supported_stages = supported_stages

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supported_stages(self) -> list[str]:
        return self._supported_stages

    def pretriage(self, context: PreTriageContext) -> PreTriageOutput:
        raise NotImplementedError(
            f"{self._model_id} does not support pretriage"
        )

    def triage(self, context: FullTriageContext) -> TriageOutput:
        raise NotImplementedError(
            f"{self._model_id} does not support triage"
        )

    def differential(self, context: FullTriageContext) -> DifferentialOutput:
        raise NotImplementedError(
            f"{self._model_id} does not support differential"
        )

    def management(self, context: FullTriageContext) -> ManagementOutput:
        raise NotImplementedError(
            f"{self._model_id} does not support management"
        )
