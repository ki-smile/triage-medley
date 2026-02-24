"""Orchestrator — parallel model dispatch with timeout and graceful degradation.

Dispatches to multiple model adapters and deterministic engines concurrently,
collects results, and handles failures gracefully.
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Any, Callable

from src.adapters.base import BaseAdapter
from src.adapters.factory import create_stage_adapters
from src.engines.pretriage_engine import evaluate as pretriage_evaluate
from src.engines.retts_engine import evaluate as retts_evaluate
from src.engines.esi_engine import evaluate as esi_evaluate
from src.engines.mts_engine import evaluate as mts_evaluate
from src.models.context import FullTriageContext, PreTriageContext
from src.models.outputs import (
    DifferentialOutput,
    ManagementOutput,
    PreTriageOutput,
    TriageOutput,
)
from src.utils.audit import log_event

logger = logging.getLogger(__name__)

# Engine registry: engine_id → evaluate function
_ENGINE_REGISTRY: dict[str, Callable] = {
    "retts": retts_evaluate,
    "esi": esi_evaluate,
    "mts": mts_evaluate,
}

# Default active engines (All three enabled by default for maximum diversity)
DEFAULT_ACTIVE_ENGINES = ["retts", "esi", "mts"]


@dataclass
class PreTriageResult:
    """Aggregated pre-triage result."""
    engine_output: PreTriageOutput
    model_output: Optional[PreTriageOutput] = None


@dataclass
class TriageEnsembleResult:
    """Aggregated triage results from all engines + AI models."""
    engine_outputs: list[TriageOutput] = field(default_factory=list)
    model_outputs: list[TriageOutput] = field(default_factory=list)
    failed_engines: list[str] = field(default_factory=list)
    failed_models: list[str] = field(default_factory=list)
    total_time_ms: int = 0

    @property
    def retts_output(self) -> Optional[TriageOutput]:
        for out in self.engine_outputs:
            if out.model_id == "retts_rules_engine":
                return out
        return None


@dataclass
class DifferentialEnsembleResult:
    """Aggregated differential diagnosis results."""
    model_outputs: list[DifferentialOutput] = field(default_factory=list)
    failed_models: list[str] = field(default_factory=list)
    total_time_ms: int = 0


@dataclass
class ManagementEnsembleResult:
    """Aggregated management plan results."""
    model_outputs: list[ManagementOutput] = field(default_factory=list)
    failed_models: list[str] = field(default_factory=list)
    total_time_ms: int = 0


# Default timeout for model inference
_DEFAULT_TIMEOUT_S = 180
_MAX_RETRIES = 2


def run_pretriage(
    context: PreTriageContext,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> PreTriageResult:
    """Run Stage A pre-triage with parallel engine and model execution."""
    start = time.monotonic()

    # Create tasks for parallel execution
    tasks = {
        "engine": lambda ctx: pretriage_evaluate(ctx),
    }
    
    adapters = create_stage_adapters("pretriage")
    if adapters:
        # Currently only medgemma_4b for pretriage
        model_id = list(adapters.keys())[0]
        # Use the adapter's configured timeout if available
        model_timeout = getattr(adapters[model_id], "_timeout_seconds", timeout_s)
        tasks["model"] = lambda ctx: adapters[model_id].pretriage(ctx)
        actual_timeout = max(timeout_s, model_timeout)
    else:
        actual_timeout = timeout_s

    results = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(fn, context): name for name, fn in tasks.items()}
        for future in as_completed(futures, timeout=actual_timeout):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                logger.error(f"Pre-triage {name} failed: {e}")
                results[name] = None

    engine_out = results.get("engine")
    model_out = results.get("model")

    if engine_out:
        log_event("model:pretriage_engine", "pretriage_result",
                  {"priority": engine_out.queue_priority.value},
                  stage="pretriage", patient_id=context.patient_id)
    
    if model_out:
        log_event("model:ai_pretriage", "pretriage_result",
                  {"priority": model_out.queue_priority.value},
                  stage="pretriage", patient_id=context.patient_id)

    return PreTriageResult(engine_output=engine_out, model_output=model_out)


def get_active_engines() -> list[str]:
    try:
        import streamlit as st
        return st.session_state.get("active_engines", DEFAULT_ACTIVE_ENGINES)
    except Exception:
        return DEFAULT_ACTIVE_ENGINES


def get_active_models() -> Optional[list[str]]:
    try:
        import streamlit as st
        return st.session_state.get("active_models", None)
    except Exception:
        return None


def run_triage_ensemble(
    context: FullTriageContext,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> TriageEnsembleResult:
    """Run Stage B triage ensemble: engines AND models in parallel."""
    start = time.monotonic()

    active_engines = get_active_engines()
    adapters = create_stage_adapters("triage")
    active_model_ids = get_active_models()
    if active_model_ids is not None:
        adapters = {k: v for k, v in adapters.items() if k in active_model_ids}

    # Combined dispatch
    engine_outputs = []
    model_outputs = []
    failed_engines = []
    failed_models = []

    with ThreadPoolExecutor(max_workers=len(active_engines) + len(adapters)) as executor:
        # Submit Engines
        engine_futures = {
            executor.submit(_ENGINE_REGISTRY[eid], context): eid 
            for eid in active_engines if eid in _ENGINE_REGISTRY
        }
        # Submit Models
        model_futures = {
            executor.submit(adapter.triage, context): mid 
            for mid, adapter in adapters.items()
        }

        # Process Engines
        for future in as_completed(engine_futures, timeout=timeout_s):
            eid = engine_futures[future]
            try:
                out = future.result()
                engine_outputs.append(out)
                log_event(f"engine:{eid}", "triage_result",
                          {"level": out.retts_level.value},
                          stage="triage", patient_id=context.patient_id)
            except Exception as e:
                failed_engines.append(eid)
                logger.error(f"Engine {eid} failed: {e}")

        # Process Models
        for future in as_completed(model_futures, timeout=timeout_s):
            mid = model_futures[future]
            try:
                out = future.result()
                model_outputs.append(out)
                log_event(f"model:{mid}", "triage_result",
                          {"level": out.retts_level.value},
                          stage="triage", patient_id=context.patient_id)
            except Exception as e:
                failed_models.append(mid)
                logger.error(f"Model {mid} failed: {e}")

    elapsed = int((time.monotonic() - start) * 1000)
    return TriageEnsembleResult(
        engine_outputs=engine_outputs,
        model_outputs=model_outputs,
        failed_engines=failed_engines,
        failed_models=failed_models,
        total_time_ms=elapsed,
    )


def run_differential_ensemble(
    context: FullTriageContext,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> DifferentialEnsembleResult:
    start = time.monotonic()
    adapters = create_stage_adapters("differential")
    model_outputs, failed = _dispatch_parallel(adapters, "differential", context, timeout_s)
    return DifferentialEnsembleResult(model_outputs=model_outputs, failed_models=failed, total_time_ms=int((time.monotonic()-start)*1000))


def run_management_ensemble(
    context: FullTriageContext,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> ManagementEnsembleResult:
    start = time.monotonic()
    adapters = create_stage_adapters("management")
    model_outputs, failed = _dispatch_parallel(adapters, "management", context, timeout_s)
    return ManagementEnsembleResult(model_outputs=model_outputs, failed_models=failed, total_time_ms=int((time.monotonic()-start)*1000))


def run_full_pipeline(
    context: FullTriageContext,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> tuple[TriageEnsembleResult, DifferentialEnsembleResult, ManagementEnsembleResult]:
    """Execute all ensemble stages in parallel for maximum performance."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_triage = executor.submit(run_triage_ensemble, context, timeout_s)
        f_diff = executor.submit(run_differential_ensemble, context, timeout_s)
        f_mgmt = executor.submit(run_management_ensemble, context, timeout_s)
        
        return f_triage.result(), f_diff.result(), f_mgmt.result()


def _dispatch_parallel(
    adapters: dict[str, BaseAdapter],
    stage: str,
    context: FullTriageContext,
    timeout_s: int,
) -> tuple[list, list[str]]:
    """Parallel dispatch with simple retry logic."""
    outputs = []
    failed = []

    def run_with_retry(adapter, ctx, stage_name):
        last_err = None
        for attempt in range(_MAX_RETRIES):
            try:
                method = getattr(adapter, stage_name)
                return method(ctx)
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1)) # linear backoff
        raise last_err

    with ThreadPoolExecutor(max_workers=len(adapters)) as executor:
        futures = {executor.submit(run_with_retry, adapter, context, stage): mid 
                   for mid, adapter in adapters.items()}
        
        for future in as_completed(futures, timeout=timeout_s):
            mid = futures[future]
            try:
                outputs.append(future.result())
            except Exception as e:
                failed.append(mid)
                logger.error(f"Stage {stage} failed for {mid} after retries: {e}")

    return outputs, failed