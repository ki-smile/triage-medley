"""Agreement Engine — ensemble consensus analysis.

Implements the MEDLEY framework's core principle: disagreement is preserved
as a clinical resource. Includes basic semantic deduplication for findings.
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

from src.models.enums import RETTSLevel
from src.models.outputs import (
    DifferentialCandidate,
    DifferentialOutput,
    ManagementOutput,
    TriageOutput,
)
from src.services.orchestrator import (
    DifferentialEnsembleResult,
    ManagementEnsembleResult,
    TriageEnsembleResult,
)


class AgreementLevel(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"


@dataclass
class TriageAgreement:
    agreement_level: AgreementLevel
    consensus_level: Optional[RETTSLevel]
    final_level: RETTSLevel
    vote_distribution: dict[str, int]
    total_voters: int
    dissenting_models: list[str]
    dont_miss_alerts: list[str]
    escalation_reason: Optional[str] = None
    requires_senior_review: bool = False

    @property
    def agreement_ratio(self) -> float:
        if not self.consensus_level or self.total_voters == 0:
            return 0.0
        # Handle both Enum and string (post-rehydration)
        lvl_key = self.consensus_level.value if hasattr(self.consensus_level, "value") else self.consensus_level
        consensus_votes = self.vote_distribution.get(lvl_key, 0)
        return consensus_votes / self.total_voters


@dataclass
class DifferentialAgreement:
    all_agree: list[DifferentialCandidate]
    some_agree: list[DifferentialCandidate]
    devil_advocate_only: list[DifferentialCandidate]
    dont_miss_all: list[str]
    agreement_level: AgreementLevel = AgreementLevel.PARTIAL


@dataclass
class ManagementAgreement:
    common_investigations: list[str]
    common_imaging: list[str]
    common_medications: list[str]
    disposition_votes: dict[str, int]
    consensus_disposition: Optional[str]
    contraindications: list[str]
    minority_investigations: list[str] = field(default_factory=list)
    minority_imaging: list[str] = field(default_factory=list)
    minority_medications: list[str] = field(default_factory=list)


def _semantic_similarity(s1: str, s2: str) -> float:
    """Basic Jaccard similarity between two strings."""
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _safe_get(obj, attr, default=None):
    if isinstance(obj, dict): return obj.get(attr, default)
    return getattr(obj, attr, default)

def _safe_val(obj, attr):
    val = _safe_get(obj, attr)
    if hasattr(val, "value"): return val.value
    return str(val) if val is not None else ""

def analyze_triage(ensemble: TriageEnsembleResult) -> TriageAgreement:
    all_outputs = ensemble.engine_outputs + ensemble.model_outputs
    votes: dict[str, list[str]] = {}

    for output in all_outputs:
        level = _safe_val(output, "retts_level")
        if level:
            if level not in votes: votes[level] = []
            votes[level].append(_safe_get(output, "model_id"))

    if not votes:
        return TriageAgreement(
            AgreementLevel.NONE, None, RETTSLevel.YELLOW, {}, 0, [], [], None, True
        )

    total = len(all_outputs)
    vote_counts = {level: len(models) for level, models in votes.items()}
    most_common_level, most_common_count = max(vote_counts.items(), key=lambda x: x[1])

    if most_common_count == total: agreement = AgreementLevel.FULL
    elif most_common_count > total / 2: agreement = AgreementLevel.PARTIAL
    else: agreement = AgreementLevel.NONE

    consensus = RETTSLevel(most_common_level)
    dissenting = [m for level, models in votes.items() if level != most_common_level for m in models]
    dont_miss = {dm for out in all_outputs for dm in _safe_get(out, "dont_miss", [])}
    
    all_levels = [RETTSLevel(level) for level in votes.keys()]
    final_level = RETTSLevel.most_severe(*all_levels)

    escalation_reason = None
    if final_level != consensus:
        escalation_reason = f"Safety Escalation: Minority flagged {final_level.value} (Consensus: {consensus.value})"

    return TriageAgreement(
        agreement_level=agreement, consensus_level=consensus, final_level=final_level,
        vote_distribution=vote_counts, total_voters=total, dissenting_models=dissenting,
        dont_miss_alerts=sorted(dont_miss), escalation_reason=escalation_reason,
        requires_senior_review=(agreement == AgreementLevel.NONE or final_level != consensus)
    )


def analyze_differential(ensemble: DifferentialEnsembleResult) -> DifferentialAgreement:
    model_outputs = _safe_get(ensemble, "model_outputs", [])
    total_models = len(model_outputs)
    if total_models == 0:
        return DifferentialAgreement([], [], [], [], AgreementLevel.NONE)

    groups: list[dict] = [] 
    all_dont_miss = set()

    for output in model_outputs:
        candidates = _safe_get(output, "candidates", [])
        for candidate in candidates:
            diag = _safe_get(candidate, "diagnosis", "Unknown")
            if _safe_get(candidate, "is_dont_miss"): 
                all_dont_miss.add(diag)
            
            found = False
            for group in groups:
                if _semantic_similarity(diag, group["canonical"]) > 0.7:
                    group["count"] += 1
                    best_prob = _safe_get(group["best"], "probability", 0) or 0
                    cand_prob = _safe_get(candidate, "probability", 0) or 0
                    if cand_prob > best_prob:
                        group["best"] = candidate
                    found = True
                    break
            if not found:
                groups.append({"canonical": diag, "count": 1, "best": candidate})

    all_agree, some_agree, devil_only = [], [], []
    for g in groups:
        ratio = g["count"] / total_models
        if ratio >= 0.8: all_agree.append(g["best"])
        elif ratio >= 0.4: some_agree.append(g["best"])
        else: devil_only.append(g["best"])

    return DifferentialAgreement(
        all_agree=all_agree, some_agree=some_agree, devil_advocate_only=devil_only,
        dont_miss_all=sorted(all_dont_miss), 
        agreement_level=AgreementLevel.FULL if len(all_agree) >= 2 else AgreementLevel.PARTIAL
    )


def analyze_management(ensemble: ManagementEnsembleResult) -> ManagementAgreement:
    model_outputs = _safe_get(ensemble, "model_outputs", [])
    total = len(model_outputs)
    if total == 0:
        return ManagementAgreement([], [], [], {}, None, [])

    def split_items(items_list: list[list[str]], threshold: float) -> tuple[list[str], list[str]]:
        flat = [i for sub in items_list for i in sub]
        if not flat: return [], []
        counts = Counter()
        canonical_map = {}
        
        for item in flat:
            found = False
            for canonical in canonical_map:
                if _semantic_similarity(item, canonical) > 0.8:
                    counts[canonical] += 1
                    found = True
                    break
            if not found:
                canonical_map[item] = item
                counts[item] = 1
        
        common = [k for k, v in counts.items() if v >= (total * threshold)]
        minority = [k for k, v in counts.items() if v < (total * threshold)]
        return common, minority

    common_inv, minor_inv = split_items([_safe_get(o, "investigations", []) for o in model_outputs], 0.5)
    common_img, minor_img = split_items([_safe_get(o, "imaging", []) for o in model_outputs], 0.5)
    common_med, minor_med = split_items([_safe_get(o, "medications", []) for o in model_outputs], 0.5)
    
    dispositions = [_safe_get(o, "disposition", "observation") for o in model_outputs]
    disp_counts = Counter(dispositions)
    contra = {c for o in model_outputs for c in _safe_get(o, "contraindications_flagged", [])}

    return ManagementAgreement(
        common_investigations=common_inv,
        common_imaging=common_img,
        common_medications=common_med,
        disposition_votes=dict(disp_counts),
        consensus_disposition=disp_counts.most_common(1)[0][0] if disp_counts else None,
        contraindications=sorted(contra),
        minority_investigations=minor_inv,
        minority_imaging=minor_img,
        minority_medications=minor_med
    )


@dataclass
class EngineDisagreement:
    engines_agree: bool
    engine_levels: dict[str, str]
    most_severe_engine: str
    least_severe_engine: str
    disagreement_explanation: str
    clinical_action: str


def analyze_engine_disagreement(engine_outputs: list[TriageOutput]) -> Optional[EngineDisagreement]:
    if len(engine_outputs) < 2: return None
    
    level_order = [RETTSLevel.RED, RETTSLevel.ORANGE, RETTSLevel.YELLOW, RETTSLevel.GREEN, RETTSLevel.BLUE]
    
    engine_levels = {}
    for out in engine_outputs:
        system = _safe_get(out, "triage_system") or _safe_get(out, "model_id")
        if system:
            engine_levels[system.upper()] = _safe_val(out, "retts_level")

    if not engine_levels: return None
    
    unique_levels = set(engine_levels.values())
    engines_agree = len(unique_levels) == 1

    def severity_rank(lvl): 
        try:
            return level_order.index(RETTSLevel(lvl) if isinstance(lvl, str) else lvl)
        except (ValueError, KeyError):
            return 99

    sorted_engines = sorted(engine_levels.items(), key=lambda x: severity_rank(x[1]))
    most_severe, least_severe = sorted_engines[0][0], sorted_engines[-1][0]

    if engines_agree:
        explanation = f"All {len(engine_outputs)} triage engines agree on {list(unique_levels)[0]}."
        action = "Proceed with standard triage workflow."
    else:
        explanation = f"Cross-system disagreement: {engine_levels}. {most_severe} assessed highest severity."
        action = f"Review reasoning. Consider {most_severe} as safety-first option."

    return EngineDisagreement(engines_agree, engine_levels, most_severe, least_severe, explanation, action)
