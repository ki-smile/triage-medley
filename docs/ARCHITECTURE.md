# Triage-Medley Technical Architecture

> **Human-in-the-Loop AI-powered Triage Decision Support System + Medical Ensemble Diagnostic system with Leveraged diversitY**
>
> Karolinska Institutet / KTH -- MedGemma Impact Challenge, February 2026

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Two-Stage Pipeline Architecture](#2-two-stage-pipeline-architecture)
3. [User Journey Sequence Diagrams](#3-user-journey-sequence-diagrams)
4. [Ensemble Pipeline Sequence Diagram](#4-ensemble-pipeline-sequence-diagram)
5. [Data Model Diagram](#5-data-model-diagram)
6. [Adapter Pattern Architecture](#6-adapter-pattern-architecture)
7. [Agreement Engine Logic](#7-agreement-engine-logic)
8. [ASR Disagreement Pipeline](#8-asr-disagreement-pipeline)
9. [Role-Based Access Control](#9-role-based-access-control)
10. [Source Code Layout](#10-source-code-layout)

---

## 1. System Architecture Overview

The system is a Streamlit multi-page application composed of six layers: Input Capture, Clinical NLP, Triage Ensemble, Differential Diagnosis, Management Plan, and HITL Visualization. It uses a mock-first approach with config-driven adapter swapping to HuggingFace Inference API models or HuggingFace Spaces (Gradio deployments).

### High-Level Component Diagram

```mermaid
graph TB
    subgraph "Frontend -- Streamlit Multi-Page App"
        APP["app.py<br/>Role-Based Router"]
        KIOSK["0_Kiosk.py<br/>Patient Walk-in"]
        QUEUE["1_Queue_View.py<br/>Charge Nurse"]
        TRIAGE["2_Triage_View.py<br/>Triage Nurse"]
        PHYSICIAN["3_Physician_View.py<br/>Physician"]
        PROMPT["4_Prompt_Editor.py<br/>Dev Tool"]
        AUDIT_PG["5_Audit_Log.py<br/>Compliance"]
        ENGINE_CFG["6_Engine_Config.py<br/>Admin Panel"]
    end

    subgraph "Services Layer"
        SESSION["Session Manager<br/>PatientSession state"]
        ORCH["Orchestrator<br/>Parallel dispatch"]
        EHR["EHR Service<br/>FHIR Bundle parser"]
        ASR["ASR Service<br/>Dual-ASR pipeline"]
        AUTH["Auth Service<br/>Role-based access"]
        PDF["PDF Service<br/>Report generation"]
    end

    subgraph "Engines Layer"
        PRETRIAGE["Pre-Triage Engine<br/>Speech + EHR rules"]
        RETTS["RETTS Engine<br/>Vitals + ESS rules"]
        ESI["ESI Engine<br/>5-level decision tree"]
        MTS["MTS Engine<br/>Flowchart discriminators"]
        AGREE["Agreement Engine<br/>Consensus analysis"]
    end

    subgraph "Adapters Layer"
        FACTORY["Adapter Factory<br/>Config-driven creation"]
        MOCK["MockAdapter<br/>JSON scenario files"]
        HF_BASE["HFBaseAdapter<br/>Inference API client"]
        HF_MG4["MedGemma4BAdapter"]
        HF_MG27["MedGemma27BAdapter"]
        HF_M42["Med42Adapter"]
        HF_QW["QwenMedAdapter"]
        HF_BM["BioMistralAdapter"]
        SPACE_BASE["SpaceBaseAdapter<br/>Gradio client"]
        SPACE_MG4["SpaceMedGemma4BAdapter"]
        PROMPT_B["PromptBuilder<br/>YAML template renderer"]
    end

    subgraph "Configuration"
        MODELS_YAML["config/models.yaml<br/>Model registry"]
        ENGINES_YAML["config/engines.yaml<br/>Engine registry"]
        PROMPTS_DIR["config/prompts/<br/>YAML templates"]
        RETTS_CFG["config/retts/<br/>ESS codes + vitals"]
        PRETRIAGE_YAML["config/pretriage.yaml<br/>Red-flag rules"]
    end

    subgraph "Data Layer"
        SCENARIOS["data/scenarios/<br/>Mock JSON responses"]
        EHR_DATA["data/ehr/<br/>Synthea FHIR Bundles"]
        SQLITE["data/triage.db<br/>Persistent Shared Queue"]
        AUDIT_LOG["data/audit/audit.jsonl<br/>Append-only log"]
    end

    APP --> KIOSK & QUEUE & TRIAGE & PHYSICIAN & PROMPT & AUDIT_PG & ENGINE_CFG
    KIOSK --> SESSION & ASR & EHR
    QUEUE --> SESSION
    TRIAGE --> SESSION & ORCH & AGREE & PDF
    PHYSICIAN --> SESSION & PDF
    AUDIT_PG --> AUDIT_LOG
    ENGINE_CFG --> MODELS_YAML & ENGINES_YAML

    SESSION --> ORCH & EHR & ASR
    ORCH --> FACTORY & PRETRIAGE & RETTS & ESI & MTS
    ORCH --> AGREE

    FACTORY --> MOCK & HF_BASE & SPACE_BASE
    FACTORY --> MODELS_YAML
    HF_BASE --> HF_MG4 & HF_MG27 & HF_M42 & HF_QW & HF_BM
    SPACE_BASE --> SPACE_MG4
    HF_BASE --> PROMPT_B
    SPACE_BASE --> PROMPT_B
    PROMPT_B --> PROMPTS_DIR

    MOCK --> SCENARIOS
    EHR --> EHR_DATA
    RETTS --> RETTS_CFG
    PRETRIAGE --> PRETRIAGE_YAML

    AUTH --> APP
```

### Component Summary

| Component | Location | Purpose |
|-----------|----------|---------|
| **app.py** | `/app.py` | Streamlit entry point; role selector, login, and `st.navigation()` router |
| **Session Manager** | `src/services/session_manager.py` | Manages `PatientSession` rehydration from DB; bridges st.session_state with SQLite |
| **DB Service** | `src/services/db_service.py` | Handles SQLite persistent storage for shared multi-device queues |
| **Orchestrator** | `src/services/orchestrator.py` | Parallel model dispatch with `ThreadPoolExecutor`; graceful degradation |
| **EHR Service** | `src/services/ehr_service.py` | Parses Synthea FHIR R4 bundles; computes risk flags from medication/condition combos |
| **ASR Service** | `src/services/asr_service.py` | Dual-ASR pipeline; word-level LCS alignment; clinical significance weighting |
| **Auth Service** | `src/services/auth_service.py` | Mock authentication; role enum; page access matrix |
| **PDF Service** | `src/services/pdf_service.py` | Generates triage and physician PDF reports using fpdf2 with KI branding |
| **Adapter Factory** | `src/adapters/factory.py` | Config-driven adapter creation (mock/huggingface/space); lazy imports |
| **Agreement Engine** | `src/engines/agreement_engine.py` | Ensemble consensus analysis; triage/differential/management agreement |
| **Audit Logger** | `src/utils/audit.py` | Thread-safe, append-only JSONL event logging |
| **Config Loader** | `src/utils/config.py` | YAML/JSON config loader with caching |

---

## 2. Two-Stage Pipeline Architecture

The pipeline enforces a strict stage separation at the type level: `PreTriageContext` has no vitals field, while `FullTriageContext` requires `VitalSigns`. The RETTS, ESI, and MTS engines refuse to run without vitals.

### Stage Flow Diagram

```mermaid
flowchart TB
    subgraph "Patient Arrival (3 Pathways)"
        WALKIN["Walk-in Self-Arrival<br/>Patient speaks at kiosk"]
        REFERRAL["1177 Vardguiden Referral<br/>Structured JSON from nurse"]
        AMBULANCE["Ambulance Arrival<br/>MobiMed ePR + SBAR"]
    end

    subgraph "Stage A: PRE-TRIAGE (< 60 seconds, no vitals)"
        direction TB
        ASR_A["Dual-ASR Processing<br/>MedASR + Whisper"]
        EHR_A["EHR Lookup<br/>FHIR Bundle parse"]
        PRE_CTX["PreTriageContext<br/>(speech + EHR, NO vitals)"]
        PRE_ENGINE["Pre-Triage Engine<br/>Red-flag keywords + EHR amplifiers"]
        MG4B_A["MedGemma 4B<br/>(only model in Stage A)"]
        PRE_OUT["PreTriageOutput<br/>Queue Priority: HIGH / MODERATE / STANDARD"]
    end

    subgraph "Stage B: FULL TRIAGE (< 30 seconds, with vitals)"
        direction TB
        VITALS_IN["Nurse Enters Vitals<br/>HR, BP, RR, SpO2, Temp, AVPU"]
        FULL_CTX["FullTriageContext<br/>(speech + EHR + VitalSigns)"]

        subgraph "Deterministic Engines"
            RETTS_E["RETTS Engine<br/>Vitals + ESS rules"]
            ESI_E["ESI Engine<br/>Decision tree + resources"]
            MTS_E["MTS Engine<br/>Flowchart discriminators"]
        end

        subgraph "LLM Models (parallel via ThreadPoolExecutor)"
            MG4B_B["MedGemma 4B"]
            MG27B["MedGemma 27B"]
            M42["Med42"]
            QW["Qwen2.5-Med"]
            BM["BioMistral"]
        end

        AGREE_E["Agreement Engine<br/>Consensus + don't-miss analysis"]

        subgraph "Outputs"
            TRIAGE_OUT["TriageOutput<br/>RETTS Level + Confidence"]
            DIFF_OUT["DifferentialOutput<br/>Ranked candidates"]
            MGMT_OUT["ManagementOutput<br/>Labs, imaging, meds, disposition"]
        end
    end

    WALKIN --> ASR_A
    REFERRAL --> ASR_A
    ASR_A --> PRE_CTX
    EHR_A --> PRE_CTX
    PRE_CTX --> PRE_ENGINE & MG4B_A
    PRE_ENGINE --> PRE_OUT
    MG4B_A --> PRE_OUT
    PRE_OUT -->|"Patient enters queue"| VITALS_IN

    AMBULANCE -->|"Skips Stage A<br/>(vitals already available)"| FULL_CTX

    VITALS_IN --> FULL_CTX
    PRE_CTX -->|"Inherits speech + EHR"| FULL_CTX

    FULL_CTX --> RETTS_E & ESI_E & MTS_E
    FULL_CTX --> MG4B_B & MG27B & M42 & QW & BM

    RETTS_E & ESI_E & MTS_E --> AGREE_E
    MG4B_B & MG27B & M42 & QW & BM --> AGREE_E

    AGREE_E --> TRIAGE_OUT & DIFF_OUT & MGMT_OUT

    style WALKIN fill:#54B986,color:#000
    style REFERRAL fill:#F59A00,color:#000
    style AMBULANCE fill:#B84145,color:#FFF
    style PRE_CTX fill:#E8EAF6,color:#000
    style FULL_CTX fill:#E8EAF6,color:#000
    style PRE_OUT fill:#C8E6C9,color:#000
    style AGREE_E fill:#4F0433,color:#FFF
```

### Stage Model Matrix

| Model | Stage A (Pre-Triage) | Stage B: Triage | Stage B: Differential | Stage B: Management |
|-------|:-------------------:|:---------------:|:--------------------:|:------------------:|
| **MedGemma 4B** | Yes | Yes | Yes | Yes |
| **MedGemma 27B** | -- | Yes | Yes | Yes |
| **Med42** | -- | Yes | Yes | -- |
| **Qwen2.5-Med** | -- | Yes | Yes | -- |
| **BioMistral** | -- | Yes | Yes | -- |
| **RETTS Engine** | -- | Yes | -- | -- |
| **ESI Engine** | -- | Yes | -- | -- |
| **MTS Engine** | -- | Yes | -- | -- |
| **Pre-Triage Engine** | Yes | -- | -- | -- |

### Data Flow Between Stages

```
PreTriageContext                    FullTriageContext
  .patient_id                        .patient_id         (inherited)
  .arrival_pathway                   .arrival_pathway     (inherited)
  .arrival_time                      .arrival_time        (inherited)
  .speech_text                       .speech_text         (inherited)
  .ehr (Optional[EHRSnapshot])       .ehr                 (inherited)
  .asr_disagreements                 .asr_disagreements   (inherited)
  .language                          .language            (inherited)
  (NO vitals field)                  .vitals              (REQUIRED - VitalSigns)
                                     .ess_category        (from pre-triage hint)
```

`FullTriageContext` inherits from `PreTriageContext` via Python class inheritance, adding the required `vitals: VitalSigns` field and the optional `ess_category` hint from Stage A.

---

## 3. User Journey Sequence Diagrams

### a. Patient Walk-in Kiosk Flow

```mermaid
sequenceDiagram
    actor Patient
    participant Kiosk as 0_Kiosk.py
    participant ASR as ASR Service
    participant EHR as EHR Service
    participant SM as Session Manager
    participant PT as Pre-Triage Engine
    participant MG4 as MedGemma 4B<br/>(MockAdapter)
    participant Audit as Audit Logger

    Patient->>Kiosk: Speaks at kiosk<br/>(or selects scenario)
    Kiosk->>SM: register_kiosk_patient(patient_id, speech_text)

    SM->>EHR: load_patient(patient_id)
    EHR-->>SM: EHRSnapshot (conditions, meds, allergies, risk_flags)<br/>or None (graceful degradation)

    SM->>ASR: process_audio(patient_id)
    ASR-->>SM: ASRResult (medasr_transcript, whisper_transcript,<br/>disagreements, confidence_score)

    SM->>SM: Create PreTriageContext<br/>(speech + EHR + ASR disagreements, NO vitals)

    SM->>PT: evaluate(PreTriageContext)
    PT->>PT: Match speech red-flag keywords
    PT->>PT: Apply EHR risk amplifiers
    PT-->>SM: PreTriageOutput<br/>(queue_priority, chief_complaint, reasoning)

    SM->>MG4: pretriage(PreTriageContext)
    MG4-->>SM: PreTriageOutput (model perspective)

    SM->>Audit: log_event("kiosk", "patient_registered", ...)
    SM->>SM: add_patient(PatientSession)
    SM-->>Kiosk: PatientSession with queue_priority

    Kiosk-->>Patient: Confirmation: "You are in the queue.<br/>Priority: HIGH / MODERATE / STANDARD"
```

### b. Triage Nurse Flow

```mermaid
sequenceDiagram
    actor Nurse as Triage Nurse
    participant Queue as 1_Queue_View.py
    participant SM as Session Manager
    participant TV as 2_Triage_View.py
    participant Orch as Orchestrator
    participant Engines as RETTS + ESI + MTS<br/>Engines
    participant Models as LLM Models<br/>(5 adapters, parallel)
    participant AE as Agreement Engine
    participant Audit as Audit Logger
    participant PDF as PDF Service

    Nurse->>Queue: View priority-ordered queue
    Queue->>SM: get_queue_ordered()
    SM-->>Queue: List[PatientSession] sorted by priority + arrival_time
    Queue-->>Nurse: Patient list with priority badges

    Nurse->>Queue: Select patient (click row)
    Queue->>SM: select_patient(patient_id)
    Nurse->>TV: Navigate to Triage View

    TV->>SM: get_selected_patient()
    SM-->>TV: PatientSession (speech, EHR, ASR, pre-triage)

    TV-->>Nurse: Display patient info, ASR transcript,<br/>EHR risk flags, pre-triage result

    Nurse->>TV: Enter vital signs<br/>(HR, BP, RR, SpO2, Temp, AVPU)
    TV->>TV: Create VitalSigns + FullTriageContext

    TV->>Orch: run_full_pipeline(FullTriageContext)

    par Deterministic Engines
        Orch->>Engines: retts_evaluate(context)
        Orch->>Engines: esi_evaluate(context)
        Orch->>Engines: mts_evaluate(context)
        Engines-->>Orch: TriageOutput per engine
    and LLM Models (ThreadPoolExecutor)
        Orch->>Models: adapter.triage(context) x5
        Orch->>Models: adapter.differential(context) x3
        Orch->>Models: adapter.management(context) x2
        Models-->>Orch: TriageOutput, DifferentialOutput,<br/>ManagementOutput per model
    end

    Orch-->>TV: TriageEnsembleResult,<br/>DifferentialEnsembleResult,<br/>ManagementEnsembleResult

    TV->>AE: analyze_triage(triage_ensemble)
    AE-->>TV: TriageAgreement<br/>(agreement_level, final_level, votes, don't-miss)

    TV->>AE: analyze_engine_disagreement(engine_outputs)
    AE-->>TV: EngineDisagreement<br/>(cross-engine comparison)

    TV-->>Nurse: Ensemble triage result:<br/>RETTS badge, vote distribution,<br/>escalation alerts, don't-miss warnings

    opt Nurse Override
        Nurse->>TV: Override RETTS level<br/>(click override, select new level, enter reason)
        TV->>Audit: log_event("nurse", "override", {original, override_to, reason})
        TV->>SM: patient.overrides.append({...})
    end

    opt Download Report
        Nurse->>TV: Click "Download Triage Report"
        TV->>PDF: generate_triage_pdf(patient)
        PDF-->>TV: PDF bytes
        TV-->>Nurse: PDF download
    end
```

### c. Physician Flow

```mermaid
sequenceDiagram
    actor Physician
    participant Queue as 1_Queue_View.py
    participant SM as Session Manager
    participant PV as 3_Physician_View.py
    participant AE as Agreement Engine
    participant Audit as Audit Logger
    participant PDF as PDF Service

    Physician->>Queue: View triaged patients
    Queue->>SM: get_queue_ordered()
    SM-->>Queue: Patient list (filtered: is_triaged = true)
    Queue-->>Physician: Patients with RETTS badges

    Physician->>Queue: Select patient
    Queue->>SM: select_patient(patient_id)
    Physician->>PV: Navigate to Physician View

    PV->>SM: get_selected_patient()
    SM-->>PV: PatientSession (with all Stage A + B data)

    PV-->>Physician: Display: demographics, vitals, ASR,<br/>EHR risk flags, triage result, nurse overrides

    PV->>AE: analyze_differential(differential_ensemble)
    AE-->>PV: DifferentialAgreement<br/>(all_agree, some_agree, devil_advocate_only)

    PV-->>Physician: Three-tier differential:<br/>1. Primary (>=80% consensus)<br/>2. Alternative (40-79%)<br/>3. Minority / devil's advocate (<40%)<br/>+ Don't-miss alerts highlighted

    PV->>AE: analyze_management(management_ensemble)
    AE-->>PV: ManagementAgreement<br/>(common_investigations, common_imaging,<br/>common_medications, disposition_votes)

    PV-->>Physician: Management plan:<br/>Consensus items + individual model extras<br/>Disposition vote distribution

    Physician->>PV: Select investigations to approve<br/>(checkboxes from consensus + individual items)
    Physician->>PV: Select imaging to approve
    Physician->>PV: Select medications to approve
    Physician->>PV: Set disposition<br/>(discharge / admission / ICU / observation)
    Physician->>PV: Enter free-text notes

    Physician->>PV: Click "Sign Off"
    PV->>SM: patient.physician_sign_off = datetime.now()
    PV->>SM: patient.physician_name = display_name
    PV->>Audit: log_event("physician", "sign_off",<br/>{investigations, imaging, medications, disposition, notes})

    opt Download Report
        Physician->>PV: Click "Download Physician Report"
        PV->>PDF: generate_physician_pdf(patient)
        PDF-->>PV: PDF bytes (includes sign-off stamp,<br/>approved plan, differential, management)
        PV-->>Physician: PDF download
    end
```

### d. Admin Flow

```mermaid
sequenceDiagram
    actor Admin
    participant App as app.py<br/>Router
    participant EC as 6_Engine_Config.py
    participant PE as 4_Prompt_Editor.py
    participant AL as 5_Audit_Log.py
    participant SM as Session Manager
    participant Config as config/*.yaml
    participant Audit as data/audit/audit.jsonl

    Admin->>App: Login (admin / 0000)
    App-->>Admin: Full access: Clinical + Development pages

    Note over Admin,EC: Engine Configuration
    Admin->>EC: Navigate to Engine Config
    EC-->>Admin: Show active engines (RETTS, ESI, MTS toggles)<br/>Show active LLM models (toggle each)
    Admin->>EC: Enable ESI + MTS engines
    EC->>SM: st.session_state.active_engines = ["retts", "esi", "mts"]
    Admin->>EC: Set HF API key
    EC->>SM: st.session_state.hf_api_key = "hf_..."
    Admin->>EC: Switch adapter mode (mock -> huggingface)
    EC-->>Admin: Configuration updated

    Note over Admin,PE: Prompt Editing
    Admin->>PE: Navigate to Prompt Editor
    PE->>Config: load_prompt_template("triage")
    Config-->>PE: YAML content (system_prompt + user_template)
    PE-->>Admin: Display editable YAML
    Admin->>PE: Edit system prompt or user template
    PE->>Config: Save updated YAML
    PE-->>Admin: "Saved successfully"

    Note over Admin,AL: Audit Review
    Admin->>AL: Navigate to Audit Log
    AL->>Audit: get_events(session_id=current)
    Audit-->>AL: List[AuditEvent]
    AL-->>Admin: Filterable table: timestamp, actor, action,<br/>patient_id, stage, payload
    Admin->>AL: Filter by patient_id or action type
    AL-->>Admin: Filtered audit trail
```

---

## 4. Ensemble Pipeline Sequence Diagram

This diagram shows the detailed internal flow of the Stage B ensemble, including all models and engines running in parallel.

```mermaid
sequenceDiagram
    participant TV as Triage View
    participant Orch as Orchestrator
    participant RETTS as RETTS Engine
    participant ESI as ESI Engine
    participant MTS as MTS Engine
    participant MG4 as MedGemma 4B
    participant MG27 as MedGemma 27B
    participant M42 as Med42
    participant QW as Qwen2.5-Med
    participant BM as BioMistral
    participant AE as Agreement Engine
    participant Audit as Audit Logger

    TV->>Orch: run_full_pipeline(FullTriageContext)

    Note over Orch: Phase 1: Triage Ensemble

    rect rgb(240, 248, 255)
        Note over Orch,MTS: Deterministic Engines (sequential per engine)
        Orch->>RETTS: evaluate(context)
        RETTS->>RETTS: _assess_vitals(thresholds)
        RETTS->>RETTS: _assess_ess(ess_category)
        RETTS->>RETTS: final = max_severity(vitals, ESS)
        RETTS-->>Orch: TriageOutput(retts_level, triage_system="retts")
        Orch->>Audit: log_event(retts_rules_engine, triage_result)

        Orch->>ESI: evaluate(context)
        ESI->>ESI: _is_esi_1? (dying patient)
        ESI->>ESI: _is_esi_2_high_risk? (altered consciousness)
        ESI->>ESI: _estimate_resources + _vitals_danger_zone
        ESI-->>Orch: TriageOutput(retts_level, triage_system="esi",<br/>native_level_detail={esi_level, resources_predicted})
        Orch->>Audit: log_event(esi_rules_engine, triage_result)

        Orch->>MTS: evaluate(context)
        MTS->>MTS: _select_flowchart(ess_category)
        MTS->>MTS: _evaluate_discriminators top-down
        MTS-->>Orch: TriageOutput(retts_level, triage_system="mts",<br/>native_level_detail={flowchart, max_wait_minutes})
        Orch->>Audit: log_event(mts_rules_engine, triage_result)
    end

    rect rgb(255, 248, 240)
        Note over Orch,BM: LLM Models (parallel via ThreadPoolExecutor)
        par
            Orch->>MG4: adapter.triage(context)
            MG4-->>Orch: TriageOutput
        and
            Orch->>MG27: adapter.triage(context)
            MG27-->>Orch: TriageOutput
        and
            Orch->>M42: adapter.triage(context)
            M42-->>Orch: TriageOutput
        and
            Orch->>QW: adapter.triage(context)
            QW-->>Orch: TriageOutput
        and
            Orch->>BM: adapter.triage(context)
            BM-->>Orch: TriageOutput
        end
        Orch->>Audit: log_event per model (triage_result)
    end

    Orch-->>TV: TriageEnsembleResult<br/>(engine_outputs[3] + model_outputs[5])

    TV->>AE: analyze_triage(ensemble)
    AE->>AE: Count votes per RETTS level
    AE->>AE: Classify: FULL / PARTIAL / NONE
    AE->>AE: final_level = most_severe across all votes
    AE->>AE: Aggregate don't-miss from ALL models
    AE->>AE: Flag if escalation (final != consensus)
    AE-->>TV: TriageAgreement

    TV->>AE: analyze_engine_disagreement(engine_outputs)
    AE->>AE: Compare RETTS vs ESI vs MTS levels
    AE->>AE: Build clinical explanation
    AE-->>TV: EngineDisagreement

    Note over Orch: Phase 2: Differential Diagnosis

    rect rgb(240, 255, 240)
        par
            Orch->>MG4: adapter.differential(context)
            MG4-->>Orch: DifferentialOutput
        and
            Orch->>MG27: adapter.differential(context)
            MG27-->>Orch: DifferentialOutput
        and
            Orch->>M42: adapter.differential(context)
            M42-->>Orch: DifferentialOutput
        and
            Orch->>QW: adapter.differential(context)
            QW-->>Orch: DifferentialOutput
        and
            Orch->>BM: adapter.differential(context)
            BM-->>Orch: DifferentialOutput
        end
    end

    Orch-->>TV: DifferentialEnsembleResult
    TV->>AE: analyze_differential(ensemble)
    AE->>AE: Count diagnosis mentions across models
    AE->>AE: Tier: all_agree (>=80%), some_agree (40-79%),<br/>devil_advocate_only (<40%)
    AE-->>TV: DifferentialAgreement (3-tier output)

    Note over Orch: Phase 3: Management Plan

    rect rgb(255, 240, 255)
        par
            Orch->>MG4: adapter.management(context)
            MG4-->>Orch: ManagementOutput
        and
            Orch->>MG27: adapter.management(context)
            MG27-->>Orch: ManagementOutput
        end
    end

    Orch-->>TV: ManagementEnsembleResult
    TV->>AE: analyze_management(ensemble)
    AE->>AE: Find common items (>50% mention threshold)
    AE->>AE: Determine disposition consensus
    AE-->>TV: ManagementAgreement
```

---

## 5. Data Model Diagram

All data models are Pydantic classes defined in `src/models/`. The key architectural decision is the type-enforced stage separation: `PreTriageContext` has no vitals field, while `FullTriageContext` extends it with a required `VitalSigns` field.

### Class Diagram

```mermaid
classDiagram
    direction TB

    class RETTSLevel {
        <<enum>>
        RED
        ORANGE
        YELLOW
        GREEN
        BLUE
        +severity_rank: int
        +most_severe(*levels) RETTSLevel
    }

    class QueuePriority {
        <<enum>>
        HIGH
        MODERATE
        STANDARD
    }

    class Confidence {
        <<enum>>
        HIGH
        MODERATE
        LOW
    }

    class ArrivalPathway {
        <<enum>>
        WALK_IN
        REFERRAL_1177
        AMBULANCE
    }

    class ConsciousnessLevel {
        <<enum>>
        Alert
        Voice
        Pain
        Unresponsive
    }

    class VitalSigns {
        +heart_rate: int
        +systolic_bp: int
        +diastolic_bp: int
        +respiratory_rate: int
        +spo2: int
        +temperature: float
        +consciousness: ConsciousnessLevel
    }

    class FHIRCondition {
        +code: str
        +display: str
        +onset_date: Optional[date]
        +status: str
    }

    class FHIRMedication {
        +code: str
        +display: str
        +dosage: Optional[str]
        +status: str
    }

    class FHIRAllergy {
        +substance: str
        +reaction: Optional[str]
        +severity: Optional[str]
    }

    class RiskFlag {
        +flag_type: str
        +description: str
        +source: str
        +severity: str
    }

    class EHRSnapshot {
        +patient_id: str
        +name: str
        +age: int
        +sex: str
        +conditions: list~FHIRCondition~
        +medications: list~FHIRMedication~
        +allergies: list~FHIRAllergy~
        +risk_flags: list~RiskFlag~
        +is_pediatric: bool
        +active_conditions: list
        +active_medications: list
        +has_medication_class(keywords)
        +has_condition_matching(keywords)
    }

    class ASRDisagreement {
        +word_index: int
        +medasr_word: str
        +whisper_word: str
        +clinical_significance: str
        +resolved: bool
        +resolved_to: Optional[str]
    }

    class PreTriageContext {
        +patient_id: str
        +arrival_pathway: ArrivalPathway
        +arrival_time: datetime
        +speech_text: str
        +ehr: Optional~EHRSnapshot~
        +asr_disagreements: list~ASRDisagreement~
        +language: str
    }

    class FullTriageContext {
        +vitals: VitalSigns
        +ess_category: Optional[str]
    }

    class PreTriageOutput {
        +model_id: str
        +queue_priority: QueuePriority
        +chief_complaint: str
        +reasoning: str
        +ess_category_hint: Optional[str]
        +risk_amplifiers_detected: list[str]
        +timestamp: datetime
        +processing_time_ms: Optional[int]
    }

    class TriageOutput {
        +model_id: str
        +retts_level: RETTSLevel
        +ess_category: Optional[str]
        +chief_complaint: str
        +clinical_reasoning: str
        +vital_sign_concerns: list[str]
        +risk_factors: list[str]
        +confidence: Confidence
        +dont_miss: list[str]
        +triage_system: Optional[str]
        +native_level_detail: Optional[dict]
        +timestamp: datetime
        +processing_time_ms: Optional[int]
    }

    class DifferentialCandidate {
        +diagnosis: str
        +probability: Optional[float]
        +supporting_evidence: list[str]
        +is_dont_miss: bool
    }

    class DifferentialOutput {
        +model_id: str
        +candidates: list~DifferentialCandidate~
        +reasoning: str
        +confidence: Confidence
        +timestamp: datetime
        +processing_time_ms: Optional[int]
    }

    class ManagementOutput {
        +model_id: str
        +investigations: list[str]
        +imaging: list[str]
        +medications: list[str]
        +disposition: str
        +contraindications_flagged: list[str]
        +reasoning: str
        +confidence: Confidence
        +timestamp: datetime
        +processing_time_ms: Optional[int]
    }

    class PatientSession {
        +patient_id: str
        +name: str
        +age: int
        +sex: str
        +arrival_pathway: ArrivalPathway
        +arrival_time: datetime
        +speech_text: str
        +asr_result: Optional~ASRResult~
        +pretriage_context: Optional~PreTriageContext~
        +pretriage_result: Optional~PreTriageResult~
        +full_context: Optional~FullTriageContext~
        +triage_ensemble: Optional~TriageEnsembleResult~
        +triage_agreement: Optional~TriageAgreement~
        +differential_ensemble: Optional~DifferentialEnsembleResult~
        +differential_agreement: Optional~DifferentialAgreement~
        +management_ensemble: Optional~ManagementEnsembleResult~
        +management_agreement: Optional~ManagementAgreement~
        +overrides: list[dict]
        +physician_approved_investigations: list[str]
        +physician_approved_imaging: list[str]
        +physician_approved_medications: list[str]
        +physician_disposition: Optional[str]
        +physician_notes: str
        +physician_sign_off: Optional[datetime]
        +physician_name: Optional[str]
        +queue_priority: Optional~QueuePriority~
        +retts_level: Optional~RETTSLevel~
        +is_triaged: bool
        +is_signed_off: bool
    }

    %% Relationships
    FullTriageContext --|> PreTriageContext : inherits
    FullTriageContext *-- VitalSigns : requires
    PreTriageContext o-- EHRSnapshot : optional
    PreTriageContext o-- ASRDisagreement : 0..*
    EHRSnapshot *-- FHIRCondition : 0..*
    EHRSnapshot *-- FHIRMedication : 0..*
    EHRSnapshot *-- FHIRAllergy : 0..*
    EHRSnapshot *-- RiskFlag : 0..*
    VitalSigns *-- ConsciousnessLevel
    PreTriageOutput *-- QueuePriority
    TriageOutput *-- RETTSLevel
    TriageOutput *-- Confidence
    DifferentialOutput *-- DifferentialCandidate : 0..*
    ManagementOutput *-- Confidence
    PatientSession o-- PreTriageContext
    PatientSession o-- FullTriageContext
```

---

## 6. Adapter Pattern Architecture

All AI models implement the `ModelAdapter` protocol. The `AdapterFactory` reads `config/models.yaml` and creates one of three adapter types: `MockAdapter` (loads JSON from `data/scenarios/`), `HFBaseAdapter` subclass (calls HuggingFace Inference API), or `SpaceBaseAdapter` subclass (calls HuggingFace Spaces via Gradio client). Swapping is a config change, not a code change.

### Adapter Class Hierarchy

```mermaid
classDiagram
    direction TB

    class ModelAdapter {
        <<protocol>>
        +model_id: str
        +model_name: str
        +supported_stages: list[str]
        +pretriage(PreTriageContext) PreTriageOutput
        +triage(FullTriageContext) TriageOutput
        +differential(FullTriageContext) DifferentialOutput
        +management(FullTriageContext) ManagementOutput
    }

    class BaseAdapter {
        <<abstract>>
        -_model_id: str
        -_model_name: str
        -_supported_stages: list[str]
        +pretriage() raises NotImplementedError
        +triage() raises NotImplementedError
        +differential() raises NotImplementedError
        +management() raises NotImplementedError
    }

    class MockAdapter {
        +pretriage(context) PreTriageOutput
        +triage(context) TriageOutput
        +differential(context) DifferentialOutput
        +management(context) ManagementOutput
        -_load_scenario(patient_id, stage) dict
        -_default_pretriage() PreTriageOutput
        -_default_triage() TriageOutput
    }

    class HFBaseAdapter {
        <<abstract>>
        -_hf_model_id: str
        -_timeout_seconds: int
        -_max_tokens: int
        -_client: InferenceClient
        +pretriage(context) PreTriageOutput
        +triage(context) TriageOutput
        +differential(context) DifferentialOutput
        +management(context) ManagementOutput
        -_chat_completion(messages) str
        -_parse_pretriage(raw, ms) PreTriageOutput
        -_parse_triage(raw, ms) TriageOutput
        -_parse_differential(raw, ms) DifferentialOutput
        -_parse_management(raw, ms) ManagementOutput
    }

    class MedGemma4BAdapter {
        hf_model_id: google/medgemma-4b-it
        stages: pretriage, triage, differential, management
        max_tokens: 2048
    }

    class MedGemma27BAdapter {
        hf_model_id: google/medgemma-27b-text-it
        stages: triage, differential, management
        max_tokens: 4096
    }

    class Med42Adapter {
        hf_model_id: m42-health/Llama3-Med42-8B
        stages: triage, differential
        max_tokens: 2048
    }

    class QwenMedAdapter {
        hf_model_id: wanglab/Qwen2.5-Med-7B
        stages: triage, differential
        max_tokens: 2048
    }

    class BioMistralAdapter {
        hf_model_id: BioMistral/BioMistral-7B
        stages: triage, differential
        max_tokens: 2048
    }

    class SpaceBaseAdapter {
        <<abstract>>
        -_space_id: str
        -_api_name: str
        -_client: gradio_client.Client
        -_chat_completion(messages) str
        -_flatten_messages(messages) str
        -_get_client() Client
    }

    class SpaceMedGemma4BAdapter {
        space_id: eduillueca/HumanInTheLoopTriage
        api_name: /predict
        stages: pretriage, triage, differential, management
    }

    class AdapterFactory {
        +create_adapter(model_id) BaseAdapter
        +create_all_adapters() dict
        +create_stage_adapters(stage) dict
        -_build_adapter(model_id, config) BaseAdapter
        -_build_hf_adapter(model_id, ...) BaseAdapter
        -_build_space_adapter(model_id, ...) BaseAdapter
    }

    class PromptBuilder {
        +build_pretriage_prompt(ctx) list[dict]
        +build_triage_prompt(ctx, model_id) list[dict]
        +build_differential_prompt(ctx, model_id) list[dict]
        +build_management_prompt(ctx, model_id) list[dict]
    }

    ModelAdapter <|.. BaseAdapter : implements
    BaseAdapter <|-- MockAdapter
    BaseAdapter <|-- HFBaseAdapter
    HFBaseAdapter <|-- MedGemma4BAdapter
    HFBaseAdapter <|-- MedGemma27BAdapter
    HFBaseAdapter <|-- Med42Adapter
    HFBaseAdapter <|-- QwenMedAdapter
    HFBaseAdapter <|-- BioMistralAdapter
    HFBaseAdapter <|-- SpaceBaseAdapter : inherits parsing
    SpaceBaseAdapter <|-- SpaceMedGemma4BAdapter

    AdapterFactory ..> MockAdapter : creates
    AdapterFactory ..> HFBaseAdapter : creates
    AdapterFactory ..> SpaceBaseAdapter : creates
    AdapterFactory ..> "config/models.yaml" : reads

    HFBaseAdapter ..> PromptBuilder : uses
    PromptBuilder ..> "config/prompts/*.yaml" : reads

    MockAdapter ..> "data/scenarios/" : reads JSON
```

### Config-Driven Switching

The `config/models.yaml` file controls which adapter type is instantiated for each model:

```yaml
# To switch MedGemma 4B from mock to live inference:
medgemma_4b:
  adapter: "mock"          # Change to "huggingface" or "space"
  hf_id: "google/medgemma-4b-it"
  space_id: "eduillueca/HumanInTheLoopTriage"  # Used only when adapter: "space"
  api_name: "/predict"                          # Gradio endpoint name
  stages: ["pretriage", "triage", "differential", "management"]
```

The factory uses lazy imports for HF and Space adapters to avoid pulling in `huggingface_hub` or `gradio_client` when running in mock mode:

```
AdapterFactory._build_adapter()
  |
  +-- adapter_type == "mock"          --> MockAdapter(model_id, model_name, stages)
  |
  +-- adapter_type == "huggingface"   --> importlib.import_module(module_path)
  |                                       HF adapter class via InferenceClient
  |
  +-- adapter_type == "space"         --> importlib.import_module(module_path)
  |                                       Space adapter class via gradio_client.Client
  |
  +-- adapter_type == "deterministic" --> Skipped (engines, not adapters)
```

The Space adapter (`SpaceBaseAdapter`) inherits all response parsing logic from `HFBaseAdapter`, overriding only the transport layer (`_get_client()` and `_chat_completion()`). It flattens the chat messages array into a single prompt string for the Gradio `/predict` endpoint.

### Mock Adapter File Lookup

```
data/scenarios/{patient_id}/{stage}/{model_id}.json

Example:
  data/scenarios/anders/triage/medgemma_4b.json
  data/scenarios/ella/differential/biomistral.json
```

If no scenario file exists, the MockAdapter returns a default response with `Confidence.LOW`.

---

## 7. Agreement Engine Logic

The Agreement Engine (`src/engines/agreement_engine.py`) implements the MEDLEY framework's core principle: **disagreement is preserved as a clinical resource, not collapsed into consensus**.

### Triage Agreement Flowchart

```mermaid
flowchart TB
    START["TriageEnsembleResult<br/>(engine_outputs + model_outputs)"]

    COUNT["Count votes per RETTS level<br/>across ALL outputs (engines + models)"]
    FIND_MAJ["Find majority level<br/>(highest vote count)"]

    CHECK_FULL{"most_common_count<br/>== total?"}
    CHECK_PARTIAL{"most_common_count<br/>> total / 2?"}

    FULL["AgreementLevel.FULL<br/>All models agree"]
    PARTIAL["AgreementLevel.PARTIAL<br/>Majority agrees, minority dissents"]
    NONE["AgreementLevel.NONE<br/>No clear majority"]

    DISSENT["Identify dissenting models<br/>(model_ids not voting with majority)"]
    DONT_MISS["Aggregate don't-miss diagnoses<br/>from ALL models (union, not intersection)"]
    FINAL["final_level = MOST SEVERE<br/>across all votes"]

    CHECK_ESC{"final_level !=<br/>consensus_level?"}
    ESC_YES["Set escalation_reason<br/>Minority model flagged higher severity"]
    ESC_NO["No escalation"]

    SENIOR{"agreement == NONE<br/>OR escalation?"}
    SENIOR_YES["requires_senior_review = True"]
    SENIOR_NO["requires_senior_review = False"]

    RESULT["TriageAgreement<br/>{agreement_level, consensus_level, final_level,<br/>vote_distribution, dissenting_models,<br/>dont_miss_alerts, escalation_reason,<br/>requires_senior_review}"]

    START --> COUNT --> FIND_MAJ --> CHECK_FULL
    CHECK_FULL -->|Yes| FULL
    CHECK_FULL -->|No| CHECK_PARTIAL
    CHECK_PARTIAL -->|Yes| PARTIAL
    CHECK_PARTIAL -->|No| NONE

    FULL & PARTIAL & NONE --> DISSENT --> DONT_MISS --> FINAL

    FINAL --> CHECK_ESC
    CHECK_ESC -->|Yes| ESC_YES --> SENIOR
    CHECK_ESC -->|No| ESC_NO --> SENIOR

    SENIOR -->|Yes| SENIOR_YES --> RESULT
    SENIOR -->|No| SENIOR_NO --> RESULT

    style FULL fill:#C8E6C9,color:#000
    style PARTIAL fill:#FFF9C4,color:#000
    style NONE fill:#FFCDD2,color:#000
    style ESC_YES fill:#FFCDD2,color:#000
    style SENIOR_YES fill:#FFCDD2,color:#000
```

### Cross-Engine Disagreement Analysis

When multiple deterministic engines are active (RETTS, ESI, MTS), the `analyze_engine_disagreement()` function compares their outputs and explains *why* different triage philosophies disagree:

| Scenario | RETTS | ESI | MTS | Clinical Meaning |
|----------|-------|-----|-----|-----------------|
| All agree on ORANGE | ORANGE | ORANGE | ORANGE | High confidence; proceed with standard workflow |
| ESI higher than RETTS | YELLOW | ORANGE | YELLOW | ESI's resource-prediction flagged higher acuity |
| MTS higher than others | GREEN | GREEN | ORANGE | MTS discriminator caught a sign others missed |
| All disagree | YELLOW | ORANGE | RED | Different philosophies see different risk; senior review mandatory |

### Differential Three-Tier Output

```mermaid
flowchart LR
    subgraph "Input: DifferentialEnsembleResult"
        M1["MedGemma 4B<br/>candidates"]
        M2["MedGemma 27B<br/>candidates"]
        M3["Med42<br/>candidates"]
        M4["Qwen2.5-Med<br/>candidates"]
        M5["BioMistral<br/>candidates"]
    end

    COUNT_DX["Count each diagnosis<br/>across models"]

    subgraph "Output: DifferentialAgreement"
        ALL["all_agree (>=80%)<br/>Primary diagnoses"]
        SOME["some_agree (40-79%)<br/>Alternative diagnoses"]
        DEVIL["devil_advocate_only (<40%)<br/>Minority / don't-miss"]
    end

    DONT_MISS_OUT["dont_miss_all<br/>(any model's is_dont_miss=true)"]

    M1 & M2 & M3 & M4 & M5 --> COUNT_DX
    COUNT_DX --> ALL & SOME & DEVIL
    COUNT_DX --> DONT_MISS_OUT

    style ALL fill:#C8E6C9,color:#000
    style SOME fill:#FFF9C4,color:#000
    style DEVIL fill:#FFCDD2,color:#000
    style DONT_MISS_OUT fill:#B84145,color:#FFF
```

### Don't-Miss Alert Propagation

The don't-miss mechanism ensures that even a single model's safety concern is surfaced:

1. **TriageOutput.dont_miss**: Each model/engine can flag don't-miss diagnoses in their output
2. **DifferentialCandidate.is_dont_miss**: Individual differential candidates can be flagged
3. **TriageAgreement.dont_miss_alerts**: Union of all don't-miss from all models (not discarded even if only 1/8 flagged it)
4. **DifferentialAgreement.dont_miss_all**: Aggregated from all models' candidates where `is_dont_miss=True`
5. **UI**: Don't-miss items are displayed with red highlighting regardless of where they appear in the agreement tiers

---

## 8. ASR Disagreement Pipeline

Based on the "From Black Box to Glass Box" research, the ASR pipeline runs two independent speech recognition models and uses their disagreement as a quality signal.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Audio as Audio Input
    participant ASR_SVC as ASR Service
    participant MedASR as MedASR<br/>(Conformer, 105M params)
    participant Whisper as Whisper Large v3<br/>(Transformer)
    participant Align as Word-Level Aligner<br/>(LCS Dynamic Programming)
    participant Classify as Clinical Significance<br/>Classifier
    participant Color as Color-Coded<br/>Transcript Builder

    Audio->>ASR_SVC: process_audio(patient_id)

    par Dual-ASR Processing
        ASR_SVC->>MedASR: Transcribe audio
        MedASR-->>ASR_SVC: medasr_transcript
    and
        ASR_SVC->>Whisper: Transcribe audio
        Whisper-->>ASR_SVC: whisper_transcript
    end

    ASR_SVC->>Align: compute_word_alignment(<br/>medasr_words, whisper_words)
    Note over Align: LCS-based dynamic programming<br/>O(m*n) alignment

    Align-->>ASR_SVC: alignment: list[(index, medasr_word, whisper_word)]

    ASR_SVC->>ASR_SVC: detect_disagreements(alignment)
    Note over ASR_SVC: Direct substitutions +<br/>Adjacent insertion/deletion pairs

    loop For each disagreement
        ASR_SVC->>Classify: classify_clinical_significance(word)
        Note over Classify: medication terms -> HIGH<br/>symptom terms -> HIGH<br/>filler words -> LOW<br/>other -> MODERATE
        Classify-->>ASR_SVC: "high" / "moderate" / "low"
    end

    ASR_SVC->>Color: build_colored_transcript(alignment, disagreements)
    Note over Color: GREEN: both agree<br/>AMBER: minor divergence<br/>RED: major divergence on clinical terms

    Color-->>ASR_SVC: HTML transcript with<br/>span classes: asr-agree,<br/>asr-minor-diverge, asr-major-diverge

    ASR_SVC-->>Audio: ASRResult {<br/>  medasr_transcript,<br/>  whisper_transcript,<br/>  merged_transcript,<br/>  disagreements[],<br/>  confidence_score<br/>}
```

### Clinical Significance Weighting

| Category | Weight | Examples | Disagreement Color |
|----------|--------|---------|-------------------|
| **Medication** | 1.0 | warfarin, metoprolol, ibuprofen | RED |
| **Dosage** | 1.0 | 5mg, 100ml | RED |
| **Symptom** | 0.8 | pain, fever, bleeding, dizzy | RED |
| **Anatomy** | 0.7 | chest, head, abdomen | AMBER/RED |
| **Temporal** | 0.5 | yesterday, 3 days ago | AMBER |
| **Qualifier** | 0.4 | severe, mild, worse | AMBER |
| **Filler** | 0.1 | um, uh, like, so | (ignored) |
| **Default** | 0.3 | Other words | AMBER |

The system maintains explicit lookup sets for medication terms (42 entries: warfarin, waran, metformin, metoprolol, etc.) and symptom terms (26 entries in both English and Swedish: pain/smartra, fever/feber, etc.).

### Example: Anders Scenario

```
MedASR:  "...I take warfarin and metoprolol..."
Whisper: "...I take waran and metoprolol..."
                     ^^^^^^^^   ^^^^^^^^^^
                     RED         GREEN
                  (high significance:
                   medication term disagrees)

Resolution: EHR confirms warfarin prescription -> resolved_to = "warfarin"
```

---

## 9. Role-Based Access Control

Authentication uses a mock credential system (`src/services/auth_service.py`) with four roles. In production, this would connect to hospital SSO/LDAP.

### Role Hierarchy and Page Access

```mermaid
graph TB
    subgraph "Roles"
        PATIENT["Patient<br/>(no login required)"]
        NURSE["Triage Nurse<br/>(PIN: 1234)"]
        PHYSICIAN["Physician<br/>(PIN: 5678)"]
        ADMIN["Admin<br/>(PIN: 0000)"]
    end

    subgraph "Pages"
        K["0_Kiosk<br/>Walk-in arrival"]
        Q["1_Queue_View<br/>Priority queue"]
        T["2_Triage_View<br/>Ensemble triage"]
        P["3_Physician_View<br/>Differential + management"]
        PE["4_Prompt_Editor<br/>YAML editing"]
        AL["5_Audit_Log<br/>Decision trail"]
        EC["6_Engine_Config<br/>Engine toggles + API keys"]
    end

    PATIENT --> K
    NURSE --> Q & T
    PHYSICIAN --> Q & P
    ADMIN --> K & Q & T & P & PE & AL & EC

    style PATIENT fill:#54B986,color:#000
    style NURSE fill:#F59A00,color:#000
    style PHYSICIAN fill:#4DB5BC,color:#000
    style ADMIN fill:#4F0433,color:#FFF
```

### Access Matrix

| Page | Patient | Triage Nurse | Physician | Admin |
|------|:-------:|:------------:|:---------:|:-----:|
| **Kiosk** | Yes | -- | -- | Yes |
| **Queue View** | -- | Yes | Yes | Yes |
| **Triage View** | -- | Yes | -- | Yes |
| **Physician View** | -- | -- | Yes | Yes |
| **Prompt Editor** | -- | -- | -- | Yes |
| **Audit Log** | -- | -- | -- | Yes |
| **Engine Config** | -- | -- | -- | Yes |

### Demo Credentials

| Username | PIN | Role | Display Name |
|----------|-----|------|-------------|
| `nurse_anna` | `1234` | Triage Nurse | Anna Lindberg, RN |
| `nurse_erik` | `1234` | Triage Nurse | Erik Holm, RN |
| `dr_nilsson` | `5678` | Physician | Dr. Sara Nilsson |
| `dr_berg` | `5678` | Physician | Dr. Magnus Berg |
| `admin` | `0000` | Admin | System Admin |

### Navigation Flow

```mermaid
flowchart TB
    START["app.py starts"]
    CHECK_ROLE{"st.session_state.role?"}
    CHECK_LOGIN{"_login_target_role set?"}

    ROLE_SEL["Role Selector Page<br/>(4 cards: Patient, Nurse, Physician, Admin)<br/>sidebar: hidden"]
    LOGIN["Login Page<br/>(username + 4-digit PIN)<br/>sidebar: hidden"]
    PAT_NAV["Patient Navigation<br/>[Kiosk only]<br/>sidebar: hidden"]
    STAFF_NAV["Staff Navigation<br/>Sidebar: brand, user info, logout,<br/>demo load button<br/>Pages: role-filtered"]
    ADMIN_NAV["Admin Navigation<br/>Sidebar: same as staff<br/>Pages grouped: Clinical + Development"]

    START --> CHECK_ROLE
    CHECK_ROLE -->|None| CHECK_LOGIN
    CHECK_LOGIN -->|No| ROLE_SEL
    CHECK_LOGIN -->|Yes| LOGIN
    CHECK_ROLE -->|PATIENT| PAT_NAV
    CHECK_ROLE -->|TRIAGE_NURSE| STAFF_NAV
    CHECK_ROLE -->|PHYSICIAN| STAFF_NAV
    CHECK_ROLE -->|ADMIN| ADMIN_NAV

    ROLE_SEL -->|"Patient card"| PAT_NAV
    ROLE_SEL -->|"Staff card"| LOGIN
    LOGIN -->|"authenticate()"| STAFF_NAV
    LOGIN -->|"back"| ROLE_SEL
```

---

## 10. Source Code Layout

```
triage-medley/
|
|-- app.py                              # Streamlit entry point: role selector, login, st.navigation() router
|-- CLAUDE.md                           # Project instructions for Claude Code (build commands, architecture, conventions)
|
|-- config/
|   |-- models.yaml                     # Model registry: adapter type (mock|huggingface|space), HF IDs, Space IDs, stages
|   |-- engines.yaml                    # Engine registry: RETTS/ESI/MTS descriptions, philosophies, defaults
|   |-- pretriage.yaml                  # Pre-triage rules: red-flag keywords, priority mappings, risk amplifiers
|   |-- prompts/
|   |   |-- pretriage.yaml              # Stage A prompt template: system_prompt + user_template
|   |   |-- triage.yaml                # Stage B triage prompt template
|   |   |-- differential.yaml          # Differential diagnosis prompt template
|   |   |-- management.yaml            # Management plan prompt template
|   |-- retts/
|   |   |-- ess_codes.json             # ESS category definitions and default severity levels
|   |   |-- vitals_cutoffs.json        # Age-stratified vital sign thresholds (adult + pediatric)
|   |-- esi/
|   |   |-- decision_tree.json         # ESI decision criteria: ESI-1/2 thresholds, danger zone, intervention keywords
|   |   |-- resource_rules.json        # Resource estimation rules per ESS category
|   |-- mts/
|       |-- flowcharts.json            # MTS complaint-specific flowcharts with discriminators
|       |-- general_discriminators.json # MTS fallback discriminators
|
|-- src/
|   |-- __init__.py
|   |
|   |-- models/                         # Pydantic data classes (Layer 0: Data Types)
|   |   |-- __init__.py                 # Re-exports all models
|   |   |-- enums.py                    # RETTSLevel, QueuePriority, Confidence, ArrivalPathway, ConsciousnessLevel
|   |   |-- vitals.py                   # VitalSigns (HR, BP, RR, SpO2, Temp, AVPU)
|   |   |-- clinical.py                # FHIRCondition, FHIRMedication, FHIRAllergy, RiskFlag, EHRSnapshot,
|   |   |                              # Symptom, ASRDisagreement
|   |   |-- context.py                 # PreTriageContext (no vitals), FullTriageContext (inherits + requires vitals)
|   |   |-- outputs.py                 # PreTriageOutput, TriageOutput, DifferentialCandidate, DifferentialOutput,
|   |                                  # ManagementOutput
|   |
|   |-- adapters/                       # Model Adapter pattern (Layer 1: Model Interface)
|   |   |-- __init__.py
|   |   |-- base.py                     # ModelAdapter protocol + BaseAdapter abstract class
|   |   |-- mock_adapter.py             # MockAdapter: loads JSON from data/scenarios/{patient_id}/{stage}/{model_id}.json
|   |   |-- factory.py                  # AdapterFactory: reads models.yaml, creates mock, HF, or Space adapters
|   |   |-- prompt_builder.py           # Renders YAML prompt templates with clinical context for HF/Space adapters
|   |   |-- hf_base.py                  # HFBaseAdapter: InferenceClient, chat_completion, JSON extraction, response parsers
|   |   |-- hf_medgemma.py              # MedGemma4BAdapter + MedGemma27BAdapter (inherits HFBaseAdapter)
|   |   |-- hf_ensemble.py              # Med42Adapter + QwenMedAdapter + BioMistralAdapter (inherits HFBaseAdapter)
|   |   |-- space_base.py               # SpaceBaseAdapter: gradio_client, predict, inherits parsing from HFBaseAdapter
|   |   |-- space_medgemma.py            # SpaceMedGemma4BAdapter (inherits SpaceBaseAdapter)
|   |
|   |-- engines/                        # Deterministic Triage Engines (Layer 2: Clinical Rules)
|   |   |-- __init__.py
|   |   |-- pretriage_engine.py         # Stage A: speech red-flag matching + EHR risk amplification -> QueuePriority
|   |   |-- retts_engine.py             # RETTS: vitals thresholds + ESS codes -> higher of two -> RETTS colour
|   |   |-- esi_engine.py               # ESI: 5-level decision tree (dying? high-risk? resources? danger zone?)
|   |   |-- mts_engine.py              # MTS: complaint-specific flowchart -> top-down discriminator walk
|   |   |-- agreement_engine.py         # Ensemble consensus: TriageAgreement, EngineDisagreement,
|   |                                   # DifferentialAgreement (3-tier), ManagementAgreement
|   |
|   |-- services/                       # Application Services (Layer 3: Orchestration)
|   |   |-- __init__.py
|   |   |-- orchestrator.py             # Parallel model dispatch (ThreadPoolExecutor), run_pretriage,
|   |   |                               # run_triage_ensemble, run_differential_ensemble, run_management_ensemble,
|   |   |                               # run_full_pipeline
|   |   |-- ehr_service.py              # FHIR Bundle parser, risk flag computation (9 rules: anticoagulation,
|   |   |                               # cardiac polypharmacy, immunosuppression, beta-blocker masking, etc.)
|   |   |-- asr_service.py              # Dual-ASR: LCS word alignment, clinical significance classification,
|   |   |                               # color-coded transcript builder, mock data for 6 scenarios
|   |   |-- auth_service.py             # Role enum, ROLE_PAGES access matrix, DEMO_CREDENTIALS, authenticate()
|   |   |-- pdf_service.py              # MedleyPDF (fpdf2): triage + physician reports, KI branding, RETTS badges
|   |   |-- session_manager.py          # PatientSession dataclass, Streamlit session state management,
|   |                                   # register_kiosk_patient(), load_demo_scenarios()
|   |
|   |-- utils/                          # Shared Utilities (Layer 4: Infrastructure)
|       |-- __init__.py
|       |-- config.py                   # YAML/JSON config loader with caching, path helpers
|       |-- audit.py                    # Thread-safe JSONL audit logger, AuditEvent model, filtering queries
|       |-- theme.py                    # KI color constants, custom CSS injection, footer rendering
|
|-- pages/                              # Streamlit Multi-Page App (Layer 5: HITL Visualization)
|   |-- 0_Kiosk.py                      # Patient walk-in: ID entry / scenario selection -> Stage A pipeline
|   |-- 1_Queue_View.py                 # Charge nurse: priority-ordered queue, patient selection, status badges
|   |-- 2_Triage_View.py               # Triage nurse: vitals entry, ensemble result, vote distribution,
|   |                                   # cross-engine comparison, nurse override, PDF download
|   |-- 3_Physician_View.py            # Physician: 3-tier differential, management plan, approve/modify,
|   |                                   # sign-off, PDF download
|   |-- 4_Prompt_Editor.py             # Admin: live YAML prompt template editing
|   |-- 5_Audit_Log.py                 # Admin: filterable audit trail viewer
|   |-- 6_Engine_Config.py             # Admin: toggle engines (RETTS/ESI/MTS), toggle LLM models,
|                                       # set HF API key, test Inference API / Space connection
|
|-- data/
|   |-- scenarios/                      # Mock JSON responses per patient per stage per model
|   |   |-- anders/                     # 68M, chest tightness (consensus scenario)
|   |   |   |-- pretriage/              # medgemma_4b.json
|   |   |   |-- triage/                 # medgemma_4b.json, medgemma_27b.json, med42.json, qwen_med.json, biomistral.json
|   |   |   |-- differential/           # (same models)
|   |   |   |-- management/             # medgemma_4b.json, medgemma_27b.json
|   |   |-- ella/                       # 4F, fever + rash (don't-miss: meningococcal)
|   |   |-- margit/                     # 81F, fall on warfarin (EHR escalation)
|   |   |-- ingrid/                     # 79F, hidden urosepsis (deceptive vitals)
|   |   |-- erik/                       # 72M, TIA + melena (aortic dissection)
|   |   |-- sofia/                      # 28F, pulled muscle (bilateral PE)
|   |
|   |-- ehr/                            # Synthea FHIR R4 Bundles (6 synthetic patients)
|   |   |-- anders.json
|   |   |-- ella.json
|   |   |-- margit.json
|   |   |-- ingrid.json
|   |   |-- erik.json
|   |   |-- sofia.json
|   |
|   |-- audit/
|       |-- audit.jsonl                 # Append-only audit log (auto-created)
|
|-- tests/
|   |-- __init__.py
|   |-- test_models.py                  # Pydantic model tests (validation, type enforcement)
|   |-- test_pretriage.py               # Pre-triage engine tests
|   |-- test_retts.py                   # RETTS engine tests
|   |-- test_esi.py                     # ESI engine tests
|   |-- test_mts.py                     # MTS engine tests
|   |-- test_ehr.py                     # EHR service tests (FHIR parsing, risk flags)
|   |-- test_audit.py                   # Audit logger tests
|   |-- test_adapters.py               # Mock adapter tests
|   |-- test_hf_adapters.py            # HuggingFace Inference API adapter tests
|   |-- test_space_adapters.py         # HuggingFace Space adapter tests
|   |-- test_multi_engine.py           # Multi-engine orchestration + agreement tests
|   |-- test_pipeline.py               # End-to-end pipeline integration tests
|   |-- test_auth.py                   # Authentication + RBAC tests
|   |-- test_pdf.py                    # PDF generation tests
|
|-- ref/                                # Reference documentation
|-- presentation/                       # Presentation assets
|-- docs/
    |-- ARCHITECTURE.md                 # This file
```

### Key File Counts

| Directory | Files | Purpose |
|-----------|------:|---------|
| `src/models/` | 5 | Pydantic data classes |
| `src/adapters/` | 9 | Model adapter protocol + implementations (mock, HF, Space) |
| `src/engines/` | 5 | Deterministic engines + agreement analysis |
| `src/services/` | 6 | Application services (orchestration, EHR, ASR, auth, PDF, session) |
| `src/utils/` | 3 | Config, audit, theme |
| `pages/` | 7 | Streamlit UI pages |
| `config/` | 11+ | YAML/JSON clinical rules and prompt templates |
| `data/scenarios/` | ~60 | Mock JSON responses (6 patients x 4 stages x ~2-5 models) |
| `data/ehr/` | 6 | Synthetic FHIR patient bundles |
| `tests/` | 13 | Unit + integration tests |

---

*Document generated from source code analysis of the Triage-Medley repository.*
*Last updated: February 2026.*
