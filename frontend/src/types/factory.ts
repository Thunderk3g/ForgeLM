// Request and response shapes of the factory service
// (src/microservices/factory_service/src/main.py). That file is the contract:
// every interface here mirrors either a pydantic request model there or the
// dict an endpoint returns, which in turn comes from a forgelm ``to_dict``.
// Field names are kept in the service's snake_case so a payload can be passed
// through without renaming.

// ---- shared ----------------------------------------------------------------

// The four planner task kinds (forgelm.planner.TASK_KINDS).
export const TASK_KINDS = [
  "text_classification",
  "tabular_classification",
  "generation",
  "document_qa",
] as const
export type TaskKind = (typeof TASK_KINDS)[number]

// The student kinds POST /classifiers accepts.
export const STUDENT_KINDS = ["tfidf", "static", "encoder", "finetuned"] as const
export type StudentKind = (typeof STUDENT_KINDS)[number]

// Evidence levels from forgelm/evidence.py, strongest first.
export const EVIDENCE_LEVELS = ["measured", "published", "reported", "assumption"] as const
export type EvidenceLevel = (typeof EVIDENCE_LEVELS)[number]

// FastAPI's error body. HTTPException gives a string; request validation
// gives a list of issues (the service's own handler keeps that shape).
export interface ValidationIssue {
  type: string
  loc: (string | number)[]
  msg: string
  input?: unknown
  ctx?: Record<string, unknown>
}
export interface ErrorBody {
  detail: string | ValidationIssue[]
}

// ---- GET /hardware ---------------------------------------------------------

// forgelm.hardware.Accelerator, via dataclasses.asdict.
export interface Accelerator {
  runtime: string // "cuda" | "rocm" | "mps" | "xpu"
  vendor: string // "nvidia" | "amd" | "apple" | "intel"
  name: string
  memory_gb: number
  count: number
  unified_memory: boolean
  // false when the OS reports the device but the installed torch cannot use
  // it; null when torch was not checked (the service probes without torch).
  torch_usable: boolean | null
  notes: string[]
}

export type DeviceClass = "cpu" | "nvidia" | "amd" | "apple" | "intel_gpu"

// forgelm.hardware.HardwareProfile.to_dict().
export interface HardwareProfile {
  os: string
  arch: string
  python: string
  cpu_model: string
  physical_cores: number
  logical_cores: number
  cpu_flags: string[]
  ram_total_gb: number
  ram_available_gb: number
  in_container: boolean
  accelerators: Accelerator[]
  packages: Record<string, string> // import name -> version ("ref" on reference profiles)
  onnx_providers: string[]
  notes: string[]
  device_class: DeviceClass
  has_vnni: boolean
}

// forgelm.generative.capacity().
export interface GenerativeCapacity {
  backend: string
  device: string
  usable_memory_gb: number
  max_params_billion: number
  basis: string
  install_hint: string[]
}

export interface HardwareResponse {
  profile: HardwareProfile
  generative_capacity: GenerativeCapacity
  reference_profiles: string[]
}

// ---- POST /plan ------------------------------------------------------------

// PlanRequest. Every field has a server default, so all are optional here.
export interface PlanRequest {
  task?: TaskKind
  n_classes?: number // >= 1
  labelled_per_class?: number // >= 0
  n_unlabelled?: number // >= 0
  avg_tokens?: number // >= 1
  multilingual?: boolean
  needs_world_knowledge?: boolean
  n_rows?: number // >= 0
  n_features?: number // >= 0
  has_text_columns?: boolean
  temporal_or_grouped?: boolean
  model_params_billion?: number // > 0, <= 2000
  knowledge_changes_often?: boolean
  llm_at_inference?: boolean
  teacher_available?: boolean
  commercial_use?: boolean
  needs_inherent_explainability?: boolean
  data_must_stay_on_prem?: boolean
  max_latency_ms?: number | null // > 0
  hardware?: string | null // reference profile to plan for; null = the service's machine
  deploy?: string | null // reference profile of the deployment machine; null = same as training
}

// forgelm.planner.TaskSpec, via asdict.
export interface TaskSpec {
  kind: TaskKind
  n_classes: number
  labelled_per_class: number
  n_unlabelled: number
  avg_tokens: number
  multilingual: boolean
  needs_world_knowledge: boolean
  n_rows: number
  n_features: number
  has_text_columns: boolean
  temporal_or_grouped: boolean
  model_params_billion: number
  knowledge_changes_often: boolean
}

// One forgelm.evidence.EVIDENCE entry, with its key added by Plan.to_dict.
export interface EvidenceEntry {
  key: string
  level: EvidenceLevel
  claim: string
  source: string // a repository path, or one or more URLs separated by " ; "
}

// forgelm.planner.Step, as Plan.to_dict writes it.
export interface PlanStep {
  name: string
  action: string
  why: string
  evidence: EvidenceEntry[]
  requires: string[]
  missing: string[] // pip names of required packages that are not installed
  feasible: boolean
  detail: Record<string, unknown>
  evidence_level: EvidenceLevel // the weakest level among ``evidence``
}

export interface NotRecommended {
  what: string
  why: string
}

// forgelm.planner.Plan.to_dict().
export interface Plan {
  task: TaskSpec
  hardware_class: string
  deploy_class: string
  approach: string
  approach_why: string
  steps: PlanStep[]
  deployment: string[]
  guardrails: string[]
  warnings: string[]
  not_recommended: NotRecommended[]
}

export interface PlanResponse {
  plan: Plan
  text: string // the same plan rendered as plain text
}

// ---- classifiers -----------------------------------------------------------

export interface TrainRequest {
  name: string // [A-Za-z0-9][A-Za-z0-9_-]{0,63}
  texts: string[]
  labels: string[]
  student?: StudentKind
  model?: string | null // one of the service's FACTORY_ALLOWED_MODELS
  holdout?: number // 0 < holdout < 0.9
  min_accuracy?: number | null
  seed?: number // >= 0
}

// forgelm.metrics.classification_report(). The service returns plain floats;
// a non-finite value would have to travel as a string, so the UI formats
// defensively.
export interface ClassificationMetrics {
  n: number
  accuracy: number
  macro_precision: number
  macro_recall: number
  macro_f1: number
  ece: number
  brier: number
  nll: number
}

// forgelm.provenance.EvalContract.check().
export interface ContractVerdict {
  contract: string
  passed: boolean
  failures: string[]
}

export interface TrainResponse {
  name: string
  labels: string[]
  n_train: number
  n_calibration: number
  n_test: number
  metrics: ClassificationMetrics
  temperature: number
  contract: ContractVerdict
  serve_requires: string[]
}

export interface ClassifierSummary {
  name: string
  created_utc: string | null
  student: string | null
  accuracy: number | null
  contract_passed: boolean | null
}

export interface ClassifierListResponse {
  classifiers: ClassifierSummary[]
}

// forgelm.provenance.RunRecord, as written to run_record.json.
export interface RunRecord {
  name: string
  task: string
  recipe: Record<string, unknown>
  data: Record<string, { n?: number; sha256?: string }>
  metrics: Record<string, unknown>
  hardware: Record<string, unknown>
  base_models: { name?: string; revision?: string; licence?: string }[]
  artefacts: Record<string, string> // relative path -> sha256
  contract: ContractVerdict | null
  notes: string[]
  run_id: string
  created_utc: string
}

export interface ClassifierCardResponse {
  run_record: RunRecord
  model_card: string // Markdown
}

export interface PredictRequest {
  texts: string[]
  threshold?: number // 0..1; abstain below this confidence
}

export interface Prediction {
  label: string | null // null when abstained
  abstained: boolean
  confidence: number
}

export interface PredictResponse {
  predictions: Prediction[]
}

// ---- POST /review-call -----------------------------------------------------

export interface ReviewCallRequest {
  utterances: string[]
  classifier?: string | null
  flag_threshold?: number // 0..1
  unsure_below?: number // 0..1
}

// forgelm.packs.bfsi.Flag.
export interface ReviewFlag {
  tier: "rule" | "classifier"
  category: string
  utterance_index: number
  span: string // the exact text that triggered the flag
  confidence: number | null // classifier flags only
}

// forgelm.packs.bfsi.CallReview.to_dict() plus the endpoint's disclaimer.
export interface ReviewCallResponse {
  disclosures: Record<string, string | null> // check -> matched span, or null when not found
  missing_disclosures: string[]
  flags: ReviewFlag[]
  needs_review: number[]
  disclaimer: string
}
