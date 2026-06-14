// TS mirror of the backend Pydantic contracts. Sources of truth:
//   - api/schemas/analyses.py
//   - agents/supervisor/supervisor_output.py
//   - agents/{mechanism,clinical_trials,literature}/*_output.py
//   - models/model_clinical_trials.py, model_evidence_summary.py, model_open_targets.py
// Optional/None fields on the backend are `T | null` here. Pydantic defensive
// defaults mean list/dict/str/int fields are always present (never null).

export type JobStatus = "pending" | "running" | "done" | "error" | "cancelled";

export type EvidenceStrength = "strong" | "moderate" | "weak" | "none";

// ----- Mechanism -----

export interface MechanismOfAction {
  mechanism_of_action: string;
  action_type: string | null;
  target_ids: string[];
  target_symbols: string[];
}

export interface MechanismCandidate {
  target_symbol: string;
  action_type: string;
  disease_name: string;
  disease_id: string | null;
  disease_description: string;
  target_function: string;
}

export interface MechanismOutput {
  drug_targets: Record<string, string>; // gene symbol → Ensembl ID
  mechanisms_of_action: MechanismOfAction[];
  candidates: MechanismCandidate[];
  summary: string;
}

// ----- Clinical trials -----

export interface Trial {
  nct_id: string;
  title: string;
  brief_summary: string | null;
  phase: string;
  overall_status: string;
  why_stopped: string | null;
  sponsor: string;
  enrollment: number | null;
  start_date: string | null;
  completion_date: string | null;
}

export interface SearchTrialsResult {
  total_count: number;
  by_status: Record<string, number>;
  trials: Trial[];
}

export interface CompletedTrialsResult {
  total_count: number;
  trials: Trial[];
}

export interface TerminatedTrialsResult {
  total_count: number;
  trials: Trial[];
}

export interface CompetitorEntry {
  sponsor: string;
  drug_name: string;
  drug_type: string | null;
  max_phase: string;
  trial_count: number;
  total_enrollment: number;
  most_recent_start: string | null;
}

export interface IndicationLandscape {
  total_trial_count: number | null;
  competitors: CompetitorEntry[];
  phase_distribution: Record<string, number>;
}

export interface ClinicalTrialsOutput {
  search: SearchTrialsResult | null;
  completed: CompletedTrialsResult | null;
  terminated: TerminatedTrialsResult | null;
  landscape: IndicationLandscape | null;
  summary: string;
  // NCTs the agent judged a different disease/drug — filtered from the trial
  // tables (mirrors the markdown report). The total_count headers stay verbatim.
  contaminated_nct_ids: string[];
}

// ----- Literature -----

export interface EvidenceSummary {
  summary: string;
  study_count: number;
  strength: EvidenceStrength;
  key_findings: string[];
  supporting_pmids: string[];
}

export interface LiteratureOutput {
  pmids: string[];
  evidence_summary: EvidenceSummary | null;
  summary: string;
}

// ----- Supervisor (top-level) -----

export interface CandidateBlurb {
  stage: string;
  literature: string;
  blocker: string;
  active_programs: string;
  key_risk: string;
  verdict: string;
  watch: string;
  prose: string;
}

export interface CandidateFindings {
  disease: string;
  source: "competitor" | "mechanism" | "both";
  literature: LiteratureOutput | null;
  clinical_trials: ClinicalTrialsOutput | null;
  blurb: CandidateBlurb | null;
}

export interface SupervisorOutput {
  drug_name: string;
  candidate_diseases: string[];
  mechanism: MechanismOutput | null;
  disease_findings: CandidateFindings[];
  top_diseases: string[];
  summary: string;
}

export interface AnalysisCreated {
  job_id: string;
  status: JobStatus;
}

export interface AnalysisStatus {
  job_id: string;
  drug_name: string;
  status: JobStatus;
  result: SupervisorOutput | null;
  error: string | null;
}

export const TERMINAL_STATUSES: JobStatus[] = ["done", "error", "cancelled"];
