// TS mirror of the backend Pydantic contracts (api/schemas/analyses.py +
// agents/supervisor/supervisor_output.py). Kept loose for nested sub-agent
// payloads (rendered in Phase 5); only the fields the run flow needs are typed.

export type JobStatus = "pending" | "running" | "done" | "error" | "cancelled";

export interface CandidateFindings {
  disease: string;
  source: "competitor" | "mechanism" | "both";
  literature: unknown | null;
  clinical_trials: unknown | null;
  blurb: unknown | null;
}

export interface SupervisorOutput {
  drug_name: string;
  candidate_diseases: string[];
  mechanism: unknown | null;
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
