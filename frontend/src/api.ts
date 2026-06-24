// API client for the analyses endpoints. Paths are same-origin (/api proxied to
// uvicorn in dev, served by FastAPI in prod).

import type { AnalysisCreated, AnalysisStatus } from "./types";

async function asJson<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const body = await resp.text();
    // FastAPI error bodies are {"detail": "..."}; surface that plainly when present.
    let detail = body;
    try {
      const parsed = JSON.parse(body);
      if (parsed && typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      /* non-JSON body: use the raw text */
    }
    throw new Error(detail);
  }
  return resp.json() as Promise<T>;
}

export async function createAnalysis(drugName: string): Promise<AnalysisCreated> {
  const resp = await fetch("/api/analyses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ drug_name: drugName }),
  });
  return asJson<AnalysisCreated>(resp);
}

export async function getAnalysis(jobId: string): Promise<AnalysisStatus> {
  return asJson<AnalysisStatus>(await fetch(`/api/analyses/${jobId}`));
}

export async function cancelAnalysis(jobId: string): Promise<void> {
  const resp = await fetch(`/api/analyses/${jobId}`, { method: "DELETE" });
  if (!resp.ok && resp.status !== 204) {
    throw new Error(`Cancel failed: ${resp.status}`);
  }
}

export async function getExample(drug: string): Promise<AnalysisStatus> {
  return asJson<AnalysisStatus>(await fetch(`/api/examples/${encodeURIComponent(drug)}`));
}

export async function getReportMarkdown(jobId: string): Promise<string> {
  const resp = await fetch(`/api/analyses/${jobId}/report.md`);
  if (!resp.ok) throw new Error(`Report unavailable: ${resp.status}`);
  return resp.text();
}

export async function getExampleReportMarkdown(drug: string): Promise<string> {
  const resp = await fetch(`/api/examples/${encodeURIComponent(drug)}/report.md`);
  if (!resp.ok) throw new Error(`Report unavailable: ${resp.status}`);
  return resp.text();
}
