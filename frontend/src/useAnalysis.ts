// Run-flow hook: submit a drug → poll status until terminal → expose result.

import { useCallback, useEffect, useRef, useState } from "react";
import { cancelAnalysis, createAnalysis, getAnalysis } from "./api";
import {
  TERMINAL_STATUSES,
  type AnalysisStatus,
  type JobStatus,
  type SupervisorOutput,
} from "./types";

const POLL_INTERVAL_MS = 1500;

export interface AnalysisState {
  jobId: string | null;
  status: JobStatus | "idle";
  data: AnalysisStatus | null;
  error: string | null;
}

const INITIAL: AnalysisState = { jobId: null, status: "idle", data: null, error: null };

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>(INITIAL);
  const timer = useRef<number | null>(null);

  const clearTimer = () => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  };

  // Stop polling on unmount.
  useEffect(() => clearTimer, []);

  const poll = useCallback((jobId: string) => {
    getAnalysis(jobId)
      .then((data) => {
        setState({ jobId, status: data.status, data, error: data.error });
        if (!TERMINAL_STATUSES.includes(data.status)) {
          timer.current = window.setTimeout(() => poll(jobId), POLL_INTERVAL_MS);
        }
      })
      .catch((err: Error) => {
        setState((s) => ({ ...s, status: "error", error: err.message }));
      });
  }, []);

  const run = useCallback(
    (drugName: string) => {
      clearTimer();
      setState({ jobId: null, status: "pending", data: null, error: null });
      createAnalysis(drugName)
        .then(({ job_id }) => {
          setState((s) => ({ ...s, jobId: job_id, status: "running" }));
          poll(job_id);
        })
        .catch((err: Error) => {
          setState({ jobId: null, status: "error", data: null, error: err.message });
        });
    },
    [poll],
  );

  const stop = useCallback(() => {
    if (!state.jobId) return;
    // Keep polling so the terminal `cancelled` status is fetched and rendered —
    // do NOT clearTimer() here, or the UI would freeze on the pre-cancel state.
    cancelAnalysis(state.jobId).catch(() => {
      /* server reflects cancelled status on next poll; ignore transient errors */
    });
  }, [state.jobId]);

  const reset = useCallback(() => {
    clearTimer();
    setState(INITIAL);
  }, []);

  // Dev-only: inject a SupervisorOutput payload directly as a finished job,
  // bypassing the POST/poll cycle. Lets the UI render without the pipeline.
  const loadSample = useCallback(
    (result: SupervisorOutput, jobId: string = "sample") => {
      clearTimer();
      setState({
        jobId,
        status: "done",
        data: {
          job_id: jobId,
          drug_name: result.drug_name,
          status: "done",
          result,
          error: null,
          progress: [],
        },
        error: null,
      });
    },
    [],
  );

  return { state, run, stop, reset, loadSample };
}
