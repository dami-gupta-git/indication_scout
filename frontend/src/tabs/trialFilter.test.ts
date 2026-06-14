import { describe, it, expect } from "vitest";
import { partitionTrials } from "./trialFilter";
import type { Trial } from "../types";

function trial(nctId: string): Trial {
  return {
    nct_id: nctId,
    title: "",
    brief_summary: null,
    phase: "Phase 3",
    overall_status: "COMPLETED",
    why_stopped: null,
    sponsor: "",
    enrollment: null,
    start_date: null,
    completion_date: null,
  };
}

describe("partitionTrials", () => {
  it("splits contaminated NCTs out while preserving order of the rest", () => {
    const trials = [trial("NCT1"), trial("NCT2"), trial("NCT3")];
    const { shown, excluded } = partitionTrials(trials, ["NCT2"]);
    expect(shown.map((t) => t.nct_id)).toEqual(["NCT1", "NCT3"]);
    expect(excluded.map((t) => t.nct_id)).toEqual(["NCT2"]);
  });

  it("returns all trials as shown when nothing is contaminated", () => {
    const trials = [trial("NCT1"), trial("NCT2")];
    const { shown, excluded } = partitionTrials(trials, []);
    expect(shown.map((t) => t.nct_id)).toEqual(["NCT1", "NCT2"]);
    expect(excluded).toEqual([]);
  });

  it("excludes every trial when all are contaminated", () => {
    const trials = [trial("NCT1"), trial("NCT2")];
    const { shown, excluded } = partitionTrials(trials, ["NCT1", "NCT2"]);
    expect(shown).toEqual([]);
    expect(excluded.map((t) => t.nct_id)).toEqual(["NCT1", "NCT2"]);
  });
});
