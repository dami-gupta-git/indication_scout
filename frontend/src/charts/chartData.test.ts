import { describe, it, expect } from "vitest";
import { phaseSlices, statusSlices } from "./chartData";
import type { Trial } from "../types";

function trial(phase: string): Trial {
  return {
    nct_id: "NCT0",
    title: "",
    brief_summary: null,
    phase,
    overall_status: "COMPLETED",
    why_stopped: null,
    sponsor: "",
    enrollment: null,
    start_date: null,
    completion_date: null,
  };
}

describe("phaseSlices", () => {
  it("counts per phase and orders early→late", () => {
    const slices = phaseSlices([
      trial("Phase 3"),
      trial("Phase 1"),
      trial("Phase 3"),
      trial("Phase 2"),
    ]);
    expect(slices).toEqual([
      { label: "Phase 1", count: 1 },
      { label: "Phase 2", count: 1 },
      { label: "Phase 3", count: 2 },
    ]);
  });

  it("labels an empty phase as Unknown and sorts it last", () => {
    const slices = phaseSlices([trial(""), trial("Phase 1")]);
    expect(slices).toEqual([
      { label: "Phase 1", count: 1 },
      { label: "Unknown", count: 1 },
    ]);
  });
});

describe("statusSlices", () => {
  it("drops zero counts and sorts largest first", () => {
    expect(
      statusSlices({ RECRUITING: 6, ACTIVE_NOT_RECRUITING: 2, WITHDRAWN: 0, UNKNOWN: 1 }),
    ).toEqual([
      { label: "RECRUITING", count: 6 },
      { label: "ACTIVE_NOT_RECRUITING", count: 2 },
      { label: "UNKNOWN", count: 1 },
    ]);
  });
});
