import { describe, it, expect } from "vitest";
import { hasStructuredBlurb } from "./blurb";
import { extractSummaryFooter } from "./summaryFooter";
import type { CandidateBlurb } from "../types";

const emptyBlurb: CandidateBlurb = {
  stage: "",
  literature: "",
  blocker: "",
  active_programs: "",
  key_risk: "",
  verdict: "",
  watch: "",
  prose: "",
};

describe("hasStructuredBlurb", () => {
  it("is true when any structured field is set", () => {
    expect(hasStructuredBlurb({ ...emptyBlurb, verdict: "Live but bottlenecked" })).toBe(true);
    expect(hasStructuredBlurb({ ...emptyBlurb, stage: "Phase 2 completed" })).toBe(true);
  });

  it("is false for a prose-only blurb (demoted/approval entry)", () => {
    expect(hasStructuredBlurb({ ...emptyBlurb, prose: "Already FDA-approved for X." })).toBe(false);
  });

  it("is false for a fully empty blurb", () => {
    expect(hasStructuredBlurb(emptyBlurb)).toBe(false);
  });
});

describe("extractSummaryFooter", () => {
  it("returns the Demoted block onward, trimmed", () => {
    const summary = [
      "Ranked repurposing signals for x:",
      "",
      "1. type 1 diabetes mellitus — moderate",
      "",
      "Demoted — approval relationship:",
      "- diabetes mellitus (broader term)",
      "",
      "Evidence gate exclusions: (none)",
    ].join("\n");
    expect(extractSummaryFooter(summary)).toBe(
      [
        "Demoted — approval relationship:",
        "- diabetes mellitus (broader term)",
        "",
        "Evidence gate exclusions: (none)",
      ].join("\n"),
    );
  });

  it("returns empty string when there is no footer", () => {
    expect(extractSummaryFooter("1. a — moderate\n2. b — weak")).toBe("");
  });
});
