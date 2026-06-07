import { describe, it, expect } from "vitest";
import { buildGridRows, sortGridRows } from "./gridRows";
import sample from "../fixtures/sample-output.json";
import type { SupervisorOutput } from "../types";

const result = sample as unknown as SupervisorOutput;

describe("buildGridRows", () => {
  const rows = buildGridRows(result);

  it("includes only candidates with a structured blurb (excludes demoted/approval)", () => {
    expect(rows.map((r) => r.disease).sort()).toEqual([
      "parkinson disease",
      "type 1 diabetes mellitus",
    ]);
  });

  it("maps each column from the payload", () => {
    const t1d = rows.find((r) => r.disease === "type 1 diabetes mellitus")!;
    expect(t1d).toMatchObject({
      verdict: "Live but bottlenecked",
      strength: "moderate",
      totalTrials: 16,
      competitors: 10,
      recruiting: 6,
      source: "both",
    });
  });
});

describe("sortGridRows", () => {
  const rows = buildGridRows(result);

  it("sorts by trial count descending", () => {
    const sorted = sortGridRows(rows, "totalTrials", "desc");
    expect(sorted.map((r) => r.disease)).toEqual([
      "type 1 diabetes mellitus", // 16
      "parkinson disease", // 1
    ]);
  });

  it("orders strength by quality, not alphabetically", () => {
    const sorted = sortGridRows(rows, "strength", "desc");
    // moderate (rank 2) before none (rank 0) — quality order, not alphabetical
    // ("moderate" < "none" alphabetically, so this also guards against that).
    expect(sorted.map((r) => r.strength)).toEqual(["moderate", "none"]);
  });

  it("sinks null cells to the bottom regardless of direction", () => {
    const withNull = [
      { disease: "a", source: "both", verdict: "", strength: null, totalTrials: 5, competitors: null, recruiting: null },
      { disease: "b", source: "both", verdict: "", strength: null, totalTrials: null, competitors: null, recruiting: null },
    ] as ReturnType<typeof buildGridRows>;
    expect(sortGridRows(withNull, "totalTrials", "asc").map((r) => r.disease)).toEqual(["a", "b"]);
    expect(sortGridRows(withNull, "totalTrials", "desc").map((r) => r.disease)).toEqual(["a", "b"]);
  });
});
