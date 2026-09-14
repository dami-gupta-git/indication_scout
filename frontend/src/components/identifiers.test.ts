import { describe, it, expect } from "vitest";
import { splitIdentifiers } from "./identifiers";

describe("splitIdentifiers", () => {
  it("returns a single text segment when there are no identifiers", () => {
    expect(splitIdentifiers("no ids here")).toEqual([{ kind: "text", value: "no ids here" }]);
  });

  it("splits out NCT ids and keeps surrounding text", () => {
    expect(splitIdentifiers("1 Phase 3 active (NCT06836128); 1 pending (NCT07120815)")).toEqual([
      { kind: "text", value: "1 Phase 3 active (" },
      { kind: "nct", value: "NCT06836128" },
      { kind: "text", value: "); 1 pending (" },
      { kind: "nct", value: "NCT07120815" },
      { kind: "text", value: ")" },
    ]);
  });

  it("splits out PMIDs, keeping the PMID prefix as text", () => {
    expect(splitIdentifiers("see PMID: 12345678 and PMID 9876543.")).toEqual([
      { kind: "text", value: "see " },
      { kind: "text", value: "PMID: " },
      { kind: "pmid", value: "12345678" },
      { kind: "text", value: " and " },
      { kind: "text", value: "PMID " },
      { kind: "pmid", value: "9876543" },
      { kind: "text", value: "." },
    ]);
  });

  it("handles an identifier alone", () => {
    expect(splitIdentifiers("NCT05973786")).toEqual([{ kind: "nct", value: "NCT05973786" }]);
  });
});
