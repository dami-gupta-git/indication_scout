import { describe, it, expect } from "vitest";
import { sortRows } from "./useSort";

interface Row {
  name: string;
  n: number | null;
}

const rows: Row[] = [
  { name: "b", n: 2 },
  { name: "a", n: 10 },
  { name: "c", n: null },
];

describe("sortRows", () => {
  it("sorts numbers ascending and descending", () => {
    expect(sortRows(rows, (r) => r.n, "asc").map((r) => r.name)).toEqual(["b", "a", "c"]);
    expect(sortRows(rows, (r) => r.n, "desc").map((r) => r.name)).toEqual(["a", "b", "c"]);
  });

  it("sinks null/empty to the bottom regardless of direction", () => {
    // 'c' (null) is last in both directions.
    const asc = sortRows(rows, (r) => r.n, "asc");
    const desc = sortRows(rows, (r) => r.n, "desc");
    expect(asc[asc.length - 1].name).toBe("c");
    expect(desc[desc.length - 1].name).toBe("c");
  });

  it("sorts strings case-insensitively via localeCompare", () => {
    const r = [{ name: "Beta", n: 0 }, { name: "alpha", n: 0 }];
    expect(sortRows(r, (x) => x.name, "asc").map((x) => x.name)).toEqual(["alpha", "Beta"]);
  });

  it("does not mutate the input array", () => {
    const input = [...rows];
    sortRows(input, (r) => r.n, "asc");
    expect(input.map((r) => r.name)).toEqual(["b", "a", "c"]);
  });
});
