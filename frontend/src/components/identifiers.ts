// Split free text into plain segments and trial / PubMed identifiers so the
// identifiers can be rendered as links. Pure so it can be unit tested.

export type Segment =
  | { kind: "text"; value: string }
  | { kind: "nct"; value: string }
  | { kind: "pmid"; value: string };

// NCT ids are eight digits; PMIDs appear as "PMID 12345678" or "PMID: 12345678".
const ID_PATTERN = /(NCT\d{8})|PMID:?\s*(\d{7,8})/g;

export function splitIdentifiers(text: string): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(ID_PATTERN)) {
    const start = m.index ?? 0;
    if (start > last) segments.push({ kind: "text", value: text.slice(last, start) });
    if (m[1]) {
      segments.push({ kind: "nct", value: m[1] });
    } else {
      segments.push({ kind: "text", value: m[0].slice(0, m[0].length - m[2].length) });
      segments.push({ kind: "pmid", value: m[2] });
    }
    last = start + m[0].length;
  }
  if (last < text.length) segments.push({ kind: "text", value: text.slice(last) });
  return segments;
}
