// Pure helper: split the supervisor summary string into the footer block that
// is NOT part of any per-disease blurb (the "Demoted — …" / "Evidence gate
// exclusions: …" trailer) so the Overview can render blurb cards for the ranked
// candidates and keep the footer below them. Mirrors the footer markers the
// report formatter's _splice_blurbs_into_summary recognizes.

// Case-insensitive match of the first footer line; everything from there down
// is the footer. Lines above it are the ranked list (replaced by blurb cards).
const FOOTER_RE = /^\s*(?:Demoted\s+—|Closed\s+signals\s*:|Evidence\s+gate\s+exclusions\s*:)/i;

/** Return the trailing footer block (joined, trimmed) or "" if none. */
export function extractSummaryFooter(summary: string): string {
  const lines = summary.split("\n");
  const idx = lines.findIndex((l) => FOOTER_RE.test(l));
  if (idx === -1) return "";
  return lines.slice(idx).join("\n").trim();
}
