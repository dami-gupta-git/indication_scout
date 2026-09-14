// External-link helpers for trial and PubMed identifiers. Mirrors the URL
// patterns app.py used (clinicaltrials.gov/study/<nct>, pubmed/<pmid>/).

import { splitIdentifiers } from "./identifiers";

export function NctLink({ nctId }: { nctId: string }) {
  return (
    <a href={`https://clinicaltrials.gov/study/${nctId}`} target="_blank" rel="noreferrer">
      {nctId}
    </a>
  );
}

export function PmidLink({ pmid }: { pmid: string }) {
  return (
    <a href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`} target="_blank" rel="noreferrer">
      {pmid}
    </a>
  );
}

// Render free text with any NCT ids / PMIDs turned into links.
export function LinkifiedText({ text }: { text: string }) {
  return (
    <>
      {splitIdentifiers(text).map((seg, i) => {
        if (seg.kind === "nct") return <NctLink key={i} nctId={seg.value} />;
        if (seg.kind === "pmid") return <PmidLink key={i} pmid={seg.value} />;
        return <span key={i}>{seg.value}</span>;
      })}
    </>
  );
}
