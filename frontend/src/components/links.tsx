// External-link helpers for trial and PubMed identifiers. Mirrors the URL
// patterns app.py used (clinicaltrials.gov/study/<nct>, pubmed/<pmid>/).

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
