# Tool-Use Loop Mockup

The Literature Agent will follow the same pattern as `ClinicalTrialsAgent` — a LangChain
ReAct agent using `create_agent` with `@tool` decorated functions. No manual loop needed.

---

## `agents/literature_tools.py` — tool wrappers

Same pattern as `clinical_trials_tools.py`: a builder function captures shared state
(drug profile, any session-level context) via closure, and each tool is a thin async
wrapper around `RetrievalService`.

```python
from langchain_core.tools import tool
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.services.retrieval import RetrievalService


def build_literature_tools(drug_profile: DrugProfile) -> list:
    """Build tool functions with drug_profile captured via closure."""

    @tool
    async def expand_search_terms(drug_name: str, disease_name: str) -> list[str]:
        """Generate diverse PubMed keyword queries for a drug-disease pair.

        Uses the drug profile (synonyms, targets, MOA, ATC codes) to produce
        5-10 varied queries. Always call this first.
        """
        async with RetrievalService() as svc:
            return await svc.expand_search_terms(drug_name, disease_name, drug_profile)

    @tool
    async def fetch_and_cache(queries: list[str]) -> dict:
        """Run PubMed searches, embed abstracts with BioLORD-2023, store in pgvector.

        Returns pmid_count and pmids.
        """
        async with RetrievalService() as svc:
            result = await svc.fetch_and_cache(queries)
        return {"pmid_count": result.pmid_count, "pmids": result.pmids}

    @tool
    async def semantic_search(pmids: list[str], top_k: int = 5) -> list[dict]:
        """Retrieve top-k abstracts from pgvector most similar to the drug-disease query.

        Restricted to the supplied PMIDs.
        """
        async with RetrievalService() as svc:
            results = await svc.semantic_search(pmids, top_k)
        return [r.model_dump() for r in results]

    @tool
    async def synthesize(abstracts: list[dict]) -> dict:
        """Synthesize retrieved abstracts into a structured EvidenceSummary.

        Returns strength (strong/moderate/weak/none), study_count, study_types,
        key_findings, has_adverse_effects, and supporting_pmids.
        Call this once you have high-similarity abstracts. If no evidence was
        found, call with whatever is available — do not skip.
        """
        async with RetrievalService() as svc:
            result = await svc.synthesize(abstracts)
        return result.model_dump()

    return [expand_search_terms, fetch_and_cache, semantic_search, synthesize]
```

---

## `agents/literature.py` — agent

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from indication_scout.agents.base import BaseAgent
from indication_scout.agents.literature_tools import build_literature_tools
from indication_scout.constants import DEFAULT_LLM_MODEL

SYSTEM_PROMPT = """You are a biomedical literature analyst assessing published
evidence for a drug-disease pair.

Follow this sequence exactly:
1. Call expand_search_terms to generate PubMed queries.
2. Call fetch_and_cache with those queries.
3. Call semantic_search with the returned PMIDs.
4. Call synthesize with the abstracts.
   If no evidence was found, still call synthesize — return strength "none",
   do not fabricate signal.

End with a plain-text confirmation that synthesis is complete."""

RECURSION_LIMIT = 10


class LiteratureAgent(BaseAgent):

    def __init__(self, model: str = DEFAULT_LLM_MODEL) -> None:
        self._model = model

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        drug_name    = input_data["drug_name"]
        disease_name = input_data["disease_name"]
        drug_profile = input_data["drug_profile"]

        tools = build_literature_tools(drug_profile=drug_profile)

        llm = ChatAnthropic(model=self._model, temperature=0, max_tokens=4096)

        agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content":
                f"Find literature evidence for '{drug_name}' in '{disease_name}'."}]},
            config={"recursion_limit": RECURSION_LIMIT},
        )

        return self._parse_result(result)

    @staticmethod
    def _parse_result(result: dict[str, Any]) -> dict[str, Any]:
        """Walk message history to extract EvidenceSummary and all PMIDs seen."""
        evidence_summary = None
        all_pmids: list[str] = []

        for msg in result.get("messages", []):
            if not (hasattr(msg, "name") and hasattr(msg, "content")):
                continue
            content = msg.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    continue

            if msg.name == "synthesize" and isinstance(content, dict):
                evidence_summary = EvidenceSummary(**content)

            if msg.name == "fetch_and_cache" and isinstance(content, dict):
                all_pmids.extend(content.get("pmids", []))

        # Deduplicate PMIDs preserving order
        all_pmids = list(dict.fromkeys(all_pmids))

        return {
            "evidence_summary": evidence_summary,
            "pmids_retrieved": all_pmids,
        }
```

---

## Comparison to clinical trials agent

| | `ClinicalTrialsAgent` | `LiteratureAgent` |
|---|---|---|
| Loop manager | LangChain `create_agent` | LangChain `create_agent` |
| Tool wrappers | `@tool` in `clinical_trials_tools.py` | `@tool` in `literature_tools.py` |
| Shared state via closure | `date_before` | `drug_profile` |
| Tool backend | `ClinicalTrialsClient` | `RetrievalService` |
| Result extraction | `_parse_result` walks message history | Same pattern |
| `query_llm_with_tools()` | Not needed | Not needed |
