# PubMed Data Sources

## Overview

PubMed is NCBI's biomedical literature database. IndicationScout queries it via the
[Entrez E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25501/) — a set of REST
endpoints that return JSON (for search) or XML (for article content).

---

## API Endpoints Used

### `esearch` — Search
Returns a list of PMIDs matching a query string. Used by `PubMedClient.search()`.

```
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
    ?db=pubmed&term=metformin+colorectal+cancer&retmax=50&retmode=json
```

Response shape (JSON):
```json
{
  "esearchresult": {
    "count": "1234",
    "idlist": ["38000001", "37999002", ...]
  }
}
```

A **PMID** (PubMed ID) is a stable integer identifier for a single record. It never changes
and is the primary key used throughout the pipeline.

### `efetch` — Fetch full records
Returns full article data for a batch of PMIDs as XML. Used by `PubMedClient.fetch_abstracts()`.

```
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
    ?db=pubmed&id=38000001,37999002&retmode=xml&rettype=abstract
```

The response is a `<PubmedArticleSet>` XML document. Each record inside it is one of two
element types depending on what the PMID refers to.

---

## XML Record Types

PubMed holds two fundamentally different kinds of records, and they have different XML
structures. The parser handles both.

### 1. `PubmedArticle` — Journal articles

The common case. A peer-reviewed article published in a journal.

```xml
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Metformin and colorectal cancer risk</ArticleTitle>
        <Abstract>
          <AbstractText>Plain abstract text.</AbstractText>
          <!-- OR structured sections: -->
          <AbstractText Label="BACKGROUND">...</AbstractText>
          <AbstractText Label="METHODS">...</AbstractText>
          <AbstractText Label="RESULTS">...</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>Jane</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>The Lancet</Title>
        </Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <History>
        <PubMedPubDate PubStatus="pubmed">
          <Year>2023</Year><Month>06</Month><Day>15</Day>
        </PubMedPubDate>
      </History>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
```

**Key fields and where they live:**

| `PubmedAbstract` field | XML path |
|---|---|
| `pmid` | `.//PMID` |
| `title` | `.//ArticleTitle` |
| `abstract` | `.//AbstractText` (joined, with `Label:` prefix if present) |
| `authors` | `.//Author` → `LastName`, `ForeName` |
| `journal` | `.//Journal/Title` |
| `pub_date` | `.//PubDate` → `Year`, `Month`, `Day` |
| `mesh_terms` | `.//MeshHeading/DescriptorName` |
| `keywords` | `.//Keyword` |

### 2. `PubmedBookArticle` — Book chapters (e.g. GeneReviews)

Used for chapters in curated reference books indexed in PubMed. The most common example is
[GeneReviews](https://www.ncbi.nlm.nih.gov/books/NBK1116/), a clinically-oriented gene-disease
reference maintained by NCBI.

The XML structure is meaningfully different from journal articles:

```xml
<PubmedArticleSet>
  <PubmedBookArticle>
    <BookDocument>                          <!-- replaces MedlineCitation -->
      <PMID>20301421</PMID>
      <Book>
        <BookTitle>GeneReviews</BookTitle>  <!-- replaces Journal/Title -->
        <PubDate><Year>1993</Year></PubDate>
        <AuthorList Type="editors">         <!-- editors of the whole book -->
          <Author>
            <LastName>Adam</LastName>
            <ForeName>Margaret P</ForeName>
          </Author>
        </AuthorList>
      </Book>
      <ArticleTitle>ALS2-Related Disorder</ArticleTitle>
      <AuthorList Type="authors">           <!-- authors of this chapter only -->
        <Author>
          <LastName>Orrell</LastName>
          <ForeName>Richard W</ForeName>
        </Author>
      </AuthorList>
      <Abstract>
        <AbstractText Label="CLINICAL CHARACTERISTICS">...</AbstractText>
        <AbstractText Label="MANAGEMENT">...</AbstractText>
        <AbstractText Label="GENETIC COUNSELING">...</AbstractText>
      </Abstract>
      <KeywordList>
        <Keyword>ALS2</Keyword>
        <Keyword>Alsin</Keyword>
      </KeywordList>
    </BookDocument>
    <PubmedBookData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">20301421</ArticleId>
      </ArticleIdList>
    </PubmedBookData>
  </PubmedBookArticle>
</PubmedArticleSet>
```

**Key differences from journal articles:**

| Concept | `PubmedArticle` | `PubmedBookArticle` |
|---|---|---|
| Content wrapper | `MedlineCitation` | `BookDocument` |
| Journal name | `Journal/Title` | `Book/BookTitle` |
| Authors | All `Author` elements | `AuthorList[@Type="authors"]` only |
| Editors | Not present | `AuthorList[@Type="editors"]` — **excluded** |
| MeSH terms | Present | **Not present** — always `[]` |
| Abstract structure | Often plain or labelled | Always labelled sections |

The `AuthorList` distinction matters: a book like GeneReviews has a fixed set of series editors
(Adam, Bick, etc.) who appear in every chapter's XML but are **not** authors of the chapter.
The parser skips any `AuthorList` with `Type="editors"`.

---

## Data Model: `PubmedAbstract`

Both record types are normalised into the same Pydantic model:

```python
class PubmedAbstract(BaseModel):
    pmid: str = ""           # e.g. "38000001"
    title: str | None        # article or chapter title
    abstract: str | None     # full text, sections joined with spaces
    authors: list[str] = []  # ["Smith, Jane", "Jones, Bob"]
    journal: str | None      # journal title or book title
    pub_date: str | None     # "2023", "2023-06", or "2023-06-15"
    mesh_terms: list[str] = []   # [] for book chapters
    keywords: list[str] = []
```

`pub_date` precision varies: some records have only a year, some have year+month, some have
all three. The parser builds `"YYYY"`, `"YYYY-MM"`, or `"YYYY-MM-DD"` accordingly.

---

## Client: `PubMedClient`

Located at [src/indication_scout/data_sources/pubmed.py](../src/indication_scout/data_sources/pubmed.py).

Extends `BaseClient` — use as an async context manager:

```python
async with PubMedClient() as client:
    pmids = await client.search("metformin AND colorectal cancer", max_results=50)
    abstracts = await client.fetch_abstracts(pmids)
```

**Methods:**

| Method | Returns | Notes |
|---|---|---|
| `search(query, max_results, date_before)` | `list[str]` | PMIDs; results cached to `_cache/` |
| `fetch_abstracts(pmids, batch_size)` | `list[PubmedAbstract]` | Batches requests in groups of 100 |
| `get_count(query, date_before)` | `int` | Fast count with no content fetch |

`fetch_abstracts` handles both `PubmedArticle` and `PubmedBookArticle` transparently.
Callers always receive `list[PubmedAbstract]` regardless of record type.

---

## Service Layer

`fetch_new_abstracts` in [src/indication_scout/services/retrieval.py](../src/indication_scout/services/retrieval.py)
wraps `fetch_abstracts` with a deduplication step: it accepts the full PMID list from a
search result plus the set of PMIDs already stored in the database, and only fetches the
difference.

```
search() → [pmids]
              ↓
get_stored_pmids(pmids, db) → stored: set[str]
              ↓
fetch_new_abstracts(pmids, stored) → [PubmedAbstract]   ← only the new ones
              ↓
embed + store in pgvector
```
