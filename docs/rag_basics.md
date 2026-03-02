RAG Pipeline Overview
The RAG (Retrieval-Augmented Generation) component is designed to solve a specific problem you 
identified during testing : raw PubMed searches return ~100 papers for a drug-disease pair, but many are irrelevant. 
For example, searching "bupropion AND obesity" returns depression papers that incidentally mention obesity, 
burying the papers about bupropion as an actual obesity treatment.
What RAG does in your pipeline:
It sits in the "evidence stacking" layer — after Path 1 (target-disease associations) and Path 2 (drug class analogy) 
identify candidate diseases, the RAG pipeline retrieves and reranks PubMed literature so the most relevant papers 
surface to the top for the Literature Agent to synthesize.
Planned architecture:

PostgreSQL + pgvector — caches PubMed abstracts with vector embeddings for semantic search
Embedding API (Voyage AI or OpenAI) — generates embeddings for reranking
Retrieval flow: PubMed search → store abstracts in pgvector → embed query → vector similarity reranking 
→ top results fed to Claude for synthesis