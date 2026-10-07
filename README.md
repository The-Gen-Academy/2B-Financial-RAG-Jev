# Financial RAG Pipeline — Apple Inc. 10-K Analysis

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built on Apple's SEC 10-K filings. The project compares two chunking strategies (fixed-size vs. semantic), adds a TypeSafe AI Jev reranking step, and formally evaluates all three pipeline configurations using RAGAS metrics.

Built as a learning project for anyone working in financial services, insurance, or regulated industries — but the techniques apply to any domain with long, dense, structured documents.

---

## Architecture

![Pipeline Architecture](architecture_jev.png)

The reranking component uses TypeSafe AI's Jev System One model. Pinecone retrieves candidate chunks, Jev scores their relevance to the query, Python sorts by the returned probability, and the top chunks continue to generation.

---

## What this project demonstrates

- How to ingest real SEC filings from EDGAR with no API key
- Why fixed-size chunking fails on financial documents (cuts tables mid-row)
- How semantic chunking preserves topical coherence in dense documents
- How Jev can be used as a System One reranker: query + retrieved candidates → typed relevance decisions → probability-based ranking
- How to formally evaluate a RAG pipeline with RAGAS (faithfulness, context precision, context recall)
- How to use two Pinecone namespaces to run controlled A/B experiments

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Orchestration | LangChain (LCEL) | Chain assembly, prompt templates, document loaders |
| Embedding | HuggingFace `BAAI/bge-base-en-v1.5` | 768-dim dense vectors, free, GPU-accelerated |
| Vector Store | Pinecone (Serverless) | Two namespaces: `fixed` and `semantic` |
| Chunking — Fixed | `RecursiveCharacterTextSplitter` | 512 chars, 64 overlap |
| Chunking — Semantic | `SemanticChunker` (LangChain Experimental) | Percentile-based breakpoints (65th) |
| Reranking | TypeSafe AI Jev | System One relevance decisions; Python ranks candidates by `P(relevant)` |
| Generation | OpenAI `gpt-4o-mini` | Context-grounded answer generation |
| Evaluation | RAGAS | Faithfulness, Context Precision, Context Recall |
| Data Source | SEC EDGAR via `sec-edgar-downloader` | Apple 10-K FY2024 + FY2025, free |

---

## Results

| Metric | A: Fixed (no rerank) | B: Semantic (no rerank) | C: Semantic + Jev Reranking |
|---|---:|---:|---:|
| **Faithfulness** | 0.400 | 0.900 | **0.967** |
| **Context Precision** | 0.000 | 0.750 | **1.000** |
| **Context Recall** | 0.000 | 0.800 | **1.000** |

**Winner: Semantic Chunking + Jev Reranking**

Semantic chunking substantially improves retrieval over fixed-size chunks. Jev reranking then raises Context Precision from 0.750 to 1.000 and Context Recall from 0.800 to 1.000, while Faithfulness increases from 0.900 to 0.967.

---

## Project Structure

```
financial-rag/
├── Financial_Document_Intelligence_Pipeline.ipynb   # Main notebook
├── financial_rag_pipeline_architecture.html         # Interactive architecture diagram
├── architecture.png                                  # Architecture diagram (static)
├── README.md                                         # Project documentation
└── SOLUTION_KIT.md                                   # Student guide
```

---

## Setup

### Prerequisites

- Google Colab (recommended) with **T4 GPU runtime** enabled
- OR local Python 3.10+ environment

### API Keys required

| Service | Where to get it | Free tier |
|---|---|---|
| Pinecone | [pinecone.io](https://pinecone.io) → API Keys | 1 index, 2GB storage |
| OpenAI | [platform.openai.com](https://platform.openai.com) → API keys | Pay-per-use |
| HuggingFace | [huggingface.co](https://huggingface.co) → Settings → Access Tokens | Free (read token) |
| TypeSafe AI | TypeSafe AI account / API key | Required for Jev System One calls |

### Running on Google Colab

1. Upload the notebook to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
3. Add your API keys to Colab Secrets (the key icon in the left sidebar):
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`
   - `HUGGINGFACEHUB_API_TOKEN`
   - `TYPESAFE_API_KEY`
4. Optionally upload `financial_rag_pipeline_architecture.html` to `/content/` to render the interactive diagram
5. Run all cells top to bottom

> **Important:** Do not install `torch` via pip in Colab — it overwrites the pre-installed CUDA-enabled build with a CPU-only version. The notebook handles this correctly.

---

## Notebook Walkthrough

| Section | What happens |
|---|---|
| **1. Setup** | Install deps, verify GPU, set API keys, test all connections |
| **2. Data Ingestion** | Download Apple 10-K (FY2024 + FY2025) from SEC EDGAR, load with LangChain |
| **3. Text Preprocessing** | Strip SGML/HTML tags, decode HTML entities, remove filing header metadata |
| **4. Chunking Strategies** | Fixed-size (12,142 chunks) and semantic (2,529 chunks) — both uploaded to Pinecone |
| **5. Retrieval Testing** | Side-by-side visual comparison of both strategies on 5 financial queries |
| **6. Reranking** | Jev evaluates top-10 semantic candidates with typed relevance questions → Python sorts by `P(relevant)` → returns top-3 |
| **7. Answer Generation** | Full RAG chain: retrieve → rerank → prompt → `gpt-4o-mini` → answer |
| **8. RAGAS Evaluation** | 10 questions × 3 experiments = formal metrics comparison |
| **9. Results** | Comparison table + winner summary with metric explanations |

---

## Key Learnings

**Why fixed-size fails on financial documents**
A 512-character window cuts Apple's revenue table mid-row. The chunk contains numbers but not their column headers — the model retrieves `$391,035` with no idea it means total net sales. Context Precision drops to 0.00.

**Why semantic chunking wins**
`SemanticChunker` detects when adjacent sentences shift topic (via cosine distance spike) and cuts there instead of at a fixed character count. Revenue tables, risk factor paragraphs, and MD&A sections stay intact as single chunks.

**Why reranking helps even after good retrieval**
The bi-encoder (Pinecone similarity search) can surface chunks that are semantically similar but not actually useful for the question. Jev receives the query and retrieved candidates as state and makes a narrow relevance decision for each candidate. Python then uses Jev's `P(relevant)` values to reorder the candidates before generation.

**The reranker limitation**
Reranking can only re-order what the bi-encoder already retrieved. If the right chunk isn't in the top-10 candidates, the reranker cannot fix it. Widening the candidate pool (`k=20` instead of `k=10`) is the remedy.

---

## Extending this project

- **Add more document types** — earnings call transcripts (conversational) alongside 10-Ks (formal) will stress-test chunking further
- **Add more companies** — multi-company retrieval requires metadata filtering to prevent cross-company confusion
- **Build a Streamlit app** — toggle chunking strategy and reranking on/off interactively
- **Try hybrid search** — combine dense vectors (semantic) with sparse BM25 (keyword) for better recall on exact financial figures
- **Experiment with chunk sizes** — the 65th percentile threshold was tuned for Apple's 10-K writing style; other document styles may need different values

---

## References

- [LangChain SemanticChunker](https://python.langchain.com/docs/how_to/semantic-chunker/)
- [Pinecone Serverless Docs](https://docs.pinecone.io/guides/get-started/quickstart)
- [RAGAS Documentation](https://docs.ragas.io)
- [TypeSafe AI — How to Build with System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)
- [BAAI/bge-base-en-v1.5 on HuggingFace](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [SEC EDGAR Full-Text Search](https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&dateRange=custom)


---

## Jev implementation note

The Jev reranker follows the TypeSafe System One pattern used in the linked documentation:

1. Python retrieves candidate chunks from Pinecone.
2. The query and candidate texts are passed as `state`.
3. One typed `Noul` relevance question is defined per candidate.
4. `TypeSafeClient.system_one(...)` evaluates the independent questions together.
5. Each Noul returns a probability for the positive decision.
6. Python sorts the candidates by that probability and returns the top documents.

Jev does not replace Pinecone, embeddings, generation, or RAGAS. It replaces only the reranking judgment.

**Required secret:** `TYPESAFE_API_KEY`
