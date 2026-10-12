# Financial RAG Pipeline — Apple Inc. 10-K Analysis

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built on Apple's SEC 10-K filings. The project compares two chunking strategies (fixed-size vs. semantic), adds a Jev Score reranking step, and formally evaluates all three pipeline configurations using RAGAS metrics.

Built as a learning project for anyone working in financial services, insurance, or regulated industries — but the techniques apply to any domain with long, dense, structured documents.

---

## Architecture

![Pipeline Architecture](architecture.png)

The pipeline moves through seven stages: data ingestion → preprocessing → dual chunking paths → embedding → vector storage → retrieval + reranking → generation → RAGAS evaluation.

---

## What this project demonstrates

- How to ingest real SEC filings from EDGAR with no API key
- Why fixed-size chunking fails on financial documents (cuts tables mid-row)
- How semantic chunking preserves topical coherence in dense documents
- How a Jev Score reranker improves retrieval precision without retraining
- How to formally evaluate a RAG pipeline with RAGAS (faithfulness, context precision, context recall)
- How to use two Pinecone namespaces to run controlled A/B experiments

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Orchestration | LangChain (LCEL) | Chain assembly, prompt templates, document loaders |
| Embedding | HuggingFace `sentence-transformers/multi-qa-mpnet-base-cos-v1` | 768-dim dense vectors, free, GPU-accelerated |
| Vector Store | Pinecone (Serverless) | Two namespaces: `fixed` and `semantic` |
| Chunking — Fixed | `RecursiveCharacterTextSplitter` | 512 chars, 64 overlap |
| Chunking — Semantic | `SemanticChunker` (LangChain Experimental) | Percentile-based breakpoints (65th) |
| Reranking | TypeSafe AI Jev `Jev Score` | TypeSafe API-based Jev Score |
| Generation | OpenAI `gpt-4o-mini` | Context-grounded answer generation |
| Evaluation | RAGAS | Faithfulness, Context Precision, Context Recall |
| Data Source | SEC EDGAR via `sec-edgar-downloader` | Apple 10-K FY2024 + FY2025, free |

---

## Results

| Metric | A: Fixed (no rerank) | B: Semantic (no rerank) | C: Semantic + Jev Score |
|---|---:|---:|---:|
| **Faithfulness** | 0.567 | 0.794 | **1.000** |
| **Context Precision** | 0.208 | 0.617 | **0.892** |
| **Context Recall** | 0.550 | 0.650 | **0.800** |

Results from the executed MPNet + Jev Score notebook.


---

## Project Structure

```
financial-rag/
├── Financial_Document_Intelligence_Pipeline.ipynb   # Main notebook
├── financial_rag_pipeline_architecture.html         # Interactive architecture diagram
├── architecture.png                                  # Architecture diagram (static)
├── README.md                                         # This file
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

### Running on Google Colab

1. Upload the notebook to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
3. Add your API keys to Colab Secrets (the key icon in the left sidebar):
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`
   - `HUGGINGFACEHUB_API_TOKEN`\n   - `TYPESAFE_API_KEY`
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
| **6. Reranking** | Jev Score re-scores top-10 semantic candidates → returns top-3 |
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
The bi-encoder (Pinecone similarity search) scores query and chunk independently and can surface off-topic chunks that happen to share vocabulary with the query. Jev Score evaluates each query and retrieved chunk against a relevance rubric — it can tell "this chunk mentions revenue but for the wrong year" from "this chunk directly answers the question."

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
- [TypeSafe AI Jev — Free local reranker](https://github.com/PrithivirajDamodaran/TypeSafe AI Jev)
- [sentence-transformers/multi-qa-mpnet-base-cos-v1 on HuggingFace](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1)
- [SEC EDGAR Full-Text Search](https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&dateRange=custom)
\n\n## Embedding and reranking configuration\n\n`multi-qa-mpnet-base-cos-v1` produces 768-dimensional embeddings. Token-aware splitting limits oversized chunks before embedding. Pinecone retrieves ten candidates; TypeSafe Jev Score grades relevance using a five-level rubric and Python selects the top three. Add `TYPESAFE_API_KEY` to Colab Secrets.\n