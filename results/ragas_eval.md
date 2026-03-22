# RAGAS Evaluation: FinSight SEC Filing RAG

**Date**: 2026-03-21 21:40 UTC  
**Testset**: 10 Q+GT pairs (1 scored per model)  
**Corpus**: AAPL, MSFT, GOOGL, AMZN, META — 10-K filings (3 years each)  
**Retrieval**: FAISS IndexFlatIP · BGE-small-en-v1.5 · top-5 chunks  
**Judge LLM**: Groq `llama-3.3-70b-versatile` (max_tokens=1024)  

---

## Model Proxy Note

> The FinSight Modal deployment runs `meta-llama/Meta-Llama-3.1-8B-Instruct` and
> `mistralai/Mistral-7B-Instruct-v0.3` via vLLM. This evaluation uses Groq API
> proxies (both map to `llama-3.1-8b-instant` — mixtral decommissioned) to avoid
> Modal GPU costs. Scores reflect model-family quality on this task.

---

## Results Summary

| Metric | LLaMA 3.1 8B | Mistral proxy |
|--------|:------------:|:-------------:|
| faithfulness | 0.0000 | N/A |
| answer_relevancy | 0.8709 | N/A |
| context_precision | 0.0000 | N/A |

### Metric Definitions

- **Faithfulness**: fraction of answer claims entailed by the retrieved context
- **Answer Relevancy**: semantic alignment between the question and the answer
- **Context Precision**: whether the most relevant chunks are ranked highest by FAISS

---

## Evaluation Setup

| Parameter | Value |
|-----------|-------|
| Judge LLM | `llama-3.3-70b-versatile` (12K TPM, 100K TPD) |
| Judge max_tokens | 1024 |
| Answer max tokens | 400 |
| Temperature | 0.0 (deterministic) |
| Retrieval k | 5 |
| Judge context truncation | 800 chars/chunk (HTML decoded, prose-filtered) |
| Inter-sample delay | 90s |
| Samples scored per model | 1 of 10 |

---

## Limitations

- Groq proxy models differ from deployed vLLM models (see proxy note above)
- Only 1 of 10 Q+GT pairs scored — limited by Groq free-tier TPD
- LLaMA and Mistral scored on separate days due to 100K TPD constraint
- Testset is hand-written; real user queries may differ in distribution
- Judge contexts truncated to 800 chars/chunk
- SEC filing chunks filtered before scoring: alpha-ratio < 0.4 AND no XBRL/table-header patterns
- `context_precision` measures retrieval rank quality, not recall coverage
