# RAGAS Evaluation: FinSight SEC Filing RAG

**Date**: 2026-03-14 22:27 UTC  
**Testset**: 10 Q+GT pairs (5 scored)  
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
| faithfulness | N/A | N/A |
| answer_relevancy | N/A | N/A |
| context_precision | N/A | N/A |

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
| Judge context truncation | 200 chars/chunk |
| Inter-sample delay | 90s |
| Samples scored | 5 of 10 |

---

## Limitations

- Groq proxy models differ from deployed vLLM models (see proxy note above)
- Only 5 of 10 Q+GT pairs scored — limited by Groq free-tier TPD
- Testset is hand-written; real user queries may differ in distribution
- Judge contexts truncated to 200 chars/chunk to fit within TPM limits
- `context_precision` measures retrieval rank quality, not recall coverage
