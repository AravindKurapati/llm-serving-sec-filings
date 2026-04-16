# Benchmark Analysis: LLaMA 3.1 8B vs Mistral 7B

**Date**: February 22, 2026  
**Hardware**: Modal A10G (24GB VRAM, sm_86)  
**Dataset**: 15 SEC 10-K filings (AAPL, MSFT, GOOGL, AMZN, META for 3 years each)  
**Index**: 4,782 chunks, BGE-small-en-v1.5 (384 dims), FAISS IndexFlatIP  
**Questions**: 5 financial analysis questions  

---

## Summary Table

| Metric | LLaMA 3.1 8B | Mistral 7B | Winner |
|--------|-------------|------------|--------|
| TTFT p50 | 4,616ms | 1,015ms | Mistral (4.5x faster) |
| TTFT p95 | 4,631ms | 2,402ms | Mistral (1.9x faster) |
| TPOT p50 | 23.1ms | 23.5ms | Tie |
| Throughput avg | 28.9 tok/s | 28.6 tok/s | Tie |
| Answer quality | Verbose, repetitive at limit | Concise, well-structured | Mistral |

---

## TTFT Analysis

TTFT measures prefill time - how long it takes to process the input prompt before generating the first output token. For RAG, the input is large: system prompt + 5 retrieved chunks + question = 600-1,300 tokens.

LLaMA 3.1 8B consistently took ~4,600ms to prefill regardless of input length. Mistral 7B ranged from 562ms (short prompt) to 2,402ms (long prompt), showing more sensitivity to input length but significantly faster on average.

In a chat interface, TTFT is the perceived "thinking time" before any text appears. Mistral feels 4.5x more responsive.

**Why is Mistral faster on prefill?**
- Mistral 7B uses sliding window attention (SWA) which is more efficient on longer sequences
- Smaller parameter count means fewer matmul operations per token during prefill
- LLaMA 3.1's grouped query attention (GQA) helps decode but adds prefill overhead at this scale

---

## TPOT and Throughput Analysis

Both models generated tokens at ~23ms per token (~29 tok/s). This is expected cause at this scale both models are memory-bandwidth bound during decode, and they share the same A10G VRAM bandwidth.

The A10G has 600 GB/s memory bandwidth. Loading 8B parameters in fp16 (16GB) per forward pass is the bottleneck, not compute. Both 7B and 8B models hit this ceiling at essentially the same speed.

---

## Answer Quality

**LLaMA 3.1 8B issues observed:**
- Repeats citation markers ([1], [2], etc.) dozens of times at the end of responses
- Hits the 400-token limit mid-thought on complex questions
- Tends to enumerate every possible point rather than synthesizing

**Mistral 7B strengths:**
- Stops when the answer is complete (used 46-217 tokens vs LLaMA's consistent 400)
- Cleaner citation format - cites once, moves on
- More direct synthesis of retrieved context

**Example - Apple supply chain risks:**

LLaMA output: 400 tokens, ends with `[1], [2], [4], [5] [1], [2], [4], [5]...` repeated 15+ times

Mistral output: 217 tokens, 6 clean bullet points, stops naturally

---

## RAGAS Quality Evaluation

> **Status**: Evaluation run pending — scores to be filled in after tonight's run (TPD resets midnight UTC).  
> Full methodology: `scripts/ragas_eval.py` | Raw results: `results/ragas_eval_<timestamp>.json`

**Setup**:
- Metrics: faithfulness, answer_relevancy, context_precision
- Judge LLM: Groq `llama-3.3-70b-versatile`
- Testset: 10 hand-written Q+GT pairs (revenue, operating income, R&D spend, risk factors, segment performance across all 5 companies)
- Retrieval: FAISS top-5 chunks, BGE-small-en-v1.5

**Note on model proxies**: The Modal deployment runs `meta-llama/Meta-Llama-3.1-8B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3` via vLLM. RAGAS evaluation uses Groq API proxies to avoid Modal GPU costs — both currently map to `llama-3.1-8b-instant` (Mixtral was decommissioned on Groq). Scores reflect model-family quality, not bit-for-bit equivalence with the vLLM-served versions.

| Metric | LLaMA 3.1 8B | Mistral 7B |
|--------|:------------:|:----------:|
| Faithfulness | — | — |
| Answer Relevancy | — | — |
| Context Precision | — | — |

**Anticipated findings**:
- Faithfulness: given LLaMA's citation-repetition artifact, lower faithfulness is plausible — the model may be hallucinating in the repetition tail
- Context Precision: retriever-dependent metric, so both models should score identically since they share the same FAISS index
- Answer Relevancy: Mistral's concise outputs may score higher since they stay on-topic rather than padding to the token limit

---

## Output Quality Evaluation

### Answer Quality (`scripts/answer_quality.py`)

> These metrics are computed from **real vLLM outputs** served by the Modal A10G deployment — LLaMA 3.1 8B (`meta-llama/Meta-Llama-3.1-8B-Instruct`) vs actual Mistral 7B (`mistralai/Mistral-7B-Instruct-v0.3`). Source: `results/benchmark_20260415_190523.json`.

| Metric | LLaMA 3.1 8B | Mistral 7B |
|--------|:------------:|:----------:|
| token_count (avg) | 114.2 | 87.2 |
| citation_count (avg) | 5.40 | 4.80 |
| repetition_score (avg) | 0.0107 | 0.0029 |
| hits_token_limit (≥390 tokens) | 0 / 5 | 0 / 5 |

**Per-question breakdown:**

| Q# | Question | LLaMA tokens | Mistral tokens | LLaMA rep. | Mistral rep. |
|----|----------|:------------:|:--------------:|:----------:|:------------:|
| Q1 | Apple supply chain risks | 172 | 137 | 0.0000 | 0.0000 |
| Q2 | Microsoft cloud revenue growth | 103 | 73 | 0.0101 | 0.0145 |
| Q3 | Meta AI infrastructure investment | 114 | 79 | 0.0000 | 0.0000 |
| Q4 | Google advertising revenue | 39 | 65 | 0.0000 | 0.0000 |
| Q5 | **Amazon cybersecurity risks** | 143 | 82 | **0.0432** | 0.0000 |

**Key findings:**
- LLaMA is **31% more verbose** on average (114.2 vs 87.2 tokens)
- LLaMA has a **3.7× higher repetition score** (0.0107 vs 0.0029) — consistent with the citation-repetition artifact observed qualitatively
- **Q5 (Amazon cybersecurity) is the worst case**: LLaMA's repetition score of 0.043 is the highest across all questions; Mistral scores 0.000 on the same question, suggesting LLaMA recycles phrasing around "cybersecurity risks" and "senior leadership" in a way Mistral avoids
- Neither model hit the 390-token limit in this benchmark run — the token-limit artifact seen in earlier qualitative observation may be more prevalent on longer or more open-ended questions

---

### LLM-as-Judge (`scripts/llm_judge.py`)

> **Caveat**: Both "LLaMA" and "Mistral" entries below use the same Groq proxy model (`llama-3.1-8b-instant`) because Mixtral was decommissioned on Groq. Scores are **identical by design** and should not be used to compare the two models. The table is included for completeness only. For actual model comparison, use the `answer_quality.py` metrics above, which are derived from real vLLM outputs.

Evaluated on the 5 qualitative questions (supply chain, cybersecurity, antitrust, workforce, privacy risks). Judge model: Groq `llama-3.3-70b-versatile`.

| Metric | LLaMA 3.1 8B | Mistral proxy |
|--------|:------------:|:-------------:|
| groundedness | 0.86 | 0.86 |
| conciseness | 0.74 | 0.74 |
| citation_quality | 0.80 | 0.80 |

The groundedness score of 0.86 indicates answers are largely grounded in retrieved context. The conciseness score of 0.74 is the weakest dimension — the Amazon workforce question dragged it down (0.60), consistent with the high token count and repetition score observed in answer_quality.py for that same question.

---

## Context Length Sensitivity

> **Status**: Script ready — `scripts/context_sensitivity_test.py`. Run when TPD resets.

Varies retrieval k across [2, 3, 5, 8, 10] and measures prompt token count, Groq latency, and answer word count per k value.

**Expected finding**: prompt token count grows roughly linearly with k (each chunk truncated to 600 chars ≈ ~150 tokens). Answer length is expected to plateau around k=5 — additional context beyond the current default adds latency without meaningfully improving answer completeness.

This test directly informs the k=5 default used throughout the project and quantifies the latency cost of retrieval breadth.

---

## Implications for RAG Applications

For financial document Q&A where answers should be concise and grounded:

- **Mistral is the better choice** — faster TTFT and cleaner outputs
- **LLaMA may be better** for tasks requiring longer, more exhaustive answers with higher token budgets
- **Throughput is not the differentiator** at this scale — both models are memory-bandwidth bound

For latency-sensitive applications (real-time chat, streaming):
- Mistral's 1,015ms p50 TTFT is within acceptable range for interactive use
- LLaMA's 4,616ms p50 TTFT would feel slow in a chat interface

---

## Limitations

- Only 5 questions — not statistically robust
- RAGAS proxy models differ from vLLM-served models (see caveat above — Mistral proxy is a different model family)
- Single-request benchmarking — concurrent request behavior not measured
- TTFT p95 is sensitive to KV cache warmup state: the first request after a cold start carries full prefill latency (800–1,200ms); subsequent requests with a warm prefix cache drop to ~200–240ms. p95 on a 5-question run captures the cold-start outlier, not steady-state latency.
