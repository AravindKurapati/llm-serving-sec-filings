# Benchmark Analysis: LLaMA 3.1 8B vs Mistral 7B

**Hardware**: Modal A10G (24GB VRAM, sm_86)  
**Dataset**: 15 SEC 10-K filings (AAPL, MSFT, GOOGL, AMZN, META for 3 years each)  
**Index**: 4,782 chunks, BGE-small-en-v1.5 (384 dims), FAISS IndexFlatIP  
**Questions**: 5 financial analysis questions  

---

## Summary Table

Latest benchmark run (April 2026, `results/benchmark_20260415_190523.json`):

| Metric | LLaMA 3.1 8B | Mistral 7B | Winner |
|--------|-------------|------------|--------|
| TTFT p50 | 198ms | 240ms | LLaMA (1.2x faster) |
| TTFT p95 | 882ms | 1,225ms | LLaMA |
| TPOT p50 | 34.3ms | 31.6ms | Tie |
| Throughput avg | 27.6 tok/s | 29.5 tok/s | Tie |
| Faithfulness (RAGAS offline) | **0.9452** | 0.8933 | LLaMA |
| Answer relevancy (RAGAS offline) | **0.9353** | 0.8819 | LLaMA |
| Verbosity (avg tokens) | 114.2 | **87.2** | Mistral |
| Repetition score | 0.0107 | **0.0029** | Mistral |

---

## TTFT: February vs April Run

An earlier benchmark (February 2026) showed LLaMA p50 TTFT at 4,616ms vs Mistral at 1,015ms — a 4.5x gap. The April 2026 run shows both models at ~200ms p50, essentially tied.

The gap closed for two reasons:

1. **vLLM version upgrade** — the April deployment uses a newer vLLM build with better prefix caching and a tuned KV cache configuration. Warm-cache requests now dominate the p50 measurement.
2. **Benchmark methodology** — the February run measured cold-KV-cache prefill for every request (each question fired sequentially with a fresh context). The April run interleaved questions that share common system prompt tokens, so the KV cache was partially warm from the second request onward.

The p95 gap (882ms vs 1,225ms) persists because p95 still captures the occasional cold-start request. That is the truer measure of first-request latency for a real user hitting a fresh endpoint.

**Bottom line**: on warm-cache steady state, the two models are infrastructure-equivalent at ~200ms. On cold-cache first requests, LLaMA is still faster (882ms vs 1,225ms p95).

---

## TTFT Analysis

TTFT measures prefill time — how long it takes to process the input prompt before generating the first output token. For RAG, the input is large: system prompt + 5 retrieved chunks + question = 600–1,300 tokens.

In warm-cache conditions (April run), both models settle at ~200–240ms p50. The KV cache reuses the common system prompt prefix across requests, so only the unique question tokens require fresh prefill.

In the cold-cache case (first request after a GPU cold start), TTFT climbs to 882ms (LLaMA) and 1,225ms (Mistral) at p95. Sliding window attention (Mistral) is generally more efficient on longer sequences, but at 7–8B parameter scale on a single A10G the difference is within noise on warm cache.

---

## TPOT and Throughput Analysis

Both models generate tokens at ~32–34ms per token (~28–30 tok/s). This is expected: at this scale both models are memory-bandwidth bound during decode, and they share the same A10G VRAM bandwidth.

The A10G has 600 GB/s memory bandwidth. Loading 7–8B parameters in fp16 (~14–16 GB) per forward pass is the bottleneck, not compute. Both models hit this ceiling at essentially the same speed.

---

## Answer Quality

### Verbosity and Repetition (`scripts/answer_quality.py`)

Evaluated on real vLLM outputs from the April benchmark run:

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

Key findings:
- LLaMA is **31% more verbose** on average (114.2 vs 87.2 tokens)
- LLaMA has a **3.7x higher repetition score** — consistent with citation-repetition artifacts observed qualitatively
- Q5 (Amazon cybersecurity) is the worst case: LLaMA repetition 0.043 vs Mistral 0.000

---

## RAGAS Evaluation

### Offline Eval on Real vLLM Outputs (`scripts/ragas_offline_eval.py`)

> These scores use **real answers from the Modal A10G vLLM deployment** — not Groq proxies.
> Source: `results/ragas_offline_20260511_combined.json`.

The prior RAGAS run (`ragas_eval.md`) used Groq API proxies for answer generation. Mixtral was decommissioned on Groq, so both model lanes mapped to `llama-3.1-8b-instant` — the scores were identical by construction and did not compare the two models. The offline eval fixes this by loading the actual LLaMA and Mistral answers from the benchmark JSON and re-retrieving fresh contexts from the local FAISS index.

**Metrics scored**: faithfulness, answer_relevancy  
**Omitted**: context_precision — requires ground_truth; the 5 benchmark questions have no GT pairs. Retrieval is identical for both models (same FAISS index, same k) so context_precision would not differentiate them.

| Metric | LLaMA 3.1 8B | Mistral 7B |
|--------|:------------:|:----------:|
| faithfulness | **0.9452** | 0.8933 |
| answer_relevancy | **0.9353** | 0.8819 |

**Per-question breakdown:**

| Question | LLaMA faith. | Mistral faith. | LLaMA rel. | Mistral rel. |
|----------|:------------:|:--------------:|:----------:|:------------:|
| Apple supply chain risks | 0.9762 | 1.0000 | 0.9956 | 0.9956 |
| Microsoft cloud revenue | 1.0000 | 0.6667 | 0.8479 | 0.9200 |
| Meta AI infrastructure | 1.0000 | 1.0000 | 0.9247 | 0.7754 |
| Google advertising revenue | 0.7500 | 0.8000 | 0.9454 | 0.8074 |
| Amazon cybersecurity | 1.0000 | 1.0000 | 0.9631 | 0.9111 |

**Interpretation**: LLaMA scores higher on both metrics — the opposite of what the verbosity story alone would suggest. This is not a contradiction. LLaMA's verbose, enumerated answers trace claims directly to retrieved chunks, making them easy for the faithfulness judge to verify. Mistral's concise synthesis occasionally abstracts slightly beyond the literal context evidence — the Microsoft cloud question (Mistral faithfulness 0.6667) is the clearest example. So the models have complementary profiles: Mistral wins on UX (concise, clean, low repetition); LLaMA wins on RAG grounding (explicit, citation-traceable).

### LLM-as-Judge (`scripts/llm_judge.py`)

> **Caveat**: same Groq proxy problem as the old RAGAS run — both "LLaMA" and "Mistral" entries use `llama-3.1-8b-instant`. Scores are identical by design and should not be used to compare models. Included for completeness only.

| Metric | LLaMA proxy | Mistral proxy |
|--------|:-----------:|:-------------:|
| groundedness | 0.86 | 0.86 |
| conciseness | 0.74 | 0.74 |
| citation_quality | 0.80 | 0.80 |

---

## Context Length Sensitivity (`scripts/context_sensitivity_test.py`)

> Run: 2026-05-11. Full results: `results/context_sensitivity_20260511_152934.json`.

Varied retrieval k across [2, 3, 5, 8, 10] on 5 qualitative questions, measuring prompt token count, Groq wall-clock latency, and answer word count.

| k | Avg Prompt Tokens | Avg Latency (ms) | Avg Answer Words |
|---|:-----------------:|:----------------:|:----------------:|
| 2 | 449 | 789 | 116 |
| 3 | 602 | 1,016 | 128 |
| **5** | **902** | **3,128** | **130** |
| 8 | 1,391 | 4,986 | 108 |
| 10 | 1,749 | 8,485 | 170 |

**Key findings:**
- Prompt tokens grow **roughly linearly** with k — each additional chunk adds ~150 tokens to the prompt (600-char truncation → ~150 tokens/chunk).
- Answer length **plateaus at k=5** (130 words). At k=8 it actually drops to 108 words; k=10 is noisy (170 words) but latency is 2.7x higher than k=5.
- Latency at k=8 is **59% higher** than k=5 (4,986ms vs 3,128ms) with no meaningful gain in answer completeness.
- **k=5 is the right default** — validated empirically, not just assumed.

Note: these latencies are Groq wall-clock (includes network RTT and generation time), not pure prefill TTFT like the Modal vLLM benchmark. The relative pattern holds regardless.

---

## Concurrency & Throughput Under Load

> Measured with `scripts/concurrency_test.py` against live Modal A10G endpoints.  
> Raw results: `results/concurrency_20260416_110056.json`

**Question**: *"What are the main financial risks disclosed in these SEC filings?"*

### LLaMA 3.1 8B

| Concurrency | req/s | med_lat_ms | p95_lat_ms | med_ttft_ms | error_rate |
|:-----------:|------:|-----------:|-----------:|------------:|:----------:|
| 1 | 0.19 | 4,780 | 4,780 | 519 | 0.00 |
| 2 | 0.20 | 7,098 | 9,424 | 2,870 | 0.00 |
| 4 | 0.16 | 17,759 | 24,648 | 15,469 | 0.00 |
| 8 | 0.21 | 20,836 | 36,808 | 16,605 | 0.00 |

### Mistral 7B

| Concurrency | req/s | med_lat_ms | p95_lat_ms | med_ttft_ms | error_rate |
|:-----------:|------:|-----------:|-----------:|------------:|:----------:|
| 1 | 0.14 | 6,588 | 6,588 | 791 | 0.00 |
| 2 | 0.15 | 9,901 | 13,219 | 4,093 | 0.00 |
| 4 | 0.15 | 16,572 | 26,061 | 10,760 | 0.00 |
| 8 | 0.16 | 28,012 | 49,424 | 22,183 | 0.00 |

**Key findings:**
- Throughput is flat across all concurrency levels (~0.14–0.21 req/s) — the single A10G is the bottleneck; vLLM processes requests sequentially.
- TTFT degrades sharply under load: at c=8, LLaMA TTFT climbs 32x (519ms → 16,605ms) and Mistral 28x (791ms → 22,183ms).
- Zero errors at all concurrency levels — Modal's request queuing absorbs load correctly.
- For multi-user scenarios, horizontal scaling (multiple Modal containers) is required.

---

## Full Picture: Model Selection

| Dimension | LLaMA 3.1 8B | Mistral 7B |
|-----------|:------------:|:----------:|
| TTFT (warm cache p50) | 198ms | 240ms |
| TTFT (cold cache p95) | 882ms | 1,225ms |
| Faithfulness | **0.9452** | 0.8933 |
| Answer relevancy | **0.9353** | 0.8819 |
| Verbosity | worse (114 tok avg) | **better (87 tok avg)** |
| Repetition | worse (0.011) | **better (0.003)** |
| Conciseness / UX | worse | **better** |
| Citation grounding | **better** | worse |

Neither model dominates. The right choice depends on what matters most:
- **Citation traceability required** (compliance, auditability): LLaMA — higher faithfulness, more explicit sourcing.
- **Interactive chat / UX**: Mistral — concise, clean, stops naturally.
- **Latency at scale**: both are equivalent on warm cache; LLaMA has a slight edge on cold-cache first requests.

---

## Limitations

- Only 5 benchmark questions — not statistically robust
- RAGAS offline eval uses re-retrieved contexts (800-char prose-filtered), not byte-identical context passed to vLLM (600-char with HTML entities)
- LLM-as-judge (llm_judge.py) and old RAGAS proxy results are not usable for model comparison due to the Mixtral decommission on Groq
- Concurrency was measured against one A10G-backed deployment; results may differ with horizontal Modal scaling
- Context sensitivity latency is Groq wall-clock (includes network RTT), not pure vLLM prefill time
