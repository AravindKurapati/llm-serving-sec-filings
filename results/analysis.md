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

## Implications for RAG Applications

For financial document Q&A where answers should be concise and grounded:

- **Mistral is the better choice** cause faster TTFT and cleaner outputs
- **LLaMA may be better** for tasks requiring longer, more exhaustive answers with higher token budgets
- **Throughput is not the differentiator** at this scale - both models are memory-bandwidth bound

For latency-sensitive applications (real-time chat, streaming):
- Mistral's 1,015ms p50 TTFT is within acceptable range for interactive use
- LLaMA's 4,616ms p50 TTFT would feel slow in a chat interface

---

## Limitations

- TTFT measurement is estimated (batch `generate()` doesn't expose streaming token times)
- Only 5 questions — not statistically robust
- No evaluation of answer factual accuracy (RAGAS evaluation pending)
- Single-request benchmarking — concurrent request behavior not measured
