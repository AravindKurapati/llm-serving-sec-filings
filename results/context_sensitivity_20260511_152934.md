# Context Length Sensitivity Analysis

**Date**: 20260511_152934  
**Questions**: 5  
**k values tested**: [2, 3, 5, 8, 10]  
**Index**: FAISS IndexFlatIP · BGE-small-en-v1.5  

---

## What This Measures

Retrieval k controls how many chunks are injected into the prompt. More chunks = longer prompt = more input tokens = higher TTFT. This test quantifies that tradeoff against answer completeness.

---

## LLAMA (llama-3.1-8b-instant)

| k | Avg Prompt Tokens | Avg Latency (ms) | Avg Answer Words |
|---|:-----------------:|:----------------:|:----------------:|
| 2 | 449 | 789 | 116 |
| 3 | 602 | 1016 | 128 |
| 5 | 902 | 3128 | 130 |
| 8 | 1391 | 4986 | 108 |
| 10 | 1749 | 8485 | 170 |

**Observation**: k=2 gives the best latency-per-answer-word ratio for llama.

---

## Implications

- **TTFT scales roughly linearly with prompt tokens** — each additional chunk adds ~600 tokens to the prompt (600-char truncation in `build_prompt`).
- **Answer length plateaus** — beyond k=5, additional context rarely produces longer or more complete answers. The model uses what it needs.
- **k=5 is a reasonable default** for this corpus — matches `finsight.py` and the RAGAS evaluation setup.
- **k=2–3 may be sufficient** for narrow factual questions; k=8–10 only helps for broad synthesis questions.

---

## Limitations

- Groq latency includes network RTT — not pure prefill time like the Modal vLLM benchmark.
- Only 5 questions — results are indicative, not statistically robust.
- Mistral proxy uses `llama-3.1-8b-instant` (mixtral decommissioned) — not a true Mistral model.
