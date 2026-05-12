# RAGAS Offline Evaluation — Real vLLM Outputs

**Date**: 2026-05-11  
**Source**: `benchmark_20260415_190523.json` (Modal A10G — LLaMA 3.1 8B + Mistral 7B)  
**Retrieval**: FAISS IndexFlatIP · BGE-small-en-v1.5 · top-5 chunks  
**Judge LLM**: Groq `llama-3.3-70b-versatile` (max_tokens=4096)  

---

## What Changed vs Prior RAGAS Run

The previous RAGAS evaluation (`ragas_eval.md`) used Groq proxies for answer
generation. Mixtral was decommissioned on Groq, so both model lanes mapped to
`llama-3.1-8b-instant` — making the LLaMA vs Mistral comparison meaningless
(scores were identical by construction).

This run uses **real answers from the deployed vLLM endpoints** (Modal A10G):
the exact outputs of `meta-llama/Meta-Llama-3.1-8B-Instruct` and
`mistralai/Mistral-7B-Instruct-v0.3`. Contexts are re-retrieved from the
local FAISS index to provide full 800-char prose-filtered chunks to the judge.

---

## Results

| Metric | LLaMA 3.1 8B | Mistral 7B | Delta |
|--------|:------------:|:----------:|:-----:|
| faithfulness | **0.9452** | 0.8933 | LLaMA +5.5% |
| answer_relevancy | **0.9353** | 0.8819 | LLaMA +6.0% |

### Per-Question Breakdown

| Question | LLaMA faith. | Mistral faith. | LLaMA rel. | Mistral rel. |
|----------|:------------:|:--------------:|:----------:|:------------:|
| Apple supply chain risks | 0.9762 | 1.0000 | 0.9956 | 0.9956 |
| Microsoft cloud revenue | 1.0000 | 0.6667 | 0.8479 | 0.9200 |
| Meta AI infrastructure | 1.0000 | 1.0000 | 0.9247 | 0.7754 |
| Google advertising revenue | 0.7500 | 0.8000 | 0.9454 | 0.8074 |
| Amazon cybersecurity | 1.0000 | 1.0000 | 0.9631 | 0.9111 |

---

## Interpretation

**LLaMA scores higher on both metrics** — the opposite of the conciseness story
from `answer_quality.py`.

This is not a contradiction. The two evals measure different things:

- `answer_quality.py` measures **verbosity and repetition** on actual vLLM outputs:
  LLaMA is 31% more verbose and 3.7x more repetitive — worse for UX.
- RAGAS faithfulness measures **whether claims are grounded in context**:
  LLaMA's verbosity works in its favour here. Its answers enumerate claims
  explicitly and trace them directly to retrieved chunks. The judge finds most
  claims verifiable because LLaMA essentially quotes the context.
- Mistral's **concise synthesis** can abstract slightly beyond the literal context
  text. The Microsoft question (faithfulness: 0.6667) is the clearest example:
  Mistral's answer synthesised a clean narrative but included a few inferred
  statements the judge couldn't directly ground in the retrieved passages.

**Practical takeaway**: LLaMA is more faithful but verbose. Mistral is more
concise but occasionally synthesises slightly beyond the retrieved evidence.
For a RAG application where citation traceability matters, LLaMA's style is
actually safer. For a chat interface where users want a crisp answer, Mistral
remains preferable.

> `context_precision` omitted — requires ground_truth; the 5 benchmark questions
> do not overlap with `data/testset.json`. Retrieval quality is identical for
> both models (same FAISS index, same k, same questions) so the metric would
> not differentiate them anyway.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Source answers | Real vLLM (Modal A10G) via `benchmark_20260415_190523.json` |
| Questions scored | 5 per model |
| Retrieval k | 5 |
| Context truncation | 800 chars/chunk (HTML decoded, prose-filtered) |
| Judge LLM | `llama-3.3-70b-versatile` |
| Judge max_tokens | 4096 |
| Inter-sample delay | 90s |

---

## Limitations

- Only 5 questions — not statistically robust
- `faithfulness` judge sees re-retrieved contexts (up to 800 chars/chunk prose-filtered),
  not byte-identical context passed to vLLM (which used 600-char truncation with HTML entities)
- Judge LLM (`llama-3.3-70b-versatile`) may have its own biases toward verbose, explicit answers
