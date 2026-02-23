# LLM Serving on SEC Filings

> RAG over real 10-K filings — LLaMA 3.1 8B vs Mistral 7B, benchmarked on A10G via Modal

This project started as a Kaggle notebook and evolved into a production Modal deployment after hitting real infrastructure walls. The repo preserves both versions — the original attempt and the working solution — because the journey is as useful as the destination.

---

## What This Does

- Downloads real SEC 10-K filings (Apple, Microsoft, Google, Amazon, Meta — 3 years each)
- Chunks and embeds them with BGE-small into a FAISS index
- Serves LLaMA 3.1 8B and Mistral 7B via vLLM on Modal A10G GPUs
- Benchmarks TTFT, TPOT, and throughput for both models
- Exposes a FastAPI endpoint for a Streamlit chat frontend

---

## Results

Benchmarked on Modal A10G (sm_86), 5 questions, 400 max tokens:

| Metric | LLaMA 3.1 8B | Mistral 7B |
|--------|-------------|------------|
| TTFT p50 | 4,616ms | 1,015ms |
| TTFT p95 | 4,631ms | 2,402ms |
| TPOT p50 | 23.1ms | 23.5ms |
| Throughput avg | 28.9 tok/s | 28.6 tok/s |

**Key finding**: Mistral is 4.5x faster on TTFT with nearly identical throughput and TPOT. The bottleneck is prefill, not decode. Mistral's smaller architecture prefills faster — important for latency-sensitive RAG applications where the prompt is long (context + retrieved chunks).

Mistral also produced more concise, better-structured answers without the citation repetition artifacts LLaMA showed at the 400-token limit.

Full results: [`results/benchmark_20260222.json`](results/benchmark_20260222.json)

---

## Repo Structure

```
llm-serving-sec-filings/
├── v1_kaggle/                  # Original Kaggle attempt
│   ├── llm-serving-claude.ipynb
│   ├── issues.md               # Dependency hell — what broke and why
│   └── README.md
│
├── v2_modal/                   # Working production version
│   ├── finsight.py             # Modal backend (vLLM + FAISS + FastAPI)
│   ├── app.py                  # Streamlit frontend
│   └── README.md
│
├── results/
│   ├── benchmark_20260222.json # Raw benchmark output
│   └── analysis.md             # Findings and interpretation
│
├── scripts/                    # Utility scripts
├── utils/                      # Shared utilities
├── docs/                       # Architecture and deployment docs
├── requirements.txt
└── .env.example
```

---

## Quick Start (v2 Modal)

### Prerequisites
- Modal account (modal.com — free tier includes $30/month)
- HuggingFace account with LLaMA and Mistral access approved

### Setup

```bash
pip install modal
modal setup          # authenticates via browser
```

Add your HuggingFace token as a Modal secret:
- Go to modal.com → Secrets → New secret → HuggingFace template
- Name it `huggingface-secret`

### First Run (builds index, benchmarks both models)

```bash
cd v2_modal
modal run finsight.py
# Takes ~15 min first time (downloads models + builds FAISS index)
# Costs ~$0.50-1.00 in Modal credits
```

### Deploy API Endpoint

```bash
modal deploy finsight.py
# Prints your public URL:
# https://your-workspace--finsight-api-query.modal.run
```

### Run Streamlit Frontend

```bash
# Paste your Modal URL into app.py's MODAL_URL variable first
pip install streamlit requests
streamlit run v2_modal/app.py
```

---

## Why Not Kaggle?

See [`v1_kaggle/issues.md`](v1_kaggle/issues.md) for the full story.

Short version: Kaggle's T4 GPUs are sm_75. Modern vLLM (0.6+) requires FlashInfer for its attention backend, and FlashInfer dropped sm_75 support. Every workaround (pinning old vLLM, pinning tokenizers, patching source files) created a new dependency conflict. After hitting the same wall on Colab, the right fix was to use Modal's A10G (sm_86) where latest vLLM just works.

---

## Architecture

```
User
 └─ Streamlit (runs locally)
     └─ POST /v1/chat (Modal public URL)
         └─ FastAPI endpoint (Modal, CPU)
             └─ vLLM LLM.generate() (Modal, A10G GPU)
                 ├─ BGE-small embedder → FAISS retrieval
                 └─ LLaMA 3.1 8B or Mistral 7B
                     └─ Modal Volume (persists index + model weights)
```

Modal Volumes mean model weights are cached after the first download — subsequent runs skip the 84-second download and go straight to inference.

---

## Roadmap

- [ ] RAGAS evaluation (faithfulness, answer relevancy, context precision)
- [ ] Streaming responses with real TTFT measurement
- [ ] Add Fastino SLM when API becomes available
- [ ] DistServe-style prefill/decode disaggregation experiment

---

## Requirements

- Python 3.11+
- Modal CLI (`pip install modal`)
- HuggingFace token with access to:
  - `meta-llama/Meta-Llama-3.1-8B-Instruct`
  - `mistralai/Mistral-7B-Instruct-v0.3`
