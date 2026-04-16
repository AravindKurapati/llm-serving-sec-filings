# LLM Serving on SEC Filings

> RAG over real 10-K filings: LLaMA 3.1 8B vs Mistral 7B, benchmarked on A10G via Modal

This project started as a Kaggle notebook and evolved into a production Modal deployment after hitting real infrastructure walls. The repo preserves both versions. (the original attempt and the working solution)

---

## What This Does

- Downloads real SEC 10-K filings (Apple, Microsoft, Google, Amazon, Meta for 3 years each)
- Chunks and embeds them with BGE-small into a FAISS index
- Serves LLaMA 3.1 8B and Mistral 7B via vLLM on Modal A10G GPUs
- Benchmarks TTFT, TPOT, and throughput for both models
- Streams tokens to a React frontend via SSE (real TTFT, no blocking)

---

## Results

Benchmarked on Modal A10G (sm_86), 5 questions, 400 max tokens. TTFT measured as real wall-clock time to first SSE token via vLLM server mode (not estimated).

| Metric | LLaMA 3.1 8B | Mistral 7B |
|--------|-------------|------------|
| TTFT p50 | 198ms | 240ms |
| TTFT p95 | 882ms | 1,225ms |
| TPOT p50 | 34.3ms | 31.6ms |
| Throughput avg | 27.6 tok/s | 29.5 tok/s |

**Key finding**: On A10G hardware these two models are infrastructure-equivalent. TTFT p50 is within 42ms of each other (~200ms both) and throughput is within 7%. The real differentiator is output quality: Mistral produces concise, well-structured answers that stop naturally; LLaMA tends toward verbosity and citation-repetition artifacts at the token limit. Choose based on answer quality requirements, not latency.

Note: p95 TTFT reflects the cold KV-cache first request. Warm-cache requests settle at ~200-240ms for both models.

Full results: [`results/benchmark_20260415_190523.json`](results/benchmark_20260415_190523.json)

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
- Modal account (modal.com with free tier includes $30/month)
- HuggingFace account with LLaMA and Mistral access approved

### Setup

```bash
pip install modal
modal setup          # authenticates via browser
```

Add your HuggingFace token as a Modal secret:
- Go to modal.com -> Secrets -> New secret -> HuggingFace template
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

### Run the Frontend

```bash
# Add your deployed URLs to .env first:
# MODAL_LLAMA_URL=https://your-workspace--finsight-llama-stream.modal.run
# MODAL_MISTRAL_URL=https://your-workspace--finsight-mistral-stream.modal.run

cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## Why Not Kaggle?

See [`v1_kaggle/issues.md`](v1_kaggle/issues.md) for the full story.

Short version: Kaggle's T4 GPUs are sm_75. Modern vLLM (0.6+) requires FlashInfer for its attention backend, and FlashInfer dropped sm_75 support. Every workaround (pinning old vLLM, pinning tokenizers, patching source files) created a new dependency conflict. After hitting the same wall on Colab, the right fix was to use Modal's A10G (sm_86) where latest vLLM just works.

---

## Architecture

```
User
 └─ React frontend (Vite, runs locally)
     └─ POST /v1/stream (Modal public URL, SSE)
         └─ FastAPI proxy (Modal, CPU)
             └─ vLLM serve subprocess (Modal, A10G GPU)
                 ├─ BGE-small embedder → FAISS retrieval
                 └─ LLaMA 3.1 8B or Mistral 7B
                     └─ Modal Volume (persists index + model weights)
```

Modal Volumes cache model weights after the first download. Subsequent cold starts skip the download and go straight to inference.

---

## Status

- Infrastructure benchmarking: complete (real TTFT via SSE streaming)
- Output quality: complete (answer quality analysis + LLM-as-judge eval)
- Concurrency: complete (stress tested up to 8 concurrent requests)
- Frontend: React (Vite) replacing Streamlit, in progress

## Requirements

- Python 3.11+
- Modal CLI (`pip install modal`)
- HuggingFace token with access to:
  - `meta-llama/Meta-Llama-3.1-8B-Instruct`
  - `mistralai/Mistral-7B-Instruct-v0.3`
