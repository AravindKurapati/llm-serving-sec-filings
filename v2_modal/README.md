# v2 — Modal Deployment

The working production version. Runs latest vLLM on Modal A10G GPUs with no dependency conflicts.

## Files

- `finsight.py` — complete Modal backend (index building, RAG engine, FastAPI endpoint)
- `app.py` — Streamlit frontend (runs locally, hits Modal endpoint)

## How to Run

### One-time setup

```bash
pip install modal
modal setup
```

Add HuggingFace token to Modal secrets:
- modal.com → Secrets → New secret → HuggingFace template
- Name: `huggingface-secret`

Make sure you have model access approved on HuggingFace for:
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### First run (builds index + benchmarks)

```bash
modal run finsight.py
```

This will:
1. Spin up an A10G container (~3.6 second cold start)
2. Download 15 SEC 10-K filings (AAPL, MSFT, GOOGL, AMZN, META — 3 years each)
3. Embed 4,782 chunks with BGE-small into a FAISS index
4. Save everything to a Modal Volume (persists between runs)
5. Benchmark LLaMA 3.1 8B and Mistral 7B on 5 questions
6. Print comparison table + save `benchmark_TIMESTAMP.json` locally

Cost: ~$0.50-1.00. Takes ~15 minutes first time, ~8 minutes on subsequent runs (index already built, just benchmarks).

### Subsequent runs (index already built)

The `build_index` call is commented out in `main()` — just runs the benchmarks directly.

```bash
modal run finsight.py
```

### Deploy persistent API endpoint

```bash
modal deploy finsight.py
```

Prints a URL like:
```
https://your-workspace--finsight-api-query.modal.run
```

The endpoint stays live until you run `modal app stop finsight`. You only pay when requests come in (scales to zero when idle).

### Run Streamlit frontend

```bash
# 1. Paste your Modal URL into app.py:
# MODAL_URL = "https://your-workspace--finsight-api-query.modal.run"

# 2. Install and run
pip install streamlit requests
streamlit run app.py
```

Opens at http://localhost:8501. Select LLaMA or Mistral from the sidebar, ask questions about the SEC filings, and see TTFT/TPOT metrics under each answer.

## Architecture

```
User
 └─ Streamlit (localhost:8501)
     └─ POST to Modal URL
         └─ api_query function (Modal, A10G)
             ├─ BGE-small embed query
             ├─ FAISS search → top 5 chunks
             └─ vLLM generate (LLaMA or Mistral)
                 └─ Modal Volume
                     ├─ chunks.faiss
                     ├─ meta.npy
                     └─ raw SEC filings
```

## Why Modal

| Problem on Kaggle/Colab | Modal solution |
|------------------------|----------------|
| T4 = sm_75, FlashInfer requires sm_80+ | A10G = sm_86, full support |
| Pre-installed packages conflict with vLLM | Clean Debian Slim container |
| Session expires, data lost | Persistent Volumes |
| No public URL without ngrok | Built-in web endpoints |
| Can't pin GPU type | Explicit `gpu="A10G"` |
