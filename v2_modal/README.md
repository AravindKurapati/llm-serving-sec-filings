# v2 — Modal Deployment

The working production version. Runs latest vLLM on Modal A10G GPUs with no dependency conflicts.

## Files

- `finsight.py` - complete Modal backend (index building, vLLM server mode, streaming FastAPI endpoints)
- `app.py` - legacy Streamlit prototype kept for reference; the React app in `frontend/` is the maintained UI

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

### Deploy persistent streaming API endpoints

```bash
modal deploy finsight.py
```

Deploys two model-specific SSE endpoints:
```
https://your-workspace--finsight-llama-stream.modal.run
https://your-workspace--finsight-mistral-stream.modal.run
```

Each endpoint exposes:
- `POST /v1/stream` for browser-safe SSE streaming
- `GET /health` and `GET /v1/status` for deployment checks

The endpoints stay live until you run `modal app stop finsight`. You only pay when requests come in (scales to zero when idle).

### Smoke test an endpoint

```bash
curl https://your-workspace--finsight-llama-stream.modal.run/health

curl -N -X POST https://your-workspace--finsight-llama-stream.modal.run/v1/stream \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are Apple's main supply chain risks?\",\"k\":3,\"max_tokens\":80}"
```

The final SSE metrics event includes latency, token counts, model metadata, and ranked source previews.

## Architecture

```
User
  -> React frontend (Vite)
  -> POST /v1/stream on one of two Modal endpoints
  -> FastAPI streaming proxy (Modal CPU)
  -> VLLMServer class (Modal A10G)
  -> vLLM OpenAI-compatible server subprocess
     -> BGE-small embed query
     -> FAISS search top-k chunks
     -> LLaMA 3.1 8B or Mistral 7B
  -> Modal Volume
     -> chunks.faiss
     -> meta.npy
     -> raw SEC filings
```

## Why Modal

| Problem on Kaggle/Colab | Modal solution |
|------------------------|----------------|
| T4 = sm_75, FlashInfer requires sm_80+ | A10G = sm_86, full support |
| Pre-installed packages conflict with vLLM | Clean Debian Slim container |
| Session expires, data lost | Persistent Volumes |
| No public URL without ngrok | Built-in web endpoints |
| Can't pin GPU type | Explicit `gpu="A10G"` |

## Frontend Quick Start (React)

The project includes a React frontend in the `frontend/` directory as an alternative to Streamlit.

### Setup

1. **Configure environment variables:**
   ```bash
   cd frontend
   cp .env.local.example .env.local
   ```
   
   Edit `.env.local` and fill in the two Modal endpoint URLs (obtained after running `modal deploy finsight.py`):
   ```
   VITE_LLAMA_URL=https://your-workspace--finsight-llama-stream.modal.run
   VITE_MISTRAL_URL=https://your-workspace--finsight-mistral-stream.modal.run
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run development server:**
   ```bash
   npm run dev
   ```
   
   Opens at http://localhost:5173

### Testing without Modal deployed

To develop/test the frontend without a live Modal deployment:

1. Run the mock server in one terminal:
   ```bash
   python -m uvicorn scripts.mock_stream_server:app --port 8001
   ```

2. In `.env.local`, set both URLs to the local mock server:
   ```
   VITE_LLAMA_URL=http://localhost:8001
   VITE_MISTRAL_URL=http://localhost:8001
   ```

3. Start the dev server in another terminal:
   ```bash
   npm run dev
   ```
