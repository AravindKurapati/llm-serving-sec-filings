# SPEC: vLLM Server Mode + React Frontend

> Read this fully before touching any code. Implement Phase 1 before Phase 2.

---

## Context

Current state:
- `v2_modal/finsight.py` runs vLLM in **library mode** (`LLM.generate()`), which blocks until full response is done
- TTFT is **estimated** via a heuristic (`input_tokens / total_time * 3`) — not real
- Frontend is Streamlit (`v2_modal/app.py`) — can't do real streaming UI
- `api_query` is a `@modal.fastapi_endpoint` that owns the full inference loop

Target state:
- vLLM runs in **server mode** (`vllm serve`) as a subprocess inside a Modal `@modal.cls`
- TTFT is measured as **wall-clock time to first SSE chunk** — real number
- Frontend is a **Vite + React app** (`frontend/`) that streams tokens and displays live metrics
- Modal exposes a **streaming FastAPI proxy** that tunnels SSE from vLLM to the browser

---

## Phase 1: vLLM Server Mode (`v2_modal/finsight.py`)

### What changes

Replace the `RAGEngine` class and `api_query` endpoint with a new architecture:

1. A `VLLMServer` Modal class that:
   - Spawns `vllm serve <model_id>` as a subprocess on `@modal.enter()`
   - Waits for the server to be ready by polling `http://localhost:8000/health`
   - Keeps FAISS + embedder loaded for retrieval (same as before)
   - Exposes a `query_stream` method that:
     - Runs FAISS retrieval
     - Builds the prompt
     - POSTs to `http://localhost:8000/v1/chat/completions` with `stream=True`
     - Records `t0` before the POST, `t_first_token` on first SSE data chunk
     - Yields SSE chunks through to the caller
     - After stream ends, appends a final `[METRICS]` SSE event with real TTFT, TPOT, throughput

2. A new `@modal.fastapi_endpoint` (`/v1/stream`) that:
   - Accepts `{ question, model, k, max_tokens }` via POST
   - Returns a `StreamingResponse` with `media_type="text/event-stream"`
   - Proxies the SSE stream from `VLLMServer.query_stream()`

### SSE event format

All events use standard SSE format (`data: ...\n\n`).

Token chunks (passthrough from vLLM OpenAI format):
```
data: {"choices": [{"delta": {"content": "token text"}}]}
```

Metrics event (appended after stream ends, before `[DONE]`):
```
data: {"type": "metrics", "ttft_ms": 312.4, "tpot_ms": 23.1, "tokens": 187, "throughput_tps": 28.6, "input_tokens": 412}
```

Done sentinel:
```
data: [DONE]
```

### TTFT measurement

```python
t0 = time.perf_counter()
response = requests.post(vllm_url, json=payload, stream=True)
for line in response.iter_lines():
    if line and line.startswith(b"data: ") and line != b"data: [DONE]":
        chunk = json.loads(line[6:])
        content = chunk["choices"][0]["delta"].get("content", "")
        if content and t_first_token is None:
            t_first_token = time.perf_counter()  # <- real TTFT
            ttft_ms = (t_first_token - t0) * 1000
        yield f"data: {line[6:].decode()}\n\n"
```

### Model switching

vLLM server mode loads one model per process. Two options:
- **Option A (simpler):** One `VLLMServer` class with `model_name` as a `modal.parameter`. Deploy two instances. Frontend picks endpoint per model. Recommended.
- **Option B:** Single server, model switching via vLLM's LoRA or swap — more complex, not needed here.

Go with Option A.

### Keep the `RAGEngine.benchmark()` method

The existing `benchmark()` method and `@app.local_entrypoint()` are used for the CLI benchmark run that produced `results/benchmark_20260222.json`. Keep them but update them to use server mode under the hood (or keep library mode for benchmark only, clearly commented). Don't break the benchmark workflow.

### What to remove

- The `api_query` `@modal.fastapi_endpoint` function (replaced by the streaming proxy)
- The library-mode `LLM()` instantiation inside `api_query`
- The TTFT estimation heuristic in `api_query`

---

## Phase 2: React Frontend (`frontend/`)

### Stack

- **Vite + React** (not Next.js — no server needed, this runs locally)
- Plain CSS or Tailwind (no component library needed)
- No state management library — `useState` + `useRef` + a custom `useStream` hook is enough

### Directory structure

```
frontend/
  index.html
  package.json
  vite.config.js
  src/
    App.jsx           # root, layout, model selector
    components/
      ChatPanel.jsx   # single model chat (receives model prop)
      MessageBubble.jsx
      MetricsBar.jsx  # shows TTFT/TPOT/throughput after each response
      SourceDrawer.jsx # collapsible retrieved chunks
    hooks/
      useStream.js    # core hook — handles SSE fetch, token accumulation, metrics
    api.js            # MODAL_URL config + fetch wrapper
    index.css
```

### `useStream` hook spec

This is the most important piece. It must:

```js
// Usage:
const { stream, isStreaming, answer, metrics, error, reset } = useStream(modalUrl)

// stream(question, model, k, maxTokens) triggers a new request
// answer: string — accumulates as tokens arrive
// metrics: { ttft_ms, tpot_ms, tokens, throughput_tps } | null — set when [METRICS] event received
// isStreaming: bool
```

Implementation notes:
- Use `fetch` with `ReadableStream`, not `EventSource` (EventSource is GET-only)
- Parse SSE lines manually from the stream reader
- On each `data:` line: if it's a token chunk, append `choices[0].delta.content` to `answer`
- On `{"type": "metrics", ...}`: set `metrics` state
- On `[DONE]`: set `isStreaming = false`
- Use `useRef` for `t0` so TTFT display is accurate (don't rely on backend-reported value for UI responsiveness display)
- Abort via `AbortController` on unmount or when `reset()` is called

### Layout

Two-panel side-by-side comparison:

```
┌─────────────────────────────────────────────────────────┐
│  FinSight — SEC 10-K RAG                          [k: 5] │
├──────────────────────────┬──────────────────────────────┤
│  LLaMA 3.1 8B            │  Mistral 7B                  │
│  [chat history]          │  [chat history]               │
│                          │                               │
│  TTFT: 4612ms            │  TTFT: 1021ms                 │
│  TPOT: 23.1ms            │  TPOT: 23.5ms                 │
├──────────────────────────┴──────────────────────────────┤
│  [question input — shared, fires both panels at once]    │
└─────────────────────────────────────────────────────────┘
```

One input fires both models simultaneously via `Promise.all([stream(q, 'llama'), stream(q, 'mistral')])` — or more precisely, calls both `useStream` hooks at the same time.

Metrics bar appears below each answer once the `[METRICS]` event is received. Before that, show a live "streaming..." indicator.

Retrieved sources are in a collapsible `<details>` below each answer.

### `api.js`

```js
export const LLAMA_URL = import.meta.env.VITE_LLAMA_URL
export const MISTRAL_URL = import.meta.env.VITE_MISTRAL_URL
```

User sets these in a `.env.local` file:
```
VITE_LLAMA_URL=https://your-workspace--finsight-llama-stream.modal.run
VITE_MISTRAL_URL=https://your-workspace--finsight-mistral-stream.modal.run
```

Add `.env.local` to `.gitignore`. Add `.env.local.example` with placeholder values.

### `package.json` dependencies

```json
{
  "dependencies": {
    "react": "^18",
    "react-dom": "^18"
  },
  "devDependencies": {
    "vite": "^5",
    "@vitejs/plugin-react": "^4"
  }
}
```

No axios, no react-query, no redux. Keep it minimal.

---

## API contract between frontend and backend

### Request

```
POST /v1/stream
Content-Type: application/json

{
  "question": "What are Apple's main supply chain risks?",
  "k": 5,
  "max_tokens": 400
}
```

Note: no `model` field in the request body — the model is baked into the endpoint (one URL per model, per Option A above).

### Response

```
Content-Type: text/event-stream

data: {"choices": [{"delta": {"content": "Apple"}}]}

data: {"choices": [{"delta": {"content": " faces"}}]}

... (more token chunks) ...

data: {"type": "metrics", "ttft_ms": 312.4, "tpot_ms": 23.1, "tokens": 187, "input_tokens": 412, "throughput_tps": 28.6, "contexts": [{"src": "AAPL_..._10K.txt", "text": "..."}]}

data: [DONE]
```

---

## Files to create / modify

### Phase 1
- **Modify**: `v2_modal/finsight.py` — replace `api_query`, update `RAGEngine` or add `VLLMServer`

### Phase 2
- **Create**: `frontend/` directory with all files listed above
- **Create**: `frontend/.env.local.example`
- **Modify**: root `.gitignore` — add `frontend/.env.local`, `frontend/node_modules/`, `frontend/dist/`
- **Modify**: `v2_modal/README.md` — update Quick Start to include `npm run dev` for frontend

### Do NOT touch
- `v2_modal/finsight.py` `build_index()` — index building is already done
- `results/` — benchmark results stay as-is
- `v1_kaggle/` — don't touch

---

## Testing checklist (before marking done)

- [ ] `modal deploy finsight.py` succeeds with two deployed endpoints (llama + mistral)
- [ ] `curl -N -X POST <llama_url>/v1/stream -H 'Content-Type: application/json' -d '{"question":"test","k":3,"max_tokens":50}'` streams SSE chunks
- [ ] Final SSE event before `[DONE]` is a `{"type": "metrics"}` object with numeric values (not estimated)
- [ ] `npm run dev` in `frontend/` starts without errors
- [ ] Both panels stream simultaneously when question is submitted
- [ ] MetricsBar appears after stream ends with TTFT, TPOT, throughput
- [ ] SourceDrawer toggles open/closed
- [ ] No hardcoded URLs in committed code (only in `.env.local`)
