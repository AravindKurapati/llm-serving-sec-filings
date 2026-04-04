import modal
import time
from pathlib import Path

app = modal.App("finsight")

# ── Docker image with everything needed ─────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "sentence-transformers",
        "faiss-cpu",
        "fastapi[standard]",
        "sec-edgar-downloader",
        "ragas",
        "datasets",
        "numpy",
        "requests",
    )
)

# ── Persistent volume — model weights + FAISS index ─────────
volume = modal.Volume.from_name("finsight-data", create_if_missing=True)
VOLUME_PATH = Path("/data")
INDEX_PATH  = VOLUME_PATH / "chunks.faiss"
META_PATH   = VOLUME_PATH / "meta.npy"

MODELS = {
    "llama":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

# ── STEP 1: Download + embed SEC filings (run once) ─────────
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def build_index():
    import re
    import numpy as np
    import faiss
    from pathlib import Path
    from sec_edgar_downloader import Downloader
    from sentence_transformers import SentenceTransformer

    RAW = VOLUME_PATH / "raw"
    RAW.mkdir(parents=True, exist_ok=True)

    dl = Downloader("FinSight", "aravind@email.com", str(VOLUME_PATH))
    for ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]:
        print(f"Downloading {ticker}...")
        dl.get("10-K", ticker, limit=3)

    def clean(text):
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text)
        return text.strip()

    filing_root = VOLUME_PATH / "sec-edgar-filings"
    for ticker_dir in filing_root.iterdir():
        ticker = ticker_dir.name
        tenk_dir = ticker_dir / "10-K"
        if not tenk_dir.exists():
            continue
        for accession_dir in sorted(tenk_dir.iterdir()):
            for fname in ["primary-document.htm", "primary-document.html", "full-submission.txt"]:
                fpath = accession_dir / fname
                if fpath.exists():
                    text = clean(fpath.read_text(errors="ignore"))
                    if len(text.split()) > 500:
                        out = RAW / f"{ticker}_{accession_dir.name[:8]}_10K.txt"
                        out.write_text(text)
                        print(f"Saved {out.name}: {len(text.split()):,} words")
                    break

    def chunk(text, size=800, overlap=200):
        words = text.split()
        out, i = [], 0
        while i < len(words):
            out.append(" ".join(words[i:i+size]))
            i += max(size - overlap, 1)
        return out

    corpus = []
    for p in RAW.glob("*.txt"):
        txt = p.read_text(errors="ignore").strip()
        company = p.stem.split("_")[0]
        for i, ch in enumerate(chunk(txt)):
            corpus.append({"doc_id": f"{p.stem}-{i}", "company": company,
                           "text": ch, "src": p.name})

    print(f"Total chunks: {len(corpus)}")

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Embedding... (BGE-small, 384 dims)")
    X = embedder.encode(
        [r["text"] for r in corpus],
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X.astype("float32"))
    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(META_PATH), np.array(corpus, dtype=object))

    volume.commit()
    print(f"Index saved: {index.ntotal} chunks, dim={X.shape[1]}")


# ── STEP 2 (server mode): vLLM subprocess + streaming RAG ───
#
# VLLMServer starts `vllm serve` as a subprocess and waits for it to
# be healthy before accepting requests.  TTFT is measured as wall-clock
# time to first SSE token — a real number, not an estimate.
#
# Deploy two instances via `modal.parameter`:
#   modal deploy finsight.py  →  two GPU containers, one per model
#   Frontend picks endpoint by model (Option A from spec).
@app.cls(
    gpu="A10G",
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=120,
)
class VLLMServer:
    model_name: str = modal.parameter(default="llama")

    @modal.enter()
    def start(self):
        import subprocess
        import requests as req
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        volume.reload()

        model_id = MODELS[self.model_name]
        print(f"Starting vLLM server for {model_id}...")

        self.proc = subprocess.Popen(
            [
                "vllm", "serve", model_id,
                "--dtype", "half",
                "--max-model-len", "2048",
                "--gpu-memory-utilization", "0.88",
                "--port", "8000",
            ],
        )

        # Poll until healthy (up to 10 minutes)
        for _ in range(120):
            try:
                r = req.get("http://localhost:8000/health", timeout=2)
                if r.status_code == 200:
                    print("vLLM server is ready!")
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise RuntimeError("vLLM server did not become healthy in time")

        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.index    = faiss.read_index(str(INDEX_PATH))
        self.meta     = np.load(str(META_PATH), allow_pickle=True).tolist()
        self.model_id = model_id
        print(f"Ready! Index has {self.index.ntotal} chunks")

    @modal.exit()
    def stop(self):
        if hasattr(self, "proc") and self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()

    def retrieve(self, question: str, k: int = 5):
        vec = self.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        _, ids = self.index.search(vec, k)
        return [self.meta[i] for i in ids[0]]

    def build_prompt(self, question: str, contexts: list) -> str:
        formatted = "\n\n".join(
            f"[{i+1}] (from {c['src']}):\n{c['text'][:600]}"
            for i, c in enumerate(contexts)
        )
        return (
            "You are a financial analyst. Answer using ONLY the context below.\n"
            "Cite sources as [1], [2] etc. Be concise and factual.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{formatted}\n\n"
            "Answer:"
        )

    @modal.method()
    def query_stream(self, question: str, k: int = 5, max_tokens: int = 400):
        """
        Generator that yields SSE-formatted strings.

        Events:
          data: {"choices": [{"delta": {"content": "..."}}]}   (passthrough vLLM chunks)
          data: {"type": "metrics", "ttft_ms": ..., ...}        (appended after stream)
          data: [DONE]
        """
        import requests as req
        import json

        contexts = self.retrieve(question, k=k)
        prompt   = self.build_prompt(question, contexts)

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        vllm_url     = "http://localhost:8000/v1/chat/completions"
        t0           = time.perf_counter()
        t_first_token = None
        n_tokens     = 0
        input_tokens = 0

        response = req.post(vllm_url, json=payload, stream=True)

        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            if not raw_line.startswith(b"data: "):
                continue
            data = raw_line[6:]
            if data == b"[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            # Capture usage from the final usage-only chunk vLLM appends
            # when stream_options.include_usage=True.
            usage = chunk.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", input_tokens)
                n_tokens     = usage.get("completion_tokens", n_tokens)

            choices = chunk.get("choices")
            if choices:
                content = choices[0].get("delta", {}).get("content", "")
                if content:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    if not usage:
                        # Manual count as fallback when usage chunk hasn't arrived
                        n_tokens += 1
                yield f"data: {data.decode()}\n\n"

        t_end      = time.perf_counter()
        total_time = t_end - t0

        ttft_ms = (
            round((t_first_token - t0) * 1000, 1)
            if t_first_token is not None
            else round(total_time * 1000, 1)
        )
        decode_time    = max(total_time - ttft_ms / 1000, 0.001)
        tpot_ms        = round(decode_time / max(n_tokens - 1, 1) * 1000, 1)
        throughput_tps = round(n_tokens / total_time, 1) if total_time > 0 else 0.0

        metrics = {
            "type":           "metrics",
            "ttft_ms":        ttft_ms,
            "tpot_ms":        tpot_ms,
            "tokens":         n_tokens,
            "input_tokens":   input_tokens,
            "throughput_tps": throughput_tps,
            "contexts":       [{"src": c["src"], "text": c["text"][:200]} for c in contexts],
        }
        yield f"data: {json.dumps(metrics)}\n\n"
        yield "data: [DONE]\n\n"


# ── STEP 3: ASGI streaming proxy (one per model) ────────────
#
# Two lightweight CPU endpoints — each proxies SSE from VLLMServer.
# CORS middleware is required for browser fetch() from the React dev server.
def _make_streaming_app(model_name: str):
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @web_app.post("/v1/stream")
    async def stream_endpoint(item: dict):
        question   = item.get("question", "")
        k          = int(item.get("k", 5))
        max_tokens = int(item.get("max_tokens", 400))

        server = VLLMServer(model_name=model_name)
        return StreamingResponse(
            server.query_stream.remote_gen(question, k, max_tokens),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control":     "no-cache",
            },
        )

    return web_app


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
)
@modal.asgi_app(label="finsight-llama-stream")
def llama_stream():
    return _make_streaming_app("llama")


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
)
@modal.asgi_app(label="finsight-mistral-stream")
def mistral_stream():
    return _make_streaming_app("mistral")


# ── STEP 2 (library mode): RAGEngine — CLI benchmark only ───
#
# Kept intact so `modal run finsight.py` / the local_entrypoint benchmark
# workflow continues to work.  Do NOT use this for the React frontend.
@app.cls(
    gpu="A10G",
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=120,
)
class RAGEngine:
    model_name: str = modal.parameter(default="llama")

    @modal.enter()
    def load(self):
        import faiss
        import numpy as np
        from vllm import LLM
        from sentence_transformers import SentenceTransformer

        volume.reload()

        model_id = MODELS[self.model_name]
        print(f"Loading {model_id}...")

        self.llm = LLM(
            model=model_id,
            dtype="half",
            max_model_len=2048,
            gpu_memory_utilization=0.88,
        )
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.index = faiss.read_index(str(INDEX_PATH))
        self.meta  = np.load(str(META_PATH), allow_pickle=True).tolist()
        print(f"Ready! Index has {self.index.ntotal} chunks")

    def retrieve(self, question, k=5):
        vec = self.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        _, ids = self.index.search(vec, k)
        return [self.meta[i] for i in ids[0]]

    def build_prompt(self, question, contexts):
        formatted = "\n\n".join(
            f"[{i+1}] (from {c['src']}):\n{c['text'][:600]}"
            for i, c in enumerate(contexts)
        )
        return f"""You are a financial analyst. Answer using ONLY the context below.
Cite sources as [1], [2] etc. Be concise and factual.

Question: {question}

Context:
{formatted}

Answer:"""

    @modal.method()
    def query(self, question: str, k: int = 5, max_tokens: int = 400):
        from vllm import SamplingParams

        contexts = self.retrieve(question, k=k)
        prompt   = self.build_prompt(question, contexts)
        params   = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        t0      = time.perf_counter()
        outputs = self.llm.generate([prompt], params)
        t_end   = time.perf_counter()

        output       = outputs[0]
        n_tokens     = len(output.outputs[0].token_ids)
        input_tokens = len(output.prompt_token_ids)
        total_time   = t_end - t0

        # TTFT estimated (library mode — not real wall-clock)
        prefill_tps = input_tokens / total_time * 3
        ttft        = input_tokens / prefill_tps
        tpot        = (total_time - ttft) / max(n_tokens - 1, 1)

        return {
            "answer":   output.outputs[0].text,
            "model":    self.model_name,
            "question": question,
            "metrics": {
                "ttft_ms":        round(ttft * 1000, 1),
                "tpot_ms":        round(tpot * 1000, 1),
                "total_time_s":   round(total_time, 2),
                "tokens":         n_tokens,
                "input_tokens":   input_tokens,
                "throughput_tps": round(n_tokens / total_time, 1),
            },
            "contexts": [{"src": c["src"], "text": c["text"][:200]} for c in contexts],
        }

    @modal.method()
    def benchmark(self, questions: list[str]):
        results = []
        for q in questions:
            r = self.query.local(q)
            results.append(r)
            print(f"Q: {q[:60]}")
            print(f"  TTFT: {r['metrics']['ttft_ms']}ms | "
                  f"TPOT: {r['metrics']['tpot_ms']}ms | "
                  f"Throughput: {r['metrics']['throughput_tps']} tok/s")

        ttfts = [r["metrics"]["ttft_ms"] for r in results]
        tpots = [r["metrics"]["tpot_ms"] for r in results]
        return {
            "model":               self.model_name,
            "n_questions":         len(results),
            "ttft_p50_ms":         round(sorted(ttfts)[len(ttfts)//2], 1),
            "ttft_p95_ms":         round(sorted(ttfts)[int(len(ttfts)*0.95)], 1),
            "tpot_p50_ms":         round(sorted(tpots)[len(tpots)//2], 1),
            "throughput_avg_tps":  round(
                sum(r["metrics"]["throughput_tps"] for r in results) / len(results), 1
            ),
            "results": results,
        }


# ── Local entrypoint — CLI benchmark (uses RAGEngine/library mode) ──
@app.local_entrypoint()
def main():
    import json
    from datetime import datetime

    questions = [
        "What are Apple's main supply chain risks?",
        "How does Microsoft describe its cloud revenue growth?",
        "What does Meta say about AI infrastructure investment?",
        "How has Google's advertising revenue changed over 3 years?",
        "What cybersecurity risks does Amazon disclose?",
    ]

    # Index is already built — keep commented out
    # print("Building index...")
    # build_index.remote()

    # Benchmark LLaMA
    print("\n=== LLaMA 3.1 8B ===")
    llama = RAGEngine(model_name="llama")
    llama_results = llama.benchmark.remote(questions)
    print(json.dumps({k: v for k, v in llama_results.items() if k != "results"}, indent=2))

    # Benchmark Mistral
    print("\n=== Mistral 7B ===")
    mistral = RAGEngine(model_name="mistral")
    mistral_results = mistral.benchmark.remote(questions)
    print(json.dumps({k: v for k, v in mistral_results.items() if k != "results"}, indent=2))

    # Comparison table
    print("\n=== COMPARISON ===")
    print(f"{'Metric':<25} {'LLaMA 3.1 8B':>15} {'Mistral 7B':>15}")
    print("-" * 55)
    for metric in ["ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "throughput_avg_tps"]:
        print(f"{metric:<25} {llama_results[metric]:>15} {mistral_results[metric]:>15}")

    # Save full results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "summary": {
            "llama":   {k: v for k, v in llama_results.items() if k != "results"},
            "mistral": {k: v for k, v in mistral_results.items() if k != "results"},
        },
        "llama_results":   llama_results["results"],
        "mistral_results": mistral_results["results"],
    }

    filename = f"benchmark_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to {filename}")
