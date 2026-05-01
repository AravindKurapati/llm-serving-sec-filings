# Mock server for frontend development — run with: uvicorn scripts.mock_stream_server:app --port 8001

import asyncio
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SYSTEM_MODES = ["concise", "detailed"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

FAKE_TOKENS = [
    "Apple", "'s", " primary", " supply", " chain", " risks", " include",
    " over", "-rel", "iance", " on", " single", "-source", " suppliers",
    ",", " geo", "pol", "itical", " tensions", ",", " and", " logistical",
    " disruptions", " that", " could", " affect", " product", " availability",
    ".", " [", "1", "]",
]


def status_payload() -> dict:
    return {
        "status": "ok",
        "service": "finsight-mock-stream",
        "model": "mock",
        "model_id": "mock",
        "embed_model": EMBED_MODEL,
        "stream_path": "/v1/stream",
        "modes": SYSTEM_MODES,
        "index_present": True,
        "metadata_present": True,
    }


async def _token_stream(question: str, k: int, max_tokens: int):
    tokens_to_send = FAKE_TOKENS[:max_tokens]

    for i, token in enumerate(tokens_to_send):
        chunk = {"choices": [{"delta": {"content": token}}]}
        yield f"data: {json.dumps(chunk)}\n\n"
        # 50ms between tokens to simulate real streaming; first token is
        # slightly delayed to give a realistic TTFT
        await asyncio.sleep(0.31 if i == 0 else 0.05)

    metrics = {
        "type": "metrics",
        "ttft_ms": 312.4,
        "tpot_ms": 23.1,
        "tokens": len(tokens_to_send),
        "input_tokens": 150,
        "throughput_tps": 28.6,
        "contexts": [
            {
                "rank": 1,
                "score": 0.8123,
                "doc_id": "AAPL_mock-0",
                "company": "AAPL",
                "src": "AAPL_10K.txt",
                "text": "fake context chunk",
            }
        ],
    }
    yield f"data: {json.dumps(metrics)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return status_payload()


@app.get("/v1/status")
async def status():
    return status_payload()


@app.post("/v1/stream")
async def stream_endpoint(item: dict):
    question   = item.get("question", "")
    k          = int(item.get("k", 5))
    max_tokens = int(item.get("max_tokens", 20))
    _mode      = item.get("mode", "concise")  # accepted, not used in mock

    return StreamingResponse(
        _token_stream(question, k, max_tokens),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
