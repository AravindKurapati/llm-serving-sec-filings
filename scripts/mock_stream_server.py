# Mock server for frontend development — run with: uvicorn scripts.mock_stream_server:app --port 8001

import asyncio
import json
import re

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SYSTEM_MODES = ["concise", "detailed"]
ALLOWED_QUESTION_HINT = (
    "Ask about SEC 10-K filings for Apple/AAPL, Microsoft/MSFT, Alphabet/Google/GOOGL, "
    "Amazon/AMZN, or Meta/META and filing topics such as revenue, risks, cloud, "
    "advertising, supply chain, cybersecurity, privacy, AI infrastructure, or workforce."
)

COMPANY_SCOPE_RE = re.compile(
    r"\b(aapl|apple|msft|microsoft|googl|google|alphabet|amzn|amazon|meta|facebook)\b",
    re.I,
)
FILING_SCOPE_RE = re.compile(
    r"\b(sec|10-k|10k|annual reports?|filings?|risk factors?|md&a|these companies|all companies|indexed companies)\b",
    re.I,
)
DISCLOSURE_TOPIC_RE = re.compile(
    r"\b(revenue|sales|income|profit|margin|cash flow|capex|capital expenditures?|r&d|"
    r"research and development|risk|risks|supply chain|cybersecurity|privacy|antitrust|"
    r"regulatory|regulation|litigation|workforce|employees|cloud|aws|azure|advertising|"
    r"ai|infrastructure|investment|segments?|services|costs?|competition|liquidity|debt|"
    r"operating|financial|disclos(?:e|es|ed|ure|ures)|growth)\b",
    re.I,
)
OFF_TOPIC_RE = re.compile(
    r"\b(joke|poem|recipe|weather|sports?|movie|song|lyrics|capital of|homework|write code|"
    r"generate code|jailbreak|ignore previous|system prompt)\b",
    re.I,
)


def validate_question_scope(question: str) -> tuple[bool, str]:
    text = " ".join(question.strip().split())
    if len(text) < 12 or len(text.split()) < 3:
        return False, f"Question is too short. {ALLOWED_QUESTION_HINT}"
    if OFF_TOPIC_RE.search(text):
        return False, f"Off-topic request rejected. {ALLOWED_QUESTION_HINT}"
    if (COMPANY_SCOPE_RE.search(text) or FILING_SCOPE_RE.search(text)) and DISCLOSURE_TOPIC_RE.search(text):
        return True, ""
    return False, f"Question is outside the FinSight filing scope. {ALLOWED_QUESTION_HINT}"

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
    status = {
        "type": "status",
        "stage": "mock_start",
        "message": "Mock stream connected. Preparing sample filing context.",
    }
    yield f"data: {json.dumps(status)}\n\n"

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
    allowed, reason = validate_question_scope(question)
    if not allowed:
        raise HTTPException(status_code=400, detail=reason)
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
