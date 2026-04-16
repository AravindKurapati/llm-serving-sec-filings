#!/usr/bin/env python3
"""
LLM-Judge Evaluation for FinSight SEC Filing RAG
=================================================
Uses a Groq LLM judge to score model answers on three custom metrics:
  - groundedness    : are all claims supported by the retrieved context?
  - conciseness     : is the answer focused without unnecessary padding?
  - citation_quality: does the answer cite sources with [1], [2] notation?

Runs over the 5 qualitative questions in data/testset.json (indices 5-9) for
both LLaMA and Mistral proxies. Saves results to results/llm_judge.json.

RUN
---
  python scripts/llm_judge.py

COST
----
  10 answer calls + 10 judge calls ≈ 30-50K TPD tokens (within 100K free tier).
"""

import html
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq, RateLimitError as GroqRateLimitError
from sentence_transformers import SentenceTransformer

load_dotenv()

ROOT            = Path(__file__).parent.parent
INDEX_DIR       = ROOT / "data" / "index"
INDEX_PATH      = INDEX_DIR / "chunks.faiss"
META_PATH       = INDEX_DIR / "meta.npy"
TESTSET_PATH    = ROOT / "data" / "testset.json"
RESULTS_DIR     = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GROQ_ANSWER_LLAMA   = "llama-3.1-8b-instant"
GROQ_ANSWER_MISTRAL = "llama-3.1-8b-instant"   # Mixtral decommissioned
GROQ_JUDGE_MODEL    = "llama-3.3-70b-versatile"

EMBED_MODEL         = "BAAI/bge-small-en-v1.5"
TOP_K               = 5
MAX_ANSWER_TOKENS   = 400
MAX_JUDGE_TOKENS    = 256
INTER_REQUEST_DELAY = 3.0   # seconds between Groq calls
RETRY_ATTEMPTS      = 4
RETRY_BASE_DELAY    = 10.0
RETRY_BACKOFF       = 2.0

# Qualitative questions are the last 5 entries in the testset
QUALITATIVE_SLICE = slice(5, 10)

JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.

Given the question, retrieved context, and model answer below, score the answer on \
three dimensions. Each score is a float from 0.0 to 1.0.

Definitions:
- groundedness    : fraction of answer claims that are directly supported by the \
provided context (1.0 = fully grounded, 0.0 = entirely hallucinated)
- conciseness     : how focused and non-padded the answer is (1.0 = perfectly concise, \
0.0 = very verbose or off-topic)
- citation_quality: how well the answer cites sources using [1], [2] etc. notation \
(1.0 = every claim cited, 0.0 = no citations at all)

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

MODEL ANSWER:
{answer}

Respond with ONLY valid JSON and nothing else:
{{"groundedness": <float 0-1>, "conciseness": <float 0-1>, "citation_quality": <float 0-1>}}"""


def retry_groq(fn, *args, **kwargs):
    delay = RETRY_BASE_DELAY
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = fn(*args, **kwargs)
            time.sleep(INTER_REQUEST_DELAY)
            return result
        except GroqRateLimitError as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            wait = delay * (RETRY_BACKOFF ** (attempt - 1))
            print(f"  [rate-limit] attempt {attempt}/{RETRY_ATTEMPTS}, sleeping {wait:.0f}s ({exc})")
            time.sleep(wait)
        except Exception as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"  [error] attempt {attempt}/{RETRY_ATTEMPTS}: {exc}")
            time.sleep(delay)


def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        sys.exit(
            f"[ERROR] Index files not found.\n"
            f"  modal volume get finsight-data /data/chunks.faiss {INDEX_PATH}\n"
            f"  modal volume get finsight-data /data/meta.npy    {META_PATH}\n"
        )
    index = faiss.read_index(str(INDEX_PATH))
    meta  = np.load(str(META_PATH), allow_pickle=True).tolist()
    print(f"[ok] Index: {index.ntotal} chunks, dim={index.d}")
    return index, meta


def retrieve(question: str, index, meta: list, embedder: SentenceTransformer) -> list[dict]:
    vec = embedder.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, ids = index.search(vec, TOP_K)
    return [meta[i] for i in ids[0]]


def build_answer_prompt(question: str, chunks: list[dict]) -> str:
    formatted = "\n\n".join(
        f"[{i+1}] (from {c.get('src', c.get('source', 'unknown'))}):\n"
        f"{html.unescape(c['text'][:600])}"
        for i, c in enumerate(chunks)
    )
    return (
        f"You are a financial analyst. Answer using ONLY the context below.\n"
        f"Cite sources as [1], [2] etc. Be concise and factual.\n\n"
        f"Question: {question}\n\nContext:\n{formatted}\n\nAnswer:"
    )


def get_answer(client: Groq, question: str, chunks: list[dict], model: str) -> str:
    prompt = build_answer_prompt(question, chunks)

    def _call():
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=MAX_ANSWER_TOKENS,
        )
        return resp.choices[0].message.content.strip()

    return retry_groq(_call)


def judge_answer(client: Groq, question: str, chunks: list[dict], answer: str) -> dict:
    context_text = "\n\n".join(
        f"[{i+1}] {html.unescape(c['text'][:500])}"
        for i, c in enumerate(chunks)
    )
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context_text,
        answer=answer,
    )

    def _call():
        resp = client.chat.completions.create(
            model=GROQ_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=MAX_JUDGE_TOKENS,
        )
        return resp.choices[0].message.content.strip()

    raw = retry_groq(_call)

    # Extract JSON — model may wrap in markdown fences
    json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not json_match:
        print(f"  [warn] judge returned unparseable output: {raw[:200]}")
        return {"groundedness": None, "conciseness": None, "citation_quality": None}

    try:
        scores = json.loads(json_match.group())
        return {
            "groundedness":     round(float(scores.get("groundedness",     0)), 4),
            "conciseness":      round(float(scores.get("conciseness",      0)), 4),
            "citation_quality": round(float(scores.get("citation_quality", 0)), 4),
        }
    except (ValueError, KeyError) as exc:
        print(f"  [warn] failed to parse judge JSON: {exc} — raw: {raw[:200]}")
        return {"groundedness": None, "conciseness": None, "citation_quality": None}


def fmt(val) -> str:
    return f"{val:.4f}" if isinstance(val, float) else " N/A "


def main():
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("[ERROR] GROQ_API_KEY is not set. Add to .env: GROQ_API_KEY=gsk_...\n")

    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)

    questions = testset[QUALITATIVE_SLICE]
    print(f"[ok] Loaded {len(questions)} qualitative questions (indices 5-9)")

    index, meta = load_index()
    embedder    = SentenceTransformer(EMBED_MODEL)
    print(f"[ok] Embedder loaded: {EMBED_MODEL}")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    models = [
        {"key": "llama",   "label": "LLaMA 3.1 8B",   "model": GROQ_ANSWER_LLAMA},
        {"key": "mistral", "label": "Mistral proxy",   "model": GROQ_ANSWER_MISTRAL},
    ]

    all_results = {}

    for model_cfg in models:
        mkey   = model_cfg["key"]
        mlabel = model_cfg["label"]
        mmodel = model_cfg["model"]

        print(f"\n{'='*60}")
        print(f"Model: {mlabel}  ({mmodel})")
        print(f"{'='*60}")

        model_rows = []
        for i, entry in enumerate(questions):
            question = entry["question"]
            print(f"\n  [Q{i+1}/5] {question[:70]}...")

            chunks = retrieve(question, index, meta, embedder)

            print(f"    generating answer ...")
            answer = get_answer(client, question, chunks, mmodel)

            print(f"    judging ...")
            scores = judge_answer(client, question, chunks, answer)

            model_rows.append({
                "question":        question,
                "answer":          answer,
                "scores":          scores,
            })
            print(f"    scores: {scores}")

        all_results[mkey] = {"label": mlabel, "groq_model": mmodel, "rows": model_rows}

    # ── Summary table ──────────────────────────────────────────────────────────
    metrics = ["groundedness", "conciseness", "citation_quality"]
    print(f"\n{'='*70}")
    print("LLM-Judge Summary  (5 qualitative questions, avg over questions)")
    print(f"{'='*70}")
    header = f"{'Metric':<22} {'LLaMA 3.1 8B':>14} {'Mistral proxy':>14}"
    print(header)
    print("-" * len(header))

    aggregated = {}
    for mkey in ("llama", "mistral"):
        aggregated[mkey] = {}
        for m in metrics:
            vals = [
                r["scores"][m]
                for r in all_results[mkey]["rows"]
                if isinstance(r["scores"].get(m), float)
            ]
            aggregated[mkey][m] = round(sum(vals) / len(vals), 4) if vals else None

    for m in metrics:
        l_val  = aggregated["llama"].get(m)
        mi_val = aggregated["mistral"].get(m)
        print(f"{m:<22} {fmt(l_val):>14} {fmt(mi_val):>14}")

    # ── Per-question detail ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Per-question scores")
    print(f"{'='*70}")
    for i, entry in enumerate(questions):
        q_short = entry["question"][:60]
        print(f"\n  Q{i+1}: {q_short}...")
        for mkey in ("llama", "mistral"):
            row    = all_results[mkey]["rows"][i]
            label  = all_results[mkey]["label"]
            s      = row["scores"]
            g  = fmt(s.get("groundedness"))
            c  = fmt(s.get("conciseness"))
            cq = fmt(s.get("citation_quality"))
            print(f"    {label:<16}  grd={g}  conc={c}  cite={cq}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "timestamp":  datetime.now().strftime("%Y%m%d_%H%M%S"),
        "judge_model": GROQ_JUDGE_MODEL,
        "embed_model": EMBED_MODEL,
        "top_k":       TOP_K,
        "questions":   [e["question"] for e in questions],
        "aggregated":  aggregated,
        "detail":      {
            mkey: {
                "label":      all_results[mkey]["label"],
                "groq_model": all_results[mkey]["groq_model"],
                "rows":       all_results[mkey]["rows"],
            }
            for mkey in ("llama", "mistral")
        },
    }
    out_path = RESULTS_DIR / "llm_judge.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[ok] Results saved to {out_path}")


if __name__ == "__main__":
    main()
