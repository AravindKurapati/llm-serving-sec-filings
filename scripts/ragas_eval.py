#!/usr/bin/env python3
"""
RAGAS Evaluation for FinSight SEC Filing RAG
=============================================
Evaluates LLaMA 3.1 8B and Mistral 7B on three RAGAS quality metrics:
  - faithfulness       : are answer claims grounded in the retrieved context?
  - answer_relevancy   : does the answer address the question?
  - context_precision  : do the most relevant chunks rank highest in retrieval?

MODEL PROXY CAVEAT
------------------
The FinSight vLLM deployment on Modal runs:
  - meta-llama/Meta-Llama-3.1-8B-Instruct  ("llama")
  - mistralai/Mistral-7B-Instruct-v0.3      ("mistral")

This evaluation uses Groq API proxies (no Modal GPU required):
  - LLaMA  proxy : groq/llama-3.1-8b-instant   (same base model, different serving)
  - Mistral proxy: groq/llama-3.1-8b-instant    (DIFFERENT MODEL: mixtral decommissioned;
                                                  llama3-8b-8192 subsequently also retired)

Scores are indicative of model family quality, not bit-for-bit equivalence
with the vLLM-served versions.

PREREQUISITES
-------------
1. Download FAISS index from Modal Volume (one-time, ~10 MB):

   mkdir -p data/index
   modal volume get finsight-data /data/chunks.faiss data/index/chunks.faiss
   modal volume get finsight-data /data/meta.npy    data/index/meta.npy

2. Set GROQ_API_KEY in your .env file or environment:

   GROQ_API_KEY=gsk_...

3. Install dependencies:

   pip install -r requirements.txt

RUN
---
   python scripts/ragas_eval.py                # uses data/testset.json (default)
   python scripts/ragas_eval.py --generate     # synthetic testset via TestsetGenerator

The bundled testset (data/testset.json) contains 10 hand-written Q+GT pairs
covering revenue, operating income, R&D spend, risk factors, and segment
performance for all five companies and is used by default.

Pass --generate to fall back to the TestsetGenerator pipeline.  The synthetic
testset is cached to results/ragas_testset.json after the first run; delete
that file to force regeneration.

Expected runtime: 12-18 minutes (Groq free-tier rate limits).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Groq / LangChain imports ──────────────────────────────────────────────────
try:
    from groq import RateLimitError as GroqRateLimitError
except ImportError:
    GroqRateLimitError = Exception  # fallback for older SDK versions

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── RAGAS v0.2 imports ────────────────────────────────────────────────────────
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Constants
# ─────────────────────────────────────────────────────────────────────────────

ROOT               = Path(__file__).parent.parent
INDEX_DIR          = ROOT / "data" / "index"
INDEX_PATH         = INDEX_DIR / "chunks.faiss"   # mirrors finsight.py:26
META_PATH          = INDEX_DIR / "meta.npy"       # mirrors finsight.py:27
HARDCODED_TESTSET  = ROOT / "data" / "testset.json"
RESULTS_DIR        = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Groq model IDs
GROQ_LLAMA_MODEL   = "llama-3.1-8b-instant"                    # proxy for LLaMA 3.1 8B
GROQ_MISTRAL_MODEL = "llama-3.1-8b-instant"                    # proxy for Mistral 7B (mixtral decommissioned; llama3-8b-8192 also retired)
GROQ_GENERATOR_LLM = "llama-3.3-70b-versatile"                 # TestsetGenerator question writer
GROQ_CRITIC_LLM    = "llama-3.3-70b-versatile"                 # TestsetGenerator critic
GROQ_EVALUATOR_LLM = "meta-llama/llama-4-scout-17b-16e-instruct"  # RAGAS judge — 30K TPM, 500K TPD

# Evaluation parameters — all match finsight.py defaults
TESTSET_SIZE        = 15
TOP_K               = 5
CORPUS_SAMPLE_SIZE  = 30
MAX_ANSWER_TOKENS   = 400
EMBED_MODEL         = "BAAI/bge-small-en-v1.5"

# Groq rate-limit settings
# llama-4-scout: 30K TPM, 500K TPD — plenty of headroom for RAGAS parallel metric calls
INTER_REQUEST_DELAY_S      = 3.0
RETRY_MAX_ATTEMPTS         = 5
RETRY_BASE_DELAY_S         = 10.0
RETRY_BACKOFF_FACTOR       = 2.0
RAGAS_INTER_SAMPLE_DELAY_S = 60       # sleep between judge calls to spread TPM usage
MIN_TPD_REMAINING          = 50_000   # abort if fewer than this many daily tokens remain
                                      # Groq TPD resets at midnight UTC (7 PM EST / 8 PM EDT)

# Truncation limit for contexts passed to the RAGAS judge.
# RAGAS builds a large prompt per metric call (question + contexts + answer +
# instructions). Truncating each retrieved chunk to 200 chars keeps the total
# judge prompt well within the TPM limit even with parallel metric calls.
JUDGE_CONTEXT_CHAR_LIMIT = 200

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Groq LLM factory + retry wrapper
# ─────────────────────────────────────────────────────────────────────────────

def make_groq_llm(model_name: str, temperature: float = 0.0, max_tokens: int = 512,
                  bypass_n: bool = False) -> LangchainLLMWrapper:
    """Create a RAGAS-compatible LangChain-wrapped Groq LLM with built-in retry.

    bypass_n=True: LangchainLLMWrapper will NOT set langchain_llm.n = n before
    calling agenerate_prompt.  Instead it fans out to n separate single-completion
    calls.  Required for Groq models that reject n>1.
    """
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=5,
        request_timeout=60,
    )
    return LangchainLLMWrapper(llm, bypass_n=bypass_n)


def groq_call_with_retry(fn, *args, **kwargs):
    """
    Wrap a Groq API call with exponential backoff on rate-limit errors.
    Also adds a proactive inter-request delay after every successful call
    to stay comfortably under free-tier RPM limits.
    """
    delay = RETRY_BASE_DELAY_S
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            result = fn(*args, **kwargs)
            time.sleep(INTER_REQUEST_DELAY_S)   # proactive throttle
            return result
        except GroqRateLimitError as exc:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
            wait = delay * (RETRY_BACKOFF_FACTOR ** (attempt - 1))
            print(f"  [rate-limit] attempt {attempt}/{RETRY_MAX_ATTEMPTS}, "
                  f"sleeping {wait:.0f}s ... ({exc})")
            time.sleep(wait)
        except Exception as exc:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
            print(f"  [error] attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {exc}")
            time.sleep(delay)


def preflight_tpd_check():
    """
    Make a minimal 1-token call to the RAGAS judge model and read the
    x-ratelimit-remaining-tokens-day response header. Exits with a clear
    error if the remaining daily token budget is below MIN_TPD_REMAINING.
    """
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    raw = client.chat.completions.with_raw_response.create(
        model=GROQ_EVALUATOR_LLM,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )
    headers   = raw.headers
    remaining = headers.get("x-ratelimit-remaining-tokens-day")
    limit     = headers.get("x-ratelimit-limit-tokens-day")

    remaining_int = int(remaining) if remaining else None
    limit_int     = int(limit) if limit else None

    if remaining_int is not None and limit_int is not None:
        print(f"[preflight] TPD remaining: {remaining_int:,} / {limit_int:,}")
    else:
        print(f"[preflight] TPD headers not returned — skipping budget check")
        print(f"  raw headers: remaining={remaining!r}, limit={limit!r}")
        return

    if remaining_int < MIN_TPD_REMAINING:
        sys.exit(
            f"[ERROR] Only {remaining_int:,} daily tokens remaining "
            f"(minimum {MIN_TPD_REMAINING:,} required).\n"
            f"Groq TPD resets at midnight UTC (7 PM EST / 8 PM EDT).\n"
            f"Wait for the reset before running."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: FAISS index loading + local retrieval
# ─────────────────────────────────────────────────────────────────────────────

def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        sys.exit(
            f"\n[ERROR] Index files not found. Download them first:\n\n"
            f"  mkdir -p {INDEX_DIR}\n"
            f"  modal volume get finsight-data /data/chunks.faiss {INDEX_PATH}\n"
            f"  modal volume get finsight-data /data/meta.npy    {META_PATH}\n"
        )
    index = faiss.read_index(str(INDEX_PATH))
    meta  = np.load(str(META_PATH), allow_pickle=True).tolist()
    print(f"[ok] Index: {index.ntotal} chunks, dim={index.d}")
    print(f"[ok] Metadata: {len(meta)} entries, keys={list(meta[0].keys())}")
    return index, meta


def load_embedder() -> SentenceTransformer:
    print(f"[ok] Loading embedder: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def retrieve(
    question: str,
    index,
    meta: list,
    embedder: SentenceTransformer,
    k: int = TOP_K,
) -> list[dict]:
    """Exact replica of RAGEngine.retrieve() from finsight.py."""
    vec = embedder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    _, ids = index.search(vec, k)
    return [meta[i] for i in ids[0]]


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Corpus preparation for TestsetGenerator
# ─────────────────────────────────────────────────────────────────────────────

def build_langchain_docs(
    meta: list[dict],
    sample_size: int = CORPUS_SAMPLE_SIZE,
) -> list[Document]:
    def _is_prose(chunk: dict) -> bool:
        t = chunk.get("text", "")
        return not (t.startswith("<?xml") or "<" in t or "auth_ref" in t or "xbrl" in t.lower())

    meta = [c for c in meta if _is_prose(c)]

    by_company: dict[str, list[dict]] = {}
    for chunk in meta:
        src     = chunk.get("src", chunk.get("source", "unknown"))
        company = chunk.get("company", src.split("_")[0] if src != "unknown" else "unknown")
        by_company.setdefault(company, []).append(chunk)

    companies   = list(by_company.keys())
    per_company = max(1, sample_size // len(companies))

    sampled = []
    for company, chunks in by_company.items():
        by_src: dict[str, list[dict]] = {}
        for c in chunks:
            src = c.get("src", c.get("source", "unknown"))
            by_src.setdefault(src, []).append(c)
        per_src = max(1, per_company // len(by_src))
        for src_chunks in by_src.values():
            sampled.extend(random.sample(src_chunks, min(per_src, len(src_chunks))))

    random.shuffle(sampled)
    sampled = sampled[:sample_size]

    docs = []
    for chunk in sampled:
        src     = chunk.get("src", chunk.get("source", "unknown"))
        company = chunk.get("company", src.split("_")[0])
        docs.append(Document(
            page_content=chunk["text"][:500],
            metadata={"source": src, "company": company, "doc_id": chunk.get("doc_id", "")},
        ))

    print(f"[ok] Prepared {len(docs)} corpus documents for TestsetGenerator "
          f"({len(companies)} companies)")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: TestsetGenerator
# ─────────────────────────────────────────────────────────────────────────────

def generate_testset(docs: list[Document]) -> list[dict]:
    print(f"\n[step 1] Generating testset ({TESTSET_SIZE} Q+GT pairs) ...")
    print("  This may take 5-10 minutes due to Groq rate limits.")

    ragas_embedder = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    generator = TestsetGenerator(
        llm=make_groq_llm(GROQ_GENERATOR_LLM, temperature=0.4),
        embedding_model=ragas_embedder,
    )

    if hasattr(generator, "generate_with_langchain_docs"):
        testset = generator.generate_with_langchain_docs(docs, testset_size=TESTSET_SIZE)
    else:
        testset = generator.generate(docs, testset_size=TESTSET_SIZE)

    pairs = []
    for sample in testset.samples:
        inner        = getattr(sample, "eval_sample", sample)
        question     = getattr(inner, "user_input",    getattr(inner, "question",     ""))
        ground_truth = getattr(inner, "reference",     getattr(inner, "ground_truth", ""))
        if not question:
            continue
        pairs.append({"question": question, "ground_truth": ground_truth})

    if len(pairs) < 5:
        print(f"  [warn] TestsetGenerator only produced {len(pairs)} pairs "
              f"(requested {TESTSET_SIZE}). Proceeding with what we have.")
    else:
        print(f"[ok] Generated {len(pairs)} Q+GT pairs")

    cache_path = RESULTS_DIR / "ragas_testset.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    print(f"[ok] Testset cached to {cache_path}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Answer generation via Groq
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(question: str, contexts: list[dict]) -> str:
    """Exact replica of RAGEngine.build_prompt() from finsight.py."""
    formatted = "\n\n".join(
        f"[{i+1}] (from {c.get('src', c.get('source', 'unknown'))}):\n{c['text'][:600]}"
        for i, c in enumerate(contexts)
    )
    return (
        f"You are a financial analyst. Answer using ONLY the context below.\n"
        f"Cite sources as [1], [2] etc. Be concise and factual.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{formatted}\n\n"
        f"Answer:"
    )


def get_answer_via_groq(question: str, contexts: list[dict], groq_model: str) -> str:
    from groq import Groq
    client   = Groq(api_key=os.environ["GROQ_API_KEY"])
    user_msg = build_prompt(question, contexts)

    def _call():
        resp = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=MAX_ANSWER_TOKENS,
        )
        return resp.choices[0].message.content.strip()

    return groq_call_with_retry(_call)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Build EvaluationDataset
# ─────────────────────────────────────────────────────────────────────────────

def build_evaluation_dataset(
    pairs:       list[dict],
    index,
    meta:        list[dict],
    embedder:    SentenceTransformer,
    model_label: str,
    groq_model:  str,
) -> EvaluationDataset:
    """
    For each Q+GT pair:
      1. Retrieve top-5 chunks via local FAISS
      2. Generate answer via Groq (full 600-char chunks)
      3. Pack into SingleTurnSample with truncated contexts for the judge
         (200 chars/chunk keeps total judge prompt within TPM limits)
    """
    print(f"\n[step 2] Generating answers for: {model_label}")
    samples = []

    for i, pair in enumerate(pairs):
        question     = pair["question"]
        ground_truth = pair["ground_truth"]

        print(f"  [{i+1:02d}/{len(pairs)}] {question[:70]}...")

        contexts      = retrieve(question, index, meta, embedder)
        context_texts = [c["text"][:JUDGE_CONTEXT_CHAR_LIMIT] for c in contexts]
        answer        = get_answer_via_groq(question, contexts, groq_model)

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=context_texts,
            response=answer,
            reference=ground_truth,
        ))

    return EvaluationDataset(samples=samples)


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas_evaluation(dataset: EvaluationDataset, model_label: str) -> dict:
    """
    Run RAGAS evaluate() one sample at a time with a deliberate delay between
    each to spread TPM usage. Per-sample scores are averaged at the end.
    NaN results are excluded from the mean.
    """
    print(f"\n[step 3] RAGAS evaluation: {model_label}")
    print(f"  metrics : faithfulness, answer_relevancy, context_precision")
    print(f"  judge   : {GROQ_EVALUATOR_LLM} (Groq, 30K TPM)")
    print(f"  mode    : sequential, {RAGAS_INTER_SAMPLE_DELAY_S}s delay between samples")
    print(f"  context : {JUDGE_CONTEXT_CHAR_LIMIT} chars/chunk (truncated for judge prompt)")

    evaluator_llm = make_groq_llm(GROQ_EVALUATOR_LLM, temperature=0.0, bypass_n=True)
    evaluator_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    run_cfg = RunConfig(max_workers=1)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision"]
    accumulated: dict[str, list[float]] = {m: [] for m in metric_names}

    for i, sample in enumerate(dataset.samples):
        print(f"  [sample {i+1:02d}/{len(dataset.samples)}] scoring ...")
        single = EvaluationDataset(samples=[sample])
        result = evaluate(
            dataset=single,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=evaluator_llm,
            embeddings=evaluator_emb,
            run_config=run_cfg,
            show_progress=False,
        )
        for m in metric_names:
            try:
                val = float(result[m])
                if not math.isnan(val):
                    accumulated[m].append(val)
            except (TypeError, ValueError):
                pass

        if i < len(dataset.samples) - 1:
            print(f"  [throttle] sleeping {RAGAS_INTER_SAMPLE_DELAY_S}s ...")
            time.sleep(RAGAS_INTER_SAMPLE_DELAY_S)

    scores = {}
    for m in metric_names:
        vals = accumulated[m]
        if vals:
            scores[m] = round(sum(vals) / len(vals), 4)
        else:
            print(f"  [warn] {m} returned NaN for all samples — setting to null")
            scores[m] = None

    print(f"  scores: {scores}")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Report writers
# ─────────────────────────────────────────────────────────────────────────────

def save_results_json(all_results: dict, testset: list[dict]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp":    timestamp,
        "testset_size": len(testset),
        "retrieval": {
            "index":       "FAISS IndexFlatIP",
            "k":           TOP_K,
            "embed_model": EMBED_MODEL,
        },
        "models":  all_results,
        "testset": testset,
    }
    path = RESULTS_DIR / f"ragas_eval_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[ok] JSON saved to {path}")
    return path


def write_markdown_report(all_results: dict, testset: list[dict]) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    metrics   = ["faithfulness", "answer_relevancy", "context_precision"]

    def fmt(val) -> str:
        return f"{val:.4f}" if isinstance(val, float) else "N/A"

    llama_scores   = all_results.get("llama",   {}).get("scores", {})
    mistral_scores = all_results.get("mistral", {}).get("scores", {})

    lines = [
        "# RAGAS Evaluation: FinSight SEC Filing RAG\n\n",
        f"**Date**: {timestamp}  \n",
        f"**Testset**: {len(testset)} Q+GT pairs  \n",
        f"**Corpus**: AAPL, MSFT, GOOGL, AMZN, META — 10-K filings (3 years each)  \n",
        f"**Retrieval**: FAISS IndexFlatIP · BGE-small-en-v1.5 · top-{TOP_K} chunks  \n",
        f"**Judge LLM**: Groq `{GROQ_EVALUATOR_LLM}`  \n\n",
        "---\n\n",
        "## Model Proxy Note\n\n",
        "> The FinSight Modal deployment runs:\n",
        "> - `meta-llama/Meta-Llama-3.1-8B-Instruct` (LLaMA)\n",
        "> - `mistralai/Mistral-7B-Instruct-v0.3` (Mistral)\n>\n",
        "> This evaluation uses Groq API proxies to avoid Modal GPU costs:\n",
        "> - **LLaMA proxy** : `llama-3.1-8b-instant` — same base model, different serving stack\n",
        "> - **Mistral proxy**: `llama-3.1-8b-instant` — mixtral decommissioned; llama3-8b-8192 also retired\n>\n",
        "> Scores reflect model-family quality on this task.\n\n",
        "---\n\n",
        "## Results Summary\n\n",
        "| Metric | LLaMA 3.1 8B | Mistral proxy |\n",
        "|--------|:------------:|:-------------:|\n",
    ]
    for m in metrics:
        lines.append(f"| {m} | {fmt(llama_scores.get(m))} | {fmt(mistral_scores.get(m))} |\n")

    lines += [
        "\n### Metric Definitions\n\n",
        "- **Faithfulness**: fraction of answer claims entailed by the retrieved context\n",
        "- **Answer Relevancy**: semantic alignment between the question and the answer\n",
        "- **Context Precision**: whether the most relevant chunks are ranked highest by FAISS\n\n",
        "---\n\n",
        "## Evaluation Setup\n\n",
        f"| Parameter | Value |\n|-----------|-------|\n",
        f"| Judge LLM | `{GROQ_EVALUATOR_LLM}` (30K TPM, 500K TPD) |\n",
        f"| Answer max tokens | {MAX_ANSWER_TOKENS} |\n",
        f"| Temperature | 0.0 (deterministic) |\n",
        f"| Retrieval k | {TOP_K} |\n",
        f"| Judge context truncation | {JUDGE_CONTEXT_CHAR_LIMIT} chars/chunk |\n\n",
        "---\n\n",
        "## Limitations\n\n",
        "- Groq proxy models differ from deployed vLLM models (see proxy note above)\n",
        f"- Only {len(testset)} Q+GT pairs — increase `TESTSET_SIZE` for statistical robustness\n",
        "- Testset is hand-written; real user queries may differ in distribution\n",
        f"- Judge contexts truncated to {JUDGE_CONTEXT_CHAR_LIMIT} chars/chunk to fit within TPM limits\n",
        "- `context_precision` measures retrieval rank quality, not recall coverage\n",
    ]

    report_path = RESULTS_DIR / "ragas_eval.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[ok] Report saved to {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FinSight RAGAS Evaluation")
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate a synthetic testset via TestsetGenerator "
             "(default: load the bundled data/testset.json)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("FinSight RAGAS Evaluation")
    print("=" * 65)

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("[ERROR] GROQ_API_KEY is not set.\nAdd it to your .env file: GROQ_API_KEY=gsk_...\n")

    preflight_tpd_check()

    index, meta = load_index()
    embedder    = load_embedder()

    if args.generate:
        docs       = build_langchain_docs(meta)
        cache_path = RESULTS_DIR / "ragas_testset.json"
        if cache_path.exists():
            print(f"\n[cache] Loading testset from {cache_path}")
            with open(cache_path, encoding="utf-8") as f:
                testset = json.load(f)
            print(f"[ok] Loaded {len(testset)} Q+GT pairs")
        else:
            testset = generate_testset(docs)
    else:
        print(f"\n[hardcoded] Loading testset from {HARDCODED_TESTSET}")
        with open(HARDCODED_TESTSET, encoding="utf-8") as f:
            testset = json.load(f)
        print(f"[ok] Loaded {len(testset)} Q+GT pairs")

    if len(testset) < 5:
        sys.exit(f"[ERROR] Too few Q+GT pairs ({len(testset)}). Check testset file.")

    models_config = [
        {"key": "llama",   "label": "LLaMA 3.1 8B  (Groq: llama-3.1-8b-instant)", "groq_model": GROQ_LLAMA_MODEL},
        {"key": "mistral", "label": "Mistral proxy  (Groq: llama-3.1-8b-instant)", "groq_model": GROQ_MISTRAL_MODEL},
    ]

    all_results: dict = {}
    for cfg in models_config:
        dataset = build_evaluation_dataset(
            pairs=testset, index=index, meta=meta, embedder=embedder,
            model_label=cfg["label"], groq_model=cfg["groq_model"],
        )
        scores = run_ragas_evaluation(dataset, model_label=cfg["label"])
        all_results[cfg["key"]] = {
            "label": cfg["label"], "groq_model": cfg["groq_model"], "scores": scores,
        }

    save_results_json(all_results, testset)
    write_markdown_report(all_results, testset)

    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    print("\n" + "=" * 65)
    print("RAGAS Evaluation Complete")
    print("=" * 65)
    print(f"{'Metric':<25} {'LLaMA 3.1 8B':>15} {'Mistral proxy':>15}")
    print("-" * 55)
    for m in metrics:
        l_val  = all_results["llama"]["scores"].get(m)
        mi_val = all_results["mistral"]["scores"].get(m)
        l_str  = f"{l_val:.4f}"  if isinstance(l_val,  float) else "N/A"
        mi_str = f"{mi_val:.4f}" if isinstance(mi_val, float) else "N/A"
        print(f"{m:<25} {l_str:>15} {mi_str:>15}")
    print()
    print(f"Full report : results/ragas_eval.md")
    print(f"Raw JSON    : results/ragas_eval_<timestamp>.json")


if __name__ == "__main__":
    main()
