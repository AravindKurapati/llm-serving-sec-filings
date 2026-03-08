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
  - Mistral proxy: groq/mixtral-8x7b-32768      (DIFFERENT: 8x7B MoE, not 7B dense)

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
   python scripts/ragas_eval.py

The testset is cached to results/ragas_testset.json after the first run.
Delete that file to force regeneration.

Expected runtime: 12-18 minutes (Groq free-tier rate limits).
"""

import json
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
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Constants
# ─────────────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
INDEX_DIR   = ROOT / "data" / "index"
INDEX_PATH  = INDEX_DIR / "chunks.faiss"   # mirrors finsight.py:26
META_PATH   = INDEX_DIR / "meta.npy"       # mirrors finsight.py:27
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Groq model IDs
GROQ_LLAMA_MODEL   = "llama-3.1-8b-instant"       # proxy for LLaMA 3.1 8B
GROQ_MISTRAL_MODEL = "mixtral-8x7b-32768"          # proxy for Mistral 7B (NOTE: MoE)
GROQ_GENERATOR_LLM = "llama-3.1-8b-instant"        # TestsetGenerator question writer
GROQ_CRITIC_LLM    = "llama-3.3-70b-versatile"     # TestsetGenerator critic
GROQ_EVALUATOR_LLM = "llama-3.3-70b-versatile"     # RAGAS judge

# Evaluation parameters — all match finsight.py defaults
TESTSET_SIZE        = 15
TOP_K               = 5
CORPUS_SAMPLE_SIZE  = 80
MAX_ANSWER_TOKENS   = 400
EMBED_MODEL         = "BAAI/bge-small-en-v1.5"

# Groq rate-limit settings (free tier: 30 RPM, 6k TPM)
INTER_REQUEST_DELAY_S  = 3.0
RETRY_MAX_ATTEMPTS     = 5
RETRY_BASE_DELAY_S     = 10.0
RETRY_BACKOFF_FACTOR   = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Groq LLM factory + retry wrapper
# ─────────────────────────────────────────────────────────────────────────────

def make_groq_llm(model_name: str, temperature: float = 0.0) -> LangchainLLMWrapper:
    """Create a RAGAS-compatible LangChain-wrapped Groq LLM with built-in retry."""
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_retries=5,
        request_timeout=60,
    )
    return LangchainLLMWrapper(llm)


def groq_call_with_retry(fn, *args, **kwargs):
    """
    Wrap a Groq API call with exponential backoff on rate-limit errors.
    Also adds a proactive inter-request delay after every successful call
    to stay comfortably under the 30 RPM / 6k TPM free-tier limits.
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
            # Catch transient network errors on the last few attempts
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
            print(f"  [error] attempt {attempt}/{RETRY_MAX_ATTEMPTS}: {exc}")
            time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: FAISS index loading + local retrieval
# ─────────────────────────────────────────────────────────────────────────────

def load_index():
    """Load local FAISS index and metadata numpy array. Fails fast if missing."""
    if not INDEX_PATH.exists() or not META_PATH.exists():
        sys.exit(
            f"\n[ERROR] Index files not found. Download them first:\n\n"
            f"  mkdir -p {INDEX_DIR}\n"
            f"  modal volume get finsight-data /data/chunks.faiss {INDEX_PATH}\n"
            f"  modal volume get finsight-data /data/meta.npy    {META_PATH}\n"
        )
    index = faiss.read_index(str(INDEX_PATH))
    # allow_pickle=True + .tolist() mirrors finsight.py:150 exactly
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
    """
    Exact replica of RAGEngine.retrieve() from finsight.py:153-158.
    Uses IndexFlatIP — normalize_embeddings=True is critical to match
    the inner-product / cosine-similarity space used during indexing.
    """
    vec = embedder.encode(
        [question],
        normalize_embeddings=True,   # MUST match finsight.py:104
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
    """
    Convert a diverse sample of metadata chunks to LangChain Documents
    for the RAGAS TestsetGenerator.

    Samples `sample_size` chunks with diversity across companies and filing
    years so the generated questions span the full corpus, not just one company.
    """
    # Group by company (handles both finsight.py "src" and build_rag_index.py "source")
    by_company: dict[str, list[dict]] = {}
    for chunk in meta:
        src     = chunk.get("src", chunk.get("source", "unknown"))
        company = chunk.get("company", src.split("_")[0] if src != "unknown" else "unknown")
        by_company.setdefault(company, []).append(chunk)

    companies   = list(by_company.keys())
    per_company = max(1, sample_size // len(companies))

    sampled = []
    for company, chunks in by_company.items():
        # Further diversify: spread across different filing years (src files)
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
            page_content=chunk["text"],
            metadata={"source": src, "company": company, "doc_id": chunk.get("doc_id", "")},
        ))

    print(f"[ok] Prepared {len(docs)} corpus documents for TestsetGenerator "
          f"({len(companies)} companies)")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: TestsetGenerator
# ─────────────────────────────────────────────────────────────────────────────

def generate_testset(docs: list[Document]) -> list[dict]:
    """
    Generate synthetic question + ground-truth pairs using RAGAS TestsetGenerator v0.2.

    Uses separate models for generation and critique to avoid circular bias:
      generator : llama-3.1-8b-instant  (produces candidate questions)
      critic    : llama-3.3-70b-versatile (filters and scores them)

    Returns list of {"question": str, "ground_truth": str}.
    """
    print(f"\n[step 1] Generating testset ({TESTSET_SIZE} Q+GT pairs) ...")
    print("  This may take 5-10 minutes due to Groq rate limits.")

    ragas_embedder = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )

    generator = TestsetGenerator(
        llm=make_groq_llm(GROQ_GENERATOR_LLM, temperature=0.4),
        critic_llm=make_groq_llm(GROQ_CRITIC_LLM, temperature=0.0),
        embeddings=ragas_embedder,
    )

    # generate_with_langchain_docs was renamed in some patch versions — handle both
    if hasattr(generator, "generate_with_langchain_docs"):
        testset = generator.generate_with_langchain_docs(docs, testset_size=TESTSET_SIZE)
    else:
        testset = generator.generate(docs, testset_size=TESTSET_SIZE)

    # Defensive field access: RAGAS v0.2 minor versions differ on attribute names
    pairs = []
    for sample in testset.samples:
        question     = getattr(sample, "user_input",    getattr(sample, "question",     ""))
        ground_truth = getattr(sample, "reference",     getattr(sample, "ground_truth", ""))
        if not question:
            continue
        pairs.append({"question": question, "ground_truth": ground_truth})

    if len(pairs) < 5:
        print(f"  [warn] TestsetGenerator only produced {len(pairs)} pairs "
              f"(requested {TESTSET_SIZE}). Proceeding with what we have.")
    else:
        print(f"[ok] Generated {len(pairs)} Q+GT pairs")

    # Cache so reruns skip regeneration
    cache_path = RESULTS_DIR / "ragas_testset.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    print(f"[ok] Testset cached to {cache_path}")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Answer generation via Groq
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(question: str, contexts: list[dict]) -> str:
    """
    Exact replica of RAGEngine.build_prompt() from finsight.py:160-173.
    The prompt format must match so that Groq answers are comparable to
    vLLM answers in the latency benchmark.
    """
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


def get_answer_via_groq(
    question: str,
    contexts: list[dict],
    groq_model: str,
) -> str:
    """Query Groq with the FinSight prompt and return the answer text."""
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
    pairs:      list[dict],
    index,
    meta:       list[dict],
    embedder:   SentenceTransformer,
    model_label: str,
    groq_model: str,
) -> EvaluationDataset:
    """
    For each Q+GT pair:
      1. Retrieve top-5 chunks via local FAISS (matches finsight.py exactly)
      2. Generate answer via Groq
      3. Pack into SingleTurnSample for RAGAS v0.2
    """
    print(f"\n[step 2] Generating answers for: {model_label}")
    samples = []

    for i, pair in enumerate(pairs):
        question     = pair["question"]
        ground_truth = pair["ground_truth"]

        print(f"  [{i+1:02d}/{len(pairs)}] {question[:70]}...")

        contexts      = retrieve(question, index, meta, embedder)
        context_texts = [c["text"] for c in contexts]
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

def run_ragas_evaluation(
    dataset:     EvaluationDataset,
    model_label: str,
) -> dict:
    """
    Run RAGAS v0.2 evaluate() with three metrics.
    Judge LLM: llama-3.3-70b-versatile (Groq).
    """
    print(f"\n[step 3] RAGAS evaluation: {model_label}")
    print(f"  metrics : faithfulness, answer_relevancy, context_precision")
    print(f"  judge   : {GROQ_EVALUATOR_LLM} (Groq)")

    evaluator_llm = make_groq_llm(GROQ_EVALUATOR_LLM, temperature=0.0)
    evaluator_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        show_progress=True,
    )

    scores = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        try:
            scores[metric] = round(float(result[metric]), 4)
        except (TypeError, ValueError):
            print(f"  [warn] {metric} returned NaN or None — setting to null")
            scores[metric] = None

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
    """Write results/ragas_eval.md with summary table and per-question breakdown."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    metrics   = ["faithfulness", "answer_relevancy", "context_precision"]

    def fmt(val) -> str:
        return f"{val:.4f}" if isinstance(val, float) else "N/A"

    llama_scores   = all_results.get("llama",   {}).get("scores", {})
    mistral_scores = all_results.get("mistral", {}).get("scores", {})

    lines = [
        "# RAGAS Evaluation: FinSight SEC Filing RAG\n",
        "\n",
        f"**Date**: {timestamp}  \n",
        f"**Testset**: {len(testset)} synthetic Q+GT pairs  \n",
        f"**Corpus**: AAPL, MSFT, GOOGL, AMZN, META — 10-K filings (3 years each)  \n",
        f"**Retrieval**: FAISS IndexFlatIP · BGE-small-en-v1.5 · top-{TOP_K} chunks  \n",
        f"**Judge LLM**: Groq `{GROQ_EVALUATOR_LLM}`  \n",
        "\n",
        "---\n",
        "\n",
        "## Model Proxy Caveat\n",
        "\n",
        "> The FinSight Modal deployment runs:\n",
        "> - `meta-llama/Meta-Llama-3.1-8B-Instruct` (LLaMA)\n",
        "> - `mistralai/Mistral-7B-Instruct-v0.3` (Mistral)\n",
        ">\n",
        "> This evaluation uses Groq API proxies to avoid Modal GPU costs:\n",
        "> - **LLaMA proxy** : `llama-3.1-8b-instant` — same base model, different serving stack\n",
        "> - **Mistral proxy**: `mixtral-8x7b-32768` — **architecturally different**: 8×7B MoE vs 7B dense\n",
        ">\n",
        "> Scores approximate model-family quality. Exact values will differ from the vLLM-served versions.\n",
        "\n",
        "---\n",
        "\n",
        "## Results Summary\n",
        "\n",
        "| Metric | LLaMA 3.1 8B | Mistral proxy (MoE) |\n",
        "|--------|:------------:|:-------------------:|\n",
    ]
    for m in metrics:
        lines.append(
            f"| {m} | {fmt(llama_scores.get(m))} | {fmt(mistral_scores.get(m))} |\n"
        )

    lines += [
        "\n",
        "### Metric Definitions\n",
        "\n",
        "- **Faithfulness**: fraction of answer claims that are entailed by the retrieved context\n",
        "- **Answer Relevancy**: semantic alignment between the question and the answer\n",
        "- **Context Precision**: whether the most relevant chunks are ranked highest by FAISS\n",
        "\n",
        "---\n",
        "\n",
        "## Evaluation Setup\n",
        "\n",
        f"| Parameter | Value |\n",
        f"|-----------|-------|\n",
        f"| TestsetGen LLM | `{GROQ_GENERATOR_LLM}` (generator), `{GROQ_CRITIC_LLM}` (critic) |\n",
        f"| Answer max tokens | {MAX_ANSWER_TOKENS} |\n",
        f"| Temperature | 0.0 (deterministic) |\n",
        f"| Corpus sample for TestsetGen | {CORPUS_SAMPLE_SIZE} chunks |\n",
        "\n",
        "---\n",
        "\n",
        "## Limitations\n",
        "\n",
        "- Groq proxy models differ from deployed vLLM models (see caveat above — Mistral especially)\n",
        f"- Only {len(testset)} Q+GT pairs — increase `TESTSET_SIZE` for statistical robustness\n",
        "- Testset is synthetic; real user queries may differ in distribution\n",
        f"- RAGAS judge (`{GROQ_EVALUATOR_LLM}`) may favour certain answer styles\n",
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
    print("=" * 65)
    print("FinSight RAGAS Evaluation")
    print("=" * 65)

    if not os.getenv("GROQ_API_KEY"):
        sys.exit(
            "[ERROR] GROQ_API_KEY is not set.\n"
            "Add it to your .env file:  GROQ_API_KEY=gsk_...\n"
        )

    # Load local FAISS index
    index, meta = load_index()
    embedder    = load_embedder()
    docs        = build_langchain_docs(meta)

    # Load or generate testset
    cache_path = RESULTS_DIR / "ragas_testset.json"
    if cache_path.exists():
        print(f"\n[cache] Loading testset from {cache_path}")
        print("  (delete this file to regenerate from scratch)")
        with open(cache_path, encoding="utf-8") as f:
            testset = json.load(f)
        print(f"[ok] Loaded {len(testset)} Q+GT pairs")
    else:
        testset = generate_testset(docs)

    if len(testset) < 5:
        sys.exit(
            f"[ERROR] Too few Q+GT pairs ({len(testset)}). "
            "Check TestsetGenerator output or provide a manual testset."
        )

    # Models to evaluate
    models_config = [
        {
            "key":        "llama",
            "label":      "LLaMA 3.1 8B  (Groq: llama-3.1-8b-instant)",
            "groq_model": GROQ_LLAMA_MODEL,
        },
        {
            "key":        "mistral",
            "label":      "Mistral proxy  (Groq: mixtral-8x7b-32768 MoE)",
            "groq_model": GROQ_MISTRAL_MODEL,
        },
    ]

    all_results: dict = {}
    for cfg in models_config:
        dataset = build_evaluation_dataset(
            pairs=testset,
            index=index,
            meta=meta,
            embedder=embedder,
            model_label=cfg["label"],
            groq_model=cfg["groq_model"],
        )
        scores = run_ragas_evaluation(dataset, model_label=cfg["label"])
        all_results[cfg["key"]] = {
            "label":      cfg["label"],
            "groq_model": cfg["groq_model"],
            "scores":     scores,
        }

    # Persist results
    save_results_json(all_results, testset)
    write_markdown_report(all_results, testset)

    # Print comparison table
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    print("\n" + "=" * 65)
    print("RAGAS Evaluation Complete")
    print("=" * 65)
    print(f"{'Metric':<25} {'LLaMA 3.1 8B':>15} {'Mistral (MoE)':>15}")
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
    print(f"Testset     : results/ragas_testset.json")


if __name__ == "__main__":
    main()
