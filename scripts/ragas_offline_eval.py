#!/usr/bin/env python3
"""
RAGAS Offline Evaluation — real vLLM outputs
=============================================
Fixes the proxy problem in ragas_eval.py: instead of calling Groq to generate
answers (where both lanes map to the same model), this script loads the actual
LLaMA 3.1 8B and Mistral 7B answers from results/benchmark_20260415_190523.json
(produced by the live Modal A10G deployment), re-retrieves full contexts from the
local FAISS index, and pipes the real output pairs through RAGAS scoring.

Metrics scored:
  - faithfulness      : are answer claims grounded in the retrieved context?
  - answer_relevancy  : does the answer address the question?

context_precision is omitted — it requires ground_truth, and the 5 benchmark
questions don't overlap with the testset.json ground-truth pairs. faithfulness
and answer_relevancy are the meaningful signals for model comparison anyway;
context_precision reflects retrieval quality (same FAISS index for both models,
so it would be identical by construction).

RUN
---
  python scripts/ragas_offline_eval.py
  python scripts/ragas_offline_eval.py --benchmark results/benchmark_20260415_190523.json

PREREQUISITES
-------------
  - GROQ_API_KEY in .env  (judge LLM only — no answer generation needed)
  - data/index/chunks.faiss + data/index/meta.npy  (local FAISS index)
  - results/benchmark_20260415_190523.json          (real vLLM outputs)
"""

import argparse
import html
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI as _OpenAIClient
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

ROOT       = Path(__file__).parent.parent
INDEX_DIR  = ROOT / "data" / "index"
INDEX_PATH = INDEX_DIR / "chunks.faiss"
META_PATH  = INDEX_DIR / "meta.npy"
RESULTS    = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEFAULT_BENCHMARK = RESULTS / "benchmark_20260415_190523.json"

EMBED_MODEL      = "BAAI/bge-small-en-v1.5"
TOP_K            = 5
JUDGE_LLM        = "llama-3.3-70b-versatile"
JUDGE_MAX_TOKENS = 4096
CONTEXT_CHAR_LIMIT = 800
INTER_SAMPLE_DELAY_S = 90

MIN_ALPHA_RATIO = 0.4
_STRUCTURAL_PATTERNS = [
    re.compile(r"\(Tables?\)", re.I),
    re.compile(r"\[Abst\b", re.I),
    re.compile(r"\d+\s+Months?\s+Ended", re.I),
    re.compile(r"auth_ref", re.I),
    re.compile(r"^\s*[\{\"]http", re.M),
]


def clean_text(text: str) -> str:
    return html.unescape(text)


def is_prose_chunk(text: str) -> bool:
    if not text:
        return False
    for window in (150, 400):
        s = text[:window]
        if not s:
            continue
        alpha = sum(c.isalpha() for c in s)
        if (alpha / len(s)) < MIN_ALPHA_RATIO:
            return False
    sample = text[:400]
    for pat in _STRUCTURAL_PATTERNS:
        if pat.search(sample):
            return False
    return True


def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        sys.exit(
            f"\n[ERROR] Index not found.\n"
            f"  modal volume get finsight-data /data/chunks.faiss {INDEX_PATH}\n"
            f"  modal volume get finsight-data /data/meta.npy    {META_PATH}\n"
        )
    index = faiss.read_index(str(INDEX_PATH))
    meta  = np.load(str(META_PATH), allow_pickle=True).tolist()
    print(f"[ok] Index: {index.ntotal} chunks")
    return index, meta


def retrieve_contexts(question: str, index, meta: list, embedder, k: int = TOP_K) -> list[str]:
    vec = embedder.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, ids = index.search(vec, k)
    chunks = [meta[i] for i in ids[0]]

    all_texts   = [clean_text(c["text"]) for c in chunks]
    prose_texts = [t for t in all_texts if is_prose_chunk(t)]
    context_texts = prose_texts if prose_texts else all_texts
    return [t[:CONTEXT_CHAR_LIMIT] for t in context_texts]


def load_benchmark(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dataset(results_key: str, benchmark: dict, index, meta, embedder) -> EvaluationDataset:
    records = benchmark[results_key]
    samples = []
    for r in records:
        question = r["question"]
        answer   = r["answer"].strip()
        contexts = retrieve_contexts(question, index, meta, embedder)
        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=answer,
        ))
    return EvaluationDataset(samples=samples)


def run_ragas(dataset: EvaluationDataset, model_label: str) -> dict:
    print(f"\n[ragas] Scoring {model_label} — {len(dataset.samples)} samples")
    print(f"  metrics : faithfulness, answer_relevancy")
    print(f"  judge   : {JUDGE_LLM} (max_tokens={JUDGE_MAX_TOKENS})")
    print(f"  delay   : {INTER_SAMPLE_DELAY_S}s between samples")

    groq_client = _OpenAIClient(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory(
        JUDGE_LLM,
        client=groq_client,
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    evaluator_emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))
    run_cfg = RunConfig(max_workers=1, max_retries=3)

    metric_names = ["faithfulness", "answer_relevancy"]
    accumulated: dict[str, list[float]] = {m: [] for m in metric_names}

    for i, sample in enumerate(dataset.samples):
        print(f"  [sample {i+1:02d}/{len(dataset.samples)}] {sample.user_input[:60]}...")
        single = EvaluationDataset(samples=[sample])
        result = evaluate(
            dataset=single,
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_emb,
            run_config=run_cfg,
            raise_exceptions=True,
            show_progress=False,
        )
        df = result.to_pandas()
        for m in metric_names:
            try:
                val = float(df[m].iloc[0])
                if not math.isnan(val):
                    accumulated[m].append(val)
                    print(f"    {m}: {val:.4f}")
            except (TypeError, ValueError, KeyError, IndexError):
                print(f"    {m}: NaN (skipped)")

        if i < len(dataset.samples) - 1:
            print(f"  [throttle] sleeping {INTER_SAMPLE_DELAY_S}s ...")
            time.sleep(INTER_SAMPLE_DELAY_S)

    scores = {}
    for m in metric_names:
        vals = accumulated[m]
        scores[m] = round(sum(vals) / len(vals), 4) if vals else None
    print(f"  -> {model_label}: {scores}")
    return scores


def save_results(all_scores: dict, benchmark_path: Path, timestamp: str):
    payload = {
        "timestamp":    timestamp,
        "benchmark":    str(benchmark_path.name),
        "source":       "real vLLM outputs (Modal A10G — LLaMA 3.1 8B + Mistral 7B)",
        "retrieval":    {"index": "FAISS IndexFlatIP", "k": TOP_K, "embed_model": EMBED_MODEL},
        "judge":        {"model": JUDGE_LLM, "max_tokens": JUDGE_MAX_TOKENS},
        "metrics_scored": ["faithfulness", "answer_relevancy"],
        "metrics_omitted": {
            "context_precision": "requires ground_truth; benchmark questions lack GT pairs"
        },
        "scores": all_scores,
    }
    json_path = RESULTS / f"ragas_offline_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[ok] JSON saved: {json_path}")

    # Update the canonical ragas_offline.md
    md_path = RESULTS / "ragas_offline.md"
    llama   = all_scores.get("llama", {})
    mistral = all_scores.get("mistral", {})

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else "N/A"

    lines = [
        "# RAGAS Offline Evaluation — Real vLLM Outputs\n\n",
        f"**Date**: {timestamp}  \n",
        f"**Source**: `{benchmark_path.name}` (Modal A10G — LLaMA 3.1 8B + Mistral 7B)  \n",
        f"**Retrieval**: FAISS IndexFlatIP · BGE-small-en-v1.5 · top-{TOP_K} chunks  \n",
        f"**Judge LLM**: Groq `{JUDGE_LLM}` (max_tokens={JUDGE_MAX_TOKENS})  \n\n",
        "---\n\n",
        "## What Changed vs Prior RAGAS Run\n\n",
        "The previous RAGAS evaluation (`ragas_eval.md`) used Groq proxies for answer\n",
        "generation. Mixtral was decommissioned on Groq, so both model lanes mapped to\n",
        "`llama-3.1-8b-instant` — making the LLaMA vs Mistral comparison meaningless.\n\n",
        "This run uses **real answers from the deployed vLLM endpoints** (Modal A10G):\n",
        "the exact outputs of `meta-llama/Meta-Llama-3.1-8B-Instruct` and\n",
        "`mistralai/Mistral-7B-Instruct-v0.3`. Contexts are re-retrieved from the\n",
        "local FAISS index to provide full 800-char prose chunks to the judge.\n\n",
        "---\n\n",
        "## Results\n\n",
        "| Metric | LLaMA 3.1 8B | Mistral 7B |\n",
        "|--------|:------------:|:----------:|\n",
        f"| faithfulness | {fmt(llama.get('faithfulness'))} | {fmt(mistral.get('faithfulness'))} |\n",
        f"| answer_relevancy | {fmt(llama.get('answer_relevancy'))} | {fmt(mistral.get('answer_relevancy'))} |\n",
        "\n",
        "> `context_precision` omitted — requires ground_truth; the 5 benchmark questions\n",
        "> do not overlap with `data/testset.json`. Retrieval quality is identical for\n",
        "> both models (same FAISS index, same k, same questions) so the metric would\n",
        "> not differentiate them anyway.\n\n",
        "---\n\n",
        "## Metric Definitions\n\n",
        "- **Faithfulness**: fraction of answer claims entailed by the retrieved context\n",
        "- **Answer Relevancy**: semantic alignment between the question and the answer\n\n",
        "---\n\n",
        "## Setup\n\n",
        f"| Parameter | Value |\n|-----------|-------|\n",
        f"| Source answers | Real vLLM (Modal A10G) via `{benchmark_path.name}` |\n",
        f"| Questions scored | 5 per model |\n",
        f"| Retrieval k | {TOP_K} |\n",
        f"| Context truncation | {CONTEXT_CHAR_LIMIT} chars/chunk (HTML decoded, prose-filtered) |\n",
        f"| Judge LLM | `{JUDGE_LLM}` |\n",
        f"| Judge max_tokens | {JUDGE_MAX_TOKENS} |\n",
        f"| Inter-sample delay | {INTER_SAMPLE_DELAY_S}s |\n\n",
        "---\n\n",
        "## Limitations\n\n",
        "- Only 5 questions — not statistically robust\n",
        "- `faithfulness` judge sees re-retrieved contexts (up to 800 chars/chunk prose-filtered),\n",
        "  not byte-identical context passed to vLLM (which used 600-char truncation with HTML entities)\n",
        "- Judge LLM (Groq `llama-3.3-70b-versatile`) may have its own biases\n",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[ok] Report saved: {md_path}")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="RAGAS offline eval using real vLLM outputs")
    parser.add_argument(
        "--benchmark", type=Path, default=DEFAULT_BENCHMARK,
        help=f"Path to benchmark JSON (default: {DEFAULT_BENCHMARK})"
    )
    parser.add_argument(
        "--model", choices=["llama", "mistral", "both"], default="both",
        help="Which model to score (default: both)"
    )
    parser.add_argument(
        "--load-llama", type=Path, default=None,
        help="Path to a JSON file with pre-computed llama scores to skip re-scoring"
    )
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("[ERROR] GROQ_API_KEY not set in .env")

    if not args.benchmark.exists():
        sys.exit(f"[ERROR] Benchmark file not found: {args.benchmark}")

    print("=" * 65)
    print("FinSight RAGAS Offline Evaluation")
    print("=" * 65)
    print(f"  benchmark : {args.benchmark.name}")
    print(f"  source    : real vLLM outputs (Modal A10G)")
    print(f"  metrics   : faithfulness, answer_relevancy")
    print(f"  judge     : {JUDGE_LLM}")

    benchmark = load_benchmark(args.benchmark)
    index, meta = load_index()
    embedder = SentenceTransformer(EMBED_MODEL)

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_scores: dict = {}

    model_map = [("llama", "llama_results"), ("mistral", "mistral_results")]
    if args.model != "both":
        model_map = [(k, r) for k, r in model_map if k == args.model]

    # Load pre-computed LLaMA scores to skip re-scoring when only Mistral is needed
    if args.load_llama and args.model in ("mistral", "both"):
        with open(args.load_llama, encoding="utf-8") as f:
            prior = json.load(f)
        all_scores["llama"] = prior.get("scores", {}).get("llama", {})
        print(f"[ok] Loaded LLaMA scores from {args.load_llama.name}: {all_scores['llama']}")
        model_map = [(k, r) for k, r in model_map if k != "llama"]

    for model_key, results_key in model_map:
        dataset = build_dataset(results_key, benchmark, index, meta, embedder)
        scores  = run_ragas(dataset, model_label=model_key)
        all_scores[model_key] = scores

    save_results(all_scores, args.benchmark, timestamp)

    print("\n" + "=" * 65)
    print("RAGAS Offline Evaluation Complete")
    print("=" * 65)
    print(f"{'Metric':<22} {'LLaMA 3.1 8B':>14} {'Mistral 7B':>12}")
    print("-" * 50)
    for m in ["faithfulness", "answer_relevancy"]:
        l = all_scores.get("llama",   {}).get(m)
        r = all_scores.get("mistral", {}).get(m)
        print(f"{m:<22} {(f'{l:.4f}' if l else 'N/A'):>14} {(f'{r:.4f}' if r else 'N/A'):>12}")


if __name__ == "__main__":
    main()
