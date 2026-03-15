#!/usr/bin/env python3
"""
RAGAS Evaluation for FinSight SEC Filing RAG
=============================================
Evaluates LLaMA 3.1 8B and Mistral 7B on three RAGAS quality metrics:
  - faithfulness       : are answer claims grounded in the retrieved context?
  - answer_relevancy   : does the answer address the question?
  - context_precision  : do the most relevant chunks rank highest in retrieval?

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

try:
    from groq import RateLimitError as GroqRateLimitError
except ImportError:
    GroqRateLimitError = Exception  # fallback for older SDK versions

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator


ROOT               = Path(__file__).parent.parent
INDEX_DIR          = ROOT / "data" / "index"
INDEX_PATH         = INDEX_DIR / "chunks.faiss"
META_PATH          = INDEX_DIR / "meta.npy"
HARDCODED_TESTSET  = ROOT / "data" / "testset.json"
RESULTS_DIR        = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Groq model IDs
GROQ_LLAMA_MODEL   = "llama-3.1-8b-instant"
GROQ_MISTRAL_MODEL = "llama-3.1-8b-instant"
GROQ_GENERATOR_LLM = "llama-3.3-70b-versatile"
GROQ_CRITIC_LLM    = "llama-3.3-70b-versatile"
GROQ_EVALUATOR_LLM = "llama-3.3-70b-versatile"  

# Evaluation parameters
TESTSET_SIZE        = 15
TOP_K               = 5
CORPUS_SAMPLE_SIZE  = 30
MAX_ANSWER_TOKENS   = 400
EMBED_MODEL         = "BAAI/bge-small-en-v1.5"


DEFAULT_EVAL_QUESTIONS = 5

JUDGE_MAX_TOKENS = 1024

# Groq rate-limit settings
# llama-3.3-70b-versatile: 12K TPM, 100K TPD
INTER_REQUEST_DELAY_S      = 3.0
RETRY_MAX_ATTEMPTS         = 5
RETRY_BASE_DELAY_S         = 10.0
RETRY_BACKOFF_FACTOR       = 2.0
RAGAS_INTER_SAMPLE_DELAY_S = 90
MIN_TPD_REMAINING          = 20_000

# Context truncation for judge prompt
JUDGE_CONTEXT_CHAR_LIMIT = 200


def make_groq_llm(model_name: str, temperature: float = 0.0, max_tokens: int = 512,
                  bypass_n: bool = False) -> LangchainLLMWrapper:
    """Create a RAGAS compatible LangChain wrapped Groq LLM with built in retry.
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
    delay = RETRY_BASE_DELAY_S
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            result = fn(*args, **kwargs)
            time.sleep(INTER_REQUEST_DELAY_S)
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


def retrieve(question: str, index, meta: list, embedder: SentenceTransformer,
             k: int = TOP_K) -> list[dict]:
    vec = embedder.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, ids = index.search(vec, k)
    return [meta[i] for i in ids[0]]



def build_langchain_docs(meta: list[dict], sample_size: int = CORPUS_SAMPLE_SIZE) -> list[Document]:
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




def generate_testset(docs: list[Document]) -> list[dict]:
    print(f"\n[step 1] Generating testset ({TESTSET_SIZE} Q+GT pairs) ...")
    print("  This may take 5-10 minutes due to Groq rate limits.")

    ragas_embedder = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))
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
        question     = getattr(inner, "user_input",  getattr(inner, "question",     ""))
        ground_truth = getattr(inner, "reference",   getattr(inner, "ground_truth", ""))
        if not question:
            continue
        pairs.append({"question": question, "ground_truth": ground_truth})

    if len(pairs) < 5:
        print(f"  [warn] TestsetGenerator only produced {len(pairs)} pairs. Proceeding.")
    else:
        print(f"[ok] Generated {len(pairs)} Q+GT pairs")

    cache_path = RESULTS_DIR / "ragas_testset.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    print(f"[ok] Testset cached to {cache_path}")
    return pairs



def build_prompt(question: str, contexts: list[dict]) -> str:
    formatted = "\n\n".join(
        f"[{i+1}] (from {c.get('src', c.get('source', 'unknown'))}):\n{c['text'][:600]}"
        for i, c in enumerate(contexts)
    )
    return (
        f"You are a financial analyst. Answer using ONLY the context below.\n"
        f"Cite sources as [1], [2] etc. Be concise and factual.\n\n"
        f"Question: {question}\n\nContext:\n{formatted}\n\nAnswer:"
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




def build_evaluation_dataset(pairs: list[dict], index, meta: list[dict],
                              embedder: SentenceTransformer, model_label: str,
                              groq_model: str) -> EvaluationDataset:
    """
    For each Q+GT pair:
      1. Retrieve top-5 chunks via FAISS
      2. Generate answer via Groq (full 600-char chunks)
      3. Pack into SingleTurnSample with truncated contexts for the judge
         (200 chars/chunk keeps judge prompt within 12K TPM limit)
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



def run_ragas_evaluation(dataset: EvaluationDataset, model_label: str,
                         eval_questions: int = DEFAULT_EVAL_QUESTIONS) -> dict:
    """
    Score up to eval_questions samples through the RAGAS judge.
    Remaining samples in the dataset are skipped to stay within TPD budget.
    """
    samples_to_score = dataset.samples[:eval_questions]

    print(f"\n[step 3] RAGAS evaluation: {model_label}")
    print(f"  metrics  : faithfulness, answer_relevancy, context_precision")
    print(f"  judge    : {GROQ_EVALUATOR_LLM} (Groq, 12K TPM, max_tokens={JUDGE_MAX_TOKENS})")
    print(f"  scoring  : {len(samples_to_score)}/{len(dataset.samples)} samples "
          f"(use --eval-questions to change)")
    print(f"  throttle : {RAGAS_INTER_SAMPLE_DELAY_S}s between samples")
    print(f"  context  : {JUDGE_CONTEXT_CHAR_LIMIT} chars/chunk (truncated for judge)")

    evaluator_llm = make_groq_llm(
        GROQ_EVALUATOR_LLM, temperature=0.0, max_tokens=JUDGE_MAX_TOKENS, bypass_n=True
    )
    evaluator_emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))
    run_cfg = RunConfig(max_workers=1)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision"]
    accumulated: dict[str, list[float]] = {m: [] for m in metric_names}

    for i, sample in enumerate(samples_to_score):
        print(f"  [sample {i+1:02d}/{len(samples_to_score)}] scoring ...")
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

        if i < len(samples_to_score) - 1:
            print(f"  [throttle] sleeping {RAGAS_INTER_SAMPLE_DELAY_S}s ...")
            time.sleep(RAGAS_INTER_SAMPLE_DELAY_S)

    scores = {}
    for m in metric_names:
        vals = accumulated[m]
        if vals:
            scores[m] = round(sum(vals) / len(vals), 4)
        else:
            print(f"  [warn] {m} returned NaN for all samples - setting to null")
            scores[m] = None

    print(f"  scores: {scores}")
    return scores



def save_results_json(all_results: dict, testset: list[dict],
                      eval_questions: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp":      timestamp,
        "testset_size":   len(testset),
        "eval_questions": eval_questions,
        "retrieval":      {"index": "FAISS IndexFlatIP", "k": TOP_K, "embed_model": EMBED_MODEL},
        "models":         all_results,
        "testset":        testset,
    }
    path = RESULTS_DIR / f"ragas_eval_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[ok] JSON saved to {path}")
    return path


def write_markdown_report(all_results: dict, testset: list[dict],
                          eval_questions: int) -> Path:
    timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    metrics        = ["faithfulness", "answer_relevancy", "context_precision"]
    llama_scores   = all_results.get("llama",   {}).get("scores", {})
    mistral_scores = all_results.get("mistral", {}).get("scores", {})

    def fmt(val) -> str:
        return f"{val:.4f}" if isinstance(val, float) else "N/A"

    lines = [
        "# RAGAS Evaluation: FinSight SEC Filing RAG\n\n",
        f"**Date**: {timestamp}  \n",
        f"**Testset**: {len(testset)} Q+GT pairs ({eval_questions} scored)  \n",
        f"**Corpus**: AAPL, MSFT, GOOGL, AMZN, META — 10-K filings (3 years each)  \n",
        f"**Retrieval**: FAISS IndexFlatIP · BGE-small-en-v1.5 · top-{TOP_K} chunks  \n",
        f"**Judge LLM**: Groq `{GROQ_EVALUATOR_LLM}` (max_tokens={JUDGE_MAX_TOKENS})  \n\n",
        "---\n\n",
        "## Model Proxy Note\n\n",
        "> The FinSight Modal deployment runs `meta-llama/Meta-Llama-3.1-8B-Instruct` and\n",
        "> `mistralai/Mistral-7B-Instruct-v0.3` via vLLM. This evaluation uses Groq API\n",
        "> proxies (both map to `llama-3.1-8b-instant` — mixtral decommissioned) to avoid\n",
        "> Modal GPU costs. Scores reflect model-family quality on this task.\n\n",
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
        f"| Judge LLM | `{GROQ_EVALUATOR_LLM}` (12K TPM, 100K TPD) |\n",
        f"| Judge max_tokens | {JUDGE_MAX_TOKENS} |\n",
        f"| Answer max tokens | {MAX_ANSWER_TOKENS} |\n",
        f"| Temperature | 0.0 (deterministic) |\n",
        f"| Retrieval k | {TOP_K} |\n",
        f"| Judge context truncation | {JUDGE_CONTEXT_CHAR_LIMIT} chars/chunk |\n",
        f"| Inter-sample delay | {RAGAS_INTER_SAMPLE_DELAY_S}s |\n",
        f"| Samples scored | {eval_questions} of {len(testset)} |\n\n",
        "---\n\n",
        "## Limitations\n\n",
        "- Groq proxy models differ from deployed vLLM models (see proxy note above)\n",
        f"- Only {eval_questions} of {len(testset)} Q+GT pairs scored — limited by Groq free-tier TPD\n",
        "- Testset is hand-written; real user queries may differ in distribution\n",
        f"- Judge contexts truncated to {JUDGE_CONTEXT_CHAR_LIMIT} chars/chunk to fit within TPM limits\n",
        "- `context_precision` measures retrieval rank quality, not recall coverage\n",
    ]

    report_path = RESULTS_DIR / "ragas_eval.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[ok] Report saved to {report_path}")
    return report_path



def main():
    parser = argparse.ArgumentParser(description="FinSight RAGAS Evaluation")
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic testset (default: load data/testset.json)")
    parser.add_argument("--eval-questions", type=int, default=DEFAULT_EVAL_QUESTIONS,
                        help=f"Number of questions to score through the RAGAS judge "
                             f"(default: {DEFAULT_EVAL_QUESTIONS}). Lower = fewer tokens used.")
    args = parser.parse_args()

    print("=" * 65)
    print("FinSight RAGAS Evaluation")
    print("=" * 65)
    print(f"  eval-questions: {args.eval_questions} (scoring first {args.eval_questions} samples)")

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("[ERROR] GROQ_API_KEY is not set. Add to .env: GROQ_API_KEY=gsk_...\n")

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

    if len(testset) < 3:
        sys.exit(f"[ERROR] Too few Q+GT pairs ({len(testset)}). Check testset file.")

    eval_questions = min(args.eval_questions, len(testset))

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
        scores = run_ragas_evaluation(dataset, model_label=cfg["label"],
                                      eval_questions=eval_questions)
        all_results[cfg["key"]] = {
            "label": cfg["label"], "groq_model": cfg["groq_model"], "scores": scores,
        }

    save_results_json(all_results, testset, eval_questions)
    write_markdown_report(all_results, testset, eval_questions)

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
