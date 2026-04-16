#!/usr/bin/env python3
"""
Answer Quality Analysis for FinSight SEC Filing RAG
====================================================
Reads results/benchmark_20260415_190523.json and computes per-answer metrics:
  - token_count       : whitespace-split word count
  - citation_count    : number of [N] citation markers (regex \[\d+\])
  - repetition_score  : fraction of 5-word ngrams that are duplicates
  - hits_token_limit  : True if token_count >= 390

Prints a comparison table LLaMA vs Mistral across all 5 questions.
Saves to results/answer_quality.json.

RUN
---
  python scripts/answer_quality.py
"""

import json
import re
from pathlib import Path

ROOT         = Path(__file__).parent.parent
BENCHMARK    = ROOT / "results" / "benchmark_20260415_190523.json"
OUT_PATH     = ROOT / "results" / "answer_quality.json"
NGRAM_SIZE   = 5
TOKEN_LIMIT  = 390


def token_count(text: str) -> int:
    return len(text.split())


def citation_count(text: str) -> int:
    return len(re.findall(r"\[\d+\]", text))


def repetition_score(text: str, n: int = NGRAM_SIZE) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    total  = len(ngrams)
    unique = len(set(ngrams))
    return round((total - unique) / total, 4)


def hits_token_limit(text: str, limit: int = TOKEN_LIMIT) -> bool:
    return token_count(text) >= limit


def analyze(answer: str) -> dict:
    tc = token_count(answer)
    return {
        "token_count":      tc,
        "citation_count":   citation_count(answer),
        "repetition_score": repetition_score(answer),
        "hits_token_limit": hits_token_limit(answer),
    }


def main():
    with open(BENCHMARK, encoding="utf-8") as f:
        data = json.load(f)

    llama_results   = data["llama_results"]
    mistral_results = data["mistral_results"]

    questions = [r["question"] for r in llama_results]
    metrics   = ["token_count", "citation_count", "repetition_score", "hits_token_limit"]

    rows = []
    for i, (lrow, mrow) in enumerate(zip(llama_results, mistral_results)):
        assert lrow["question"] == mrow["question"], "question mismatch"
        lm = analyze(lrow["answer"])
        mm = analyze(mrow["answer"])
        rows.append({
            "question_index": i,
            "question":       lrow["question"],
            "llama":          lm,
            "mistral":        mm,
        })

    # ── Comparison table ───────────────────────────────────────────────────────
    Q_WIDTH = 42
    COL     = 10

    def fmt(val) -> str:
        if isinstance(val, bool):
            return "YES" if val else "no"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    header_metric = f"{'Metric':<20}"
    header_llama  = f"{'LLaMA':>{COL}}"
    header_mis    = f"{'Mistral':>{COL}}"

    print(f"\n{'='*80}")
    print("Answer Quality  —  LLaMA 3.1 8B  vs  Mistral proxy")
    print(f"{'='*80}")

    for row in rows:
        q_short = row["question"][:Q_WIDTH]
        print(f"\nQ{row['question_index']+1}: {q_short}{'...' if len(row['question']) > Q_WIDTH else ''}")
        print(f"  {header_metric}{header_llama}{header_mis}")
        print(f"  {'-'*40}")
        for m in metrics:
            lv = fmt(row["llama"][m])
            mv = fmt(row["mistral"][m])
            print(f"  {m:<20}{lv:>{COL}}{mv:>{COL}}")

    # ── Averages (numeric metrics only) ───────────────────────────────────────
    numeric_metrics = ["token_count", "citation_count", "repetition_score"]
    print(f"\n{'='*80}")
    print("Averages across 5 questions")
    print(f"{'='*80}")
    print(f"  {header_metric}{header_llama}{header_mis}")
    print(f"  {'-'*40}")
    for m in numeric_metrics:
        l_avg = sum(r["llama"][m]   for r in rows) / len(rows)
        m_avg = sum(r["mistral"][m] for r in rows) / len(rows)
        l_str = f"{l_avg:.1f}" if m == "token_count" else f"{l_avg:.4f}"
        m_str = f"{m_avg:.1f}" if m == "token_count" else f"{m_avg:.4f}"
        print(f"  {m:<20}{l_str:>{COL}}{m_str:>{COL}}")

    hits_l = sum(1 for r in rows if r["llama"]["hits_token_limit"])
    hits_m = sum(1 for r in rows if r["mistral"]["hits_token_limit"])
    print(f"  {'hits_token_limit':<20}{f'{hits_l}/5':>{COL}}{f'{hits_m}/5':>{COL}}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "source_benchmark": str(BENCHMARK.name),
        "token_limit":      TOKEN_LIMIT,
        "ngram_size":       NGRAM_SIZE,
        "questions":        rows,
        "averages": {
            "llama":   {
                m: round(sum(r["llama"][m] for r in rows) / len(rows), 4)
                for m in numeric_metrics
            } | {"hits_token_limit": hits_l},
            "mistral": {
                m: round(sum(r["mistral"][m] for r in rows) / len(rows), 4)
                for m in numeric_metrics
            } | {"hits_token_limit": hits_m},
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[ok] Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
