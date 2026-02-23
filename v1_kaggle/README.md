# v1 — Kaggle Attempt

This directory preserves the original Kaggle notebook and documents every issue encountered trying to run vLLM on Kaggle T4 GPUs.

## What Was Attempted

- Dual T4 GPU setup (tensor-parallel-size 2)
- vLLM 0.6.3 → latest vLLM
- Various dependency pinning strategies
- Colab as an alternative platform

## What Worked

- SEC filing downloads via `sec-edgar-downloader`
- BGE-small embeddings + FAISS index (4,782 chunks from 15 filings)
- Basic RAG pipeline logic

## What Didn't

Latest vLLM cannot run on T4 (sm_75) without dependency conflicts. See [`issues.md`](issues.md) for the full breakdown.

## Files

- `llm-serving-claude.ipynb` — original notebook with all attempts documented
- `issues.md` — complete error timeline and root cause analysis
