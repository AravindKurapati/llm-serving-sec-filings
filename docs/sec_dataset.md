# SEC Filing Dataset Processing Guide

This guide explains how to download, process, and index large-scale SEC filing datasets.

## Overview

The project supports two dataset sizes:

1. **Sample documents** - Small test files for development
2. **SEC filings** - Production-scale 10-K and 10-Q filings

## Getting Started with SEC Filings

### 1. Get SEC API Key

Sign up at [sec-api.io](https://sec-api.io/) to get a free API key (500 requests/month free tier).

Add to your `.env`:
```bash
SEC_API_KEY=your_api_key_here
```

### 2. Download Filings

Download filings for specific companies:
```bash
python scripts/download_sec_filings.py \
  --tickers AAPL MSFT GOOGL AMZN TSLA \
  --filing-types 10-K 10-Q \
  --years 2022 2023 2024 \
  --output data/raw/sec_filings
```

This creates files like:
