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
data/raw/sec_filings/
├── AAPL_10-K_2024-11-01.txt
├── AAPL_10-Q_2024-08-02.txt
├── MSFT_10-K_2024-07-30.txt
└── ...
### 3. Build Index

Process and index all downloaded filings:
```bash
python scripts/batch_index_sec.py \
  --input data/raw/sec_filings \
  --output data/processed/sec_index \
  --chunk-size 800 \
  --batch-size 32
```

### 4. Query the Index
```python
python examples/rag_demo.py \
  --index data/processed/sec_index \
  "What are Apple's main revenue streams according to recent 10-K?"
```

## Performance Considerations

### Memory Management

- **Small datasets** (< 1000 docs): Use default batch size 64
- **Medium datasets** (1000-10000 docs): Reduce to batch size 32
- **Large datasets** (10000+ docs): Use batch size 16, process in stages

### Chunk Size Optimization

| Document Type | Recommended Chunk Size | Overlap |
|--------------|----------------------|---------|
| Short forms (8-K) | 500 tokens | 100 |
| Quarterly (10-Q) | 800 tokens | 200 |
| Annual (10-K) | 1000 tokens | 250 |

### Indexing Time Estimates

| Dataset Size | Documents | Chunks | Index Time (est.) |
|-------------|-----------|--------|-------------------|
| Small | 10-50 | 100-500 | 1-2 min |
| Medium | 100-500 | 1K-5K | 10-30 min |
| Large | 1000+ | 10K+ | 1-3 hours |

## Advanced Usage

### Section-Specific Indexing

Extract and index only specific sections:
```python
from utils.sec_parser import extract_sections

# Extract only risk factors and MD&A
sections = extract_sections(filing_text)
risk_text = sections.get('risk_factors', '')
mda_text = sections.get('mda', '')
```

### Multi-Index Strategy

Create separate indexes for different filing types:
```bash
# 10-K index (annual reports)
python scripts/batch_index_sec.py \
  --input data/raw/sec_filings/10K \
  --output data/processed/10k_index

# 10-Q index (quarterly reports)
python scripts/batch_index_sec.py \
  --input data/raw/sec_filings/10Q \
  --output data/processed/10q_index
```

## Troubleshooting

### Rate Limiting

If you hit SEC API rate limits:
- Free tier: 500 requests/month
- Paid tier: 10,000+ requests/month
- Add delays: `time.sleep(0.5)` between requests

### Memory Issues

If indexing fails with OOM:
```bash
# Reduce batch size
python scripts/batch_index_sec.py --batch-size 16

# Or process in stages
python scripts/batch_index_sec.py --input data/raw/sec_filings/batch1
python scripts/batch_index_sec.py --input data/raw/sec_filings/batch2
```

### Empty Chunks

Some SEC filings contain tables/images that don't convert well to text. Filter out short chunks:
```python
# In build_rag_index.py, add minimum chunk length
if len(chunk.split()) < 50:
    continue  # Skip very short chunks
```

## Example Workflows

### Comparative Analysis
```bash
# Download and index competitors
python scripts/download_sec_filings.py \
  --tickers AAPL MSFT GOOGL \
  --filing-types 10-K \
  --years 2024

# Query across companies
python examples/rag_demo.py \
  "Compare cloud revenue strategies across major tech companies [2024 10-K]"
```

### Time-Series Analysis
```bash
# Download multiple years
python scripts/download_sec_filings.py \
  --tickers TSLA \
  --filing-types 10-K 10-Q \
  --years 2020 2021 2022 2023 2024

# Track changes over time
python examples/rag_demo.py \
  "How has Tesla's risk disclosure changed from 2020 to 2024?"
```
