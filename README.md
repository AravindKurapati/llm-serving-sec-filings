# vLLM Serving 

> **Fast, production-grade LLM serving with built-in RAG and performance benchmarking**

A complete starter kit for serving LLMs with vLLM, featuring retrieval-augmented generation (RAG) examples and tools to measure prefill/decode performance. Optimized for Kaggle dual T4 GPUs or single local GPU setups.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-latest-green.svg)](https://github.com/vllm-project/vllm)

---

## What This Does

- **Production LLM serving** using vLLM's OpenAI-compatible API
- **RAG implementation** with FAISS vector search and BGE embeddings
- **Performance benchmarking** tools for TTFT, TPOT, and throughput analysis
- **Prefill/decode optimization** insights and measurement utilities

**Core principle:** Prefill is compute-bound, decoding is memory bandwidth-bound. This project helps you measure and optimize each phase independently.

---

##  Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/vllm-serving-rag
cd vllm-serving-rag
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Set your Hugging Face token
export HF_TOKEN=hf_your_token_here
export MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
export PORT=8000
```

### 3. Start the Server

**Local (single GPU):**
```bash
./scripts/start_server.sh
```

**Kaggle (dual T4):**
```bash
./scripts/start_server_kaggle.sh
```

### 4. Test It

```bash
# Simple chat completion
python examples/simple_chat.py

# RAG with citations
python examples/rag_demo.py "What are Apple's main risk factors?"

# Run benchmarks
python examples/benchmark.py
```

---

## Project Structure

```
vllm-serving-rag/
├── scripts/
│   ├── start_server.sh           # Local server startup
│   ├── start_server_kaggle.sh    # Kaggle-specific startup
│   └── build_rag_index.py        # Build FAISS index from docs
├── examples/
│   ├── simple_chat.py            # Basic chat completion
│   ├── streaming_chat.py         # Streaming responses
│   ├── rag_demo.py               # RAG with citations
│   ├── benchmark.py              # TTFT/TPOT measurements
│   └── load_test.py              # Concurrent request testing
├── notebooks/
│   └── kaggle_full_demo.ipynb    # Complete Kaggle notebook
├── data/
│   ├── raw/                      # Raw documents for RAG
│   └── processed/                # FAISS indexes and metadata
├── docs/
│   ├── architecture.md           # System design details
│   ├── performance.md            # Optimization guide
│   └── deployment.md             # Production deployment tips
├── requirements.txt
└── README.md
```

---

##  Features

### vLLM Server
- Continuous batching for high throughput
- PagedAttention for efficient memory usage
- OpenAI-compatible API endpoints
- Chunked prefill support
- Multi-GPU tensor parallelism

### RAG Capabilities
- FAISS vector search integration
- BGE embeddings for retrieval
- Citation-aware response generation
- Custom document chunking strategies

### Benchmarking Tools
- **TTFT** (Time To First Token) - Prefill latency
- **TPOT** (Time Per Output Token) - Decode speed
- **Throughput** - Tokens per second
- **Load testing** - Concurrent request handling

---

## Example Usage

### Simple Chat
```python
from openai import OpenAI
import os

client = OpenAI(
    base_url=f"http://127.0.0.1:{os.getenv('PORT', '8000')}/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[{"role": "user", "content": "Explain transformers in one sentence."}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

### RAG Query
```bash
python examples/rag_demo.py "What are Microsoft's cloud revenue drivers?"
```

Output:
```
Microsoft's cloud revenue is driven by Azure growth and Office 365 adoption [1][2].
Key risks include macroeconomic factors and foreign exchange volatility [1].

Sources:
[1] MSFT_10K_mda.txt
[2] MSFT_10K_risk.txt
```

### Benchmark
```bash
python examples/benchmark.py --prompt "Explain GPU memory bandwidth" --max-tokens 128
```

Output:
```
TTFT: 0.23s
TPOT: 0.042s/token
Throughput: 23.8 tokens/s
Total tokens: 128
```

---

##  Deployment Options

### Kaggle (2× T4 GPUs)
1. Upload notebook from `notebooks/kaggle_full_demo.ipynb`
2. Add `HF_TOKEN` to Kaggle Secrets
3. Enable GPU (T4 x2) accelerator
4. Run all cells

See [docs/deployment.md](docs/deployment.md#kaggle) for details.

### Local Development
- **Single GPU**: Works out of the box with 16GB+ VRAM
- **Multi-GPU**: Set `CUDA_VISIBLE_DEVICES` and adjust tensor parallelism
- **CPU**: Not recommended (extremely slow)

See [docs/deployment.md](docs/deployment.md#local) for configuration details.

### Production
- Use docker-compose for orchestration
- Configure load balancing and autoscaling
- Monitor with Prometheus/Grafana

See [docs/deployment.md](docs/deployment.md#production) for production setup.

---

##  Documentation

- **[Architecture Guide](docs/architecture.md)** - System design and component overview
- **[Performance Tuning](docs/performance.md)** - Optimize prefill and decode phases
- **[Deployment Guide](docs/deployment.md)** - Production deployment strategies
- **[API Reference](docs/api.md)** - Complete API documentation

---



### Roadmap: DistServe-Style Disaggregation
Future plans to separate prefill and decode into independent services:
- Dedicated prefill instances (GPU compute optimized)
- Dedicated decode instances (memory bandwidth optimized)
- KV cache transfer between services
- SLO enforcement: TTFT ≤ 250ms, TPOT ≤ 50ms/token

---

##  Requirements

- **Python**: 3.10 or higher
- **GPU**: CUDA-compatible (16GB+ VRAM recommended)
- **Hugging Face**: Read access token for model downloads

See [requirements.txt](requirements.txt) for complete dependency list.

---

##  Troubleshooting

| Issue | Solution |
|-------|----------|
| `401 Unauthorized` on model download | Check `HF_TOKEN` is set correctly |
| CUDA Out of Memory | Reduce `--max-model-len` or enable tensor parallelism |
| Server not responding | Check logs: `tail -f /path/to/vllm.log` |
| Slow performance on T4 | FlashAttention v2 not available; using PyTorch fallback (expected) |

See [docs/troubleshooting.md](docs/troubleshooting.md) for more solutions.

---

##  Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

##  License

MIT License - see [LICENSE](LICENSE) for details.

---

##  Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for the serving engine
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings

