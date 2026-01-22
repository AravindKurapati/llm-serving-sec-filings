#!/usr/bin/env python3
"""Load test with concurrent requests"""

import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

VLLM_URL = f"http://127.0.0.1:{os.getenv('PORT', '8000')}/v1"
MODEL = os.getenv("MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

def single_request(client, prompt="Write one sentence about machine learning."):
    """Execute a single request and return latency"""
    start = time.time()
    try:
        client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.0
        )
        return time.time() - start
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_test(num_requests, max_workers):
    """Run load test with specified concurrency"""
    client = OpenAI(base_url=VLLM_URL, api_key="not-needed")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        latencies = list(executor.map(
            lambda _: single_request(client),
            range(num_requests)
        ))
    
    # Filter out failed requests
    latencies = [l for l in latencies if l is not None]
    
    if not latencies:
        return None
    
    return {
        "count": len(latencies),
        "mean": np.mean(latencies),
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
    }

def main():
    print("Load Test")
    print("=" * 60)
    
    concurrency_levels = [1, 4, 8, 16, 32]
    
    for concurrency in concurrency_levels:
        print(f"\nTesting concurrency: {concurrency}")
        results = load_test(num_requests=concurrency, max_workers=concurrency)
        
        if results:
            print(f"  Requests: {results['count']}")
            print(f"  Mean: {results['mean']:.3f}s")
            print(f"  P50: {results['p50']:.3f}s")
            print(f"  P95: {results['p95']:.3f}s")
            print(f"  P99: {results['p99']:.3f}s")
        else:
            print("  Failed")

if __name__ == "__main__":
    main()
