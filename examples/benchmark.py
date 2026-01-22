#!/usr/bin/env python3
"""Benchmark TTFT, TPOT, and throughput"""

import os
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

VLLM_URL = f"http://127.0.0.1:{os.getenv('PORT', '8000')}/v1"
MODEL = os.getenv("MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

def measure_ttft(client, prompt, max_tokens=128):
    """Measure Time To First Token using streaming"""
    start = time.time()
    first_token_time = None
    
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content and first_token_time is None:
            first_token_time = time.time()
            break
    
    # Consume rest of stream
    for _ in stream:
        pass
    
    ttft = first_token_time - start if first_token_time else None
    return ttft

def measure_generation(client, prompt, max_tokens=128):
    """Measure full generation metrics"""
    start = time.time()
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0
    )
    
    end = time.time()
    total_time = end - start
    
    tokens_generated = response.usage.completion_tokens
    tpot = total_time / tokens_generated if tokens_generated > 0 else None
    throughput = tokens_generated / total_time if total_time > 0 else None
    
    return {
        "total_time": total_time,
        "tokens_generated": tokens_generated,
        "tpot": tpot,
        "throughput": throughput
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM server")
    parser.add_argument("--prompt", default="Explain how GPUs accelerate deep learning in detail.", help="Prompt to benchmark")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    args = parser.parse_args()
    
    client = OpenAI(base_url=VLLM_URL, api_key="not-needed")
    
    print("vLLM Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs: {args.runs}")
    print("=" * 60)
    
    ttfts = []
    tpots = []
    throughputs = []
    
    for i in range(args.runs):
        print(f"\nRun {i+1}/{args.runs}...")
        
        # Measure TTFT
        ttft = measure_ttft(client, args.prompt, args.max_tokens)
        if ttft:
            ttfts.append(ttft)
            print(f"  TTFT: {ttft:.3f}s")
        
        # Measure generation
        metrics = measure_generation(client, args.prompt, args.max_tokens)
        if metrics["tpot"]:
            tpots.append(metrics["tpot"])
            throughputs.append(metrics["throughput"])
            print(f"  TPOT: {metrics['tpot']:.4f}s/token")
            print(f"  Throughput: {metrics['throughput']:.2f} tokens/s")
            print(f"  Tokens: {metrics['tokens_generated']}")
    
    print("\n" + "=" * 60)
    print("RESULTS (averaged over {} runs)".format(args.runs))
    print("=" * 60)
    if ttfts:
        print(f"TTFT: {sum(ttfts)/len(ttfts):.3f}s")
    if tpots:
        print(f"TPOT: {sum(tpots)/len(tpots):.4f}s/token")
    if throughputs:
        print(f"Throughput: {sum(throughputs)/len(throughputs):.2f} tokens/s")

if __name__ == "__main__":
    main()
