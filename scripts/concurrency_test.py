#!/usr/bin/env python3
"""
Concurrency Test for FinSight Modal Endpoints
==============================================
Sends N concurrent POST requests to a deployed Modal /v1/stream endpoint and
measures throughput degradation as concurrency increases.

RUN
---
  # Against both models using URLs from .env:
  python scripts/concurrency_test.py

  # Against a specific URL:
  python scripts/concurrency_test.py --url https://your-modal-url.modal.run

  # Custom concurrency levels and question:
  python scripts/concurrency_test.py --concurrency 1 4 8 16 \\
      --question "What are Apple's main supply chain risks?"

  # Test only LLaMA:
  python scripts/concurrency_test.py --url "$MODAL_LLAMA_URL"

REQUIREMENTS
------------
  pip install httpx[http2] python-dotenv
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONCURRENCY = [1, 2, 4, 8]
DEFAULT_QUESTION    = "What are the main financial risks disclosed in these SEC filings?"
REQUEST_TIMEOUT_S   = 600.0   # vLLM cold start can take 5-10 min on first hit
WARMUP_TIMEOUT_S    = 600.0   # same — warmup waits for full cold start


async def stream_request(
    client: httpx.AsyncClient,
    url: str,
    question: str,
) -> dict:
    """
    Fire one POST /v1/stream request, consume the full SSE stream, and return
    timing metrics.

    Returns:
        {
          "ttft_ms":       float | None,   # time to first content token
          "total_ms":      float,           # wall time for full response
          "tokens":        int,             # completion tokens (from metrics event)
          "error":         str | None,      # set if request failed
        }
    """
    payload = {"question": question, "k": 5, "max_tokens": 400, "mode": "concise"}
    t0 = time.perf_counter()
    ttft_ms   = None
    tokens    = 0
    error_msg = None

    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Server-appended metrics event
                if chunk.get("type") == "metrics":
                    # Use server-measured TTFT when available (more accurate)
                    if ttft_ms is None and chunk.get("ttft_ms"):
                        ttft_ms = chunk["ttft_ms"]
                    tokens = chunk.get("tokens", tokens)
                    continue

                choices = chunk.get("choices")
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content and ttft_ms is None:
                        # Fall back to client-side TTFT if server didn't send metrics
                        ttft_ms = (time.perf_counter() - t0) * 1000

    except httpx.HTTPStatusError as exc:
        error_msg = f"HTTP {exc.response.status_code}"
    except (httpx.TimeoutException, asyncio.TimeoutError):
        error_msg = f"timeout after {REQUEST_TIMEOUT_S}s"
    except Exception as exc:
        error_msg = str(exc)

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "ttft_ms":  ttft_ms,
        "total_ms": round(total_ms, 1),
        "tokens":   tokens,
        "error":    error_msg,
    }


async def run_concurrency_level(
    url: str,
    question: str,
    concurrency: int,
) -> dict:
    """Fire `concurrency` requests simultaneously and return aggregate metrics."""
    print(f"  concurrency={concurrency:2d} — firing {concurrency} request(s) ...", end="", flush=True)
    t_wall_start = time.perf_counter()

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks   = [stream_request(client, url, question) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)

    wall_s = time.perf_counter() - t_wall_start

    errors   = [r for r in results if r["error"]]
    successes = [r for r in results if not r["error"]]

    latencies  = [r["total_ms"] for r in successes]
    ttfts      = [r["ttft_ms"]  for r in successes if r["ttft_ms"] is not None]

    req_per_sec   = concurrency / wall_s if wall_s > 0 else 0
    median_lat_ms = statistics.median(latencies) if latencies else None
    p95_lat_ms    = (
        sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else
        latencies[0] if latencies else None
    )
    median_ttft_ms = statistics.median(ttfts) if ttfts else None
    error_rate     = len(errors) / concurrency

    med_lat_str = f"{median_lat_ms:.0f}ms" if median_lat_ms is not None else "N/A"
    print(
        f"  done in {wall_s:.1f}s  |  {req_per_sec:.2f} req/s  |  "
        f"med_lat={med_lat_str}  |  errors={len(errors)}/{concurrency}"
    )

    return {
        "concurrency":      concurrency,
        "wall_time_s":      round(wall_s, 2),
        "requests_per_sec": round(req_per_sec, 3),
        "median_latency_ms": round(median_lat_ms, 1) if median_lat_ms is not None else None,
        "p95_latency_ms":    round(p95_lat_ms,    1) if p95_lat_ms    is not None else None,
        "median_ttft_ms":    round(median_ttft_ms, 1) if median_ttft_ms is not None else None,
        "error_rate":        round(error_rate, 4),
        "n_errors":          len(errors),
        "error_samples":     [e["error"] for e in errors[:3]],
    }


async def warmup(url: str, question: str) -> bool:
    """
    Send one request and wait for a full response to ensure the Modal container
    (including vLLM cold start) is ready before measuring concurrency.
    Returns True if warmup succeeded, False if it timed out or errored.
    """
    print(f"  [warmup] waiting for container to be ready (up to {WARMUP_TIMEOUT_S:.0f}s) ...", end="", flush=True)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        result = await stream_request(client, url, question)
    if result["error"]:
        print(f" FAILED: {result['error']}")
        return False
    print(f" ready in {result['total_ms']/1000:.1f}s (TTFT {result['ttft_ms']:.0f}ms)")
    return True


async def test_url(url: str, question: str, concurrency_levels: list[int]) -> list[dict]:
    """Warm up the endpoint then run all concurrency levels sequentially."""
    ok = await warmup(url, question)
    if not ok:
        print("  [warn] warmup failed — proceeding anyway, results may include cold-start latency")

    level_results = []
    for c in concurrency_levels:
        result = await run_concurrency_level(url, question, c)
        level_results.append(result)
        # Brief pause between levels to let the endpoint stabilize
        if c != concurrency_levels[-1]:
            await asyncio.sleep(3)
    return level_results


def print_table(label: str, level_results: list[dict]) -> None:
    COL = 14
    print(f"\n{'='*75}")
    print(f"Results: {label}")
    print(f"{'='*75}")
    headers = ["concurrency", "req/s", "med_lat_ms", "p95_lat_ms", "med_ttft_ms", "error_rate"]
    print("  " + "".join(f"{h:>{COL}}" for h in headers))
    print("  " + "-" * (COL * len(headers)))
    for r in level_results:
        def fmt(v) -> str:
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)
        row = [
            fmt(r["concurrency"]),
            fmt(r["requests_per_sec"]),
            fmt(r["median_latency_ms"]),
            fmt(r["p95_latency_ms"]),
            fmt(r["median_ttft_ms"]),
            fmt(r["error_rate"]),
        ]
        print("  " + "".join(f"{v:>{COL}}" for v in row))


async def main_async(args) -> None:
    question = args.question

    def normalize_url(url: str) -> str:
        """Ensure URL ends with /v1/stream — the Modal endpoint path."""
        url = url.rstrip("/")
        if not url.endswith("/v1/stream"):
            url += "/v1/stream"
        return url

    # Resolve URLs
    targets: list[dict] = []
    if args.url:
        targets.append({"label": "custom", "url": normalize_url(args.url)})
    else:
        llama_url   = os.getenv("MODAL_LLAMA_URL")
        mistral_url = os.getenv("MODAL_MISTRAL_URL")
        if not llama_url and not mistral_url:
            sys.exit(
                "[ERROR] No --url provided and neither MODAL_LLAMA_URL nor "
                "MODAL_MISTRAL_URL is set in .env.\n"
                "Add them to .env or pass --url explicitly."
            )
        if llama_url:
            targets.append({"label": "llama",   "url": normalize_url(llama_url)})
        if mistral_url:
            targets.append({"label": "mistral", "url": normalize_url(mistral_url)})

    concurrency_levels = sorted(set(args.concurrency))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*75}")
    print("FinSight Concurrency Test")
    print(f"{'='*75}")
    print(f"  question    : {question[:70]}{'...' if len(question) > 70 else ''}")
    print(f"  concurrency : {concurrency_levels}")
    print(f"  targets     : {[t['label'] for t in targets]}")
    print(f"  timeout     : {REQUEST_TIMEOUT_S}s per request\n")

    all_results = {}
    for target in targets:
        print(f"\n--- {target['label']}  ({target['url']}) ---")
        level_results = await test_url(target["url"], question, concurrency_levels)
        all_results[target["label"]] = {
            "url":    target["url"],
            "levels": level_results,
        }

    # Print tables
    for label, data in all_results.items():
        print_table(label, data["levels"])

    # Save JSON
    output = {
        "timestamp":          timestamp,
        "question":           question,
        "concurrency_levels": concurrency_levels,
        "results":            all_results,
    }
    out_path = RESULTS_DIR / f"concurrency_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[ok] Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Modal endpoint throughput degradation under concurrent load."
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Modal endpoint URL (e.g. https://xxx.modal.run). "
             "Overrides MODAL_LLAMA_URL / MODAL_MISTRAL_URL from .env.",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=DEFAULT_CONCURRENCY,
        metavar="N",
        help=f"Concurrency levels to test (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question to send to the endpoint.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
