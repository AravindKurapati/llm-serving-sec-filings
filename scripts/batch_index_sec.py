#!/usr/bin/env python3
"""Batch process and index large SEC filing datasets"""

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.build_rag_index import build_index

def main():
    parser = argparse.ArgumentParser(
        description="Batch process SEC filings and build index"
    )
    parser.add_argument("--input", default="data/raw/sec_filings",
                       help="Directory containing SEC filing .txt files")
    parser.add_argument("--output", default="data/processed/sec_index",
                       help="Output directory for index")
    parser.add_argument("--chunk-size", type=int, default=800,
                       help="Chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=200,
                       help="Overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embeddings (reduce if OOM)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SEC Filing Batch Indexing")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    build_index(
        args.input,
        args.output,
        args.chunk_size,
        args.overlap,
        args.batch_size
    )

if __name__ == "__main__":
    main()
