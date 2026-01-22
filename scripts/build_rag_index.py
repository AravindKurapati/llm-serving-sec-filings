#!/usr/bin/env python3
"""Build FAISS index from documents in data/raw/"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def chunk_text(text, chunk_size=800, overlap=200):
    """Split text into overlapping chunks"""
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += max(chunk_size - overlap, 1)
    return chunks

def build_index():
    """Build FAISS index from documents"""
    print("Loading documents...")
    corpus = []
    
    for doc_path in RAW_DIR.glob("*.txt"):
        print(f"Processing {doc_path.name}...")
        text = doc_path.read_text()
        
        for i, chunk in enumerate(chunk_text(text)):
            corpus.append({
                "doc_id": f"{doc_path.stem}-{i}",
                "text": chunk,
                "source": doc_path.name
            })
    
    print(f"Total chunks: {len(corpus)}")
    
    # Generate embeddings
    print("Generating embeddings...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(
        [item["text"] for item in corpus],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    
    # Save
    faiss.write_index(index, str(PROCESSED_DIR / "chunks.faiss"))
    np.save(PROCESSED_DIR / "metadata.npy", np.array(corpus, dtype=object))
    
    print(f" Index saved to {PROCESSED_DIR}")
    print(f"  - {len(corpus)} chunks indexed")
    print(f"  - Dimension: {dimension}")

if __name__ == "__main__":
    if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.txt")):
        print(f"Error: No .txt files found in {RAW_DIR}")
        print("Please add documents to data/raw/ first")
        exit(1)
    
    build_index()
