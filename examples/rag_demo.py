#!/usr/bin/env python3
"""RAG demo with FAISS retrieval and citations"""

import os
import sys
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = Path("data/processed")
VLLM_URL = f"http://127.0.0.1:{os.getenv('PORT', '8000')}/v1"
MODEL = os.getenv("MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

class RAGSystem:
    def __init__(self):
        print("Loading RAG system...")
        self.index = faiss.read_index(str(PROCESSED_DIR / "chunks.faiss"))
        self.metadata = np.load(PROCESSED_DIR / "metadata.npy", allow_pickle=True)
        self.retriever = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.client = OpenAI(base_url=VLLM_URL, api_key="not-needed")
        print("✓ RAG system ready")
    
    def retrieve(self, query, k=4):
        """Retrieve top-k relevant chunks"""
        query_vec = self.retriever.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        
        distances, indices = self.index.search(query_vec, k)
        return [self.metadata[idx].item() for idx in indices[0]]
    
    def build_prompt(self, query, context):
        """Build prompt with retrieved context"""
        citations = "\n\n".join([
            f"[{i+1}] {chunk['text'][:500]}... (source: {chunk['source']})"
            for i, chunk in enumerate(context)
        ])
        
        return f"""You are a helpful assistant. Answer the question using the provided context.
Include citations like [1], [2] for specific claims.

Question: {query}

Context:
{citations}

Answer:"""
    
    def ask(self, query):
        """Ask a question with RAG"""
        print(f"\nQuery: {query}")
        print("Retrieving context...")
        
        context = self.retrieve(query)
        prompt = self.build_prompt(query, context)
        
        print("Generating answer...")
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        
        answer = response.choices[0].message.content
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print(answer)
        print("\n" + "=" * 60)
        print("\nSOURCES:")
        for i, chunk in enumerate(context, 1):
            print(f"[{i}] {chunk['source']}")
        print()
        
        return answer

def main():
    if not PROCESSED_DIR.exists():
        print("Error: No index found. Run 'python scripts/build_rag_index.py' first")
        return
    
    rag = RAGSystem()
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        rag.ask(query)
    else:
        print("\nInteractive RAG Demo")
        print("Type 'exit' to quit\n")
        
        while True:
            query = input("Ask a question: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if query:
                rag.ask(query)

if __name__ == "__main__":
    main()
