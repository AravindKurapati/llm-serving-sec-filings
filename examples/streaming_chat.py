#!/usr/bin/env python3
"""Streaming chat completion example"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
    client = OpenAI(
        base_url=f"http://127.0.0.1:{os.getenv('PORT', '8000')}/v1",
        api_key="not-needed"
    )
    
    model = os.getenv("MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    print("Streaming Chat Example")
    print("=" * 50)
    print("\nResponse: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Write a haiku about GPUs."}
        ],
        max_tokens=100,
        temperature=0.8,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")

if __name__ == "__main__":
    main()
