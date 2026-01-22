#!/usr/bin/env python3
"""Simple chat completion example"""

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
    
    print("Simple Chat Example")
    print("=" * 50)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Explain what a transformer model is in 2 sentences."}
        ],
        max_tokens=100,
        temperature=0.7
    )
    
    print("\nResponse:")
    print(response.choices[0].message.content)
    print(f"\nTokens used: {response.usage.total_tokens}")

if __name__ == "__main__":
    main()
