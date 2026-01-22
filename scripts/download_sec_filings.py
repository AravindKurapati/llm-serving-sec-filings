#!/usr/bin/env python3
"""Download SEC filings for specified companies"""

import argparse
import os
from pathlib import Path
from sec_api import QueryApi, ExtractorApi
from tqdm import tqdm
import time

def download_filings(tickers, filing_types, years, output_dir, sec_api_key=None):
    """Download SEC filings using SEC API"""
    
    if not sec_api_key:
        sec_api_key = os.getenv("SEC_API_KEY")
        if not sec_api_key:
            print("Error: SEC_API_KEY not found. Get one at https://sec-api.io/")
            return
    
    query_api = QueryApi(api_key=sec_api_key)
    extractor_api = ExtractorApi(api_key=sec_api_key)
    
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    for ticker in tickers:
        for filing_type in filing_types:
            for year in years:
                print(f"\nSearching {ticker} {filing_type} filings from {year}...")
                
                query = {
                    "query": f'ticker:{ticker} AND formType:"{filing_type}" AND filedAt:[{year}-01-01 TO {year}-12-31]',
                    "from": "0",
                    "size": "10",
                    "sort": [{"filedAt": {"order": "desc"}}]
                }
                
                try:
                    filings = query_api.get_filings(query)
                    
                    for filing in tqdm(filings.get('filings', []), desc=f"{ticker} {filing_type} {year}"):
                        filing_url = filing['linkToFilingDetails']
                        filed_date = filing['filedAt'][:10]
                        
                        # Extract the full text
                        text = extractor_api.get_section(filing_url, "all", "text")
                        
                        # Save to file
                        filename = f"{ticker}_{filing_type}_{filed_date}.txt"
                        filepath = output / filename
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(text)
                        
                        print(f"  Saved: {filename}")
                        time.sleep(0.2)  # Rate limiting
                        
                except Exception as e:
                    print(f"Error downloading {ticker} {filing_type} {year}: {e}")
                    continue
    
    print(f"\n✓ Downloads complete! Files saved to {output}")

def main():
    parser = argparse.ArgumentParser(description="Download SEC filings")
    parser.add_argument("--tickers", nargs='+', required=True,
                       help="Stock tickers (e.g., AAPL MSFT GOOGL)")
    parser.add_argument("--filing-types", nargs='+', default=['10-K', '10-Q'],
                       help="Filing types to download")
    parser.add_argument("--years", nargs='+', type=int, required=True,
                       help="Years to download (e.g., 2022 2023 2024)")
    parser.add_argument("--output", default="data/raw/sec_filings",
                       help="Output directory")
    parser.add_argument("--api-key", help="SEC API key (or set SEC_API_KEY env var)")
    
    args = parser.parse_args()
    
    download_filings(
        args.tickers,
        args.filing_types,
        args.years,
        args.output,
        args.api_key
    )

if __name__ == "__main__":
    main()
