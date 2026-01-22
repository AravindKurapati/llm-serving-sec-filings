"""SEC filing parsing utilities"""

from bs4 import BeautifulSoup
import re

def clean_sec_text(html_text):
    """Clean and extract text from SEC filing HTML"""
    
    # Parse HTML
    soup = BeautifulSoup(html_text, 'lxml')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # Remove excessive newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text

def extract_sections(filing_text):
    """Extract major sections from SEC filing"""
    
    sections = {}
    
    # Common section patterns
    patterns = {
        'risk_factors': r'Item\s+1A\.?\s+Risk Factors',
        'mda': r'Item\s+7\.?\s+Management.*Discussion.*Analysis',
        'business': r'Item\s+1\.?\s+Business',
        'financial_statements': r'Item\s+8\.?\s+Financial Statements'
    }
    
    for section_name, pattern in patterns.items():
        match = re.search(pattern, filing_text, re.IGNORECASE)
        if match:
            start = match.start()
            # Find next section or end
            rest = filing_text[start:]
            next_match = re.search(r'\nItem\s+\d+[A-Z]?\.', rest[100:], re.IGNORECASE)
            if next_match:
                end = start + 100 + next_match.start()
                sections[section_name] = filing_text[start:end]
            else:
                sections[section_name] = rest
    
    return sections
