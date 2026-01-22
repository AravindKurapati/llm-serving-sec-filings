"""Text chunking utilities"""

def chunk_text(text, chunk_size=800, overlap=200):
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in tokens (approximate)
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    tokens = text.split()
    chunks = []
    i = 0
    
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += max(chunk_size - overlap, 1)
    
    return chunks
