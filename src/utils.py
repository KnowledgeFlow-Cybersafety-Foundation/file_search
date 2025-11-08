import os
import re
from pathlib import Path
from docx import Document
from typing import List, Dict, Any

def extract_text_from_docx(file_path: str) -> str:
    """Extract text content from a Word document"""
    try:
        doc = Document(file_path)
        paragraphs = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text.strip())
        
        return "\n".join(paragraphs)
    except Exception as e:
        raise Exception(f"Error reading {file_path}: {str(e)}")

def get_document_metadata(file_path: str) -> Dict[str, Any]:
    """Get metadata about a document"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file stats
    stats = path.stat()
    
    try:
        # Extract text to count words
        text = extract_text_from_docx(file_path)
        word_count = len(text.split())
        char_count = len(text)
        
        # Count paragraphs (non-empty lines)
        paragraph_count = len([line for line in text.split('\n') if line.strip()])
        
    except Exception:
        word_count = 0
        char_count = 0
        paragraph_count = 0
    
    return {
        'filename': path.name,
        'path': str(path),
        'size_bytes': stats.st_size,
        'size_kb': round(stats.st_size / 1024, 2),
        'word_count': word_count,
        'char_count': char_count,
        'paragraph_count': paragraph_count,
        'modified_time': stats.st_mtime
    }

def find_docx_files(folder_path: str) -> List[str]:
    """Find all .docx files in a folder"""
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")
    
    return [str(file) for file in folder.glob("*.docx") if not file.name.startswith("~")]

def highlight_text(text: str, query: str, max_length: int = 200) -> str:
    """Highlight query terms in text and return context"""
    if not query or not text:
        return text[:max_length] + ("..." if len(text) > max_length else "")
    
    # Find first occurrence of any query word
    query_words = query.lower().split()
    text_lower = text.lower()
    
    positions = []
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1:
            positions.append(pos)
    
    if not positions:
        return text[:max_length] + ("..." if len(text) > max_length else "")
    
    # Start context around first match
    start_pos = min(positions)
    context_start = max(0, start_pos - max_length // 2)
    context_end = min(len(text), start_pos + max_length // 2)
    
    context = text[context_start:context_end]
    
    # Add ellipsis if truncated
    if context_start > 0:
        context = "..." + context
    if context_end < len(text):
        context = context + "..."
    
    return context

def clean_text(text: str) -> str:
    """Clean and normalize text for better searching"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text.strip()