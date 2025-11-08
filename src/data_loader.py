from pathlib import Path
from typing import Dict

import streamlit as st
from docx import Document

from config import DOCUMENTS_DIR


@st.cache_data(show_spinner=False)
def load_documents(dir_path: Path = DOCUMENTS_DIR) -> Dict[str, str]:
    """Load .docx files from the documents directory into a dict of name->text."""
    data: Dict[str, str] = {}
    if not dir_path.exists():
        return data
    for file in dir_path.glob("*.docx"):
        try:
            doc = Document(str(file))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            content = "\n".join(paragraphs)
            data[file.name] = content
        except Exception as e:  # pragma: no cover - informational
            data[file.name] = f"<Error reading document: {e}>"
    return data


@st.cache_data(show_spinner=False)
def aggregate_text(docs: Dict[str, str]) -> str:
    """Concatenate all document text for downstream processing."""
    return " \n ".join(docs.values())
