# Document Search Dashboard

A Streamlit-based application for searching through Word documents with an intuitive web interface.

## New Minimal Word Cloud Search Mode

This repository now includes a simplified Streamlit app that:

- Loads all `.docx` files from `src/documents/`
- Generates a word cloud of the most frequent words
- Lets you click on top words or manually type a term to search documents
- Shows contextual snippets for each matching document

The minimal interface lives in `src/app.py` (default run target).

## Quick Start

### Local Development

1. Install Python 3.11+ and dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app (word cloud search):
```bash
streamlit run src/app.py
```

3. Open your browser to `http://localhost:8501`
