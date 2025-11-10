# Document Word Cloud Search

A small Streamlit app to build a word cloud from multiple `.docx` files and search those documents with contextual snippets and simple extractive summaries.

## Quick start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run src/app.py
```

3. Open your browser at http://localhost:8501

## Features

- Load `.docx` files from `src/documents/`
- Build a word cloud and show top terms as clickable buttons
- Search documents with contextual snippets
- Per-document extractive summaries and direct download of source `.docx`

## Project layout

- `src/app.py` — Streamlit UI composition
- `src/data_loader.py` — document loading & aggregation
- `src/text_processing.py` — cleaning, wordcloud, summaries
- `src/search.py` — search & snippet extraction
- `src/ui_components.py` — reusable UI fragments
- `src/documents/` — place your `.docx` files here

## Notes

- Caching is used for document loading and wordcloud building for faster reruns.
- Summaries use a lightweight extractive approach and fall back if optional libraries are missing.
