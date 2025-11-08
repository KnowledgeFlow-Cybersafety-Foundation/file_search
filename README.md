# Document Search Dashboard

A Streamlit-based application for searching through Word documents with an intuitive web interface.

## Features

The Streamlit app provides:

- Bulk loading of `.docx` files from `src/documents/`
- Word cloud visualization of most frequent terms
- Clickable top-word tags to auto-populate the search box
- Free-text search across all loaded documents with contextual snippets
- Caching for faster repeated interactions (document load + wordcloud build)
- Per-result document summary and direct download button


## Modular Structure

The original monolithic `app.py` has been refactored for clarity and reuse:

```
src/
	app.py              # Streamlit UI composition
	config.py           # Configuration constants (paths, limits, page title)
	data_loader.py      # Document loading + aggregation helpers
	text_processing.py  # Text cleanup, word cloud generation, top word extraction
		ui_components.py    # Reusable UI fragments (top terms, search results)
	search.py           # Search/matching logic for term snippets
	documents/          # Place your .docx files here
```

You can import and reuse logic outside Streamlit, e.g. for tests or CLI tools:

```python
from config import DOCUMENTS_DIR
from data_loader import load_documents
from text_processing import clean_text, build_wordcloud, extract_top_words
from search import find_matches
```

## Quick Start

### Local Development

1. Install Python 3.11+ and dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run src/app.py
```

3. Open your browser to `http://localhost:8501`

## Testing Core Logic (Optional)

Without starting Streamlit you can smoke-test the pipeline:

```bash
python - <<'PY'
import sys; sys.path.append('src')
from data_loader import load_documents, aggregate_text
from text_processing import clean_text, build_wordcloud, extract_top_words
from search import find_matches
docs = load_documents()
text = clean_text(aggregate_text(docs))
wc = build_wordcloud(text)
print('Top words:', extract_top_words(wc)[:5])
print('Search sample:', find_matches(docs, extract_top_words(wc)[0][0])[:1])
PY
```

## Next Ideas

- Add unit tests under `tests/` for pure functions
- Expose a REST API variant using FastAPI for integration
- Support additional file types (`.txt`, `.pdf`)
- Improve summarization (e.g., use NLP library for semantic summaries)
