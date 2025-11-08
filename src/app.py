import streamlit as st

from config import DOCUMENTS_DIR, PAGE_TITLE
from data_loader import aggregate_text, load_documents
from search import find_matches
from text_processing import build_wordcloud, clean_text, extract_top_words
from ui_components import render_results, render_top_terms

# --------- UI ---------


st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title("📄 " + PAGE_TITLE)

# Load documents
with st.spinner("Loading documents..."):
    documents = load_documents()

if not documents:
    st.warning(f"No .docx documents found in '{DOCUMENTS_DIR}'. Add files and refresh.")
    st.stop()

st.caption(f"Loaded {len(documents)} document(s) from '{DOCUMENTS_DIR}'.")

# Aggregate text and build word cloud
raw_text = aggregate_text(documents)
cleaned = clean_text(raw_text)

if not cleaned:
    st.error("Documents contained no usable text.")
    st.stop()

wc = build_wordcloud(cleaned)

# Display top words as clickable tags (buttons)
selected_word = render_top_terms(wc, extract_top_words)

# Search input
default_search = selected_word if selected_word else ""
search_term = st.text_input(
    "Search documents by term",
    value=default_search,
    placeholder="Enter a word from the cloud or any term...",
)

# Perform search
if search_term.strip():
    matches = find_matches(documents, search_term.strip())
    render_results(matches, search_term, docs=documents, documents_dir=DOCUMENTS_DIR)
else:
    st.info("Select a word from the cloud above or enter a search term.")

st.markdown("---")
st.caption(
    "Word cloud built from aggregated document text. Frequency score approximates relative importance."
)
