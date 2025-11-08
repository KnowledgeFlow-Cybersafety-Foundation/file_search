import re
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from docx import Document
from wordcloud import STOPWORDS, WordCloud

# --------- Configuration ---------
DOCUMENTS_DIR = Path(__file__).parent / "documents"
TOP_N_WORDS = 50
MIN_WORD_LENGTH = 3

# --------- Helpers ---------


def clean_text(text: str) -> str:
    # Basic cleanup: lowercase, remove punctuation except intra-word hyphens
    text = text.lower()
    text = re.sub(r"[\n\r]", " ", text)
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_data(show_spinner=False)
def load_documents(dir_path: Path = DOCUMENTS_DIR) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not dir_path.exists():
        return data
    for file in dir_path.glob("*.docx"):
        try:
            doc = Document(str(file))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            content = "\n".join(paragraphs)
            data[file.name] = content
        except Exception as e:
            data[file.name] = f"<Error reading document: {e}>"
    return data


@st.cache_data(show_spinner=False)
def aggregate_text(docs: Dict[str, str]) -> str:
    return " \n ".join(docs.values())


@st.cache_resource(show_spinner=False)
def build_wordcloud(text: str) -> WordCloud:
    stopwords = set(STOPWORDS)
    # Domain-specific additions (can tune later)
    stopwords.update({"the", "and", "for", "with", "that", "this"})
    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        stopwords=stopwords,
        collocations=False,
        min_word_length=MIN_WORD_LENGTH,
    )
    wc.generate(text)
    return wc


def extract_top_words(wc: WordCloud, limit: int = TOP_N_WORDS) -> List[Tuple[str, int]]:
    # word_frequencies already sorted descending by frequency
    items = wc.words_.items()  # word: relative freq (0-1)
    # Convert relative freq to an int score for display
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)[:limit]
    # Scale to approximate counts (optional); just keep relative for simplicity
    return [(w, int(freq * 1000)) for w, freq in sorted_items]


def find_matches(docs: Dict[str, str], term: str) -> List[Tuple[str, str]]:
    term_lower = term.lower()
    results = []
    pattern = re.compile(rf"(.{{0,120}}\b{re.escape(term_lower)}\b.*?\. )", re.IGNORECASE)
    for name, content in docs.items():
        if term_lower in content.lower():
            # Get first matching snippet
            match = pattern.search(content.replace("\n", " "))
            snippet = (
                match.group(0).strip()
                if match
                else content[:160] + ("..." if len(content) > 160 else "")
            )
            results.append((name, snippet))
    return results


# --------- UI ---------


st.set_page_config(page_title="Document Word Cloud Search", layout="wide")
st.title("📄 Document Word Cloud Search")

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
st.subheader("Top Terms")
cols = st.columns(6)
selected_word = st.session_state.get("selected_word")

for i, (word, score) in enumerate(extract_top_words(wc)):
    col = cols[i % len(cols)]
    if col.button(word):
        st.session_state["selected_word"] = word
        selected_word = word

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
    st.markdown(f"### Results for '{search_term}' ({len(matches)} document(s))")
    if matches:
        for doc_name, snippet in matches:
            highlighted = re.sub(
                rf"\b({re.escape(search_term)})\b", r"**\1**", snippet, flags=re.IGNORECASE
            )
            with st.expander(doc_name):
                st.write(highlighted)
    else:
        st.info("No documents matched your search term.")
else:
    st.info("Select a word from the cloud above or enter a search term.")

st.markdown("---")
st.caption(
    "Word cloud built from aggregated document text. Frequency score approximates relative importance."
)
