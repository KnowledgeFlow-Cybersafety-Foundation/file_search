import re
from typing import List, Tuple

import streamlit as st


def render_top_terms(wc, extract_top_words_fn, columns: int = 6) -> str | None:
    """Render top terms as clickable buttons and return the selected word if any."""
    st.subheader("Top Terms")
    cols = st.columns(columns)
    selected_word = st.session_state.get("selected_word")
    for i, (word, _score) in enumerate(extract_top_words_fn(wc)):
        col = cols[i % len(cols)]
        if col.button(word):
            st.session_state["selected_word"] = word
            selected_word = word
    return selected_word


def render_results(matches: List[Tuple[str, str]], search_term: str) -> None:
    """Render search results with highlighted terms inside expandable sections."""
    st.markdown(f"### Results for '{search_term}' ({len(matches)} document(s))")
    if not matches:
        st.info("No documents matched your search term.")
        return
    for doc_name, snippet in matches:
        highlighted = re.sub(
            rf"\\b({re.escape(search_term)})\\b", r"**\\1**", snippet, flags=re.IGNORECASE
        )
        with st.expander(doc_name):
            st.write(highlighted)
