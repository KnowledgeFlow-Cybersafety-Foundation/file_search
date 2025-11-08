import re
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from text_processing import summarize_text


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


def render_results(
    matches: List[Tuple[str, str]],
    search_term: str,
    docs: Dict[str, str] | None = None,
    documents_dir: Path | None = None,
) -> None:
    """Render search results with summary, highlighted snippets, and download button.

    - docs: mapping of filename -> full text used for summary (optional but recommended)
    - documents_dir: folder path to load the .docx for download button (optional)
    """
    st.markdown(f"### Results for '{search_term}' ({len(matches)} document(s))")
    if not matches:
        st.info("No documents matched your search term.")
        return
    for doc_name, snippet in matches:
        highlighted = re.sub(
            rf"\\b({re.escape(search_term)})\\b", r"**\\1**", snippet, flags=re.IGNORECASE
        )
        with st.expander(doc_name):
            # Summary
            full_text = docs.get(doc_name) if docs else None
            if full_text:
                summary = summarize_text(full_text)
                if summary:
                    st.markdown("**Summary**")
                    st.write(summary)
            # Snippet with highlights
            st.markdown("**Match snippet**")
            st.write(highlighted)
            # Download button
            if documents_dir is not None:
                file_path = documents_dir / doc_name
                if file_path.exists():
                    with file_path.open("rb") as f:
                        data = f.read()
                    st.download_button(
                        label="Download document",
                        data=data,
                        file_name=doc_name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                else:
                    st.caption("File not found for download.")
