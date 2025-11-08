import re
from typing import Iterable, List, Tuple

import streamlit as st
from wordcloud import STOPWORDS, WordCloud

from config import DEFAULT_EXTRA_STOPWORDS, MIN_WORD_LENGTH, TOP_N_WORDS


def clean_text(text: str) -> str:
    """Basic cleanup: lowercase, remove punctuation except intra-word hyphens."""
    text = text.lower()
    text = re.sub(r"[\n\r]", " ", text)
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_stopwords(additional: Iterable[str] | None = None) -> set[str]:
    """Base STOPWORDS plus optional additional entries."""
    stopwords = set(STOPWORDS)
    stopwords.update(DEFAULT_EXTRA_STOPWORDS)
    if additional:
        stopwords.update([s.strip() for s in additional if s and s.strip()])
    return stopwords


@st.cache_resource(show_spinner=False)
def build_wordcloud(text: str, stopwords: Tuple[str, ...] | None = None) -> WordCloud:
    # Note: stopwords is a tuple for caching; converted to set internally.
    sw = set(stopwords) if stopwords else prepare_stopwords()
    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        stopwords=sw,
        collocations=False,
        min_word_length=MIN_WORD_LENGTH,
    )
    wc.generate(text)
    return wc


def extract_top_words(wc: WordCloud, limit: int = TOP_N_WORDS) -> List[Tuple[str, int]]:
    """Return top words as (word, approx_score) pairs from a generated WordCloud."""
    items = wc.words_.items()  # word: relative freq (0-1)
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)[:limit]
    return [(w, int(freq * 1000)) for w, freq in sorted_items]


def summarize_text(text: str, max_sentences: int = 2, max_chars: int = 400) -> str:
    """Lightweight extractive summary: take first non-empty sentences up to limits.

    Splits on period boundaries. This avoids external dependencies while giving a
    quick preview. If the text is already short, returns it unchanged.
    """
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    # Naive sentence split
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", cleaned) if s.strip()]
    selected: List[str] = []
    for s in sentences:
        if len(" ".join(selected + [s])) > max_chars:
            break
        selected.append(s)
        if len(selected) >= max_sentences:
            break
    summary = " ".join(selected).strip()
    return (
        summary
        if summary
        else cleaned[:max_chars].rstrip() + ("..." if len(cleaned) > max_chars else "")
    )
